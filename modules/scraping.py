"""Scraping layer — Selenium-based HKEx scraping for ESG filings and company basics.

ChromeDriver is selected automatically:
  - Chromium is used when its binary is found on PATH (e.g. HF Spaces / Linux
    with packages.txt installing chromium-browser).
  - Standard Chrome is used otherwise (local Windows dev environment).
"""

import re
import shutil
import time
from datetime import datetime
from typing import Dict, List, Union

import pytz
import streamlit as st
from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException, TimeoutException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager
from webdriver_manager.core.os_manager import ChromeType

from modules.db import supabase


def normalize_market_cap(market_cap_str: str) -> float:
    """Convert a market cap string (e.g. "HK$1,234.5B") to billions as a float."""
    match = re.search(r"([\d,]+\.?\d*)\s*([BM])", market_cap_str.upper())
    if not match:
        raise ValueError(f"Could not parse market cap string: '{market_cap_str}'")
    value_str, suffix = match.groups()
    value = float(value_str.replace(",", ""))
    if suffix == "M":
        return round(value / 1000, 2)
    if suffix == "B":
        return round(value, 2)
    return 0.00


def init_chromedriver() -> webdriver.Chrome:
    """Initialize ChromeDriver, auto-selecting Chromium or Chrome based on PATH."""
    # "chromium" on Debian/Ubuntu; "chromium-browser" on some older distros.
    chromium_bin = shutil.which("chromium") or shutil.which("chromium-browser")
    if chromium_bin:
        service = Service(ChromeDriverManager(chrome_type=ChromeType.CHROMIUM).install())
    else:
        service = Service(ChromeDriverManager().install())

    options = Options()
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gcm")
    options.add_argument("--disable-notifications")
    options.add_experimental_option(
        "prefs",
        {"profile.default_content_setting_values.notifications": 2},
    )
    return webdriver.Chrome(service=service, options=options)


def edit_listco_info_title_search(
    driver: webdriver.Chrome,
    stock_code: str,
    esg_filings_only: bool = False,
) -> None:
    url = st.secrets.FILINGS_URL
    driver.get(url)

    stock_input = driver.find_element(By.ID, "searchStockCode")
    stock_input.clear()
    stock_input.send_keys(stock_code)
    # NOTE: visibility_of_element_located ensures element is present and visible for clicking
    autocomplete_suggestion = WebDriverWait(driver, 5).until(
        EC.visibility_of_element_located(
            (By.CSS_SELECTOR, "#autocomplete-list-0 table tr.autocomplete-suggestion.narrow")
        )
    )
    if autocomplete_suggestion.text == "View More":
        raise ValueError(
            "Please check your stock code and retry, as there is no autocomplete suggestion."
        )
    autocomplete_suggestion.click()

    search_type__all = WebDriverWait(driver, 5).until(
        EC.element_to_be_clickable(
            (By.CSS_SELECTOR, "a.combobox-field[data-value='rbAll']")
        )
    )
    search_type__all.click()
    search_type__headline = WebDriverWait(driver, 5).until(
        EC.element_to_be_clickable(
            (By.CSS_SELECTOR, "div.droplist-item[data-value='rbAfter2006']")
        )
    )
    search_type__headline.click()

    doc_type__all = WebDriverWait(driver, 5).until(
        EC.element_to_be_clickable(
            (By.CSS_SELECTOR, "#rbAfter2006 a.combobox-field[data-value='-2']")
        )
    )
    doc_type__all.click()
    doc_type__esg = WebDriverWait(driver, 5).until(
        EC.visibility_of_element_located(
            (By.CSS_SELECTOR, "#rbAfter2006 ul li[data-value='40000']")
        )
    )
    # Dropdown items are hidden until scrolled into the viewport; scrollIntoView
    # is required before click() or Selenium raises ElementNotInteractableError.
    driver.execute_script("arguments[0].scrollIntoView(true);", doc_type__esg)
    doc_type__esg.click()

    data_value = "40400" if esg_filings_only else "-2"
    doc_type__esg_all = WebDriverWait(driver, 5).until(
        EC.visibility_of_element_located(
            (
                By.CSS_SELECTOR,
                f"#rbAfter2006 ul li[data-value='40000'] ul li[data-value='{data_value}']",
            )
        )
    )
    doc_type__esg_all.click()

    search = driver.find_element(
        By.CSS_SELECTOR, "div.filter__buttonGroup a[class^=filter__btn-applyFilters-js]"
    )
    search.click()

    time.sleep(2)
    WebDriverWait(driver, 5).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, "#titleSearchResultPanel"))
    )


def get_last_updated_filings_at(stock_code: str) -> datetime | None:
    condition = st.session_state.control_df["stock_code"] == stock_code
    result_df = st.session_state.control_df[condition][["last_updated_filings_at"]]
    result = (
        result_df["last_updated_filings_at"].iloc[0] if not result_df.empty else None
    )
    return result


def get_earliest_release_time(driver: webdriver.Chrome) -> datetime | None:
    result_rows = driver.find_elements(
        By.CSS_SELECTOR, "#titleSearchResultPanel table tbody tr"
    )
    if result_rows:
        last_row = result_rows[-1]
        last_row_cells = last_row.find_elements(By.TAG_NAME, "td")
        earliest_release_time_str = last_row_cells[0].text
        return pytz.timezone("Asia/Hong_Kong").localize(
            datetime.strptime(earliest_release_time_str, "%d/%m/%Y %H:%M")
        )
    return None


def load_more_records(driver: webdriver.Chrome) -> None:
    load_more = WebDriverWait(driver, 5).until(
        EC.element_to_be_clickable(
            (
                By.CSS_SELECTOR,
                (
                    "#recordCountPanel2 div.component-loadmore__dropdown-container"
                    " ul a[href='javascript:loadMore();']"
                ),
            )
        )
    )
    driver.execute_script("arguments[0].scrollIntoView(true);", load_more)
    time.sleep(2)
    load_more.click()


def scrape(stock_code: str, *, save_to_db: bool = True) -> None:
    """Scrape HKEx website and extract key filings."""
    driver = init_chromedriver()
    data_lst: List[Dict[str, Union[datetime, str]]] = []
    try:
        for search_option in [True, False]:
            edit_listco_info_title_search(driver, stock_code, esg_filings_only=search_option)
            result_rows = driver.find_elements(
                By.CSS_SELECTOR, "#titleSearchResultPanel table tbody tr"
            )
            if result_rows:
                break

        last_updated_at = get_last_updated_filings_at(stock_code)
        while True:
            if last_updated_at is not None:
                earliest_release_time = get_earliest_release_time(driver)
                if (earliest_release_time is not None) and (
                    earliest_release_time <= last_updated_at
                ):
                    break
            try:
                load_more_records(driver)
            except (TimeoutException, NoSuchElementException):
                break
        time.sleep(2)

        result_rows = driver.find_elements(
            By.CSS_SELECTOR, "#titleSearchResultPanel table tbody tr"
        )

        # Stamp created_at explicitly
        now_utc = datetime.now(pytz.UTC).isoformat()
        for row in result_rows:
            cells = row.find_elements(By.TAG_NAME, "td")
            release_time_str = cells[0].text
            release_time = pytz.timezone("Asia/Hong_Kong").localize(
                datetime.strptime(release_time_str, "%d/%m/%Y %H:%M")
            )
            # Results are ordered newest-first; once we reach a filing already in
            # the DB, all subsequent rows are also already stored — stop early.
            if (last_updated_at is not None) and (last_updated_at > release_time):
                break
            doc_cell = cells[3].find_element(By.CSS_SELECTOR, "div.doc-link a")
            doc_title = doc_cell.text
            doc_url = doc_cell.get_attribute("href")
            data_lst.append(
                {
                    "stock_code": stock_code,
                    "release_time": release_time.isoformat(),
                    "title": doc_title,
                    "url": doc_url,
                    "created_at": now_utc,
                }
            )
    finally:
        driver.quit()

    if save_to_db:
        if data_lst:
            # URL is the natural dedup key; ignore_duplicates makes re-scraping safe.
            supabase.table("esg_filings").upsert(
                data_lst, ignore_duplicates=True, on_conflict="url"
            ).execute()
        supabase.table("control").update(
            {"last_updated_filings_at": datetime.now(pytz.UTC).isoformat()}
        ).eq("stock_code", stock_code).execute()


def get_company_basics(stock_code: str, *, save_to_db: bool = True) -> None:
    """Scrape market cap and sector from HKEx company page."""
    driver = init_chromedriver()

    url = st.secrets.BASICS_URL.format(int(stock_code))
    driver.get(url)
    time.sleep(2)

    company_name_element = WebDriverWait(driver, 10).until(
        EC.visibility_of_element_located((By.CSS_SELECTOR, "h1[class='col_longname']"))
    )
    match = re.search(r"^(.*?)\s*\(\d+\)", company_name_element.text)
    company_name = match.group(1).strip() if match else None

    hsic_element = WebDriverWait(driver, 10).until(
        EC.visibility_of_element_located(
            (By.CSS_SELECTOR, "span[class='col_industry_hsic']")
        )
    )
    sector = hsic_element.text.split(" - ")[0]

    market_cap_element = WebDriverWait(driver, 10).until(
        EC.visibility_of_element_located((By.CSS_SELECTOR, "dt.ico_data.col_mktcap"))
    )
    market_cap = normalize_market_cap(market_cap_element.text)

    driver.quit()

    if save_to_db:
        # .or_() guard: only overwrite if currently empty, preserving any
        # manual corrections a user may have made via the data editor.
        supabase.table("control").update({"name": company_name}).eq(
            "stock_code", stock_code
        ).or_('name.is.null, name.eq.""').execute()
        supabase.table("control").update({"sector": sector}).eq(
            "stock_code", stock_code
        ).or_('sector.is.null, sector.eq.""').execute()
        supabase.table("control").update(
            {
                "market_cap": market_cap,
                "last_updated_market_cap_at": datetime.now(pytz.UTC).isoformat(),
            }
        ).eq("stock_code", stock_code).execute()
