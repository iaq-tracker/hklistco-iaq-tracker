"""
App deployed on Streamlit Community Cloud
"""

from datetime import datetime
from datetime import timedelta
from email.message import EmailMessage
from email.policy import default
import re
import time
from typing import List
from typing import Dict
from typing import Union
from io import BytesIO
import random

from dateutil import parser
from google import genai
from google.genai import types
import pandas as pd
from pandas.api.types import (
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
)
from pydantic import BaseModel
import pytz
import sqlalchemy.exc
import streamlit as st
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from selenium.common.exceptions import NoSuchElementException
from supabase import create_client
from webdriver_manager.chrome import ChromeDriverManager
from webdriver_manager.core.os_manager import ChromeType


# -------------------------- CSS for Navigation Bar -------------------------- #
st.markdown(
    """
    <style>
        /* Container for radio group */
        div.stRadio > div[role="radiogroup"] {
            flex-direction: row;
            border-bottom: 1px solid rgba(255, 255, 255, 0.2);
            margin-bottom: 20px;
            flex-wrap: nowrap;
            overflow-x: auto;
        }
        /* Hide radio circles */
        div.stRadio > div[role="radiogroup"] > label[data-baseweb="radio"] > div:first-child {
            display: none !important;
        }
        /* Style individual tabs (labels) */
        div.stRadio > div[role="radiogroup"] > label[data-baseweb="radio"] > div {
            padding: 10px 20px;
            border: 1px solid rgba(255, 255, 255, 0.2);
            border-bottom: none;
            background-color: var(--secondary-background-color, #333333);
            color: var(--text-color, #ffffff);
            cursor: pointer;
            margin-right: -1px;
            border-radius: 4px 4px 0 0;
            transition: background-color 0.3s;
            white-space: nowrap;
        }
        /* Hover effect for unselected tabs */
        div.stRadio > div[role="radiogroup"] > label[data-baseweb="radio"] > div:hover {
            background-color: rgba(255, 255, 255, 0.1);
        }
        /* Selected tab: Force primary color */
        div.stRadio > div[role="radiogroup"] > label[data-baseweb="radio"] > input[type="radio"]:checked + div {
            background-color: var(--primary-color, #ff4b4b) !important;
            border-bottom: 1px solid var(--primary-color, #ff4b4b) !important;
            color: #ffffff !important;
            font-weight: bold;
        }
    </style>
""",
    unsafe_allow_html=True,
)


# ------------------------------- DB Connection ------------------------------ #
@st.cache_resource
def init_connection():
    """
    Initialize connection to Supabase database.
    """
    url = st.secrets.connections.supabase.SUPABASE_URL
    key = st.secrets.connections.supabase.SUPABASE_KEY
    return create_client(url, key)


supabase = init_connection()


# ---------------------------------------------------------------------------- #
#                                     Utils                                    #
# ---------------------------------------------------------------------------- #
class Contact(BaseModel):
    """
    Schema used in LLM as structured output for contact details.
    """

    email: str
    name: str | None
    tel: str | None
    title: str | None
    citations: list[str]


class Grade(BaseModel):
    """
    Schema used in LLM as structured output for IAQ gradings.
    """

    grade: str
    overview: str
    justification: str
    extracts: str


# ---------------------------------------------------------------------------- #
#                                   Services                                   #
# ---------------------------------------------------------------------------- #
def get_stock_codes_tbu(
    *,
    update_market_cap: bool = False,
    update_filings: bool = False,
    update_contacts: bool = False,
    update_before: datetime | None = None,
) -> list[str]:
    """
    Get list of stock codes for which company basics, ESG filings, or IR contacts require updating.
    Optional: set update_before to get stock codes updated before a specific date
    """
    # Validate inputs
    if not (update_market_cap or update_filings or update_contacts):
        msg = "At least one of update_market_cap, update_filings, or update_contacts must be True"
        raise ValueError(msg)

    # Determine the field to filter
    if update_market_cap:
        field = "last_updated_market_cap_at"
    elif update_filings:
        field = "last_updated_filings_at"
    else:
        field = "last_updated_contacts_at"

    # Create condition to select rows where field is NULL
    condition = st.session_state.control_df[field].isna()

    # Add condition to include rows where field is before update_before
    if update_before:
        # Convert to the timezone-aware pandas Timestamp
        update_before = pd.to_datetime(update_before, utc=True)
        condition |= st.session_state.control_df[field] <= update_before

    # Query dataframe
    result_df = st.session_state.control_df[condition][["stock_code"]]
    return result_df["stock_code"].dropna().tolist()


def get_llm_client() -> genai.Client:
    """
    Set up Gemini API client
    """
    # Set up counter in session state
    if "api_key_counter" not in st.session_state:
        st.session_state.api_key_counter = 0

    # Count number of api keys available
    api_keys = st.secrets.GEMINI_API_KEYS
    api_keys_count = len(api_keys)

    # Rotate and return api key
    client = genai.Client(
        api_key=api_keys[
            (st.session_state.api_key_counter + random.randint(1, api_keys_count))
            % api_keys_count
        ]
    )

    # Update api key counter
    st.session_state.api_key_counter += 1

    return client


def get_citations(response: types.GenerateContentResponse) -> list[str]:
    """
    Extract citations from prompt response.
    """
    citations = []
    if response.candidates:
        if metadata := response.candidates[0].grounding_metadata:
            chunks = getattr(metadata, "grounding_chunks")
            if chunks:
                for chunk in chunks:
                    citations.append(chunk.web.uri)
    return citations


def embed_citations(response: types.GenerateContentResponse) -> str:
    """
    Embed citations to prompt response
    """
    text = response.text
    supports = response.candidates[0].grounding_metadata.grounding_supports
    chunks = response.candidates[0].grounding_metadata.grounding_chunks

    if (supports is not None) and (chunks is not None):
        # Sort supports by end_index in descending order to avoid shifting issues when inserting.
        sorted_supports = sorted(
            supports, key=lambda s: s.segment.end_index, reverse=True
        )

        for support in sorted_supports:
            end_index = support.segment.end_index
            if support.grounding_chunk_indices:
                # Create citation string like [1](link1)[2](link2)
                citation_links = []
                for i in support.grounding_chunk_indices:
                    if i < len(chunks):
                        uri = chunks[i].web.uri
                        citation_links.append(f"[{i + 1}]({uri})")

                citation_string = "\n".join(citation_links)
                text = text[:end_index] + citation_string + text[end_index:]
    return text


def filter_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a UI on top of a dataframe to let viewers filter columns.
    """
    modify = st.checkbox("Add Filters")
    if not modify:
        return df
    df = df.copy()
    # Try to convert datetimes into a standard format (datetime, no timezone)
    for col in df.columns:
        if is_object_dtype(df[col]):
            try:
                df[col] = pd.to_datetime(df[col])
            except Exception:
                pass
        if is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.tz_localize(None)
    modification_container = st.container()
    with modification_container:
        to_filter_columns = st.multiselect("Filter dataframe on", df.columns)
        for column in to_filter_columns:
            left, right = st.columns((1, 20))
            # Treat columns with < 10 unique values as categorical
            if is_categorical_dtype(df[column]) or df[column].nunique() < 10:
                user_cat_input = right.multiselect(
                    f"Values for {column}",
                    df[column].unique(),
                    default=list(df[column].unique()),
                )
                df = df[df[column].isin(user_cat_input)]
            elif is_numeric_dtype(df[column]):
                _min = float(df[column].min())
                _max = float(df[column].max())
                step = (_max - _min) / 100
                user_num_input = right.slider(
                    f"Values for {column}",
                    min_value=_min,
                    max_value=_max,
                    value=(_min, _max),
                    step=step,
                )
                df = df[df[column].between(*user_num_input)]
            elif is_datetime64_any_dtype(df[column]):
                user_date_input = right.date_input(
                    f"Values for {column}",
                    value=(
                        df[column].min(),
                        df[column].max(),
                    ),
                )
                if len(user_date_input) == 2:
                    user_date_input = tuple(map(pd.to_datetime, user_date_input))
                    start_date, end_date = user_date_input
                    df = df.loc[df[column].between(start_date, end_date)]
            else:
                user_text_input = right.text_input(
                    f"Substring or regex in {column}",
                )
                if user_text_input:
                    df = df[df[column].astype(str).str.contains(user_text_input)]
    return df


def normalize_market_cap(market_cap_str: str) -> float:
    """
    Converts a market cap string (e.g., "HK$1,234.5B" or "567.8M")
    into a float representing the value in billions.

    Args:
        market_cap_str: The market cap string from the website.

    Returns:
        The market cap value in billions as a float.
    """
    # Clean the string: remove currency, commas, and whitespace
    match = re.search(r"([\d,]+\.?\d*)\s*([BM])", market_cap_str.upper())
    if not match:
        raise ValueError(f"Could not parse market cap string: '{market_cap_str}'")

    # Convert the numeric part to a float
    value_str, suffix = match.groups()
    value = float(value_str.replace(",", ""))

    # Convert to billions if necessary
    if suffix == "M":
        return round(value / 1000, 2)
    if suffix == "B":
        return round(value, 2)
    return 0.00


def get_company_basics(
    stock_code: str,
    *,
    save_to_db: bool = True,
) -> None:
    """
    Extract market cap
    """
    # ----------------------- Step 1 - set up chromedriver ----------------------- #
    service = Service(ChromeDriverManager(chrome_type=ChromeType.CHROMIUM).install())
    options = Options()
    options.add_argument("--window-size=1920,1080")  # set window size
    options.add_argument("--headless")  # headless mode
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gcm")  # disable GCM registration
    options.add_argument("--disable-notifications")  # disable push notification
    options.add_experimental_option(
        "prefs",
        {
            "profile.default_content_setting_values.notifications": 2  # Block notifications
        },
    )
    driver = webdriver.Chrome(service=service, options=options)

    # ---------- Step 2: visit "Listed Company Information Title Search" --------- #
    url = st.secrets.BASICS_URL.format(int(stock_code))
    driver.get(url)

    time.sleep(2)

    company_name_element = WebDriverWait(driver, 10).until(
        EC.visibility_of_element_located((By.CSS_SELECTOR, "h1[class='col_longname']"))
    )
    match = re.search(r"^(.*?)\s*\(\d+\)", company_name_element.text)
    if match:
        company_name = match.group(1).strip()
    else:
        company_name = None

    hsic_element = WebDriverWait(driver, 10).until(
        EC.visibility_of_element_located(
            (By.CSS_SELECTOR, "span[class='col_industry_hsic']")
        )
    )
    hsic: str = hsic_element.text
    sector = hsic.split(" - ")[0]

    market_cap_element = WebDriverWait(driver, 10).until(
        EC.visibility_of_element_located((By.CSS_SELECTOR, "dt.ico_data.col_mktcap"))
    )
    market_cap = normalize_market_cap(market_cap_element.text)

    driver.quit()

    if save_to_db:
        # add company name
        supabase.table("control").update({"name": company_name}).eq(
            "stock_code", stock_code
        ).or_('name.is.null, name.eq.""').execute()
        # add hsic
        supabase.table("control").update({"sector": sector}).eq(
            "stock_code", stock_code
        ).or_('sector.is.null, sector.eq.""').execute()
        # add market cap
        supabase.table("control").update(
            {
                "market_cap": market_cap,
                "last_updated_market_cap_at": datetime.now(pytz.UTC).isoformat(),
            }
        ).eq("stock_code", stock_code).execute()


# -------------------------------- ESG Filings ------------------------------- #
def get_last_updated_filings_at(stock_code: str) -> datetime | None:
    """
    Get the timestamp of last updated at for a stock code.
    """
    # Filter control_df for the given stock_code and select last_updated_filings_at
    condition = st.session_state.control_df["stock_code"] == stock_code
    result_df = st.session_state.control_df[condition][["last_updated_filings_at"]]

    # Extract the first value if the DataFrame is not empty, else return None
    result = (
        result_df["last_updated_filings_at"].iloc[0] if not result_df.empty else None
    )
    return result


def get_earliest_release_time(driver: webdriver.Chrome) -> datetime | None:
    """
    Get the release time of the earliest record displayed in results page of HKEx website.
    """
    result_rows = driver.find_elements(
        By.CSS_SELECTOR,
        "#titleSearchResultPanel table tbody tr",
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
    """
    Load additional 100 records on results page if available.
    """
    # check if "LOAD MORE" button is present
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
    # scroll to the button to ensure it is in view
    driver.execute_script("arguments[0].scrollIntoView(true);", load_more)
    time.sleep(2)
    # click on "LOAD MORE"
    load_more.click()


def scrape(
    stock_code: str,
    *,
    save_to_db: bool = True,
) -> None:
    """
    Scrape HKEx website and extract key filings.
    """
    # ----------------------- Step 1 - set up chromedriver ----------------------- #
    service = Service(ChromeDriverManager(chrome_type=ChromeType.CHROMIUM).install())
    options = Options()
    options.add_argument("--window-size=1920,1080")  # set window size
    options.add_argument("--headless")  # headless mode
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gcm")  # disable GCM registration
    options.add_argument("--disable-notifications")  # disable push notification
    options.add_experimental_option(
        "prefs",
        {
            "profile.default_content_setting_values.notifications": 2  # Block notifications
        },
    )
    driver = webdriver.Chrome(service=service, options=options)

    # ---------- Step 2: visit "Listed Company Information Title Search" --------- #
    url = st.secrets.FILINGS_URL
    driver.get(url)

    # a) enter Stock Code
    stock_input = driver.find_element(By.ID, "searchStockCode")
    stock_input.clear()
    stock_input.send_keys(stock_code)
    # wait till autocomplete suggestion for stock code appears
    # NOTE: visibility_of_element_located ensures element is present and visible for clicking
    autocomplete_suggestion = WebDriverWait(driver, 5).until(
        EC.visibility_of_element_located(
            (
                By.CSS_SELECTOR,
                "#autocomplete-list-0 table tr.autocomplete-suggestion.narrow",
            )
        )
    )
    if autocomplete_suggestion.text == "View More":
        msg = "Please check your stock code and retry, as there is no autocomplete suggestion."
        raise ValueError(msg)
    # click on autocomplete suggestion
    autocomplete_suggestion.click()

    # b) choose Headline Category
    # click on "ALL" under Search Type
    search_type__all = WebDriverWait(driver, 5).until(
        EC.element_to_be_clickable(
            (By.CSS_SELECTOR, "a.combobox-field[data-value='rbAll']")
        )
    )
    search_type__all.click()
    # click on "Headline Category" under Search Type
    search_type__headline = WebDriverWait(driver, 5).until(
        EC.element_to_be_clickable(
            (By.CSS_SELECTOR, "div.droplist-item[data-value='rbAfter2006']")
        )
    )
    search_type__headline.click()

    # c) choose Document Type
    # click on "ALL" under Document Type
    doc_type__all = WebDriverWait(driver, 5).until(
        EC.element_to_be_clickable(
            (By.CSS_SELECTOR, "#rbAfter2006 a.combobox-field[data-value='-2']")
        )
    )
    doc_type__all.click()
    # click on "Financial Statements/ESG Information" under Document Type
    doc_type__esg = WebDriverWait(driver, 5).until(
        EC.visibility_of_element_located(
            (By.CSS_SELECTOR, "#rbAfter2006 ul li[data-value='40000']")
        )
    )
    driver.execute_script("arguments[0].scrollIntoView(true);", doc_type__esg)
    doc_type__esg.click()
    # then click on "ALL"
    doc_type__esg_all = WebDriverWait(driver, 5).until(
        EC.visibility_of_element_located(
            (
                By.CSS_SELECTOR,
                "#rbAfter2006 ul li[data-value='40000'] ul li[data-value='40400']",
            )
        )
    )
    doc_type__esg_all.click()

    # d) search result
    search = driver.find_element(
        By.CSS_SELECTOR, "div.filter__buttonGroup a[class^=filter__btn-applyFilters-js]"
    )
    search.click()

    # ------------------ Step 3: loop through available reports ------------------ #
    time.sleep(2)

    # a) wait till results table appears
    WebDriverWait(driver, 5).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, "#titleSearchResultPanel"))
    )

    # b) load more records if i) there is no last_updated_at
    # or ii) earliest_release_time is after last_updated_at
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

    # c) locate results table and extract key data
    result_rows = driver.find_elements(
        By.CSS_SELECTOR,
        "#titleSearchResultPanel table tbody tr",
    )

    # loop through each row
    data_lst: List[Dict[str, Union[datetime, str]]] = []
    for row in result_rows:
        cells = row.find_elements(By.TAG_NAME, "td")
        # extract and convert release_time
        release_time_str = cells[0].text
        release_time = pytz.timezone("Asia/Hong_Kong").localize(
            datetime.strptime(release_time_str, "%d/%m/%Y %H:%M")
        )
        # break if release_time is after last_updated_at
        if (last_updated_at is not None) and (last_updated_at > release_time):
            break
        # extract title and url of Document
        doc_cell = cells[3].find_element(By.CSS_SELECTOR, "div.doc-link a")
        doc_title = doc_cell.text
        doc_url = doc_cell.get_attribute("href")
        # add key data to list
        data_lst.append(
            {
                "stock_code": stock_code,
                "release_time": release_time.isoformat(),
                "title": doc_title,
                "url": doc_url,
            }
        )
    # close browser
    driver.quit()

    # d) save key data to esg_filings tab
    if save_to_db:
        if data_lst:
            supabase.table("esg_filings").upsert(
                data_lst,
                ignore_duplicates=True,
                on_conflict="url",
            ).execute()
        # update last_updated_filings_at and company name in control table
        supabase.table("control").update(
            {"last_updated_filings_at": datetime.now(pytz.UTC).isoformat()}
        ).eq("stock_code", stock_code).execute()


# -------------------------------- IAQ Grading ------------------------------- #
def grade_iaq(
    stock_code: str,
    *,
    save_to_db: bool = True,
) -> str:
    """
    Grade IAQ discloures of listed company in its ESG reports by LLM
    NOTE: As of 20 Aug 2025, url context can only process up to 20 URLs
    per request. And the maximum size for content retrieved from a
    single URL is 34MB
    See: https://ai.google.dev/gemini-api/docs/url-context
    NOTE: As of 29 Aug 2025, Gemini 2.5 Pro has a 1 million token context
    window (2 million coming soon), roughly 8 average length English novels.
    Trial and error shows that it is enough for 10-15 filings at a time.
    See: https://ai.google.dev/gemini-api/docs/long-context
    """
    # Get stock code, company name from control df
    company_name = get_company_name(stock_code=stock_code) or ""

    # Get all filings from esg_filings df
    condition = st.session_state.esg_filings_df["stock_code"] == stock_code
    filings_df = st.session_state.esg_filings_df[condition][
        ["title", "url", "release_time"]
    ].sort_values(by="release_time", ascending=False)

    # Raise if no filings found
    if filings_df.empty:
        raise ValueError(f"No filings found in database for {stock_code}")

    # Init list to store responses
    responses = ""

    # Chunk data and process in batches of 10
    chunk_size = 10
    for i in range(0, len(filings_df), chunk_size):
        chunk_df = filings_df.iloc[i : i + chunk_size]
        filings = "\n".join(
            [f"{row['title']}: {row['url']}" for _, row in chunk_df.iterrows()]
        )

        # Create prompt
        prompt = f"""You are an expert ESG analyst specializing in evaluating corporate disclosures for Hong Kong listed companies under the Hong Kong Stock Exchange (HKEX) ESG reporting guidelines.
        Your task is to evaluate the ESG disclosures of the company {company_name} with a stock ticker of '{stock_code}' specifically on the topic of indoor air quality (IAQ). This includes any mentions of IAQ management, monitoring, policies, risks, mitigation strategies, emissions (e.g., VOCs, PM2.5, CO2 levels), ventilation systems, employee health impacts, building certifications (e.g., BEAM Plus, LEED), or related initiatives in its operation.
        You are provided with below list of URLs to all of the company's ESG filings published onHKEx. Read content from these URLs, then extract and summarize only the sections relevant to indoor air quality.
        {filings}
        Evaluation Criteria:
        Focus solely on indoor air quality disclosures. Grade based on:

        # Length and Detail: Short/vague mentions (e.g., one sentence) vs. dedicated sections with explanations, data, and examples.
        # Key Performance Indicators (KPIs): Presence of quantifiable metrics (e.g., IAQ monitoring results, reduction targets for pollutants, compliance rates with standards like Hong Kong IAQ Objectives).
        # Consistency: How regularly KPIs are reported over time; improvements or expansions in disclosure (e.g., adding new metrics or deeper analysis in recent years).
        # Progression: Emphasis on the last three years to assess if disclosure has improved.

        # Grading Scale:
        # Use a three-tier grading system. Assign one grade only, with a brief justification tied to the criteria above. Buckets:

        # Low (No or Minimal Disclosure): Little to no mention of IAQ across reports; generic statements without details, KPIs, or data; no evidence of consistency or improvement.
        # Medium (Emerging Disclosure): Basic mentions starting or increasing in the recent three years; some details or initial KPIs introduced recently, but inconsistent reporting or limited depth/trends.
        # High (Strong Disclosure): Comprehensive, detailed sections on IAQ with multiple KPIs disclosed consistently over years; evidence of adding more information (e.g., year-over-year data, targets, audits) and progressive improvements in recent reports.

        # Output Format:
        # Structure your response as follows:
        # Company Overview: Brief 1-2 sentence summary of the company's business and why IAQ might be material (e.g., based on industry like real estate or hospitality).
        # Key Extracts: Bullet points of relevant IAQ excerpts from each report (cite year and URL).
        # Grade: [Low/Medium/High]
        # Justification for grade: Explanation in 3-5 sentences.
        # """

        client = get_llm_client()
        # Send prompt
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                tools=[
                    {"url_context": {}},
                ],
                temperature=0.3,
            ),
        )

        # Raise if response is null
        if not response.text:
            msg = "Null value in response text."
            raise ValueError(msg)

        if save_to_db:
            citations = get_citations(response)
            supabase.table("llm_logs").insert(
                {
                    "stock_code": stock_code,
                    "prompt": prompt,
                    "response": response.text,
                    "citations": "\n".join(citations) if citations else "None",
                }
            ).execute()

        responses += response.text
    return responses


def format_grading(
    stock_code: str,
    response_text: str,
    *,
    save_to_db: bool = True,
) -> Grade:
    """
    Format text response to specified JSON schema.
    """
    client = get_llm_client()
    prompt = f"""You are an expert data extraction tool. Your task is to analyze the text provided below and extract key details from one or more ESG grading report(s) prepared by an expert ESG analyst.

    From the text, identify the following for each report found:
    - grade: make sure it is one of [Low/Medium/High]. prioritize grade that are based on more recent ESG filings
    - company overview
    - key extracts
    - Justification for grade

    The output must be a JSON object of the grade.

    **Text to Analyze:**
    ---
    {response_text}
    ---

    **Expected JSON Output:**
    """
    response = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.0,
            response_mime_type="application/json",
            response_schema=Grade,
        ),
    )
    grade: Grade = response.parsed

    if save_to_db and grade:
        # Add to iaq_gradings table
        supabase.table("iaq_gradings").insert(
            {
                "stock_code": stock_code,
                "grade": grade.grade,
                "overview": grade.overview,
                "justification": grade.justification,
                "extracts": grade.extracts,
                "grading_date": datetime.now(pytz.UTC).isoformat(),
            },
        ).execute()
        # Update iaq_grade and last_updated_grade_at in control table
        supabase.table("control").update(
            {
                "iaq_grade": grade.grade,
                "last_updated_grade_at": datetime.now(pytz.UTC).isoformat(),
            }
        ).eq("stock_code", stock_code).execute()
        # Add to llm logs table
        citations = get_citations(response)
        supabase.table("llm_logs").insert(
            {
                "stock_code": stock_code,
                "prompt": prompt,
                "response": response.text,
                "citations": "\n".join(citations) if citations else "None",
            }
        ).execute()

    return grade


# -------------------------------- IR Contacts ------------------------------- #
def search_contacts(
    stock_code: str,
    *,
    save_to_db: bool = True,
) -> str:
    """
    Google search contacts of listed company
    NOTE: As of 5 Aug 2025, it is not possible to configure a single Gemini API call
    to simultaneously use a grounding tool and enforce a structured JSON output.
    See: https://github.com/googleapis/python-genai/issues/665
    NOTE: As of 9 Sep 2025, Grounding with Google Search for Gemini 2.5 Pro (free tier)
    is not supported.
    See: https://ai.google.dev/gemini-api/docs/pricing#standard
    """
    # Filter control_df for the given stock_code and get name
    condition = st.session_state.control_df["stock_code"] == stock_code
    result_df = st.session_state.control_df[condition][["name"]]
    company_name = result_df["name"].iloc[0] if not result_df.empty else ""

    client = get_llm_client()
    # Create prompt
    prompt = (
        "Imagine you are an ESG reporting consultant trying to reach "
        "out to the Investor Relations department of the Hong Kong listed "
        f"company {company_name} with a stock ticker of '{stock_code}' "
        "ideally via email. Find all official, up-to-date, available "
        "contact details, including A) general departmental contact "
        "information (e.g. ir@aia.com); and B) details for specific "
        "individuals (e.g. title, name, email, telephone numbers). "
        "Ignore Company Share Registrar. Only reference official sources "
        "such as company websites or filings. Do not reference third-party "
        "sources e.g. Wikipedia."
    )

    # Send prompt
    grounding_tool = types.Tool(google_search=types.GoogleSearch())
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            tools=[grounding_tool],
            temperature=0.0,
        ),
    )

    # Raise if response is null
    if not response.text:
        msg = "Null value in response text."
        raise ValueError(msg)

    if save_to_db:
        citations = get_citations(response)
        supabase.table("llm_logs").insert(
            {
                "stock_code": stock_code,
                "prompt": prompt,
                "response": response.text,
                "citations": "\n".join(citations) if citations else "None",
            }
        ).execute()
    return embed_citations(response)


def format_contacts(
    stock_code: str,
    response_text: str,
    *,
    save_to_db: bool = True,
) -> list[Contact]:
    """
    Format key data from grounded information to specified JSON schema.
    """
    client = get_llm_client()
    prompt = f"""You are an expert data extraction tool. Your task is to analyze the text provided below and extract all available contact details for the Investor Relations department.

    From the text, identify the following for each contact found:
    - name: The full name of the person or a descriptor for a general contact (e.g., "Investor Relations Department").
    - title: The job title of the person. If it is a general contact, this can be null or an empty string.
    - email: The email address.
    - phone: The phone number.
    - citations: A list of URLs included in the text next to the contact details

    It is crucial that you include general, department-wide contact information. If you find a departmental email like "ir@company.com" or a general phone number, create a contact entry for it. For these general contacts, use a descriptive name like "Investor Relations Department" for the "name" field.

    The output must be a JSON object containing a list of contacts.

    **Text to Analyze:**
    ---
    {response_text}
    ---

    **Expected JSON Output:**
    """
    response = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.0,
            response_mime_type="application/json",
            response_schema=list[Contact],
        ),
    )
    contacts: list[Contact] = response.parsed

    # keep contacts with valid email pattern
    email_pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    valid_contacts = [
        {
            "stock_code": stock_code,
            "email": contact.email,
            "name": contact.name,
            "tel": contact.tel,
            "title": contact.title,
            "citations": "\n".join(contact.citations) if contact.citations else None,
        }
        for contact in contacts
        if re.match(email_pattern, contact.email)
    ]

    if save_to_db and contacts:
        # Add to contacts table
        supabase.table("ir_contacts").upsert(
            valid_contacts,
            ignore_duplicates=True,
            on_conflict="stock_code,email",
        ).execute()
        # Update last_updated_contacts_at in control table
        supabase.table("control").update(
            {"last_updated_contacts_at": datetime.now(pytz.UTC).isoformat()}
        ).eq("stock_code", stock_code).execute()
        # Add to llm logs table
        citations = get_citations(response)
        supabase.table("llm_logs").insert(
            {
                "stock_code": stock_code,
                "prompt": prompt,
                "response": response.text,
                "citations": "\n".join(citations) if citations else "None",
            }
        ).execute()
    return contacts


# ----------------------------------- Email ---------------------------------- #
def draft_email():
    """
    Generate email content with AI.
    """
    # Get IR contact names to address them in email
    contact_names = "Sir/Madam"
    if not st.session_state.ir_contacts_df.empty:
        names = st.session_state.ir_contacts_df["name"].dropna().tolist()
        if names:
            # Filter out generic names if more specific names are available
            specific_names = [
                name for name in names if "department" not in name.lower()
            ]
            if specific_names:
                contact_names = ", ".join(specific_names)

    # Retrieve the reference email from secrets
    reference_email = st.secrets.EMAIL_TEMPLATE

    # Constuct prompt
    prompt = f"""Imagine you are an ESG consultant from the Hong Kong-based NGO ({st.secrets.NGO_URL}). Your task is to draft a professional outreach email to {st.session_state.selected_company_name} (stock code: {st.session_state.selected_stock_code}-HK).

    The email should be addressed to {contact_names}.

    **Your primary goal is to secure a meeting to discuss potential collaborations on Indoor Air Quality (IAQ) initiatives.**

    **Reference Email (for tone, style, and content):**
    ---
    {reference_email}
    ---

    **Instructions for your draft:**
    1.  **Adopt the Tone and Style:** Mirror the professional, appreciative, and collaborative tone of the reference email.
    2.  **Maintain Core Message:** Keep the brief introduction of CAN and the explanation of IAQ's importance for public health in Hong Kong.
    3.  **Personalize the Opening:** Start with a polite and personalized salutation addressing {contact_names}. In the first paragraph, acknowledge the company's current IAQ disclosure status. Use the following assessment to make your opening specific and show you've done your research:
    """

    # Add the company's specific IAQ justification to the prompt if it exists
    if (
        f"justification_{st.session_state.selected_stock_code}" in st.session_state
        and st.session_state[f"justification_{st.session_state.selected_stock_code}"]
    ):
        prompt += f"\n**IAQ Assessment:** {st.session_state[f'justification_{st.session_state.selected_stock_code}']}\n"
    else:
        prompt += "\n**IAQ Assessment:** (No specific assessment available, make a general but positive opening remark about their ESG reporting.)\n"

    # Add the remaining instructions to the prompt
    prompt += """
    4.  **Incorporate Key Proposals:** You MUST include the three initiatives mentioned in the reference email: Leadership Case Study, Expert Presentations and Awareness Workshops, and the ESG Award.
    5.  **Call to Action:** End with a clear call to action, requesting a brief meeting (virtual or in-person).
    6.  **Format:** Return ONLY the body of the email in plain text that is suitable for an email body. Do not use any markdown formatting, such as asterisks for bolding (`**text**`) or bullet points (`* text`). Write paragraphs in standard block format. Do not include the subject line, recipient line (e.g., "To:"), or signature. Keep it concise (within 400 words).
    """

    client = get_llm_client()
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            tools=[
                {"url_context": {}},
            ],
            temperature=0.8,
        ),
    )
    st.session_state.email_content = response.text


def generate_email():
    """
    Generate .eml file.
    """
    custom_policy = default.clone(max_line_length=0, linesep="\r\n")

    # Create .eml file content with custom policy
    msg = EmailMessage(policy=custom_policy)
    msg.set_content(
        st.session_state.get("email_content", None),
        subtype="plain",
        charset="utf-8",
        cte="8bit",
    )  # Set cte='8bit' directly

    # Email headers
    msg["Subject"] = st.session_state.get("email_subject", None)
    msg["To"] = st.session_state.get("email_contacts", None)
    msg["Date"] = datetime.now(pytz.timezone("Asia/Hong_Kong")).strftime(
        "%a, %d %b %Y %H:%M:%S %z"
    )
    msg["X-Unsent"] = "1"  # marked as draft

    return msg.as_string()


# ---------------------------------------------------------------------------- #
#                               Streamlit Helpers                              #
# ---------------------------------------------------------------------------- #
# ---------------------------------- Control --------------------------------- #
def load_control_df():
    """
    Load ir_contacts table from database to session state
    """
    response = supabase.table("control").select("*").order("stock_code").execute()
    df = pd.DataFrame(response.data)

    # convert datetime fields
    if not df.empty:
        datetime_columns = [
            "last_updated_market_cap_at",
            "last_updated_filings_at",
            "last_updated_grade_at",
            "last_updated_contacts_at",
            "created_at",
        ]
        hkt_tz = pytz.timezone("Asia/Hong_Kong")
        for col in datetime_columns:
            if col in df.columns:
                df[col] = df[col].apply(
                    lambda x: parser.parse(x).replace(tzinfo=hkt_tz)
                    if pd.notnull(x)
                    else pd.NaT
                )

    st.session_state.control_df = df


def load_tabs():
    """
    Load ESG filing, IAQ grading and IR contact tabs from database to session state
    """
    st.session_state.esg_filings_df = load_esg_filings(
        stock_code=st.session_state.selected_stock_code
    )
    st.session_state.iaq_gradings_df = load_iaq_gradings(
        stock_code=st.session_state.selected_stock_code
    )
    st.session_state.ir_contacts_df = load_ir_contacts(
        stock_code=st.session_state.selected_stock_code
    )


def edit_control_df():
    """
    Save changes to control table in database
    """
    control_df = st.session_state.control_df
    control_key = st.session_state.control_key

    # Update database based on edited_rows
    if edited_rows := control_key["edited_rows"]:
        for row_idx, changes in edited_rows.items():
            code = control_df.iloc[row_idx]["stock_code"]
            # Skip if no change other than change to stock_code
            changes.pop("stock_code", None)
            if not changes:
                continue
            supabase.table("control").update(
                changes,
            ).eq("stock_code", code).execute()

    # Add stock_codes to database based on added_rows
    if added_rows := control_key["added_rows"]:
        supabase.table("control").upsert(added_rows, on_conflict="stock_code").execute()
        for row in added_rows:
            get_company_basics(stock_code=row["stock_code"], save_to_db=True)

    # Remove deleted rows from database
    if deleted_rows := control_key["deleted_rows"]:
        codes_to_delete = [
            control_df.iloc[row_idx]["stock_code"] for row_idx in deleted_rows
        ]
        supabase.table("control").delete().in_("stock_code", codes_to_delete).execute()

    # Reset edit_control toggle
    st.session_state.control_toggle = False

    if edited_rows or added_rows or deleted_rows:
        # Delete control_key from session state
        del st.session_state.control_key
        # Reset control_df from session state to force reloading
        load_control_df()
        # Show success message
        st.success("Changes to the table saved successfully!")


def get_company_name(stock_code: str) -> str | None:
    """
    Get company name from control table.
    """
    condition = st.session_state.control_df["stock_code"] == stock_code
    result_df = st.session_state.control_df[condition][["name"]]
    return result_df["name"].iloc[0] if not result_df.empty else None


def load_all_iaq_gradings():
    """
    Load all historical iaq_gradings from the database for the chart.
    """
    response = (
        supabase.table("iaq_gradings")
        .select("stock_code, grade, grading_date")
        .execute()
    )
    df = pd.DataFrame(response.data)

    if not df.empty:
        df["grading_date"] = pd.to_datetime(
            df["grading_date"], errors="coerce", utc=True
        )
    return df


def prepare_chart_data() -> pd.DataFrame:
    """
    Loads all historical gradings and processes them to show the
    latest grade count per year for all companies.
    """
    all_gradings_df = load_all_iaq_gradings()

    if all_gradings_df.empty or "grading_date" not in all_gradings_df.columns:
        return pd.DataFrame()

    all_gradings_df.dropna(subset=["grading_date"], inplace=True)
    all_gradings_df["year"] = all_gradings_df["grading_date"].dt.year
    all_gradings_df.sort_values("grading_date", inplace=True)

    latest_grades_per_year_df = all_gradings_df.drop_duplicates(
        subset=["stock_code", "year"], keep="last"
    )

    chart_data = pd.crosstab(
        index=latest_grades_per_year_df["year"],
        columns=latest_grades_per_year_df["grade"],
    )

    for grade in ["Low", "Medium", "High"]:
        if grade not in chart_data.columns:
            chart_data[grade] = 0

    return chart_data[["Low", "Medium", "High"]]


# ---------------------------------- Basics ---------------------------------- #
def update_basics(stock_codes: list[str]):
    for code in stock_codes:
        get_company_basics(stock_code=code, save_to_db=True)
    load_control_df()


# -------------------------------- ESG Filings ------------------------------- #
def load_esg_filings(
    stock_code: str | None,
) -> pd.DataFrame:
    """
    Load esg_filings table from database to session state
    """
    q = supabase.table("esg_filings").select(
        "id",
        "stock_code",
        "release_time",
        "title",
        "url",
        "created_at",
    )
    if stock_code:
        q.eq("stock_code", stock_code)

    response = (
        q.order("stock_code", desc=False).order("release_time", desc=True).execute()
    )
    df = pd.DataFrame(response.data)

    # convert datetime fields
    if not df.empty:
        datetime_columns = ["release_time", "created_at"]
        for col in datetime_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce", utc=False)
                # Localize to Asia/Hong_Kong
                df[col] = df[col].dt.tz_localize(
                    "Asia/Hong_Kong", ambiguous="raise", nonexistent="raise"
                )
    return df


def edit_esg_filings_df():
    """
    Save changes to esg_filings table in database
    """
    # Update database based on edited_rows
    if edited_rows := st.session_state.esg_filings_key["edited_rows"]:
        for row_idx, changes in edited_rows.items():
            pid = int(st.session_state.esg_filings_df.iloc[row_idx]["id"])
            # Skip if no change other than change to stock_code
            changes.pop("stock_code", None)
            if not changes:
                continue
            supabase.table("esg_filings").update(
                changes,
            ).eq("id", pid).execute()

    # Add new rows to database based on added_rows
    if added_rows := st.session_state.esg_filings_key["added_rows"]:
        # Define allowed fields for insertion
        allowed_fields = ["stock_code", "release_time", "title", "url"]
        rows_to_insert = [
            {field: row[field] for field in allowed_fields if field in row}
            for row in added_rows
        ]
        if rows_to_insert:
            supabase.table("esg_filings").upsert(
                rows_to_insert,
                ignore_duplicates=True,
                on_conflict="url",
            ).execute()

    # Remove deleted rows from esg_filings table
    if deleted_rows := st.session_state.esg_filings_key["deleted_rows"]:
        ids_to_delete = [
            int(st.session_state.esg_filings_df.iloc[row_idx]["id"])
            for row_idx in deleted_rows
        ]
        supabase.table("esg_filings").delete().in_("id", ids_to_delete).execute()

    # Reset esg_filings_toggle
    st.session_state.esg_filings_toggle = False

    # Reset session state if there are any changes
    if edited_rows or added_rows or deleted_rows:
        # Delete esg_filings_key from session state
        del st.session_state.esg_filings_key
        # Reload esg_filings_df
        st.session_state.esg_filings_df = load_esg_filings(
            stock_code=st.session_state.selected_stock_code
        )

    st.success("Changes to the table saved successfully!")


def update_esg_filings_df(
    stock_codes: list[str],
):
    """
    Fetch ESG filings with Gemini, and trigger reload of esg_filings_df and control_df
    """
    for code in stock_codes:
        try:
            scrape(stock_code=code, save_to_db=True)
        except sqlalchemy.exc.IntegrityError:
            pass
        else:
            st.success(f"ESG filings successfully updated for {code}!")

    # Reset session state
    load_control_df()
    if "selected_stock_code" in st.session_state:
        st.session_state.esg_filings_df = load_esg_filings(
            stock_code=st.session_state.selected_stock_code
        )


# -------------------------------- IAQ Grading ------------------------------- #
def load_iaq_gradings(
    stock_code: str | None,
):
    """
    Load iaq_gradings from database to session state.
    """
    q = supabase.table("iaq_gradings").select("*")
    if stock_code:
        q.eq("stock_code", stock_code)

    # Sort by grading_date to show the most recent gradings first
    response = q.order("grading_date", desc=True).execute()
    df = pd.DataFrame(response.data)

    # convert datetime fields
    if not df.empty:
        datetime_columns = ["grading_date"]
        for col in datetime_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce", utc=False)
                # Localize to Asia/Hong_Kong
                df[col] = df[col].dt.tz_localize(
                    "Asia/Hong_Kong", ambiguous="raise", nonexistent="raise"
                )

    return df


def edit_iaq_grading(
    grade_id: int,
    stock_code: str,
):
    """
    Save changes to a specific historical record in the iaq_gradings table.
    If the edited record is the latest one, also update the control table.
    """
    # Fetch values from session state using grade_id
    grade = st.session_state.get(f"grade_{grade_id}", "")
    overview = st.session_state.get(f"overview_{grade_id}", "")
    justification = st.session_state.get(f"justification_{grade_id}", "")
    extracts = st.session_state.get(f"extracts_{grade_id}", "")

    # Update the specific historical record in iaq_gradings table
    supabase.table("iaq_gradings").update(
        {
            "grade": grade,
            "overview": overview,
            "justification": justification,
            "extracts": extracts,
        }
    ).eq("id", grade_id).execute()

    # Check if the edited grade is the most recent one.
    latest_grade_id = st.session_state.iaq_gradings_df["id"].iloc[0]
    if grade_id == latest_grade_id:
        # Update the control table to keep it in sync.
        supabase.table("control").update(
            {
                "iaq_grade": grade,
                "last_updated_grade_at": datetime.now(pytz.UTC).isoformat(),
            }
        ).eq("stock_code", stock_code).execute()
        # Reload control_df to reflect the change on the main dashboard
        load_control_df()

    # After saving, force a reload of the data to show changes
    st.session_state.iaq_gradings_df = load_iaq_gradings(stock_code)
    st.success("Changes to the grading report have been saved.")


def update_iaq_grading(
    stock_code: str,
):
    """
    Generate IAQ grading report with Gemini, and trigger reload of control_df and iaq_grading
    """
    try:
        # Grade IAQ discloures with Gemini
        response_text = grade_iaq(
            stock_code=stock_code,
            save_to_db=True,
        )
        # Format grade with Gemini and save to database
        format_grading(
            stock_code=stock_code,
            response_text=response_text,
            save_to_db=True,
        )
    except ValueError as exc:
        if "No filings found in database for" in str(exc):
            msg = "No filings found in database. Please fetch filings first."
            st.warning(msg, icon="⚠️")
        elif "Null value in response text." in str(exc):
            msg = (
                "Error encountered while using Gemini API to grade IAQ "
                f"disclosures for {stock_code}. "
                "This may be due to rate limits—please try again later."
            )
            st.warning(msg, icon="⚠️")
        else:
            st.warning(exc, icon="⚠️")
    else:
        st.success(f"IAQ grading report successfully generated for {stock_code}!")

    # Reset session state
    load_control_df()
    st.session_state.iaq_gradings_df = load_iaq_gradings(stock_code)


# -------------------------------- IR Contacts ------------------------------- #
def load_ir_contacts(
    stock_code: str | None,
):
    """
    Load ir_contacts from database to session state
    """
    q = supabase.table("ir_contacts").select("*")
    if stock_code:
        q.eq("stock_code", stock_code)

    response = q.order("stock_code", desc=False).execute()
    df = pd.DataFrame(response.data)

    # convert datetime fields
    if not df.empty:
        datetime_columns = ["created_at"]
        for col in datetime_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce", utc=False)
                # Localize to Asia/Hong_Kong
                df[col] = df[col].dt.tz_localize(
                    "Asia/Hong_Kong", ambiguous="raise", nonexistent="raise"
                )

    return df


def edit_ir_contacts_df():
    """
    Save changes to ir_contacts table in database
    """
    # Update database based on edited_rows
    if edited_rows := st.session_state.ir_contacts_key["edited_rows"]:
        for row_idx, changes in edited_rows.items():
            pid = int(st.session_state.ir_contacts_df.iloc[row_idx]["id"])
            # Skip if no change other than change to stock_code
            changes.pop("stock_code", None)
            if not changes:
                continue
            supabase.table("ir_contacts").update(changes).eq("id", pid).execute()

    # Add new rows to database based on added_rows
    if added_rows := st.session_state.ir_contacts_key["added_rows"]:
        # Define allowed fields for insertion
        allowed_fields = [
            "stock_code",
            "email",
            "name",
            "tel",
            "title",
            "citations",
        ]
        rows_to_insert = [
            {field: row[field] for field in allowed_fields if field in row}
            for row in added_rows
        ]
        if rows_to_insert:
            supabase.table("ir_contacts").upsert(
                rows_to_insert,
                ignore_duplicates=True,
                on_conflict="stock_code,email",
            ).execute()

    # Remove deleted rows from ir_contacts table
    if deleted_rows := st.session_state.ir_contacts_key["deleted_rows"]:
        ids_to_delete = [
            int(st.session_state.ir_contacts_df.iloc[row_idx]["id"])
            for row_idx in deleted_rows
        ]
        supabase.table("ir_contacts").delete().in_("id", ids_to_delete).execute()

    # Reset ir_contacts_toggle
    st.session_state.ir_contacts_toggle = False

    if edited_rows or added_rows or deleted_rows:
        # Delete ir_contacts_key from session state
        del st.session_state.ir_contacts_key
        # Reload ir_contacts_df
        st.session_state.ir_contacts_df = load_ir_contacts(
            stock_code=st.session_state.selected_stock_code
        )
        st.success("Changes to the table saved successfully!")


def update_ir_contacts_df(
    stock_codes: list[str],
):
    """
    Update IR contacts with Gemini, and trigger reload of ir_contacts_df and control_df
    """
    for code in stock_codes:
        # NOTE: Rate limit at 5 RPM, 250k TPM, 100 RPD for Gemini 2.5 Pro
        # https://ai.google.dev/gemini-api/docs/rate-limits
        try:
            # Search contacts with Gemini
            response_text = search_contacts(
                stock_code=code,
                save_to_db=True,
            )
            # Format contacts with Gemini and save to database
            format_contacts(
                stock_code=code,
                response_text=response_text,
                save_to_db=True,
            )
        except ValueError as exc:
            if "Null value in response text." in str(exc):
                msg = (
                    "Error encountered while using Gemini API to grade IAQ "
                    f"disclosures for {code}. "
                    "This may be due to rate limits—please try again later."
                )
                st.warning(msg, icon="⚠️")
            else:
                st.warning(exc, icon="⚠️")
        except genai.errors.ServerError:
            msg = "Server error from Gemini API. Please retry later."
            st.warning(msg, icon="⚠️")
        else:
            st.success(f"IR contacts updated for {code}!")

    # Reset session state
    load_control_df()
    if "selected_stock_code" in st.session_state:
        st.session_state.ir_contacts_df = load_ir_contacts(
            stock_code=st.session_state.selected_stock_code
        )


# --------------------------------- Download --------------------------------- #
def load_llm_logs() -> pd.DataFrame:
    """
    Load llm_logs table from database
    """
    response = (
        supabase.table("llm_logs").select("*").order("created_at", desc=True).execute()
    )
    df = pd.DataFrame(response.data)

    # convert datetime fields
    if not df.empty:
        datetime_columns = ["created_at"]
        for col in datetime_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce", utc=False)
                # Localize to Asia/Hong_Kong
                df[col] = df[col].dt.tz_localize(
                    "Asia/Hong_Kong", ambiguous="raise", nonexistent="raise"
                )
    return df


def write_to_excel():
    """
    Export dataframes to Excel
    """
    # List of dataframes
    dfs = [
        st.session_state.control_df,
        load_esg_filings(stock_code=None),
        load_iaq_gradings(stock_code=None),
        load_ir_contacts(stock_code=None),
        load_llm_logs(),
    ]
    df_names = [
        "Control",
        "ESG Filings",
        "IAQ Gradings",
        "IR Contacts",
        "LLM Logs",
    ]

    # Create an in-memory buffer for the Excel file
    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        for df, name in zip(dfs, df_names):
            df_to_export = df.copy()

            for col in df_to_export.columns:
                # Check if the column dtype is a timezone-aware datetime
                if (
                    pd.api.types.is_datetime64_any_dtype(df_to_export[col])
                    and df_to_export[col].dt.tz is not None
                ):
                    # Convert to timezone-unaware (naive) datetime
                    df_to_export[col] = df_to_export[col].dt.tz_localize(None)

            # Convert id column to int
            if "id" in df_to_export.columns:
                df_to_export["id"] = df_to_export["id"].astype("Int64").fillna(pd.NA)
                # Sort dataframe by id
                df_to_export = df_to_export.sort_values(by="id")

            # Write the modified dataframe to Excel
            df_to_export.to_excel(writer, sheet_name=name, index=False)
    output.seek(0)
    return output


# ---------------------------------------------------------------------------- #
#                                     Main                                     #
# ---------------------------------------------------------------------------- #
st.set_page_config(page_title="HK ListCo IAQ Tracker", page_icon=":material/aq_indoor:")
st.title(":material/aq_indoor: Hong Kong ListCo IAQ Tracker")
st.write(
    """
    Welcome to the **Hong Kong ListCo IAQ Tracker** — an AI-powered tool that helps you monitor how Hong Kong–listed companies disclose their **Indoor Air Quality (IAQ)** information.
    """
)
with st.expander("Getting Started"):
    st.markdown("""
    **🧭 How to Navigate**
    - **Dashboard:** View all companies in your watchlist.
      - Click a row to open its company profile.
      - Toggle *Manage Watchlist* to add or remove companies.
    - **Company Profile Tabs:**
      - **Basics:** Retrieve the company’s market cap and sector classification.
      - **Filings:** Pull the latest ESG filings directly from HKEx.
      - **Grading:** Use AI to create and review historical IAQ disclosure quality reports.
      - **Contacts:** Find up-to-date Investor Relations contact information.
      - **Outreach:** Automatically draft a professional email to engage the company.
    ---
    **⚙️ Other Tools**
    - **Bulk Updates:** Refresh market data, ESG filings, and IR contacts for multiple companies at once.
    - **Data Visualization:** Analyze historical grade trends across your watchlist with an interactive chart.
    - **Data Export:** Download all your tracked data to a single Excel file for reporting.
    """)

st.divider()


# ---------------------------------- Control --------------------------------- #
st.subheader("📊 Dashboard")
st.write("""
    Your central watchlist of Hong Kong–listed companies.
    Use this page to **view**, **filter**, and **manage** the companies you’re tracking for IAQ disclosure updates.
    """)
with st.expander("Detailed Instructions"):
    st.markdown("""
    👀 View Mode (Default)
    - **Open a Company Profile:** Click the **checkbox in the first column** beside a company's name to select it. The company’s detailed profile will appear below the dashboard.
    - **Filter Your List:** Turn on **Add Filters** (above the table) to search or narrow companies by sector, market cap, or IAQ grade.
    - **Sort Columns:** Click any column header — for example, *Market Cap* or *IAQ Grade* — to sort ascending or descending.

    ✏️ Manage Watchlist (Edit Mode)
    - **Switch to Manage Mode:** Toggle **Manage Watchlist** at the top-left of the dashboard.
      This lets you add or remove companies from your list.
    - **Add a Company:** Scroll to the blank row at the bottom and enter a 5-digit stock code (e.g. `00005`). The company’s name, sector, and market data will fill in automatically.
    - **Update Info:** Click a cell to edit its value, then press *Enter* key or click away to confirm.
    - **Remove a Company:** Tick the checkbox next to a company, then press **Delete** key.
    - **Save Your Watchlist:** Click **Save Watchlist** when finished to apply your changes. *Your updates won’t be stored until you save.*
    """)

# init control dataframe and edit_control
if "control_df" not in st.session_state:
    load_control_df()
if "control_toggle" not in st.session_state:
    st.session_state.control_toggle = False

# display edit control toggle
edit_control = st.toggle("Manage Watchlist", key="control_toggle")

# show dataframe in display mode
control_col_order = [
    "stock_code",
    "name",
    "sector",
    "market_cap",
    "iaq_grade",
    "last_updated_filings_at",
    "last_updated_grade_at",
    "last_updated_contacts_at",
]

if not edit_control:
    # user to select a stock code
    selected_row = st.dataframe(
        filter_dataframe(st.session_state.control_df),
        use_container_width=True,
        hide_index=True,
        column_order=control_col_order,
        column_config={
            "id": None,
            "market_cap": st.column_config.NumberColumn("market_cap (HK$bn)"),
            "last_updated_filings_at": st.column_config.DatetimeColumn(
                format="YYYY-MM-DD"
            ),
            "last_updated_grade_at": st.column_config.DatetimeColumn(
                format="YYYY-MM-DD"
            ),
            "last_updated_contacts_at": st.column_config.DatetimeColumn(
                format="YYYY-MM-DD"
            ),
        },
        on_select="rerun",
        selection_mode="single-row",
    )
    # save selected stock code to session state
    if selected_row["selection"]["rows"]:
        st.session_state.selected_stock_code = st.session_state.control_df.iloc[
            selected_row["selection"]["rows"][0]
        ]["stock_code"]


# show data editor in edit mode
else:
    st.caption("""
        ✏️ **Manage Watchlist** mode: You can add, edit, or delete companies. Click **Save Watchlist** when done.
        """)
    st.data_editor(
        st.session_state.control_df,
        use_container_width=True,
        hide_index=True,
        column_order=control_col_order,
        column_config={
            "id": None,
            "stock_code": st.column_config.TextColumn(
                max_chars=5,
                validate=r"^\d{5}$",
                required=True,
            ),
            "iaq_grade": st.column_config.SelectboxColumn(
                options=["Low", "Medium", "High"],
            ),
        },
        disabled=[
            "created_at",
        ],
        key="control_key",
        num_rows="dynamic",
    )
    done_edit = st.button("Save Watchlist", type="primary", on_click=edit_control_df)

st.divider()


# # ------------------------------ Company Profile ----------------------------- #
if "selected_stock_code" not in st.session_state:
    st.subheader("🏢 Company Profile")
    st.info("Select a stock code from *Dashboard* to view its details.")
else:
    # init session state variables if changes in selected stock code
    if ("prev_selected_stock_code" not in st.session_state) or (
        st.session_state.selected_stock_code
        != st.session_state.prev_selected_stock_code
    ):
        # reload dataframes
        load_tabs()
        # reload company name
        st.session_state.selected_company_name = get_company_name(
            stock_code=st.session_state.selected_stock_code
        )
        # reset email body and email
        if "email_content" in st.session_state:
            del st.session_state.email_content
        if "email" in st.session_state:
            del st.session_state.email
        # update prev_selected_stock_code
        st.session_state.prev_selected_stock_code = st.session_state.selected_stock_code
    # rewrite subheader
    if (
        "selected_company_name" in st.session_state
        and st.session_state.selected_company_name
    ):
        st.subheader(
            f"Company Profile: {st.session_state.selected_company_name} ({st.session_state.selected_stock_code})"
        )
    else:
        st.subheader(f"Company Profile: {st.session_state.selected_stock_code}")

    # init navigation bar
    tab_lst = ["Basics", "Filings", "Grading", "Contacts", "Outreach"]
    active_tab = st.radio("", tab_lst, horizontal=True, key="active_tab")

    # Tab 1: Basics
    if active_tab == tab_lst[0]:
        st.write(
            """
            This data is automatically fetched from HKEx when a stock code is first added.
            Click the button below to refresh the company's name, sector, and market capitalization.
            """
        )

        # Fetch the row for the selected stock code from control_df
        data = st.session_state.control_df[
            st.session_state.control_df["stock_code"]
            == st.session_state.selected_stock_code
        ].iloc[0]

        st.metric("HSICS Sector", data.get("sector", "N/A"))
        col1, col2 = st.columns(2)
        col1.metric("Market Cap (HK$bn)", f"{data.get('market_cap', 0):,.2f}")
        col2.metric(
            "Last Updated",
            data.get("last_updated_market_cap_at", pd.NaT).strftime("%Y-%m-%d")
            if pd.notna(data.get("last_updated_market_cap_at"))
            else "N/A",
        )

        st.button(
            "Refresh Basics",
            type="primary",
            on_click=update_basics,
            kwargs={
                "stock_codes": [st.session_state.selected_stock_code],
            },
        )

    # Tab 2: ESG Filings
    elif active_tab == tab_lst[1]:
        st.write(
            "Review official ESG reports from HKEx, sorted by the most recent date. Use the 'Refresh' button to fetch the latest filings."
        )

        if st.session_state.esg_filings_df.empty:
            st.info(
                f"No ESG filings are currently stored for {st.session_state.selected_stock_code}. Fetch latest filings from HKEx below!"
            )
            st.button(
                "Fetch Filings",
                type="primary",
                key="fetch_esg_filings_when_empty",
                on_click=update_esg_filings_df,
                kwargs={
                    "stock_codes": [st.session_state.selected_stock_code],
                },
            )
        else:
            # init esg_filings_toggle
            if "esg_filings_toggle" not in st.session_state:
                st.session_state.esg_filings_toggle = False
            # display edit esg_filings toggle
            edit_esg_filings = st.toggle("Manage Filings", key="esg_filings_toggle")

            if not edit_esg_filings:
                st.dataframe(
                    st.session_state.esg_filings_df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "id": None,
                        "url": st.column_config.LinkColumn(display_text="Open Link"),
                    },
                )
                st.button(
                    "Refresh Filings",
                    type="primary",
                    key="fetch_esg_filings",
                    on_click=update_esg_filings_df,
                    kwargs={
                        "stock_codes": [st.session_state.selected_stock_code],
                    },
                )
            else:
                st.caption(
                    "✏️ **Manage Filings** mode: You can now add, edit, or delete filings. Click 'Save Changes' when done."
                )
                st.data_editor(
                    st.session_state.esg_filings_df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "id": None,
                        "stock_code": st.column_config.TextColumn(
                            max_chars=5,
                            validate=r"^\d{5}$",
                            required=True,
                        ),
                        "release_time": st.column_config.DatetimeColumn(
                            required=True,
                        ),
                        "title": st.column_config.TextColumn(
                            required=True,
                        ),
                        "url": st.column_config.TextColumn(
                            required=True,
                        ),
                    },
                    disabled=[
                        "created_at",
                    ],
                    key="esg_filings_key",
                    num_rows="dynamic",
                )
                st.button("Save Changes", type="primary", on_click=edit_esg_filings_df)

    # Tab 3: IAQ Grading
    elif active_tab == tab_lst[2]:
        st.write(
            """
            Access AI-generated reports on the company's IAQ disclosure quality. Each report is a **point-in-time snapshot** of their performance.
            """
        )
        st.info(
            """
            ⭐ **Tip:** For the most accurate assessment, always refresh the ESG filings first and generate a new report whenever a new filing is published to track progress over time.
            """
        )

        st.markdown("#### Historical Reports")

        if st.session_state.iaq_gradings_df.empty:
            st.info(
                "No IAQ grading reports are currently stored for this company. You can generate one below."
            )
        else:
            st.write("Click on a report to expand its details and edit if needed.")
            # Iterate through each historical report and create an expander for it
            for index, report_data in st.session_state.iaq_gradings_df.iterrows():
                grade_id = report_data["id"]
                expander_title = f"Report from {report_data['grading_date'].strftime('%d %B %Y')}  |  Grade: {report_data.get('grade', 'N/A')}"

                with st.expander(expander_title):
                    # Contextual "Outdated Report" warning INSIDE the relevant expander
                    if not st.session_state.esg_filings_df.empty:
                        latest_filing_date = st.session_state.esg_filings_df[
                            "release_time"
                        ].iloc[0]
                        if report_data["grading_date"] < latest_filing_date:
                            st.warning(
                                "This report may be outdated as newer ESG filings have been published since it was created.",
                                icon="⚠️",
                            )

                    # The form for editing is neatly contained within the expander
                    with st.form(f"iaq_grading_form_{grade_id}"):
                        st.selectbox(
                            "IAQ Grade",
                            options=["Low", "Medium", "High"],
                            index=["Low", "Medium", "High"].index(
                                report_data.get("grade", "Low")
                            ),
                            key=f"grade_{grade_id}",
                        )
                        st.text_area(
                            "Company Overview",
                            value=report_data.get("overview", ""),
                            height=100,
                            key=f"overview_{grade_id}",
                        )
                        st.text_area(
                            "Justification",
                            value=report_data.get("justification", ""),
                            height=150,
                            key=f"justification_{grade_id}",
                        )
                        st.text_area(
                            "Key Extracts",
                            value=report_data.get("extracts", ""),
                            height=200,
                            key=f"extracts_{grade_id}",
                        )

                        st.form_submit_button(
                            "Save Changes",
                            type="primary",
                            on_click=edit_iaq_grading,
                            kwargs={
                                "grade_id": grade_id,
                                "stock_code": st.session_state.selected_stock_code,
                            },
                        )

        st.markdown("#### Generate New Report")
        st.write("Create a new AI-powered assessment of the company's IAQ disclosures.")

        # The evaluation criteria are now logically placed with the creation action
        with st.expander("View Evaluation Criteria"):
            st.markdown("""
            The AI grades the company's IAQ disclosures based on the following criteria:
            - **Length and Detail:** Short/vague mentions vs. dedicated sections with data.
            - **Key Performance Indicators (KPIs):** Presence of quantifiable metrics.
            - **Consistency:** How regularly KPIs are reported over time.
            - **Progression:** Emphasis on improvements in recent years.
            """)

        if st.button("Generate Report", type="primary"):
            update_iaq_grading(stock_code=st.session_state.selected_stock_code)
            st.rerun()

    # Tab 4: IR Contacts
    elif active_tab == tab_lst[3]:
        st.write(
            "Find and manage the Investor Relations contact details needed for your outreach efforts."
        )

        if st.session_state.ir_contacts_df.empty:
            st.info(
                f"No IR contacts are currently stored for {st.session_state.selected_stock_code}. Fetch contacts below!"
            )
            st.button(
                "Fetch Contacts",
                type="primary",
                key="fetch_ir_contacts_when_empty",
                on_click=update_ir_contacts_df,
                kwargs={"stock_codes": [st.session_state.selected_stock_code]},
            )
        else:
            # init ir_contacts_toggle
            if "ir_contacts_toggle" not in st.session_state:
                st.session_state.ir_contacts_toggle = False
            # display edit ir_contacts toggle
            edit_ir_contacts = st.toggle("Manage Contacts", key="ir_contacts_toggle")

            if not edit_ir_contacts:
                st.dataframe(
                    st.session_state.ir_contacts_df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={"id": None},
                )
                st.button(
                    "Refresh Contacts",
                    type="primary",
                    key="fetch_ir_contacts",
                    on_click=update_ir_contacts_df,
                    kwargs={"stock_codes": [st.session_state.selected_stock_code]},
                )
            else:
                st.caption(
                    "✏️ **Manage Contacts** mode: You can now add, edit, or delete contacts. Click 'Save Changes' when done."
                )
                st.data_editor(
                    st.session_state.ir_contacts_df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "id": None,
                        "stock_code": st.column_config.TextColumn(
                            max_chars=5,
                            validate=r"^\d{5}$",
                            required=True,
                        ),
                        "email": st.column_config.TextColumn(
                            validate=r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$",
                            required=False,
                        ),
                    },
                    disabled=[
                        "created_at",
                    ],
                    key="ir_contacts_key",
                    num_rows="dynamic",
                )
                st.button("Save Changes", type="primary", on_click=edit_ir_contacts_df)

    # Tab 5: Outreach Email
    elif active_tab == tab_lst[4]:
        st.write(
            """
            Draft a professional outreach email to the company's IR department. The AI will use the company's IAQ grade and contact details to create a personalized first draft.
            """
        )
        st.info(
            """
            ⭐ **Tip:** The generated email is temporary, so remember to generate and download the `.eml` file to save a copy.
            """
        )

        if "email_content" not in st.session_state:
            st.session_state.email_content = ""

        # set up email form
        with st.form("email_form"):
            st.markdown("##### Step 1: Confirm Recipients and Subject")
            st.text_input(
                "To",
                value=st.session_state.ir_contacts_df["email"]
                .dropna()
                .str.cat(sep=", ")
                if not st.session_state.ir_contacts_df.empty
                else "",
                key="email_contacts",
            )
            st.text_input(
                "Subject",
                value="Opportunities to Enhance and Showcase Your Leadership in IAQ",
                key="email_subject",
            )
            st.markdown("##### Step 2: Generate and Refine Email Body")
            st.text_area(
                "Content",
                key="email_content",
            )

            col1, col2, _ = st.columns([0.25, 0.25, 0.5])
            with col1:
                st.form_submit_button(
                    "Draft with AI",
                    type="secondary",
                    on_click=draft_email,
                    use_container_width=True,
                )
            with col2:
                submitted = st.form_submit_button(
                    "Generate .eml File", type="primary", use_container_width=True
                )
            if submitted:
                st.session_state.email = generate_email()
                st.session_state.email_filename = f"""Outreach___{st.session_state.selected_stock_code}___{
                    datetime.now(
                        tz=pytz.timezone('Asia/Hong_Kong')
                    ).strftime(
                        '%Y%m%d%H%M%S'
                    )
                }.eml"""
                st.success("Email generated! Ready for download.")
        if "email" in st.session_state:
            st.download_button(
                label="Download .eml File",
                data=st.session_state.email,
                file_name=st.session_state.email_filename,
                mime="message/rfc822",
                icon=":material/download:",
            )

st.divider()


# ------------------------------- Bulk Updates ------------------------------- #
st.subheader("⚡ Bulk Updates")
st.write(
    """
    Save time by refreshing ESG filings and IR contacts for multiple companies at once.
    This tool will find all companies in your watchlist that haven't been updated
    within your chosen timeframe and refresh their data.
    """
)

with st.form("bulk_update_form"):
    weeks = st.number_input(
        label=("Refresh data for companies not updated in the last (weeks):"),
        min_value=0,
        value=12,
    )

    col1, col2, col3, _ = st.columns([0.25, 0.25, 0.25, 0.25])

    with col1:
        st.form_submit_button(
            "Fetch Basics",
            type="primary",
            on_click=update_basics,
            kwargs={
                "stock_codes": get_stock_codes_tbu(
                    update_market_cap=True,
                    update_before=datetime.now(pytz.UTC) - timedelta(weeks=int(weeks)),
                )
            },
            use_container_width=True,
        )
    with col2:
        st.form_submit_button(
            "Fetch Filings",
            type="primary",
            on_click=update_esg_filings_df,
            kwargs={
                "stock_codes": get_stock_codes_tbu(
                    update_filings=True,
                    update_before=datetime.now(pytz.UTC) - timedelta(weeks=int(weeks)),
                )
            },
            use_container_width=True,
        )
    with col3:
        st.form_submit_button(
            "Fetch Contacts",
            type="primary",
            on_click=update_ir_contacts_df,
            kwargs={
                "stock_codes": get_stock_codes_tbu(
                    update_contacts=True,
                    update_before=datetime.now(pytz.UTC) - timedelta(weeks=int(weeks)),
                )
            },
            use_container_width=True,
        )

st.divider()


# ---------------------------- Data Visualization ---------------------------- #
st.subheader("🎨 Data Visualization")
st.write(
    """
    Analyze the historical IAQ grade distribution across your entire watchlist.
    This helps you track overall progress year by year.
    """
)

# Prepare data and get available years
chart_df = prepare_chart_data()
if not chart_df.empty:
    available_years = chart_df.index.unique().tolist()

    # Add a multiselect to filter by year
    selected_years = st.multiselect(
        "Filter by Year:", options=available_years, default=available_years
    )

    if selected_years:
        filtered_chart_df = chart_df[chart_df.index.isin(selected_years)]

        # Use columns for a side-by-side view of the chart and data
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown("#### Grade Distribution Over Time")
            st.bar_chart(filtered_chart_df)
        with col2:
            st.markdown("#### Grade Summary")
            st.dataframe(filtered_chart_df, use_container_width=True)
    else:
        st.info("Please select at least one year to display trends.")

st.divider()


# --------------------------------- Download --------------------------------- #
st.subheader("📥 Data Export")
st.write(
    """
    Download your entire dataset to a single Excel file.
    Export includes your full dataset (companies, filings, IAQ grades, contacts, and AI logs) in Excel format.
    """
)

if st.button("Generate .xlsx File", type="primary"):
    excel_output = write_to_excel()
    st.session_state.excel_data = excel_output.getvalue()
    st.session_state.excel_filename = f"""ListCo IAQ tracker__{
        datetime.now(
            tz=pytz.timezone('Asia/Hong_Kong')
        ).strftime(
            '%Y%m%d%H%M%S'
        )
    }.xlsx"""
    st.success("Excel file generated! Ready for download.")

if "excel_data" in st.session_state:
    st.download_button(
        label="Download .xlsx File",
        data=st.session_state.excel_data,
        file_name=st.session_state.excel_filename,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        icon=":material/download:",
    )
