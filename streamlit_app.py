"""
App to be deployed on Streamlit Cloud
"""

from datetime import datetime
from datetime import timedelta
import re
import time
from typing import List
from typing import Dict
from io import BytesIO

from google import genai
from google.genai import types
import pandas as pd
from pydantic import BaseModel
import pytz
import sqlalchemy.exc
from sqlalchemy import text as sql_text
import streamlit as st
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from selenium.common.exceptions import NoSuchElementException
from webdriver_manager.chrome import ChromeDriverManager


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


# ---------------------------------------------------------------------------- #
#                                   Services                                   #
# ---------------------------------------------------------------------------- #
def get_stock_codes_tbu(
    *,
    update_filings: bool = False,
    update_contacts: bool = False,
    update_before: datetime | None = None,
) -> list[str]:
    """
    Get list of stock codes for which ESG filings or IR contacts require updating.
    Optional: set update_before to get stock codes updated before a specific date
    """
    # Validate inputs
    if not (update_filings or update_contacts):
        msg = "Either update_filings or update_contacts must be True"
        raise ValueError(msg)

    # Load control_df if it is not loaded in session state
    if "control_df" not in st.session_state:
        load_control_df()

    # Determine the field to filter
    field = "last_updated_filings_at" if update_filings else "last_updated_contacts_at"

    # Ensure the datetime column is timezone-aware (UTC)
    if st.session_state.control_df[field].dtype == "datetime64[ns]":
        st.session_state.control_df[field] = st.session_state.control_df[
            field
        ].dt.tz_localize("UTC")

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


# -------------------------------- ESG Filings ------------------------------- #
def get_last_updated_at(
    stock_code: str, *, in_local_tz: bool = True
) -> datetime | None:
    """
    Get the timestamp of last updated at for a stock code.
    """
    # Load control_df if it is not loaded in session state
    if "control_df" not in st.session_state:
        load_control_df()

    # Filter control_df for the given stock_code and select last_updated_filings_at
    condition = st.session_state.control_df["stock_code"] == stock_code
    result_df = st.session_state.control_df[condition][["last_updated_filings_at"]]

    # Extract the first value if the DataFrame is not empty, else return None
    result = (
        result_df["last_updated_filings_at"].iloc[0] if not result_df.empty else None
    )

    # convert to local timezone if specified
    if in_local_tz and result:
        # Localize the naive timestamp to UTC and then convert to the target timezone
        result = result.tz_localize("UTC").tz_convert("Asia/Hong_Kong")
        # Convert the timezone-aware pandas Timestamp to a timezone-aware python datetime object
        result = result.to_pydatetime()
    return result


def get_earliest_release_time(driver: webdriver.Chrome) -> datetime | None:
    """
    Get the release time of the earliest record displayed in results page of HKEx website.
    """
    result_rows = driver.find_elements(
        By.CSS_SELECTOR,
        "#titleSearchResultPanel table tbody tr",
    )
    if result_rows is not None:
        last_row = result_rows[-1]
        last_row_cells = last_row.find_elements(By.TAG_NAME, "td")
        earliest_release_time_str = last_row_cells[0].text
        return datetime.strptime(earliest_release_time_str, "%d/%m/%Y %H:%M")
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
    # service = Service(ChromeDriverManager(chrome_type=ChromeType.CHROMIUM).install())
    service = Service(ChromeDriverManager().install())
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
    url = st.secrets.URL
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
    last_updated_at = get_last_updated_at(
        stock_code=stock_code,
        in_local_tz=True,
    )
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
    data_lst: List[Dict[datetime | str]] = []
    company_name = None
    for row in result_rows:
        cells = row.find_elements(By.TAG_NAME, "td")
        # extract and convert release_time
        release_time_str = cells[0].text
        release_time = datetime.strptime(release_time_str, "%d/%m/%Y %H:%M")
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
                "release_time": release_time,
                "title": doc_title,
                "url": doc_url,
            }
        )
        # extract company name
        if not company_name:
            company_name = cells[2].text.split("\n")[0]
    # close browser
    driver.quit()

    # d) save key data to esg_filings tab of master excel
    if save_to_db:
        db_conn = st.connection("neon", type="sql")
        for data in data_lst:
            with db_conn.session as session:
                session.execute(
                    sql_text("""
                    INSERT INTO esg_filings (stock_code, release_time, title, url)
                    VALUES (:stock_code, :release_time, :title, :url)
                    ON CONFLICT (url) DO NOTHING
                    """),
                    {
                        "stock_code": stock_code,
                        "release_time": data.get("release_time"),
                        "title": data.get("title"),
                        "url": data.get("url"),
                    },
                )
                session.commit()
        # update last_updated_filings_at and company name in control table
        with db_conn.session as session:
            session.execute(
                sql_text("""
                UPDATE control
                SET last_updated_filings_at = CURRENT_TIMESTAMP
                WHERE stock_code = :stock_code
                """),
                params={"stock_code": stock_code},
            )
            if company_name:
                session.execute(
                    sql_text("""
                    UPDATE control
                    SET name = :name
                    WHERE stock_code = :stock_code AND name IS NULL
                    """),
                    params={"name": company_name, "stock_code": stock_code},
                )
            session.commit()


# -------------------------------- IR Contacts ------------------------------- #
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
    return genai.Client(
        api_key=api_keys[st.session_state.api_key_counter % api_keys_count]
    )


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


def search_contacts(
    client: genai.Client,
    stock_code: str,
    *,
    save_to_db: bool = True,
) -> str:
    """
    Google search contacts of listed company
    NOTE: As of 5 Aug 2025, it is not possible to configure a single Gemini API call
    to simultaneously use a grounding tool and enforce a structured JSON output.
    See: https://github.com/googleapis/python-genai/issues/665
    """
    # Load control_df if it is not loaded in session state
    if "control_df" not in st.session_state:
        load_control_df()

    # Filter control_df for the given stock_code and get name
    condition = st.session_state.control_df["stock_code"] == stock_code
    result_df = st.session_state.control_df[condition][["name"]]
    company_name = result_df["name"].iloc[0] if not result_df.empty else ""

    # Create prompt
    model = "gemini-2.5-pro"
    prompt = (
        "Imagine you are an ESG reporting consultant trying to reach "
        "out to the Investor Relations department of the Hong Kong listed "
        f"company {company_name} with a stock ticker of '{stock_code}' "
        "ideally via email. Find all official, up-to-date, available "
        "contact details, including A)s general departmental contact "
        "information (e.g. ir@aia.com); and B) details for specific "
        "individuals (e.g. title, name, email, telephone numbers). "
        "Ignore Company Share Registrar. Only reference official sources "
        "such as company websites or filings. Do not reference third-party "
        "sources e.g. Wikipedia."
    )

    # Send prompt
    grounding_tool = types.Tool(google_search=types.GoogleSearch())
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            tools=[grounding_tool],
            temperature=0.0,
        ),
    )
    if save_to_db:
        db_conn = st.connection("neon", type="sql")
        citations = get_citations(response)
        with db_conn.session as session:
            session.execute(
                sql_text("""
                INSERT INTO llm_logs (stock_code, prompt, response, citations)
                VALUES (:stock_code, :prompt, :response, :citations)
                """),
                {
                    "stock_code": stock_code,
                    "prompt": prompt,
                    "response": response.text,
                    "citations": "\n".join(citations) if citations else "None",
                },
            )
            session.commit()

    # Update api key counter
    st.session_state.api_key_counter += 1

    return embed_citations(response)


def format_contacts(
    client: genai.Client,
    stock_code: str,
    response_text: str,
    *,
    save_to_db: bool = True,
) -> list[Contact]:
    """
    Format key data from grounded information to specified JSON schema.
    """
    model = "gemini-2.5-flash-lite"
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
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.0,
            response_mime_type="application/json",
            response_schema=list[Contact],
        ),
    )
    contacts: list[Contact] = response.parsed

    if save_to_db and contacts:
        db_conn = st.connection("neon", type="sql")
        for contact in contacts:
            # Check if email is valid
            email_pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
            if not re.match(email_pattern, contact.email):
                continue
            # Add to contacts table
            with db_conn.session as session:
                session.execute(
                    sql_text("""
                    INSERT INTO ir_contacts (stock_code, email, name, tel, title, citations)
                    VALUES (:stock_code, :email, :name, :tel, :title, :citations)
                    ON CONFLICT (email) DO NOTHING
                    """),
                    {
                        "stock_code": stock_code,
                        "email": contact.email,
                        "name": contact.name,
                        "tel": contact.tel,
                        "title": contact.title,
                        "citations": "\n".join(contact.citations)
                        if contact.citations
                        else None,
                    },
                )
                session.commit()
        # Update last_updated_contacts_at in control table
        with db_conn.session as session:
            session.execute(
                sql_text("""
                UPDATE control
                SET last_updated_contacts_at = CURRENT_TIMESTAMP
                WHERE stock_code = :stock_code
                """),
                {
                    "stock_code": stock_code,
                },
            )
            session.commit()

    # Update api key counter
    st.session_state.api_key_counter += 1

    return contacts


# ---------------------------------------------------------------------------- #
#                               Streamlit Helpers                              #
# ---------------------------------------------------------------------------- #
# ---------------------------------- Control --------------------------------- #
def load_control_df():
    """
    Load ir_contacts table from database to session state
    """
    db_conn = st.connection("neon", type="sql")
    st.session_state.control_df = db_conn.query(
        "SELECT id, stock_code, name, last_updated_filings_at, "
        "last_updated_contacts_at, created_at "
        "FROM control ORDER BY created_at",
        ttl=0,
    )


def edit_control_df():
    """
    Save changes to control table in database
    """
    db_conn = st.connection("neon", type="sql")
    control_df = st.session_state.control_df
    control_key = st.session_state.control_key

    # Update database based on edited_rows
    if edited_rows := control_key["edited_rows"]:
        for row_idx, changes in edited_rows.items():
            code = control_df.iloc[row_idx]["stock_code"]
            # Skip if no change to name
            name = changes.get("name")
            if not name:
                continue
            with db_conn.session as session:
                session.execute(
                    sql_text("""
                    UPDATE control
                    SET name = :name
                    WHERE stock_code = :stock_code
                    """),
                    {
                        "name": name,
                        "stock_code": code,
                    },
                )
                session.commit()

    # Add stock_codes to database based on added_rows
    if added_rows := control_key["added_rows"]:
        for row in added_rows:
            code = row.get("stock_code")
            name = row.get("name")
            with db_conn.session as session:
                session.execute(
                    sql_text("""
                    INSERT INTO control (stock_code, name)
                    VALUES (:code, :name)
                    ON CONFLICT (stock_code) DO NOTHING
                    """),
                    {
                        "code": code,
                        "name": name,
                    },
                )
                session.commit()

    # Remove deleted rows from database
    if deleted_rows := control_key["deleted_rows"]:
        for row_idx in deleted_rows:
            code = control_df.iloc[row_idx]["stock_code"]
            with db_conn.session as session:
                session.execute(
                    sql_text("DELETE FROM control WHERE stock_code = :code"),
                    {"code": code},
                )
                session.commit()

    # Reset edit_control toggle
    st.session_state.edit_control = False

    # Reset session state if there are any changes
    if edited_rows or added_rows or deleted_rows:
        # Delete control_key from session state
        del st.session_state.control_key
        # Delete control_df from session state to force reloading
        del st.session_state.control_df

    st.success("Changes to Summary table saved!")


# -------------------------------- ESG Filings ------------------------------- #
def load_esg_filings_df():
    """
    Load esg_filings table from database to session state
    """
    db_conn = st.connection("neon", type="sql")
    st.session_state.esg_filings_df = db_conn.query(
        "SELECT id, stock_code, release_time, title, "
        "url, created_at "
        "FROM esg_filings ORDER BY created_at",
        ttl=0,
    )


def edit_esg_filings_df():
    """
    Save changes to esg_filings table in database
    """
    db_conn = st.connection("neon", type="sql")
    esg_filings_df = st.session_state.esg_filings_df
    esg_filings_key = st.session_state.esg_filings_key

    # Update database based on edited_rows
    if edited_rows := esg_filings_key["edited_rows"]:
        for row_idx, changes in edited_rows.items():
            pid = int(esg_filings_df.iloc[row_idx]["id"])
            # Skip if no change other than change to stock_code
            changes.pop("stock_code", None)
            if not changes:
                continue
            # Construct sql query
            set_clause = ", ".join(f"{key} = :{key}" for key in changes.keys())
            params = dict(changes.items())
            params["id"] = pid

            with db_conn.session as session:
                session.execute(
                    sql_text(f"""
                    UPDATE esg_filings
                    SET {set_clause}
                    WHERE id = :id
                    """),
                    params,
                )
                session.commit()

    # Add new rows to database based on added_rows
    if added_rows := esg_filings_key["added_rows"]:
        for row in added_rows:
            # Define allowed fields for insertion
            allowed_fields = ["stock_code", "release_time", "title", "url"]
            insert_fields = [field for field in allowed_fields if field in row]
            # Construct sql query
            fields = ", ".join(insert_fields)
            placeholders = ", ".join(f":{field}" for field in insert_fields)
            params = {field: row[field] for field in insert_fields}

            with db_conn.session as session:
                session.execute(
                    sql_text(f"""
                    INSERT INTO esg_filings ({fields})
                    VALUES ({placeholders})
                    ON CONFLICT (url) DO NOTHING
                    """),
                    params,
                )
                session.commit()

    # Remove deleted rows from esg_filings table
    if deleted_rows := esg_filings_key["deleted_rows"]:
        for row_idx in deleted_rows:
            pid = int(esg_filings_df.iloc[row_idx]["id"])
            with db_conn.session as session:
                session.execute(
                    sql_text("DELETE FROM esg_filings WHERE id = :id"),
                    {"id": pid},
                )
                session.commit()

    # Reset edit_control toggle
    st.session_state.edit_esg_filings = False

    # Reset session state if there are any changes
    if edited_rows or added_rows or deleted_rows:
        # Delete esg_filings_key from session state
        del st.session_state.esg_filings_key
        # Delete esg_filings_df from session state to force reloading
        del st.session_state.esg_filings_df

    st.success("Changes to ESG Filings table saved!")


def update_esg_filings_df(
    weeks: int,
):
    """
    Fetch ESG filings with Gemini, and trigger reload of esg_filings_df and control_df
    """
    # Calculate update_before
    update_before = datetime.now(pytz.UTC) - timedelta(weeks=int(weeks))

    # Get stock codes to update
    stock_codes = get_stock_codes_tbu(update_filings=True, update_before=update_before)

    for code in stock_codes:
        try:
            scrape(stock_code=code, save_to_db=True)
        except sqlalchemy.exc.IntegrityError:
            pass

    # Reset session state
    del st.session_state.esg_filings_df
    del st.session_state.control_df

    st.success(
        f"ESG filings updated for stock codes that are last fetched {weeks} weeks ago!"
    )


# -------------------------------- IR Contacts ------------------------------- #
def load_ir_contacts_df():
    """
    Load ir_contacts table from database to session state
    """
    db_conn = st.connection("neon", type="sql")
    st.session_state.ir_contacts_df = db_conn.query(
        "SELECT id, stock_code, email, name, tel, title, "
        "citations, created_at "
        "FROM ir_contacts ORDER BY created_at",
        ttl=0,
    )


def edit_ir_contacts_df():
    """
    Save changes to ir_contacts table in database
    """
    db_conn = st.connection("neon", type="sql")
    ir_contacts_df = st.session_state.ir_contacts_df
    ir_contacts_key = st.session_state.ir_contacts_key

    # Update database based on edited_rows
    if edited_rows := ir_contacts_key["edited_rows"]:
        for row_idx, changes in edited_rows.items():
            pid = int(ir_contacts_df.iloc[row_idx]["id"])
            # Skip if no change other than change to stock_code
            changes.pop("stock_code", None)
            if not changes:
                continue
            # Construct sql query
            set_clause = ", ".join(f"{key} = :{key}" for key in changes.keys())
            params = dict(changes.items())
            params["id"] = pid

            with db_conn.session as session:
                session.execute(
                    sql_text(f"""
                    UPDATE ir_contacts
                    SET {set_clause}
                    WHERE id = :id
                    """),
                    params,
                )
                session.commit()

    # Add new rows to database based on added_rows
    if added_rows := ir_contacts_key["added_rows"]:
        for row in added_rows:
            # Define allowed fields for insertion
            allowed_fields = [
                "stock_code",
                "email",
                "name",
                "tel",
                "title",
                "citations",
            ]
            insert_fields = [field for field in allowed_fields if field in row]
            # Construct sql query
            fields = ", ".join(insert_fields)
            placeholders = ", ".join(f":{field}" for field in insert_fields)
            params = {field: row[field] for field in insert_fields}

            with db_conn.session as session:
                session.execute(
                    sql_text(f"""
                    INSERT INTO ir_contacts ({fields})
                    VALUES ({placeholders})
                    ON CONFLICT (email) DO NOTHING
                    """),
                    params,
                )
                session.commit()

    # Remove deleted rows from ir_contacts table
    if deleted_rows := ir_contacts_key["deleted_rows"]:
        for row_idx in deleted_rows:
            pid = int(ir_contacts_df.iloc[row_idx]["id"])
            with db_conn.session as session:
                session.execute(
                    sql_text("DELETE FROM ir_contacts WHERE id = :id"),
                    {"id": pid},
                )
                session.commit()

    # Reset edit_control toggle
    st.session_state.edit_ir_contacts = False

    # Reset session state if there are any changes
    if edited_rows or added_rows or deleted_rows:
        # Delete ir_contacts_key from session state
        del st.session_state.ir_contacts_key
        # Delete ir_contacts_df from session state to force reloading
        del st.session_state.ir_contacts_df

    st.success("Changes to IR Contacts table saved!")


def update_ir_contacts_df(
    weeks: int,
):
    """
    Update IR contacts with Gemini, and trigger reload of ir_contacts_df and control_df
    """
    # Calculate update_before
    update_before = datetime.now(pytz.UTC) - timedelta(weeks=int(weeks))

    # Get stock codes to update
    stock_codes = get_stock_codes_tbu(update_contacts=True, update_before=update_before)

    for code in stock_codes:
        # NOTE: Rate limit at 5 RPM, 250k TPM, 100 RPD for Gemini 2.5 Pro
        # https://ai.google.dev/gemini-api/docs/rate-limits
        try:
            # Search contacts with Gemini
            response_text = search_contacts(
                client=get_llm_client(),
                stock_code=code,
                save_to_db=True,
            )
            # Format contacts with Gemini and save to database
            format_contacts(
                client=get_llm_client(),
                stock_code=code,
                response_text=response_text,
                save_to_db=True,
            )
        except sqlalchemy.exc.IntegrityError as exc:
            if 'null value in column "response"' in str(exc):
                msg = (
                    f"Error found while using Gemini API to extract contacts for {code}. "
                    "Might have hit rate limit. Please retry later."
                )
                st.write(msg)
            else:
                st.write(exc)
        else:
            st.success(f"IR contacts updated for {code}!")

    # Reset session state
    del st.session_state.ir_contacts_df
    del st.session_state.control_df


# --------------------------------- Download --------------------------------- #
def load_llm_logs_df():
    """
    Load llm_logs table from database to session state
    """
    db_conn = st.connection("neon", type="sql")
    st.session_state.llm_logs_df = db_conn.query(
        "SELECT id, stock_code, prompt, response, "
        "citations, created_at "
        "FROM llm_logs ORDER BY created_at",
        ttl=0,
    )


def write_to_excel():
    """
    Export dataframes to Excel
    """
    # List of dataframes
    dfs = [
        st.session_state.control_df,
        st.session_state.esg_filings_df,
        st.session_state.ir_contacts_df,
        st.session_state.llm_logs_df,
    ]
    df_names = [
        "Control",
        "ESG Filings",
        "IR Contacts",
        "LLM Logs",
    ]

    # Create an in-memory buffer for the Excel file
    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        for df, name in zip(dfs, df_names):
            # Convert id column to int
            df["id"] = df["id"].astype("Int64").fillna(pd.NA)
            # Sort dataframe by id
            df = df.sort_values(by="id")
            # Write to Excel
            df.to_excel(writer, sheet_name=name, index=False)
    output.seek(0)
    return output


# ---------------------------------------------------------------------------- #
#                                     Main                                     #
# ---------------------------------------------------------------------------- #
# Show app title and description
st.set_page_config(page_title="ListCo IAQ tracker", page_icon=":material/aq_indoor:")
st.title(":material/aq_indoor: Hong Kong ListCo IAQ Tracker")
st.write(
    """
    This app helps you track ESG disclosures on indoor air quality (IAQ)
    for HKEx-listed companies. It allows you to:
    - Maintain a wishlist of stock codes
    - Automatically download and update the latest ESG filings
    - Use AI (powered by Gemini LLM) to extract contact information of
    the IR department of listed companies for outreach
    - Export organized data to Excel for analysis and reporting
    """
)
st.divider(width="stretch")


# ---------------------------------- Control --------------------------------- #
st.subheader("Summary of Latest Filings and Contact Updates")
st.write(
    """
    View a summary of the stock codes you're tracking, including the last update dates for:
    - ESG filings from HKEx
    - IR contacts
    """
)
st.info(
    """
    ✍️ Tips:
    - Sort by column when "Edit" is toggled off.
    - Enable an edit mode by toggling "Edit" to make changes to stock code or company name.
    - To add a row, click the empty row at the bottom of the table and enter the data
    for each column. Make sure any newly added stock code is 5 digits (e.g. 00001).
    - To remove a row, click the checkbox in the first column of that row and press "Delete".
    - To modify a row, double-click the cell and type the new value.
    - Press "Enter" or "Tab", or click outside the cell to confirm after every edit.
    - Once you are done with editing, click the "Done" button to commit all of the changes.
    """,
)

# init control dataframe and edit_control
if "control_df" not in st.session_state:
    load_control_df()
if "edit_control" not in st.session_state:
    st.session_state.edit_control = False

edit_control = st.toggle("Edit", key="edit_control")
if not edit_control:
    st.dataframe(
        st.session_state.control_df,
        use_container_width=True,
        hide_index=True,
        column_config={"id": None},
    )
else:
    st.data_editor(
        st.session_state.control_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "id": None,
            "stock_code": st.column_config.TextColumn(
                max_chars=5,
                validate=r"^\d{5}$",
                required=True,
            ),
        },
        disabled=[
            "last_updated_filings_at",
            "last_updated_contacts_at",
            "created_at",
        ],
        key="control_key",
        num_rows="dynamic",
    )
    done_edit = st.button("Done", type="primary", on_click=edit_control_df)

st.divider(width="stretch")


# -------------------------------- ESG Filings ------------------------------- #
st.subheader("ESG Filings")
st.write(
    """
    View the latest ESG filings for your tracked HKEx stock codes, scraped from the HKEx website.
    Each filing includes the release date, title, and URL for easy access.
    Use the form below to fetch new filings.
    """
)

# init esg_filings dataframe and edit_esg_filings
if "esg_filings_df" not in st.session_state:
    load_esg_filings_df()
if "edit_esg_filings" not in st.session_state:
    st.session_state.edit_esg_filings = False

edit_esg_filings = st.toggle("Edit", key="edit_esg_filings")
if not edit_esg_filings:
    st.dataframe(
        st.session_state.esg_filings_df,
        use_container_width=True,
        hide_index=True,
        column_config={"id": None},
    )
else:
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
    st.button("Done", type="primary", on_click=edit_esg_filings_df)

with st.form("esg_filings_form"):
    update_threshold_weeks = st.number_input(
        label="Update filings older than this many weeks "
        "(e.g. enter 12 to refresh anything not updated in the last 3 months):",
        min_value=0,
        value=12,
    )
    submitted = st.form_submit_button(
        "Update",
        type="primary",
        on_click=update_esg_filings_df,
        kwargs={"weeks": update_threshold_weeks},
    )

st.divider(width="stretch")


# -------------------------------- IR Contacts ------------------------------- #
st.subheader("Investor Relations Contacts")
st.write(
    """
    This is a list of IR contacts for your tracked stock codes extracted
    by AI (Gemini LLM). Use the form below to refresh contacts that haven't
    been updated recently, AI will re-scan and extract any new or changed info.
    """
)
st.info(
    """
    ✍️ Notes:
    Updating IR contacts requires the use of Gemini 2.5 Pro. Since the model has
    a lower rate limit, there may be times that the update will fail mid process.
    Please retry the update after a few minutes.
    """,
)


# init ir_contacts dataframe and edit_ir_contacts
if "ir_contacts_df" not in st.session_state:
    load_ir_contacts_df()
if "edit_ir_contacts" not in st.session_state:
    st.session_state.edit_ir_contacts = False

edit_ir_contacts = st.toggle("Edit", key="edit_ir_contacts")
if not edit_ir_contacts:
    st.dataframe(
        st.session_state.ir_contacts_df,
        use_container_width=True,
        hide_index=True,
        column_config={"id": None},
    )
else:
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
                required=True,
            ),
        },
        disabled=[
            "created_at",
        ],
        key="ir_contacts_key",
        num_rows="dynamic",
    )
    st.button("Done", type="primary", on_click=edit_ir_contacts_df)

with st.form("ir_contacts_form"):
    update_threshold_weeks = st.number_input(
        label="Update contacts older than this many weeks "
        "(e.g. enter 12 to refresh anything not updated in the last 3 months):",
        min_value=0,
        value=12,
    )
    submitted = st.form_submit_button(
        "Update",
        type="primary",
        on_click=update_ir_contacts_df,
        kwargs={"weeks": update_threshold_weeks},
    )

st.divider(width="stretch")


# --------------------------------- Download --------------------------------- #
st.subheader("Download Data")
st.write(
    """
    Export full dataset, including summary, ESG filings,
    IR contacts, and conversation logs with AI to Excel.
    """
)

if st.button("Generate Excel", type="primary"):
    load_llm_logs_df()
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
        label="Download",
        data=st.session_state.excel_data,
        file_name=st.session_state.excel_filename,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        icon=":material/download:",
    )
