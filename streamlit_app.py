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
            background-color: var(--secondary-background-color, #333333); /* Fallback to dark gray */
            color: var(--text-color, #ffffff); /* Fallback to white */
            cursor: pointer;
            margin-right: -1px;
            border-radius: 4px 4px 0 0;
            transition: background-color 0.3s;
        }
        /* Hover effect for unselected tabs */
        div.stRadio > div[role="radiogroup"] > label[data-baseweb="radio"] > div:hover {
            background-color: rgba(255, 255, 255, 0.1);
        }
        /* Selected tab: Force primary color */
        div.stRadio > div[role="radiogroup"] > label[data-baseweb="radio"] > input[type="radio"]:checked + div {
            background-color: var(--primary-color, #ff4b4b) !important; /* Fallback to red */
            border-bottom: 1px solid var(--primary-color, #ff4b4b) !important;
            color: #ffffff !important; /* White for contrast */
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

    # Determine the field to filter
    field = "last_updated_filings_at" if update_filings else "last_updated_contacts_at"

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
    company_name = None
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
        # extract company name
        if (not company_name) or (company_name == ""):
            company_name = cells[2].text.split("\n")[0]
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
        if company_name:
            supabase.table("control").update({"name": company_name}).eq(
                "stock_code", stock_code
            ).is_("name", None).execute()


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
        supabase.table("iaq_gradings").upsert(
            {
                "stock_code": stock_code,
                "overview": grade.overview,
                "justification": grade.justification,
                "extracts": grade.extracts,
                "updated_at": datetime.now(pytz.UTC).isoformat(),
            },
            on_conflict="stock_code",
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
        model="gemini-2.5-pro",
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
    prompt = f"""Imagine you represent an ESG consultant from the Hong Kong-based NGO ({st.secrets.NGO_URL}). Draft a professional introduction email to the Investor Relations (IR) department of the company with a stock ticker of '{st.session_state.selected_stock_code}'"""

    if (
        "selected_company_name" in st.session_state
        and st.session_state.selected_company_name
    ):
        prompt += f", {st.session_state.selected_company_name}"

    if f"justification_{st.session_state.selected_stock_code}" in st.session_state:
        prompt += f". Consider the following assessment on its disclosures with regard to IAQ: {st.session_state[f'justification_{st.session_state.selected_stock_code}']}"

    prompt += """. The goal is to discuss ways to implement Indoor Air Quality (IAQ) best practices and improve
    IAQ disclosures in their ESG reports. Key points to include:
    - Polite introduction and context on HKEX ESG guidelines.
    - Suggest next steps (e.g., meeting to share best practices).
    - End with a call to action.

    Draft and return the body of the email only. Keep it concise (200-300 words), professional, and positive. Do not include email subject, recipient line and signature etc.
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

    response = q.order("stock_code", desc=False).execute()
    df = pd.DataFrame(response.data)

    # convert datetime fields
    if not df.empty:
        datetime_columns = ["updated_at"]
        for col in datetime_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce", utc=False)
                # Localize to Asia/Hong_Kong
                df[col] = df[col].dt.tz_localize(
                    "Asia/Hong_Kong", ambiguous="raise", nonexistent="raise"
                )
        # Merge with control_df to add iaq_grade column
        df = df.merge(
            st.session_state.control_df[["stock_code", "iaq_grade"]],
            on="stock_code",
            how="left",
        )

    return df


def edit_iaq_grading(
    stock_code: str,
):
    """
    Save changes to control and/or iaq_gradings table
    """
    # Fetch values from session state using keys
    grade = st.session_state.get(f"grade_{stock_code}", "")
    overview = st.session_state.get(f"overview_{stock_code}", "")
    justification = st.session_state.get(f"justification_{stock_code}", "")
    extracts = st.session_state.get(f"extracts_{stock_code}", "")

    # Update control if changes made to grade
    if grade != st.session_state.iaq_gradings_df["iaq_grade"].iloc[0]:
        supabase.table("control").update({"iaq_grade": grade}).eq(
            "stock_code", stock_code
        ).execute()
        # Reload control_df from session state
        load_control_df()
        # Reload iaq_grading from session state
        st.session_state.iaq_gradings_df = load_iaq_gradings(stock_code)

    # Update iaq_gradings if changes made to overview, justification or extracts
    if (
        overview != st.session_state.iaq_gradings_df["overview"].iloc[0]
        or justification != st.session_state.iaq_gradings_df["justification"].iloc[0]
        or extracts != st.session_state.iaq_gradings_df["extracts"].iloc[0]
    ):
        supabase.table("iaq_gradings").update(
            {
                "overview": overview,
                "justification": justification,
                "extracts": extracts,
            }
        ).eq("stock_code", stock_code).execute()
        # Reload iaq_grading from session state
        st.session_state.iaq_gradings_df = load_iaq_gradings(stock_code)


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
    This tool helps you monitor and analyze Indoor Air Quality (IAQ) disclosures
    in the ESG reports of Hong Kong-listed companies.

    **Key Features:**
    *   **Dashboard:** Manage your watchlist of companies, see their AI-generated IAQ grades, and track when their data was last refreshed.
    *   **Company Profiles:** Select a company to dive deeper and view its key data, including:
        *   **ESG Filings:** Pull the latest ESG filings directly from the HKEx website.
        *   **IAQ Grading:** Generate a detailed report and grade on the company's IAQ disclosure quality with AI.
        *   **IR Contacts:** Use AI to find and extract up-to-date Investor Relations contacts.
        *   **Outreach Email:** Draft a professional outreach email based on the aforementioned data.
    *   **Bulk Updates:** Save time by refreshing ESG filings and IR contacts for multiple companies at once.
    *   **Data Export:** Download your entire dataset, including dashboard, ESG filings, IAQ gradings, IR contacts, and AI logs, to a single Excel file.
    """
)
st.divider()


# ---------------------------------- Control --------------------------------- #
st.subheader("Dashboard")
st.write(
    """
    This is your central dashboard. Add companies you want to monitor to the watchlist,
    see their latest IAQ grade, and check when their data was last refreshed.
    Toggle the 'Edit Mode' button to switch between Display and Edit Mode.

    **Display Mode:** Your main view for browsing and selecting companies.
    - **Select a company:** Select the checkbox next to the row to load its detailed profile in the next section.
    - **Sort data:** Click on a column header to sort the table by that column.

    **Edit Mode:** Your view for managing your watchlist.
    - **Add a company:** Scroll to the bottom and fill the blank row at the bottom with a 5-digit stock code (e.g., 00005). The company name is optional and will be auto-populated when you fetch its ESG filings.
    - **Edit a cell:** Double-click any cell to make changes, then press the `Enter` key or click away to confirm.
    - **Delete a company:** Select the checkbox next to the row and press the `Delete` key.
    - Click **"Save Changes"** below to apply all your edits to the database.
    """
)

# init control dataframe and edit_control
if "control_df" not in st.session_state:
    load_control_df()
if "control_toggle" not in st.session_state:
    st.session_state.control_toggle = False

# display edit control toggle
edit_control = st.toggle("Edit Mode", key="control_toggle")

# show dataframe in display mode
if not edit_control:
    # user to select a stock code
    selected_row = st.dataframe(
        st.session_state.control_df,
        use_container_width=True,
        hide_index=True,
        column_config={"id": None},
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
    done_edit = st.button("Save Changes", type="primary", on_click=edit_control_df)

st.divider()


# # ------------------------------ Company Profile ----------------------------- #
if "selected_stock_code" not in st.session_state:
    st.subheader("Company Profile")
    st.info("Select a stock code from dashboard to view its details.")
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
    tab_lst = ["ESG Filings", "IAQ Grading", "IR Contacts", "Outreach Email"]
    active_tab = st.radio("", tab_lst, horizontal=True, key="active_tab")
    # Tab 1: ESG Filings
    if active_tab == tab_lst[0]:
        st.write(
            "Review the official ESG reports and announcements for this company, fetched directly from HKEx and sorted by the most recent date."
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
            edit_esg_filings = st.toggle("Edit Mode", key="esg_filings_toggle")

            if not edit_esg_filings:
                st.dataframe(
                    st.session_state.esg_filings_df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={"id": None},
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

    # Tab 2: IAQ Grading
    elif active_tab == tab_lst[1]:
        st.write("""
            Access the AI-generated report that grades the company's IAQ disclosure quality based on the filings in the previous tab.

            ⭐ **Tips:** For the most accurate assessment, refresh the filings first if they seem outdated.
        """)

        if st.session_state.iaq_gradings_df.empty:
            st.info(
                f"No IAQ grading report is currently stored for {st.session_state.selected_stock_code}. Generate one below!"
            )
        else:
            # Compile data
            data = st.session_state.iaq_gradings_df.iloc[0]

            # Flag if report is outdated
            if (
                data.get("updated_at")
                < st.session_state.esg_filings_df.iloc[0]["release_time"]
            ):
                st.info(
                    "⚠️ Report Outdated: Newer ESG filings have been fetched since this IAQ report was generated. Regenerate the report to include the latest data in the analysis.",
                )

            # Define keys based on selected stock code
            with st.form("iaq_grading_form"):
                st.text_input(
                    "IAQ Grade",
                    value=data.get("iaq_grade", ""),
                    key=f"grade_{st.session_state.selected_stock_code}",
                )
                st.text_area(
                    "Company Overview",
                    value=data.get("overview", ""),
                    height=150,
                    key=f"overview_{st.session_state.selected_stock_code}",
                )
                st.text_area(
                    "Justification",
                    value=data.get("justification", ""),
                    height=200,
                    key=f"justification_{st.session_state.selected_stock_code}",
                )
                st.text_area(
                    "Extracts",
                    value=data.get("extracts", ""),
                    height=400,
                    key=f"extracts_{st.session_state.selected_stock_code}",
                )
                st.form_submit_button(
                    "Save Changes",
                    type="primary",
                    on_click=edit_iaq_grading,
                    kwargs={"stock_code": st.session_state.selected_stock_code},
                )

        # Generate/Update IAQ Grading
        st.button(
            label="Generate Report"
            if st.session_state.iaq_gradings_df.empty
            else "Regenerate Report",
            type="primary",
            on_click=update_iaq_grading,
            kwargs={
                "stock_code": st.session_state.selected_stock_code,
            },
        )

    # Tab 3: IR Contacts
    elif active_tab == "IR Contacts":
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
            edit_ir_contacts = st.toggle("Edit Mode", key="ir_contacts_toggle")

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

    # Tab 4: Outreach Email
    elif active_tab == "Outreach Email":
        st.write(
            """
        Draft a professional outreach email based on data from the previous tabs and
        generate a .eml file. Use AI to autofill the content for a first draft.

        ⭐ **Tips:** The email content is temporary and will not be saved. To keep a copy, please generate and download the email as a .eml file.
        """
        )

        if "email_content" not in st.session_state:
            st.session_state.email_content = ""

        # set up email form
        with st.form("email_form"):
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
                value=f"Enhancing ESG Disclosures on Indoor Air Quality: A Partnership Opportunity with {st.secrets.NGO_NAME}",
                key="email_subject",
            )
            st.text_area(
                "Content",
                key="email_content",
            )

            col1, col2, _ = st.columns([0.25, 0.25, 0.5])
            with col1:
                st.form_submit_button(
                    "Autofill Content",
                    type="secondary",
                    on_click=draft_email,
                    use_container_width=True,
                )
            with col2:
                submitted = st.form_submit_button(
                    "Generate Email", type="primary", use_container_width=True
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
                label="Download Email",
                data=st.session_state.email,
                file_name=st.session_state.email_filename,
                mime="message/rfc822",
                icon=":material/download:",
            )

st.divider()


# ------------------------------- Bulk Updates ------------------------------- #
st.subheader("Bulk Updates")
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

    col1, col2, _ = st.columns([0.25, 0.25, 0.5])
    with col1:
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
    with col2:
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


# --------------------------------- Download --------------------------------- #
st.subheader("Data Export")
st.write(
    """
    Download your entire dataset to a single Excel file.
    This includes your dashboard watchlist, all collected ESG filings, IAQ gradings,
    IR contacts, and the AI interaction (LLM) logs.
    """
)

if st.button("Generate Excel", type="primary"):
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
        label="Download Excel",
        data=st.session_state.excel_data,
        file_name=st.session_state.excel_filename,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        icon=":material/download:",
    )
