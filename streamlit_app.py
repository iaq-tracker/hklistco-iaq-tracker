"""Hong Kong ListCo IAQ Tracker — Streamlit entry point.

All business logic lives in the modules/ package:
  modules/db.py       — Supabase CRUD and session-state helpers
  modules/llm.py      — Gemini AI calls (grading, contacts, email drafting)
  modules/scraping.py — Selenium-based HKEx scraping

Usage:
    $ streamlit run streamlit_app.py
"""

from datetime import datetime
from datetime import timedelta
from email.message import EmailMessage
from email.policy import default

from google import genai
import pandas as pd
from pandas.api.types import (
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
)
import pytz
import sqlalchemy.exc
import streamlit as st
import streamlit_authenticator as stauth

from modules.db import (
    supabase,
    load_control_df,
    load_tabs,
    load_esg_filings,
    load_iaq_gradings,
    load_ir_contacts,
    get_company_name,
    get_stock_codes_tbu,
    prepare_chart_data,
    write_to_excel,
    edit_esg_filings_df,
    edit_iaq_grading,
    edit_ir_contacts_df,
)
from modules.llm import (
    grade_iaq,
    format_grading,
    search_contacts,
    format_contacts,
    draft_email,
)
from modules.scraping import scrape, get_company_basics


# ---------------------------------------------------------------------------- #
#                                     Main                                     #
# ---------------------------------------------------------------------------- #
st.set_page_config(
    page_title="HK ListCo IAQ Tracker",
    page_icon=":material/aq_indoor:",
    initial_sidebar_state="collapsed",
)

# ------------------------------------ Auth ---------------------------------- #
try:
    _authenticator = stauth.Authenticate(
        st.secrets["auth"]["credentials"].to_dict(),
        st.secrets["auth"]["cookie"]["name"],
        st.secrets["auth"]["cookie"]["key"],
        st.secrets["auth"]["cookie"]["expiry_days"],
    )
    _authenticator.login()
except Exception as e:
    st.error(f"Authentication error: {e}")
    st.stop()

if st.session_state.get("authentication_status") is False:
    st.error("Incorrect username or password.")
    st.stop()
elif st.session_state.get("authentication_status") is not True:
    st.warning("Please enter your username and password.")
    st.stop()

with st.sidebar:
    st.write(f"Logged in as **{st.session_state.get('name', 'Unknown')}**")
    _authenticator.logout()

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


# ---------------------------------------------------------------------------- #
#                               UI Utilities                                   #
# ---------------------------------------------------------------------------- #
def filter_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Adds a UI on top of a dataframe to let viewers filter columns."""
    modify = st.checkbox("Add Filters")
    if not modify:
        return df
    df = df.copy()
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
            if isinstance(df[column].dtype, pd.CategoricalDtype) or df[column].nunique() < 10:
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
                    df = df[df[column].astype(str).str.contains(user_text_input, case=False, na=False)]
    return df


def generate_email() -> str:
    """Build a .eml string from session-state email fields."""
    custom_policy = default.clone(max_line_length=0, linesep="\r\n")
    msg = EmailMessage(policy=custom_policy)
    msg.set_content(
        st.session_state.get("email_content", None),
        subtype="plain",
        charset="utf-8",
        cte="8bit",
    )
    msg["Subject"] = st.session_state.get("email_subject", None)
    msg["To"] = st.session_state.get("email_contacts", None)
    msg["Date"] = datetime.now(pytz.timezone("Asia/Hong_Kong")).strftime(
        "%a, %d %b %Y %H:%M:%S %z"
    )
    msg["X-Unsent"] = "1"
    return msg.as_string()


# ---------------------------------------------------------------------------- #
#                           Callback Orchestrators                             #
# ---------------------------------------------------------------------------- #
def edit_control_df() -> None:
    """Save watchlist edits to DB, trigger scraping for newly added companies."""
    control_df = st.session_state.control_df
    control_key = st.session_state.control_key

    if edited_rows := control_key["edited_rows"]:
        for row_idx, changes in edited_rows.items():
            code = control_df.iloc[row_idx]["stock_code"]
            changes.pop("stock_code", None)
            if not changes:
                continue
            supabase.table("control").update(changes).eq("stock_code", code).execute()

    if added_rows := control_key["added_rows"]:
        supabase.table("control").upsert(added_rows, on_conflict="stock_code").execute()
        for row in added_rows:
            get_company_basics(stock_code=row["stock_code"], save_to_db=True)

    if deleted_rows := control_key["deleted_rows"]:
        codes_to_delete = [
            control_df.iloc[row_idx]["stock_code"] for row_idx in deleted_rows
        ]
        supabase.table("iaq_gradings").delete().in_(
            "stock_code", codes_to_delete
        ).execute()
        supabase.table("control").delete().in_("stock_code", codes_to_delete).execute()

    st.session_state.control_toggle = False

    if edited_rows or added_rows or deleted_rows:
        del st.session_state.control_key
        load_control_df()
        st.success("Changes to the table saved successfully!")


def update_basics(stock_codes: list[str]) -> None:
    for code in stock_codes:
        get_company_basics(stock_code=code, save_to_db=True)
    load_control_df()


def update_esg_filings_df(stock_codes: list[str]) -> None:
    for code in stock_codes:
        try:
            scrape(stock_code=code, save_to_db=True)
        except sqlalchemy.exc.IntegrityError:
            pass
        else:
            st.success(f"ESG filings successfully updated for {code}!")
    load_control_df()
    if "selected_stock_code" in st.session_state:
        st.session_state.esg_filings_df = load_esg_filings(
            stock_code=st.session_state.selected_stock_code
        )


def generate_iaq_grading(stock_code: str) -> bool | None:
    try:
        response_text = grade_iaq(stock_code=stock_code, save_to_db=True)
        format_grading(stock_code=stock_code, response_text=response_text, save_to_db=True)
    except ValueError as exc:
        if "No filings found in database for" in str(exc):
            msg = "No filings found in database. Please fetch filings first."
            print(msg)
            st.warning(msg, icon="⚠️")
        elif "Null value in response text." in str(exc):
            msg = (
                "Error encountered while using Gemini API to grade IAQ "
                f"disclosures for {stock_code}. "
                "This may be due to rate limits—please try again later."
            )
            print(msg)
            st.warning(msg, icon="⚠️")
    except genai.errors.ServerError:
        msg = "The AI service is currently unavailable due to a server error. Please try again in a few moments."
        print(msg)
        st.warning(msg, icon="⚠️")
    else:
        st.success(f"IAQ grading report successfully generated for {stock_code}!")
        load_control_df()
        st.session_state.iaq_gradings_df = load_iaq_gradings(stock_code)
        return True


def generate_iaq_gradings(stock_codes: list[str]) -> None:
    for code in stock_codes:
        generate_iaq_grading(stock_code=code)


def update_ir_contacts_df(stock_codes: list[str]) -> None:
    for code in stock_codes:
        try:
            response_text = search_contacts(stock_code=code, save_to_db=True)
            format_contacts(stock_code=code, response_text=response_text, save_to_db=True)
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
            st.warning("Server error from Gemini API. Please retry later.", icon="⚠️")
        else:
            st.success(f"IR contacts updated for {code}!")
    load_control_df()
    if "selected_stock_code" in st.session_state:
        st.session_state.ir_contacts_df = load_ir_contacts(
            stock_code=st.session_state.selected_stock_code
        )


# ---------------------------------------------------------------------------- #
#                                     UI                                       #
# ---------------------------------------------------------------------------- #
st.set_page_config(page_title="HK ListCo IAQ Tracker", page_icon=":material/aq_indoor:")
st.title("Hong Kong ListCo IAQ Tracker")
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
      - **Basics:** Retrieve the company's market cap and sector classification.
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
    Use this page to **view**, **filter**, and **manage** the companies you're tracking for IAQ disclosure updates.
    """)
with st.expander("Detailed Instructions"):
    st.markdown("""
    👀 View Mode (Default)
    - **Open a Company Profile:** Click the **checkbox in the first column** beside a company's name to select it. The company's detailed profile will appear below the dashboard.
    - **Filter Your List:** Turn on **Add Filters** (above the table) to search or narrow companies by sector, market cap, or IAQ grade.
    - **Sort Columns:** Click any column header — for example, *Market Cap* or *IAQ Grade* — to sort ascending or descending.

    ✏️ Manage Watchlist (Edit Mode)
    - **Switch to Manage Mode:** Toggle **Manage Watchlist** at the top-left of the dashboard.
      This lets you add or remove companies from your list.
    - **Add a Company:** Scroll to the blank row at the bottom and enter a 5-digit stock code (e.g. `00005`). The company's name, sector, and market data will fill in automatically.
    - **Update Info:** Click a cell to edit its value, then press *Enter* key or click away to confirm.
    - **Remove a Company:** Tick the checkbox next to a company, then press **Delete** key.
    - **Save Your Watchlist:** Click **Save Watchlist** when finished to apply your changes. *Your updates won't be stored until you save.*
    """)

if "control_df" not in st.session_state:
    load_control_df()
if "control_toggle" not in st.session_state:
    st.session_state.control_toggle = False

edit_control = st.toggle("Manage Watchlist", key="control_toggle")

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
    if selected_row["selection"]["rows"]:
        st.session_state.selected_stock_code = st.session_state.control_df.iloc[
            selected_row["selection"]["rows"][0]
        ]["stock_code"]

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
        disabled=["created_at"],
        key="control_key",
        num_rows="dynamic",
    )
    st.button("Save Watchlist", type="primary", on_click=edit_control_df)

st.divider()


# ------------------------------ Company Profile ----------------------------- #
if "selected_stock_code" not in st.session_state:
    st.subheader("🏢 Company Profile")
    st.info("Select a stock code from *Dashboard* to view its details.")
else:
    if ("prev_selected_stock_code" not in st.session_state) or (
        st.session_state.selected_stock_code
        != st.session_state.prev_selected_stock_code
    ):
        load_tabs()
        st.session_state.selected_company_name = get_company_name(
            stock_code=st.session_state.selected_stock_code
        )
        if "email_content" in st.session_state:
            del st.session_state.email_content
        if "email" in st.session_state:
            del st.session_state.email
        st.session_state.prev_selected_stock_code = st.session_state.selected_stock_code

    if (
        "selected_company_name" in st.session_state
        and st.session_state.selected_company_name
    ):
        st.subheader(
            f"Company Profile: {st.session_state.selected_company_name} ({st.session_state.selected_stock_code})"
        )
    else:
        st.subheader(f"Company Profile: {st.session_state.selected_stock_code}")

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
        st.caption(
            "Note: HKEx calculates market cap based on HKEx-listed shares only which may not reflect the company's total global market capitalization (e.g., for H-shares, Depositary Receipts (DRs), or dual-listed companies)."
        )

        st.button(
            "Refresh Basics",
            type="primary",
            on_click=update_basics,
            kwargs={"stock_codes": [st.session_state.selected_stock_code]},
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
                kwargs={"stock_codes": [st.session_state.selected_stock_code]},
            )
        else:
            if "esg_filings_toggle" not in st.session_state:
                st.session_state.esg_filings_toggle = False
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
                    kwargs={"stock_codes": [st.session_state.selected_stock_code]},
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
                        "release_time": st.column_config.DatetimeColumn(required=True),
                        "title": st.column_config.TextColumn(required=True),
                        "url": st.column_config.TextColumn(required=True),
                    },
                    disabled=["created_at"],
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
            ⭐ **Tip:** For the most accurate assessment, always refresh ESG filings first. The AI will then generate a new report based on the **15 most recent filings** to track progress over time.
            """
        )

        st.markdown("#### Historical Reports")

        if st.session_state.iaq_gradings_df.empty:
            st.info(
                "No IAQ grading reports are currently stored for this company. You can generate one below."
            )
        else:
            st.write("Click on a report to expand its details and edit if needed.")
            for index, report_data in st.session_state.iaq_gradings_df.iterrows():
                grade_id = report_data["id"]
                expander_title = f"Report from {report_data['grading_date'].strftime('%d %B %Y')}  |  Grade: {report_data.get('grade', 'N/A')}"

                with st.expander(expander_title):
                    if not st.session_state.esg_filings_df.empty:
                        latest_filing_date = st.session_state.esg_filings_df[
                            "release_time"
                        ].iloc[0]
                        if report_data["grading_date"] < latest_filing_date:
                            st.warning(
                                "This report may be outdated as newer ESG filings have been published since it was created.",
                                icon="⚠️",
                            )

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

        with st.expander("View Evaluation Criteria"):
            st.markdown("""
            The AI grades the company's IAQ disclosures based on the following criteria:
            - **Length and Detail:** Short/vague mentions vs. dedicated sections with data.
            - **Key Performance Indicators (KPIs):** Presence of quantifiable metrics.
            - **Consistency:** How regularly KPIs are reported over time.
            - **Progression:** Emphasis on improvements in recent years.
            """)

        if st.button("Generate Report", type="primary"):
            success = generate_iaq_grading(stock_code=st.session_state.selected_stock_code)
            if success:
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
            if "ir_contacts_toggle" not in st.session_state:
                st.session_state.ir_contacts_toggle = False
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
                    disabled=["created_at"],
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
            st.text_area("Content", key="email_content")

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
        value=52,
    )

    col1, col2, col3, col4 = st.columns([0.25, 0.25, 0.25, 0.25])

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
            "Generate Reports",
            type="primary",
            on_click=generate_iaq_gradings,
            kwargs={
                "stock_codes": get_stock_codes_tbu(
                    update_grades=True,
                    update_before=datetime.now(pytz.UTC) - timedelta(weeks=int(weeks)),
                )
            },
            use_container_width=True,
        )
    with col4:
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

chart_df = prepare_chart_data()
if not chart_df.empty:
    available_years = chart_df.index.unique().tolist()

    selected_years = st.multiselect(
        "Filter by Year:", options=available_years, default=available_years
    )

    if selected_years:
        filtered_chart_df = chart_df[chart_df.index.isin(selected_years)]

        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown("#### Grade Distribution Over Time")
            st.bar_chart(filtered_chart_df)
        with col2:
            st.markdown("#### Grade Summary")
            st.dataframe(filtered_chart_df, use_container_width=True)
    else:
        st.info("Please select at least one year to display trends.")
else:
    st.info(
        "Not enough historical data to generate a trend chart. Start by generating some grading reports!"
    )

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
