"""Database layer — Supabase CRUD operations and session-state data helpers."""

from datetime import datetime
from io import BytesIO

from dateutil import parser
import pandas as pd
import pytz
import streamlit as st
from supabase import create_client


# @st.cache_resource keeps one client alive for the whole Streamlit process;
# without it a new client would be created on every page rerun.
@st.cache_resource
def init_connection():
    url = st.secrets.connections.supabase.SUPABASE_URL
    key = st.secrets.connections.supabase.SUPABASE_KEY
    return create_client(url, key)


# Module-level singleton — llm.py and scraping.py import this same object,
# so all three modules share the one cached client.
supabase = init_connection()


def _localize_datetime_cols(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Convert listed columns to Asia/Hong_Kong-localized datetimes in-place."""
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", utc=False)
            # HK observes no DST so ambiguous/nonexistent transitions can never
            # occur in practice — "raise" here acts as a data-integrity guard.
            df[col] = df[col].dt.tz_localize(
                "Asia/Hong_Kong", ambiguous="raise", nonexistent="raise"
            )
    return df


# ---------------------------------------------------------------------------- #
#                                  Data Loaders                                #
# ---------------------------------------------------------------------------- #
def load_control_df() -> None:
    response = (
        supabase.table("control").select("*").order("market_cap", desc=True).execute()
    )
    df = pd.DataFrame(response.data)

    if not df.empty:
        hkt_tz = pytz.timezone("Asia/Hong_Kong")
        datetime_cols = [
            "last_updated_market_cap_at",
            "last_updated_filings_at",
            "last_updated_grade_at",
            "last_updated_contacts_at",
            "created_at",
        ]
        for col in datetime_cols:
            if col in df.columns:
                # dateutil.parser handles the mixed ISO formats Supabase returns.
                # .astimezone() converts from the stored UTC value to HKT.
                df[col] = df[col].apply(
                    lambda x: parser.parse(x).astimezone(hkt_tz)
                    if pd.notnull(x)
                    else pd.NaT
                )

    st.session_state.control_df = df


def load_tabs() -> None:
    st.session_state.esg_filings_df = load_esg_filings(
        stock_code=st.session_state.selected_stock_code
    )
    st.session_state.iaq_gradings_df = load_iaq_gradings(
        stock_code=st.session_state.selected_stock_code
    )
    st.session_state.ir_contacts_df = load_ir_contacts(
        stock_code=st.session_state.selected_stock_code
    )


def load_esg_filings(stock_code: str | None) -> pd.DataFrame:
    q = supabase.table("esg_filings").select(
        "id", "stock_code", "release_time", "title", "url", "created_at"
    )
    if stock_code:
        q = q.eq("stock_code", stock_code)
    response = (
        q.order("stock_code", desc=False).order("release_time", desc=True).execute()
    )
    df = pd.DataFrame(response.data)
    if not df.empty:
        df = _localize_datetime_cols(df, ["release_time", "created_at"])
    return df


def load_iaq_gradings(stock_code: str | None) -> pd.DataFrame:
    q = supabase.table("iaq_gradings").select("*")
    if stock_code:
        q = q.eq("stock_code", stock_code)
    response = q.order("grading_date", desc=True).execute()
    df = pd.DataFrame(response.data)
    if not df.empty:
        df = _localize_datetime_cols(df, ["grading_date"])
    return df


def load_ir_contacts(stock_code: str | None) -> pd.DataFrame:
    q = supabase.table("ir_contacts").select("*")
    if stock_code:
        q = q.eq("stock_code", stock_code)
    response = q.order("stock_code", desc=False).execute()
    df = pd.DataFrame(response.data)
    if not df.empty:
        df = _localize_datetime_cols(df, ["created_at"])
    return df


def load_llm_logs() -> pd.DataFrame:
    response = (
        supabase.table("llm_logs").select("*").order("created_at", desc=True).execute()
    )
    df = pd.DataFrame(response.data)
    if not df.empty:
        df = _localize_datetime_cols(df, ["created_at"])
    return df


def load_all_iaq_gradings() -> pd.DataFrame:
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


def write_to_excel() -> BytesIO:
    dfs = [
        st.session_state.control_df,
        load_esg_filings(stock_code=None),
        load_iaq_gradings(stock_code=None),
        load_ir_contacts(stock_code=None),
        load_llm_logs(),
    ]
    df_names = ["Control", "ESG Filings", "IAQ Gradings", "IR Contacts", "LLM Logs"]

    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        for df, name in zip(dfs, df_names):
            df_to_export = df.copy()
            for col in df_to_export.columns:
                if (
                    pd.api.types.is_datetime64_any_dtype(df_to_export[col])
                    and df_to_export[col].dt.tz is not None
                ):
                    # xlsxwriter cannot write tz-aware datetimes; strip tz info.
                    df_to_export[col] = df_to_export[col].dt.tz_localize(None)
            if "id" in df_to_export.columns:
                df_to_export["id"] = df_to_export["id"].astype("Int64").fillna(pd.NA)
                df_to_export = df_to_export.sort_values(by="id")
            df_to_export.to_excel(writer, sheet_name=name, index=False)
    output.seek(0)
    return output


# ---------------------------------------------------------------------------- #
#                                   Helpers                                    #
# ---------------------------------------------------------------------------- #
def get_company_name(stock_code: str) -> str | None:
    condition = st.session_state.control_df["stock_code"] == stock_code
    result_df = st.session_state.control_df[condition][["name"]]
    return result_df["name"].iloc[0] if not result_df.empty else None


def get_stock_codes_tbu(
    *,
    update_market_cap: bool = False,
    update_filings: bool = False,
    update_grades: bool = False,
    update_contacts: bool = False,
    update_before: datetime | None = None,
) -> list[str]:
    if not (update_market_cap or update_filings or update_grades or update_contacts):
        raise ValueError(
            "At least one of update_market_cap, update_filings, update_grades, "
            "or update_contacts must be True"
        )
    if update_market_cap:
        field = "last_updated_market_cap_at"
    elif update_filings:
        field = "last_updated_filings_at"
    elif update_grades:
        field = "last_updated_grade_at"
    else:
        field = "last_updated_contacts_at"

    condition = st.session_state.control_df[field].isna()
    if update_before:
        update_before = pd.to_datetime(update_before, utc=True)
        # |= extends the boolean mask to also include rows with a stale timestamp.
        condition |= st.session_state.control_df[field] <= update_before

    result_df = st.session_state.control_df[condition][["stock_code"]]
    return result_df["stock_code"].dropna().tolist()


# ---------------------------------------------------------------------------- #
#                                Data Editors                                  #
# ---------------------------------------------------------------------------- #
def edit_esg_filings_df() -> None:
    if edited_rows := st.session_state.esg_filings_key["edited_rows"]:
        for row_idx, changes in edited_rows.items():
            pid = int(st.session_state.esg_filings_df.iloc[row_idx]["id"])
            changes.pop("stock_code", None)
            if not changes:
                continue
            supabase.table("esg_filings").update(changes).eq("id", pid).execute()

    if added_rows := st.session_state.esg_filings_key["added_rows"]:
        allowed_fields = ["stock_code", "release_time", "title", "url"]
        rows_to_insert = [
            {field: row[field] for field in allowed_fields if field in row}
            for row in added_rows
        ]
        # Drop rows where the user added a blank row without filling any fields.
        rows_to_insert = [r for r in rows_to_insert if r]
        for row in rows_to_insert:
            row["created_at"] = datetime.now(pytz.UTC).isoformat()
        if rows_to_insert:
            supabase.table("esg_filings").upsert(
                rows_to_insert, ignore_duplicates=True, on_conflict="url"
            ).execute()

    if deleted_rows := st.session_state.esg_filings_key["deleted_rows"]:
        ids_to_delete = [
            int(st.session_state.esg_filings_df.iloc[row_idx]["id"])
            for row_idx in deleted_rows
        ]
        supabase.table("esg_filings").delete().in_("id", ids_to_delete).execute()

    st.session_state.esg_filings_toggle = False

    if edited_rows or added_rows or deleted_rows:
        # Delete the widget key so Streamlit re-renders the data_editor with
        # fresh data; leaving it in place would re-apply stale edit state.
        del st.session_state.esg_filings_key
        st.session_state.esg_filings_df = load_esg_filings(
            stock_code=st.session_state.selected_stock_code
        )
        st.success("Changes to the table saved successfully!")


def edit_iaq_grading(grade_id: int, stock_code: str) -> None:
    grade = st.session_state.get(f"grade_{grade_id}", "")
    overview = st.session_state.get(f"overview_{grade_id}", "")
    justification = st.session_state.get(f"justification_{grade_id}", "")
    extracts = st.session_state.get(f"extracts_{grade_id}", "")

    supabase.table("iaq_gradings").update(
        {
            "grade": grade,
            "overview": overview,
            "justification": justification,
            "extracts": extracts,
        }
    ).eq("id", grade_id).execute()

    latest_grade_id = st.session_state.iaq_gradings_df["id"].iloc[0]
    if grade_id == latest_grade_id:
        supabase.table("control").update(
            {
                "iaq_grade": grade,
                "last_updated_grade_at": datetime.now(pytz.UTC).isoformat(),
            }
        ).eq("stock_code", stock_code).execute()
        load_control_df()

    st.session_state.iaq_gradings_df = load_iaq_gradings(stock_code)
    st.success("Changes to the grading report have been saved.")


def edit_ir_contacts_df() -> None:
    if edited_rows := st.session_state.ir_contacts_key["edited_rows"]:
        for row_idx, changes in edited_rows.items():
            pid = int(st.session_state.ir_contacts_df.iloc[row_idx]["id"])
            changes.pop("stock_code", None)
            if not changes:
                continue
            supabase.table("ir_contacts").update(changes).eq("id", pid).execute()

    if added_rows := st.session_state.ir_contacts_key["added_rows"]:
        allowed_fields = ["stock_code", "email", "name", "tel", "title", "citations"]
        rows_to_insert = [
            {field: row[field] for field in allowed_fields if field in row}
            for row in added_rows
        ]
        if rows_to_insert:
            supabase.table("ir_contacts").upsert(
                rows_to_insert, ignore_duplicates=True, on_conflict="stock_code,email"
            ).execute()

    if deleted_rows := st.session_state.ir_contacts_key["deleted_rows"]:
        ids_to_delete = [
            int(st.session_state.ir_contacts_df.iloc[row_idx]["id"])
            for row_idx in deleted_rows
        ]
        supabase.table("ir_contacts").delete().in_("id", ids_to_delete).execute()

    st.session_state.ir_contacts_toggle = False

    if edited_rows or added_rows or deleted_rows:
        # Same pattern as edit_esg_filings_df: clear widget key before refresh.
        del st.session_state.ir_contacts_key
        st.session_state.ir_contacts_df = load_ir_contacts(
            stock_code=st.session_state.selected_stock_code
        )
        st.success("Changes to the table saved successfully!")
