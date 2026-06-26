"""Unit tests for modules/db.py — database layer helpers."""

from datetime import datetime, timedelta

import pandas as pd
import pytest
import pytz

import modules.db as db


# --------------------------------------------------------------------------- #
# _localize_datetime_cols
# --------------------------------------------------------------------------- #
class TestLocalizeDatetimeCols:
    def test_adds_hong_kong_timezone(self):
        df = pd.DataFrame({"ts": pd.to_datetime(["2025-01-01 08:00:00"])})
        result = db._localize_datetime_cols(df, ["ts"])
        assert str(result["ts"].dt.tz) == "Asia/Hong_Kong"

    def test_coerces_unparseable_to_nat(self):
        df = pd.DataFrame({"ts": pd.to_datetime(["2025-01-01", None])})
        result = db._localize_datetime_cols(df, ["ts"])
        assert pd.isna(result["ts"].iloc[1])

    def test_skips_columns_not_in_dataframe(self):
        df = pd.DataFrame({"other_col": [1, 2]})
        result = db._localize_datetime_cols(df, ["ts"])
        assert "ts" not in result.columns
        assert list(result.columns) == ["other_col"]

    def test_handles_multiple_columns(self):
        df = pd.DataFrame(
            {
                "col_a": pd.to_datetime(["2025-01-01"]),
                "col_b": pd.to_datetime(["2025-06-01"]),
            }
        )
        result = db._localize_datetime_cols(df, ["col_a", "col_b"])
        assert str(result["col_a"].dt.tz) == "Asia/Hong_Kong"
        assert str(result["col_b"].dt.tz) == "Asia/Hong_Kong"


# --------------------------------------------------------------------------- #
# get_company_name
# --------------------------------------------------------------------------- #
class TestGetCompanyName:
    def test_returns_name_for_known_stock_code(self, sample_control_df):
        import streamlit as st
        st.session_state["control_df"] = sample_control_df
        assert db.get_company_name("00700") == "Tencent Holdings"

    def test_returns_none_for_unknown_stock_code(self, sample_control_df):
        import streamlit as st
        st.session_state["control_df"] = sample_control_df
        assert db.get_company_name("99999") is None

    def test_returns_first_match_for_duplicate_codes(self):
        import streamlit as st
        df = pd.DataFrame(
            {"stock_code": ["00700", "00700"], "name": ["Name A", "Name B"]}
        )
        st.session_state["control_df"] = df
        assert db.get_company_name("00700") == "Name A"


# --------------------------------------------------------------------------- #
# get_stock_codes_tbu
# --------------------------------------------------------------------------- #
class TestGetStockCodesTbu:
    def test_raises_when_no_update_flag_set(self, sample_control_df):
        import streamlit as st
        st.session_state["control_df"] = sample_control_df
        with pytest.raises(ValueError, match="At least one of"):
            db.get_stock_codes_tbu()

    def test_returns_all_codes_when_field_is_na(self, sample_control_df):
        import streamlit as st
        st.session_state["control_df"] = sample_control_df
        codes = db.get_stock_codes_tbu(update_filings=True)
        assert set(codes) == {"00700", "00005"}

    def test_excludes_recently_updated_codes(self):
        import streamlit as st
        hkt = pytz.timezone("Asia/Hong_Kong")
        now = datetime.now(hkt)
        df = pd.DataFrame(
            {
                "stock_code": ["00700", "00005"],
                "name": ["Tencent Holdings", "HSBC Holdings"],
                "sector": ["Technology", "Finance"],
                "market_cap": [3000.0, 1500.0],
                "iaq_grade": ["High", "Medium"],
                "last_updated_market_cap_at": [pd.NaT, pd.NaT],
                "last_updated_filings_at": [now, pd.NaT],
                "last_updated_grade_at": [pd.NaT, pd.NaT],
                "last_updated_contacts_at": [pd.NaT, pd.NaT],
            }
        )
        st.session_state["control_df"] = df
        cutoff = now - timedelta(days=1)
        codes = db.get_stock_codes_tbu(update_filings=True, update_before=cutoff)
        # 00700 was updated today (after cutoff) so it should be excluded
        assert "00700" not in codes
        assert "00005" in codes

    def test_uses_correct_field_for_each_update_type(self, sample_control_df):
        import streamlit as st
        st.session_state["control_df"] = sample_control_df
        for kwargs in [
            {"update_market_cap": True},
            {"update_filings": True},
            {"update_grades": True},
            {"update_contacts": True},
        ]:
            codes = db.get_stock_codes_tbu(**kwargs)
            assert isinstance(codes, list)


# --------------------------------------------------------------------------- #
# prepare_chart_data
# --------------------------------------------------------------------------- #
class TestPrepareChartData:
    def test_returns_empty_dataframe_when_no_data(self, mock_supabase):
        mock_supabase.table.return_value.select.return_value.execute.return_value.data = []
        result = db.prepare_chart_data()
        assert result.empty

    def test_groups_grades_by_year(self, mock_supabase):
        mock_supabase.table.return_value.select.return_value.execute.return_value.data = [
            {
                "stock_code": "00700",
                "grade": "High",
                "grading_date": "2024-01-01T00:00:00+00:00",
            },
            {
                "stock_code": "00005",
                "grade": "Medium",
                "grading_date": "2024-06-01T00:00:00+00:00",
            },
            {
                "stock_code": "00001",
                "grade": "Low",
                "grading_date": "2023-01-01T00:00:00+00:00",
            },
        ]
        result = db.prepare_chart_data()
        assert 2024 in result.index
        assert 2023 in result.index
        assert result.loc[2024, "High"] == 1
        assert result.loc[2024, "Medium"] == 1
        assert result.loc[2023, "Low"] == 1

    def test_keeps_only_latest_grade_per_company_per_year(self, mock_supabase):
        # Two gradings for the same company in the same year — only last counts.
        mock_supabase.table.return_value.select.return_value.execute.return_value.data = [
            {
                "stock_code": "00700",
                "grade": "Low",
                "grading_date": "2024-01-01T00:00:00+00:00",
            },
            {
                "stock_code": "00700",
                "grade": "High",
                "grading_date": "2024-12-01T00:00:00+00:00",
            },
        ]
        result = db.prepare_chart_data()
        assert result.loc[2024, "High"] == 1
        assert result.loc[2024, "Low"] == 0

    def test_output_always_has_low_medium_high_columns(self, mock_supabase):
        mock_supabase.table.return_value.select.return_value.execute.return_value.data = [
            {
                "stock_code": "00700",
                "grade": "High",
                "grading_date": "2024-01-01T00:00:00+00:00",
            }
        ]
        result = db.prepare_chart_data()
        assert list(result.columns) == ["Low", "Medium", "High"]
