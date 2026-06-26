"""Shared fixtures and streamlit/supabase mocks for all unit tests.

These mocks are injected into sys.modules at import time so that the app
modules never try to connect to real external services during tests.
"""

import sys
from unittest.mock import MagicMock

import pandas as pd
import pytest


# --------------------------------------------------------------------------- #
# Streamlit mock
# Must be placed in sys.modules BEFORE any app module is imported, because
# modules/db.py calls st.cache_resource and reads st.secrets at module level.
# --------------------------------------------------------------------------- #
class _FakeSessionState(dict):
    """Dict that also supports attribute access, mirroring Streamlit's session_state."""

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError:
            raise AttributeError(key)


class _FakeSecrets:
    """Minimal replica of the secrets.toml structure used by the app."""

    GEMINI_API_KEYS = ["test-key-1", "test-key-2"]
    GROUNDING_MODEL = "gemini-2.5-flash"
    EXTRACTION_MODEL = "gemini-2.5-flash-lite"
    GRADE_IAQ_CHUNK_SIZE = 3
    GRADE_IAQ_MAX_BATCHES = 5
    FILINGS_URL = "https://example.com/search"
    BASICS_URL = "https://example.com/basics/{}"
    NGO_URL = "https://example.com/ngo"
    NGO_NAME = "Test NGO"
    EMAIL_TEMPLATE = "Dear [Recipient], ..."

    def get(self, key, default=None):
        return getattr(self, key, default)

    class connections:
        class supabase:
            SUPABASE_URL = "https://fake.supabase.co"
            SUPABASE_KEY = "fake-jwt"


_mock_st = MagicMock()
_mock_st.secrets = _FakeSecrets()
_mock_st.session_state = _FakeSessionState()
# Strip the caching decorator so init_connection() is called directly in tests.
_mock_st.cache_resource = lambda f: f

sys.modules.setdefault("streamlit", _mock_st)

# --------------------------------------------------------------------------- #
# Supabase library mock
# create_client() would validate the URL and attempt network I/O; replace it
# with a MagicMock so module-level `supabase = init_connection()` succeeds.
# --------------------------------------------------------------------------- #
_mock_supabase_lib = MagicMock()
_mock_supabase_lib.create_client.return_value = MagicMock()
sys.modules.setdefault("supabase", _mock_supabase_lib)


# --------------------------------------------------------------------------- #
# Shared fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture(autouse=True)
def reset_session_state():
    """Clear session_state between tests to prevent state leakage."""
    import streamlit as st
    st.session_state.clear()
    yield
    st.session_state.clear()


@pytest.fixture
def mock_supabase(monkeypatch):
    """Fresh MagicMock supabase client patched into modules.db."""
    import modules.db as db
    client = MagicMock()
    monkeypatch.setattr(db, "supabase", client)
    return client


@pytest.fixture
def sample_control_df():
    """Minimal control DataFrame for use as st.session_state.control_df."""
    return pd.DataFrame(
        {
            "stock_code": ["00700", "00005"],
            "name": ["Tencent Holdings", "HSBC Holdings"],
            "sector": ["Technology", "Finance"],
            "market_cap": [3000.0, 1500.0],
            "iaq_grade": ["High", "Medium"],
            "last_updated_market_cap_at": [pd.NaT, pd.NaT],
            "last_updated_filings_at": [pd.NaT, pd.NaT],
            "last_updated_grade_at": [pd.NaT, pd.NaT],
            "last_updated_contacts_at": [pd.NaT, pd.NaT],
        }
    )
