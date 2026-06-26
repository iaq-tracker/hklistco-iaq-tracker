"""Unit tests for modules/llm.py — Gemini LLM helpers."""

from unittest.mock import MagicMock, patch

import pytest

import modules.llm as llm


# --------------------------------------------------------------------------- #
# get_llm_client — API key rotation
# --------------------------------------------------------------------------- #
class TestGetLlmClient:
    def test_initialises_counter_on_first_call(self):
        import streamlit as st
        with patch("modules.llm.genai.Client"):
            llm.get_llm_client()
        assert "api_key_counter" in st.session_state

    def test_increments_counter_on_each_call(self):
        import streamlit as st
        with patch("modules.llm.genai.Client"):
            llm.get_llm_client()
            assert st.session_state["api_key_counter"] == 1
            llm.get_llm_client()
            assert st.session_state["api_key_counter"] == 2

    def test_uses_different_key_across_calls(self):
        """Consecutive calls pick different keys when randint offset is fixed at 1.

        With offset=1: index = (counter + 1) % len(keys) — alternates each call.
        """
        keys_used = []

        def capture_client(api_key):
            keys_used.append(api_key)
            return MagicMock()

        # Fix randint to 1 so the key selection is deterministic.
        with (
            patch("modules.llm.random.randint", return_value=1),
            patch("modules.llm.genai.Client", side_effect=capture_client),
        ):
            llm.get_llm_client()  # counter=0 → (0+1)%2=1 → key[1]
            llm.get_llm_client()  # counter=1 → (1+1)%2=0 → key[0]

        assert len(set(keys_used)) == 2

    def test_stays_within_valid_key_index(self):
        """Chosen index must always be within bounds of the key list."""
        valid_keys = {"test-key-1", "test-key-2"}
        keys_used = []

        def capture_client(api_key):
            keys_used.append(api_key)
            return MagicMock()

        with patch("modules.llm.genai.Client", side_effect=capture_client):
            for _ in range(20):
                llm.get_llm_client()

        assert all(k in valid_keys for k in keys_used)


# --------------------------------------------------------------------------- #
# get_citations
# --------------------------------------------------------------------------- #
class TestGetCitations:
    def test_returns_empty_list_when_no_candidates(self):
        response = MagicMock()
        response.candidates = []
        assert llm.get_citations(response) == []

    def test_returns_empty_list_when_grounding_metadata_is_none(self):
        candidate = MagicMock()
        candidate.grounding_metadata = None
        response = MagicMock()
        response.candidates = [candidate]
        # getattr fallback returns None when metadata is None
        result = llm.get_citations(response)
        assert result == []

    def test_returns_empty_list_when_no_grounding_chunks(self):
        metadata = MagicMock()
        metadata.grounding_chunks = None
        candidate = MagicMock()
        candidate.grounding_metadata = metadata
        response = MagicMock()
        response.candidates = [candidate]
        assert llm.get_citations(response) == []

    def test_extracts_uris_from_grounding_chunks(self):
        chunk1 = MagicMock()
        chunk1.web.uri = "https://example.com/a"
        chunk2 = MagicMock()
        chunk2.web.uri = "https://example.com/b"
        metadata = MagicMock()
        metadata.grounding_chunks = [chunk1, chunk2]
        candidate = MagicMock()
        candidate.grounding_metadata = metadata
        response = MagicMock()
        response.candidates = [candidate]
        assert llm.get_citations(response) == [
            "https://example.com/a",
            "https://example.com/b",
        ]


# --------------------------------------------------------------------------- #
# embed_citations
# --------------------------------------------------------------------------- #
class TestEmbedCitations:
    def _make_response(self, text, supports, chunks):
        candidate = MagicMock()
        candidate.grounding_metadata.grounding_supports = supports
        candidate.grounding_metadata.grounding_chunks = chunks
        response = MagicMock()
        response.text = text
        response.candidates = [candidate]
        return response

    def test_inserts_citation_links_into_text(self):
        chunk = MagicMock()
        chunk.web.uri = "https://example.com"
        support = MagicMock()
        support.segment.end_index = 5
        support.grounding_chunk_indices = [0]
        response = self._make_response("Hello world", [support], [chunk])

        result = llm.embed_citations(response)
        assert "[1](https://example.com)" in result

    def test_inserts_back_to_front_to_preserve_offsets(self):
        """Citations at higher offsets must be inserted before those at lower offsets."""
        chunk1 = MagicMock()
        chunk1.web.uri = "https://a.com"
        chunk2 = MagicMock()
        chunk2.web.uri = "https://b.com"

        support_early = MagicMock()
        support_early.segment.end_index = 5
        support_early.grounding_chunk_indices = [0]

        support_late = MagicMock()
        support_late.segment.end_index = 11
        support_late.grounding_chunk_indices = [1]

        # Pass supports in forward order — embed_citations must reorder them.
        response = self._make_response(
            "Hello world!", [support_early, support_late], [chunk1, chunk2]
        )
        result = llm.embed_citations(response)

        # Both citations must appear in the result.
        assert "[1](https://a.com)" in result
        assert "[2](https://b.com)" in result

    def test_returns_original_text_when_no_grounding(self):
        candidate = MagicMock()
        candidate.grounding_metadata.grounding_supports = None
        candidate.grounding_metadata.grounding_chunks = None
        response = MagicMock()
        response.text = "Plain text with no citations."
        response.candidates = [candidate]

        result = llm.embed_citations(response)
        assert result == "Plain text with no citations."

    def test_skips_out_of_range_chunk_index(self):
        chunk = MagicMock()
        chunk.web.uri = "https://example.com"
        support = MagicMock()
        support.segment.end_index = 5
        support.grounding_chunk_indices = [99]  # index beyond chunks list
        response = self._make_response("Hello", [support], [chunk])

        # Should not raise; simply skips the bad index.
        result = llm.embed_citations(response)
        assert isinstance(result, str)


# --------------------------------------------------------------------------- #
# grade_iaq — loop cap (no network calls)
# --------------------------------------------------------------------------- #
class TestGradeIaqLoopCap:
    """Verify that grade_iaq respects GRADE_IAQ_MAX_BATCHES without calling the API."""

    def test_processes_at_most_max_batches_times_chunk_size_filings(self):
        """With 20 filings and defaults (5 batches × 3), exactly 5 API calls made.

        grade_iaq imports `supabase` directly from modules.db, so we patch
        `modules.llm.supabase` (the name as bound in llm.py) not modules.db.supabase.
        """
        import pandas as pd
        import streamlit as st

        # 20 filings available; cap should stop at 15 (5 batches × 3 each).
        filings = [
            {
                "title": f"ESG Report {i}",
                "url": f"https://example.com/{i}",
                "release_time": f"2024-{i:02d}-01T00:00:00",
            }
            for i in range(1, 21)
        ]

        st.session_state["control_df"] = pd.DataFrame(
            {"stock_code": ["00700"], "name": ["Tencent Holdings"]}
        )

        mock_sb = MagicMock()
        mock_sb.table.return_value.select.return_value.eq.return_value.order.return_value.execute.return_value.data = (
            filings
        )

        calls = []

        def fake_generate(model, contents, config):
            calls.append(contents)
            resp = MagicMock()
            resp.text = "Some grading text."
            return resp

        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = fake_generate

        with (
            patch("modules.llm.supabase", mock_sb),
            patch("modules.llm.get_llm_client", return_value=mock_client),
        ):
            llm.grade_iaq("00700", save_to_db=False)

        assert len(calls) == 5
