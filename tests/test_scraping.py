"""Unit tests for modules/scraping.py — HKEx Selenium scraping helpers."""

from unittest.mock import MagicMock, call, patch

import pytest

from modules.scraping import normalize_market_cap, init_chromedriver


# --------------------------------------------------------------------------- #
# normalize_market_cap
# --------------------------------------------------------------------------- #
class TestNormalizeMarketCap:
    def test_parses_billions(self):
        assert normalize_market_cap("HK$3,291.5B") == 3291.5

    def test_parses_millions_and_converts_to_billions(self):
        assert normalize_market_cap("HK$500.0M") == 0.5

    def test_handles_no_currency_prefix(self):
        assert normalize_market_cap("1,000.0B") == 1000.0

    def test_rounds_to_two_decimal_places(self):
        # 1500M / 1000 = 1.5 — no floating-point drift
        result = normalize_market_cap("1,500M")
        assert result == 1.5

    def test_raises_on_unparseable_string(self):
        with pytest.raises(ValueError, match="Could not parse"):
            normalize_market_cap("N/A")

    def test_raises_on_empty_string(self):
        with pytest.raises(ValueError, match="Could not parse"):
            normalize_market_cap("")


# --------------------------------------------------------------------------- #
# init_chromedriver
# --------------------------------------------------------------------------- #
class TestInitChromedriver:
    def test_uses_chromium_service_when_chromium_on_path(self):
        """When 'chromium' binary is found, ChromeType.CHROMIUM is passed."""
        mock_driver = MagicMock()
        with (
            patch(
                "modules.scraping.shutil.which",
                side_effect=lambda name: "/usr/bin/chromium" if name == "chromium" else None,
            ),
            patch("modules.scraping.ChromeDriverManager") as mock_mgr,
            patch("modules.scraping.webdriver.Chrome", return_value=mock_driver),
        ):
            result = init_chromedriver()

        # ChromeDriverManager must have been called with chrome_type kwarg
        assert mock_mgr.called
        _, kwargs = mock_mgr.call_args
        assert "chrome_type" in kwargs

    def test_uses_chromium_browser_binary_name_as_fallback(self):
        """When 'chromium' is missing but 'chromium-browser' is found, still uses CHROMIUM type."""
        mock_driver = MagicMock()
        with (
            patch(
                "modules.scraping.shutil.which",
                side_effect=lambda name: "/usr/bin/chromium-browser"
                if name == "chromium-browser"
                else None,
            ),
            patch("modules.scraping.ChromeDriverManager") as mock_mgr,
            patch("modules.scraping.webdriver.Chrome", return_value=mock_driver),
        ):
            init_chromedriver()

        _, kwargs = mock_mgr.call_args
        assert "chrome_type" in kwargs

    def test_uses_standard_chrome_when_chromium_not_on_path(self):
        """When no Chromium binary is found, ChromeDriverManager() is called without chrome_type."""
        mock_driver = MagicMock()
        with (
            patch("modules.scraping.shutil.which", return_value=None),
            patch("modules.scraping.ChromeDriverManager") as mock_mgr,
            patch("modules.scraping.webdriver.Chrome", return_value=mock_driver),
        ):
            init_chromedriver()

        mock_mgr.assert_called_once_with()

    def test_returns_chrome_driver_instance(self):
        mock_driver = MagicMock()
        with (
            patch("modules.scraping.shutil.which", return_value=None),
            patch("modules.scraping.ChromeDriverManager"),
            patch("modules.scraping.webdriver.Chrome", return_value=mock_driver),
        ):
            result = init_chromedriver()

        assert result is mock_driver

    def test_headless_flag_is_always_set(self):
        """Headless mode must always be enabled — the app has no display."""
        captured_options = []

        def capture_chrome(service, options):
            captured_options.append(options)
            return MagicMock()

        with (
            patch("modules.scraping.shutil.which", return_value=None),
            patch("modules.scraping.ChromeDriverManager"),
            patch("modules.scraping.webdriver.Chrome", side_effect=capture_chrome),
        ):
            init_chromedriver()

        args = captured_options[0].arguments
        assert any("headless" in arg for arg in args)
