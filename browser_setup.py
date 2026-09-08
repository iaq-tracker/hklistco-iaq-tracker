"""Browser setup for Selenium scraping."""

from selenium import webdriver
from selenium.webdriver.chrome.options import Options


def create_driver() -> webdriver.Chrome:
    """Create a headless Chrome driver using Selenium Manager."""
    options = Options()
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--headless=new")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gcm")
    options.add_argument("--disable-notifications")
    options.add_experimental_option(
        "prefs",
        {
            "profile.default_content_setting_values.notifications": 2,
        },
    )
    return webdriver.Chrome(options=options)
