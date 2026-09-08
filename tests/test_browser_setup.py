from browser_setup import create_driver
import browser_setup


def test_create_driver_uses_selenium_manager(monkeypatch):
    captured = {}

    def fake_chrome(*, options):
        captured["options"] = options
        return object()

    monkeypatch.setattr(browser_setup.webdriver, "Chrome", fake_chrome)

    driver = create_driver()

    assert driver is not None
    assert "--headless=new" in captured["options"].arguments
    assert "--no-sandbox" in captured["options"].arguments
    assert "--disable-dev-shm-usage" in captured["options"].arguments
