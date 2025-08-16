# HKEx ListCo IAQ Tracker

A Streamlit app to track ESG disclosures on indoor air quality (IAQ) for HKEx-listed companies. Features:
- Wishlist of stock tickers for ESG filings download.
- LLM (Gemini) extraction of IR contacts.
- Excel export for data organization.

## Disclaimer
- This app is designed for non-commercial use and retrieves public ESG filings. Ensure compliance with HKEx terms and conditions before use.
- Rate limits apply to LLM API.
- No warranties; use at your own risk.

## Setup
- Configure Streamlit secrets: `GEMINI_API_KEYS` (array of keys), `URL`, and Neon DB connection.
- Install dependencies: `pip install -r requirements.txt`.

### How to run it on your own machine
1. Install the requirements

   ```
   $ pip install -r requirements.txt
   ```

2. Run the app

   ```
   $ streamlit run streamlit_app.py
   ```
