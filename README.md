# Hong Kong ListCo Indoor Air Quality (IAQ) Tracker

This Streamlit application is an AI-powered tool designed to monitor, analyze, and engage with Hong Kong-listed companies regarding their disclosures related to Indoor Air Quality (IAQ).

The tool leverages Google's Gemini models for analysis and content generation, and provides a user-friendly interface to manage a watchlist, review data, and draft outreach emails.

## Key Features

*   **Dashboard:** Manage a watchlist of companies. View their AI-generated IAQ disclosure grades and track data freshness at a glance.
*   **AI-Powered IAQ Grading:** Utilize Google's Gemini 2.5 models to analyze all of a company's ESG filings, extract relevant sections on IAQ, and assign a "Low," "Medium," or "High" grade based on the quality and consistency of their disclosures.
*   **AI-Powered Contact Discovery:** Use Google Search integrated with the Gemini API to find and extract up-to-date Investor Relations (IR) contact information from official company sources.
*   **AI-Assisted Email Drafting:** Generate a professional, context-aware outreach email to a company's IR department based on its specific IAQ grading report. The generated `.eml` file can be downloaded and opened in any email client.
*   **Bulk Updates:** Efficiently refresh ESG filings and IR contacts for multiple companies at once, based on a user-defined timeframe (e.g., companies not updated in the last 12 weeks).
*   **Data Export:** Download the entire dataset—including your watchlist, all filings, IAQ gradings, IR contacts, and AI interaction logs—to a single, organized Excel file.

## Disclaimer
- This app is designed for non-commercial use and retrieves public ESG filings. Ensure compliance with website's terms and conditions before use.
- No warranties; use at your own risk.

## Tech Stack

*   **Framework:** [Streamlit](https://streamlit.io/)
*   **Database:** [Supabase](https://supabase.io/) (PostgreSQL)
*   **AI/LLM:** [Google Gemini API](https://ai.google.dev/) (2.5 Pro, 2.5 Flash)
*   **Web Scraping:** [Selenium](https://www.selenium.dev/) & [webdriver-manager](https://pypi.org/project/webdriver-manager/)
*   **Data Handling:** [Pandas](https://pandas.pydata.org/)
*   **Deployment:** Streamlit Community Cloud

## Local Setup and Installation

Follow these steps to run the application on your local machine.

### 1. Prerequisites

*   Python 3.9+
*   Git

### 2. Clone the Repository

```bash
git clone https://github.com/ernesthung/hklistco-esg-tracker.git
cd hklistco-esg-tracker
```

### 3. Install Dependencies

It's recommended to use a virtual environment.

```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

pip install -r requirements.txt
```

### 4. Set Up Supabase
See: https://docs.streamlit.io/develop/tutorials/databases/supabase

### 5. Run the Application
Once the setup is complete, run the following command in your terminal:

```bash
streamlit run streamlit_app.py
```
