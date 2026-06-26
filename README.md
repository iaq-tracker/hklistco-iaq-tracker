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
*   **Authentication:** [streamlit-authenticator](https://github.com/mkhorasani/Streamlit-Authenticator) (v0.4.2)
*   **Database:** [Supabase](https://supabase.io/) (PostgreSQL)
*   **AI/LLM:** [Google Gemini API](https://ai.google.dev/) (2.5 Flash, 2.5 Flash-Lite)
*   **Web Scraping:** [Selenium](https://www.selenium.dev/) & [webdriver-manager](https://pypi.org/project/webdriver-manager/)
*   **Data Handling:** [Pandas](https://pandas.pydata.org/)
*   **Deployment:** [Hugging Face Spaces](https://huggingface.co/spaces)

## Project Structure

```
streamlit_app.py        # Main Streamlit entry point (UI + orchestration)
modules/
  db.py                 # Database layer — Supabase CRUD and session-state helpers
  llm.py                # LLM layer — Gemini calls for grading, contacts, email drafting
  scraping.py           # Scraping layer — Selenium/HKEx for filings and company basics
load_data.py            # One-time script to seed the master Excel into Supabase
.github/
  workflows/
    keep-alive.yml      # Daily cron job to prevent Supabase/HF Space inactivity pauses
.streamlit/
  secrets.toml          # Local secrets (not committed — see setup below)
packages.txt            # System packages for HF Spaces (Chromium)
requirements.txt        # Python dependencies
```

## Local Setup and Installation

### 1. Prerequisites

*   Python 3.9+
*   Git
*   Google Chrome (for local Selenium scraping)

### 2. Clone the Repository

```bash
git clone https://github.com/ernesthung/hklistco-iaq-tracker.git
cd hklistco-iaq-tracker
```

### 3. Install Dependencies

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 4. Set Up Supabase

Follow the [Streamlit + Supabase guide](https://docs.streamlit.io/develop/tutorials/databases/supabase) to create a project and get your URL and service key.

### 5. Configure Secrets

Create `.streamlit/secrets.toml` with the following structure:

```toml
GEMINI_API_KEYS = ["your-key-1", "your-key-2"]

FILINGS_URL = "https://..."   # HKEx filing search URL
BASICS_URL  = "https://..."   # HKEx company page URL

NGO_URL       = "https://..."  # Your organisation's website
NGO_NAME      = "Your Org"
EMAIL_TEMPLATE = "..."         # Outreach email template

[connections.supabase]
SUPABASE_URL = "https://your-project.supabase.co"
SUPABASE_KEY = "your-service-key"

[auth]
cookie_name    = "iaq_tracker_auth"
cookie_key     = "some-random-secret"
cookie_expiry_days = 30

[auth.credentials.usernames.admin]
email    = "admin@example.com"
name     = "Admin"
password = "plain-or-pre-hashed-password"
```

To pre-hash a password (recommended):

```bash
python -c "import streamlit_authenticator as s; print(s.Hasher(['YOUR_PASSWORD']).generate())"
```

### 6. Seed the Database (first run only)

```bash
python load_data.py
```

### 7. Run the Application

```bash
streamlit run streamlit_app.py
```

## Deployment (Hugging Face Spaces)

1. Create a new Space on Hugging Face (Streamlit SDK).
2. Connect your GitHub repo in the HF Spaces UI — pushes to `main` trigger automatic redeploys.
3. Add every key from `.streamlit/secrets.toml` as a **Space Secret** (they replace the file in production).
4. `packages.txt` (installs Chromium) and `requirements.txt` are picked up automatically.

### Keep-Alive Cron Job

`.github/workflows/keep-alive.yml` runs daily at **6 am KST (9 pm UTC)** via GitHub Actions:

- Pings Supabase to prevent the 7-day inactivity pause.
- Optionally pings the HF Space URL to prevent the 48-hour inactivity sleep.

Required **GitHub Actions secrets** (repo → Settings → Secrets and variables → Actions):

| Secret | Purpose |
|--------|---------|
| `SUPABASE_URL` | Supabase project URL |
| `SUPABASE_KEY` | Supabase service key |
| `HF_SPACE_URL` | Your HF Space URL (set after deploying; leave unset to skip) |

## Development

```bash
# Lint (pylint + ruff via pre-commit)
pre-commit run --all-files

# Ruff only
ruff check . --fix
ruff format .
```
