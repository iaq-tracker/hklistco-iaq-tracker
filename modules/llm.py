"""LLM layer — Gemini AI calls for IAQ grading, contact search, and email drafting.

Model names and batch sizes are configured via secrets.toml:
  GROUNDING_MODEL     — model used for web-grounded generation (default: gemini-2.5-flash)
  EXTRACTION_MODEL    — model used for JSON extraction (default: gemini-2.5-flash-lite)
  GRADE_IAQ_CHUNK_SIZE   — URLs per grading batch (default: 3)
  GRADE_IAQ_MAX_BATCHES  — max batches per grading run (default: 5)
"""

import random
import re
from datetime import datetime

from google import genai
from google.genai import types
from pydantic import BaseModel
import pandas as pd
import pytz
import streamlit as st

from modules.db import supabase, get_company_name


class Contact(BaseModel):
    """Structured output schema for IR contact details."""

    email: str
    name: str | None
    tel: str | None
    title: str | None
    citations: list[str]


class Grade(BaseModel):
    """Structured output schema for IAQ disclosure grades."""

    grade: str
    overview: str
    justification: str
    extracts: str


# ---------------------------------------------------------------------------- #
#                                LLM Client                                    #
# ---------------------------------------------------------------------------- #
def get_llm_client() -> genai.Client:
    if "api_key_counter" not in st.session_state:
        st.session_state.api_key_counter = 0
    api_keys = st.secrets.GEMINI_API_KEYS
    api_keys_count = len(api_keys)
    # randint(1, count) never returns 0, so the offset always picks a key
    # different from the previous call — distributes load across all keys.
    client = genai.Client(
        api_key=api_keys[
            (st.session_state.api_key_counter + random.randint(1, api_keys_count))
            % api_keys_count
        ]
    )
    st.session_state.api_key_counter += 1
    return client


def get_citations(response: types.GenerateContentResponse) -> list[str]:
    citations = []
    if response.candidates:
        if metadata := response.candidates[0].grounding_metadata:
            chunks = getattr(metadata, "grounding_chunks")
            if chunks:
                for chunk in chunks:
                    citations.append(chunk.web.uri)
    return citations


def embed_citations(response: types.GenerateContentResponse) -> str:
    text = response.text
    metadata = response.candidates[0].grounding_metadata
    if metadata is None:
        return text
    supports = metadata.grounding_supports
    chunks = metadata.grounding_chunks

    if (supports is not None) and (chunks is not None):
        # Sort descending so we insert citations from the end of the string
        # first; inserting front-to-back would shift all subsequent end_index
        # offsets and corrupt the remaining insertions.
        sorted_supports = sorted(
            supports, key=lambda s: s.segment.end_index, reverse=True
        )
        for support in sorted_supports:
            end_index = support.segment.end_index
            if support.grounding_chunk_indices:
                citation_links = []
                for i in support.grounding_chunk_indices:
                    if i < len(chunks):
                        uri = chunks[i].web.uri
                        citation_links.append(f"[{i + 1}]({uri})")
                citation_string = "\n".join(citation_links)
                text = text[:end_index] + citation_string + text[end_index:]
    return text


# ---------------------------------------------------------------------------- #
#                                IAQ Grading                                   #
# ---------------------------------------------------------------------------- #
def grade_iaq(stock_code: str, *, save_to_db: bool = True) -> str:
    """
    Grade IAQ disclosures of listed company in its ESG reports by LLM.
    NOTE: url context can only process up to 20 URLs per request (as of Aug 2025).
    See: https://ai.google.dev/gemini-api/docs/url-context
    """
    company_name = get_company_name(stock_code=stock_code) or ""

    response = (
        supabase.table("esg_filings")
        .select("title, url, release_time")
        .eq("stock_code", stock_code)
        .order("release_time", desc=True)
        .execute()
    )
    filings_df = pd.DataFrame(response.data)

    if filings_df.empty:
        raise ValueError(f"No filings found in database for {stock_code}")

    grounding_model = st.secrets.get("GROUNDING_MODEL", "gemini-2.5-flash")
    chunk_size = int(st.secrets.get("GRADE_IAQ_CHUNK_SIZE", 3))
    max_batches = int(st.secrets.get("GRADE_IAQ_MAX_BATCHES", 5))

    responses = ""

    # Cap at max_batches * chunk_size total filings; step by chunk_size so each
    # iteration sends the next batch of URLs to the url_context tool.
    for i in range(0, min(len(filings_df), max_batches * chunk_size), chunk_size):
        chunk_df = filings_df.iloc[i : i + chunk_size]
        filings = "\n".join(
            [f"{row['title']}: {row['url']}" for _, row in chunk_df.iterrows()]
        )

        prompt = f"""You are an expert ESG analyst specializing in evaluating corporate disclosures based on Hong Kong Stock Exchange (HKEX) ESG reporting guidelines.
        Your task is to evaluate the ESG disclosures of the company {company_name} with a stock ticker of '{stock_code}' specifically on the topic of indoor air quality (IAQ). This includes any mentions of IAQ management, monitoring, policies, risks, mitigation strategies, emissions (e.g., VOCs, PM2.5, CO2 levels), ventilation systems, employee health impacts, building certifications (e.g., BEAM Plus, LEED), or related initiatives in its operation.
        Below are list of URLs to all of the company's ESG filings published on HKEx. Browse these URLs, then extract and summarize only the sections relevant to indoor air quality.
        {filings}
        Evaluation Criteria:
        Focus solely on indoor air quality disclosures. Grade based on:

        # Length and Detail: Vague mentions (e.g., one sentence) vs. dedicated sections with explanations, data, and examples.
        # Key Performance Indicators (KPIs): Quantifiable metrics (e.g., IAQ monitoring results, reduction targets for pollutants, compliance rates with standards like Hong Kong IAQ Objectives).
        # Consistency: How regularly KPIs are reported over time; improvements or expansions in disclosure (e.g., adding new metrics or deeper analysis in recent years).
        # Progression: Emphasis on the last three years to assess if disclosure has improved.

        # Grading Scale:
        # Use a three-tier grading system. Assign one grade only, with a brief justification tied to the criteria above. Buckets:

        # Low (No or Minimal Disclosure): Little to no mention of IAQ across reports; generic statements without details, KPIs, or data; no evidence of consistency or improvement.
        # Medium (Emerging Disclosure): Basic mentions starting or increasing in the recent three years; some details or initial KPIs introduced recently, but inconsistent reporting or limited depth/trends.
        # High (Strong Disclosure): Comprehensive, detailed sections on IAQ with multiple KPIs disclosed consistently over years; evidence of adding more information (e.g., year-over-year data, targets, audits) and progressive improvements in recent reports.

        # Output Format:
        # Structure your response as follows:
        # Company Overview: Brief 1-2 sentence summary of the company's business and why IAQ might be material (e.g., based on industry like real estate or hospitality).
        # Key Extracts: Bullet points of relevant IAQ excerpts from each report (cite year and URL).
        # Grade: [Low/Medium/High]
        # Justification for grade: Explanation in 3-5 sentences.
        # """

        client = get_llm_client()
        response = client.models.generate_content(
            model=grounding_model,
            contents=prompt,
            config=types.GenerateContentConfig(
                tools=[{"url_context": {}}],
                temperature=0.3,
            ),
        )

        if not response.text:
            raise ValueError("Null value in response text.")

        if save_to_db:
            citations = get_citations(response)
            supabase.table("llm_logs").insert(
                {
                    "stock_code": stock_code,
                    "prompt": prompt,
                    "response": response.text,
                    "citations": "\n".join(citations) if citations else "None",
                }
            ).execute()

        responses += response.text
    return responses


def format_grading(
    stock_code: str, response_text: str, *, save_to_db: bool = True
) -> Grade:
    """Format text response to specified JSON schema."""
    extraction_model = st.secrets.get("EXTRACTION_MODEL", "gemini-2.5-flash-lite")
    client = get_llm_client()
    prompt = f"""You are an expert data extraction tool. Your task is to analyze the text provided below and extract key details from one or more ESG grading report(s) prepared by an expert ESG analyst.

    From the text, identify the following for each report found:
    - grade: make sure it is one of [Low/Medium/High]. prioritize grade that are based on more recent ESG filings
    - company overview
    - key extracts
    - Justification for grade

    The output must be a JSON object of the grade.

    **Text to Analyze:**
    ---
    {response_text}
    ---

    **Expected JSON Output:**
    """
    response = client.models.generate_content(
        model=extraction_model,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.0,
            response_mime_type="application/json",
            response_schema=Grade,
        ),
    )
    # google-genai auto-deserializes into the Pydantic model when response_schema is set.
    grade: Grade = response.parsed

    if save_to_db and grade:
        supabase.table("iaq_gradings").insert(
            {
                "stock_code": stock_code,
                "grade": grade.grade,
                "overview": grade.overview,
                "justification": grade.justification,
                "extracts": grade.extracts,
                "grading_date": datetime.now(pytz.UTC).isoformat(),
            },
        ).execute()
        supabase.table("control").update(
            {
                "iaq_grade": grade.grade,
                "last_updated_grade_at": datetime.now(pytz.UTC).isoformat(),
            }
        ).eq("stock_code", stock_code).execute()
        citations = get_citations(response)
        supabase.table("llm_logs").insert(
            {
                "stock_code": stock_code,
                "prompt": prompt,
                "response": response.text,
                "citations": "\n".join(citations) if citations else "None",
            }
        ).execute()

    return grade


# ---------------------------------------------------------------------------- #
#                               IR Contacts                                    #
# ---------------------------------------------------------------------------- #
def search_contacts(stock_code: str, *, save_to_db: bool = True) -> str:
    """
    Google search contacts of listed company.
    NOTE: As of 5 Aug 2025, a single Gemini API call cannot simultaneously use a
    grounding tool and enforce structured JSON output.
    See: https://github.com/googleapis/python-genai/issues/665
    NOTE: As of 9 Sep 2025, Google Search grounding for Gemini 2.5 Pro (free tier)
    is not supported.
    See: https://ai.google.dev/gemini-api/docs/pricing#standard
    """
    grounding_model = st.secrets.get("GROUNDING_MODEL", "gemini-2.5-flash")
    company_name = get_company_name(stock_code) or ""

    client = get_llm_client()
    prompt = (
        "Imagine you are an ESG reporting consultant trying to reach "
        "out to the Investor Relations department of the Hong Kong listed "
        f"company {company_name} with a stock ticker of '{stock_code}' "
        "ideally via email. Find all official, up-to-date, available "
        "contact details, including A) general departmental contact "
        "information (e.g. ir@aia.com); and B) details for specific "
        "individuals (e.g. title, name, email, telephone numbers). "
        "Ignore Company Share Registrar. Only reference official sources "
        "such as company websites or filings. Do not reference third-party "
        "sources e.g. Wikipedia."
    )

    grounding_tool = types.Tool(google_search=types.GoogleSearch())
    response = client.models.generate_content(
        model=grounding_model,
        contents=prompt,
        config=types.GenerateContentConfig(
            tools=[grounding_tool],
            temperature=0.0,
        ),
    )

    if not response.text:
        raise ValueError("Null value in response text.")

    if save_to_db:
        citations = get_citations(response)
        supabase.table("llm_logs").insert(
            {
                "stock_code": stock_code,
                "prompt": prompt,
                "response": response.text,
                "citations": "\n".join(citations) if citations else "None",
            }
        ).execute()
    return embed_citations(response)


def format_contacts(
    stock_code: str, response_text: str, *, save_to_db: bool = True
) -> list[Contact]:
    """Format key data from grounded information to specified JSON schema."""
    extraction_model = st.secrets.get("EXTRACTION_MODEL", "gemini-2.5-flash-lite")
    client = get_llm_client()
    prompt = f"""You are an expert data extraction tool. Your task is to analyze the text provided below and extract all available contact details for the Investor Relations department.

    From the text, identify the following for each contact found:
    - name: The full name of the person or a descriptor for a general contact (e.g., "Investor Relations Department").
    - title: The job title of the person. If it is a general contact, this can be null or an empty string.
    - email: The email address.
    - phone: The phone number.
    - citations: A list of URLs included in the text next to the contact details

    It is crucial that you include general, department-wide contact information. If you find a departmental email like "ir@company.com" or a general phone number, create a contact entry for it. For these general contacts, use a descriptive name like "Investor Relations Department" for the "name" field.

    The output must be a JSON object containing a list of contacts.

    **Text to Analyze:**
    ---
    {response_text}
    ---

    **Expected JSON Output:**
    """
    response = client.models.generate_content(
        model=extraction_model,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.0,
            response_mime_type="application/json",
            response_schema=list[Contact],
        ),
    )
    contacts: list[Contact] = response.parsed

    # Only keep contacts with a syntactically valid email; the grounding model
    # sometimes returns placeholder text like "(email not found)".
    email_pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    valid_contacts = [
        {
            "stock_code": stock_code,
            "email": contact.email,
            "name": contact.name,
            "tel": contact.tel,
            "title": contact.title,
            "citations": "\n".join(contact.citations) if contact.citations else None,
        }
        for contact in contacts
        if re.match(email_pattern, contact.email)
    ]

    if save_to_db and valid_contacts:
        supabase.table("ir_contacts").upsert(
            valid_contacts, ignore_duplicates=True, on_conflict="stock_code,email"
        ).execute()
        supabase.table("control").update(
            {"last_updated_contacts_at": datetime.now(pytz.UTC).isoformat()}
        ).eq("stock_code", stock_code).execute()
        citations = get_citations(response)
        supabase.table("llm_logs").insert(
            {
                "stock_code": stock_code,
                "prompt": prompt,
                "response": response.text,
                "citations": "\n".join(citations) if citations else "None",
            }
        ).execute()
    return contacts


# ---------------------------------------------------------------------------- #
#                               Outreach Email                                 #
# ---------------------------------------------------------------------------- #
def draft_email() -> None:
    """Generate email body with AI and store result in st.session_state.email_content."""
    contact_names = "Sir/Madam"
    if not st.session_state.ir_contacts_df.empty:
        names = st.session_state.ir_contacts_df["name"].dropna().tolist()
        if names:
            specific_names = [n for n in names if "department" not in n.lower()]
            if specific_names:
                contact_names = ", ".join(specific_names)

    reference_email = st.secrets.EMAIL_TEMPLATE
    grounding_model = st.secrets.get("GROUNDING_MODEL", "gemini-2.5-flash")
    stock_code = st.session_state.selected_stock_code

    prompt = f"""Imagine you are an ESG consultant from the Hong Kong-based NGO ({st.secrets.NGO_URL}). Your task is to draft a professional outreach email to {st.session_state.selected_company_name} (stock code: {stock_code}-HK).

    The email should be addressed to {contact_names}.

    **Your primary goal is to secure a meeting to discuss potential collaborations on Indoor Air Quality (IAQ) initiatives.**

    **Reference Email (for tone, style, and content):**
    ---
    {reference_email}
    ---

    **Instructions for your draft:**
    1.  **Adopt the Tone and Style:** Mirror the professional, appreciative, and collaborative tone of the reference email.
    2.  **Maintain Core Message:** Keep the brief introduction of CAN and the explanation of IAQ's importance for public health in Hong Kong.
    3.  **Personalize the Opening:** Start with a polite and personalized salutation addressing {contact_names}. In the first paragraph, acknowledge the company's current IAQ disclosure status. Use the following assessment to make your opening specific and show you've done your research:
    """

    if (
        f"justification_{stock_code}" in st.session_state
        and st.session_state[f"justification_{stock_code}"]
    ):
        prompt += f"\n**IAQ Assessment:** {st.session_state[f'justification_{stock_code}']}\n"
    else:
        prompt += "\n**IAQ Assessment:** (No specific assessment available, make a general but positive opening remark about their ESG reporting.)\n"

    prompt += """
    4.  **Incorporate Key Proposals:** You MUST include the three initiatives mentioned in the reference email: Leadership Case Study, Expert Presentations and Awareness Workshops, and the ESG Award.
    5.  **Call to Action:** End with a clear call to action, requesting a brief meeting (virtual or in-person).
    6.  **Format:** Return ONLY the body of the email in plain text that is suitable for an email body. Do not use any markdown formatting, such as asterisks for bolding (`**text**`) or bullet points (`* text`). Write paragraphs in standard block format. Do not include the subject line, recipient line (e.g., "To:"), or signature. Keep it concise (within 400 words).
    """

    client = get_llm_client()
    response = client.models.generate_content(
        model=grounding_model,
        contents=prompt,
        config=types.GenerateContentConfig(
            tools=[{"url_context": {}}],
            temperature=0.8,
        ),
    )
    st.session_state.email_content = response.text
