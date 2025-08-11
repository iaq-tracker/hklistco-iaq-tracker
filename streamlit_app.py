"""
App to be deployed on Streamlit Cloud
"""

import streamlit as st
from google import genai


# Show app title and description.
st.set_page_config(page_title="ListCo ESG tracker", page_icon="🎫")

with st.form("add_ticket_form"):
    prompt = st.text_area("Write your prompt")
    submitted = st.form_submit_button("Submit")

if submitted:
    client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
    response = client.models.generate_content(
        model="gemini-2.5-pro",
        contents=prompt,
    )
    st.write(response)
