import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.title("🤖 LLM Chatbot")

model_name = st.selectbox(
    "Select Model",
    ["google/flan-t5-small", "bert-base-uncased", "distilgpt2"],
)

user_input = st.text_input("Enter your question:")

if st.button("Submit"):
    if not user_input.strip():
        st.warning("Please type a question!")
    else:
        try:
            response = requests.post(
                f"{API_URL}/chat",
                json={"message": user_input, "model_name": model_name},
                timeout=10,
            )
            if response.status_code == 200:
                st.success(f"Response: {response.json().get('response')}")
            else:
                st.error(f"API error: {response.status_code}")
        except Exception as e:
            st.error(f"Could not reach API: {e}")
