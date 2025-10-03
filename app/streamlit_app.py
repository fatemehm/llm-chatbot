# app/streamlit_app.py
import streamlit as st
import requests

st.title("Tech Support Chatbot")

# Input query
query = st.text_input(
    "Enter your tech support question:", "How to fix a blue screen error?"
)

# Button to submit query
if st.button("Get Answer"):
    try:
        # Send request to FastAPI backend
        response = requests.post(
            "http://localhost:8000/chat",
            json={"text": query},
            headers={"Content-Type": "application/json"},
        )
        response.raise_for_status()  # Raise error for bad status
        answer = response.json().get("response", "No response received")
        st.write(f"**Answer**: {answer}")
    except requests.RequestException as e:
        st.error(f"Error: Could not connect to backend - {e}")
