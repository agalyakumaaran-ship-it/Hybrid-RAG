
import os
import requests
import streamlit as st

st.set_page_config(
    page_title="Hybrid RAG Virtual Assistant",
    page_icon="🤖",
    layout="centered",
)

# Use Docker's backend service URL when deployed.
# Use localhost when running the app directly on your Mac.
BACKEND_URL = os.getenv(
    "BACKEND_URL",
    "http://127.0.0.1:8000"
)

API_URL = f"{BACKEND_URL}/ask"

st.title("🤖 Hybrid RAG Virtual Assistant")
st.write("Ask a question about the document and get an AI-generated answer.")

# Question input
with st.form("question_form"):
    question = st.text_input(
        "Your question",
        placeholder="e.g., What services are used in the architecture?",
    )
    submitted = st.form_submit_button("Ask")

# Submit question to the backend
if submitted:
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        try:
            with st.spinner("Finding an answer..."):
                response = requests.post(
                    API_URL,
                    json={"question": question.strip()},
                    timeout=120,
                )
                response.raise_for_status()
                result = response.json()

            st.subheader("Answer")
            st.markdown(
                result.get("answer", "No answer was returned.")
            )

        except requests.RequestException as error:
            st.error(
                "Could not get an answer from the backend. "
                "Make sure the backend is running and the document "
                f"has loaded successfully.\n\nDetails: {error}"
            )
