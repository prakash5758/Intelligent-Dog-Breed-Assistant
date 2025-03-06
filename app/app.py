import streamlit as st
import requests

st.title("🐶 Intelligent Dog Breed Assistant")

# Input field for user's query
query = st.text_input("Enter your query about dog breeds:")

if st.button("Get Recommendations") and query:
    payload = {"user_id": "Prakash", "query": query}
    try:
        # Change the URL if needed (e.g., if running in Docker, use the container name)
        response = requests.post("http://api:8000/ask", json=payload)
        if response.status_code == 200:
            data = response.json()
            recommendations = data.get("recommendations", [])
            st.write("### Breed Recommendations:")
            for rec in recommendations:
                st.write(f"**Breed:** {rec['Dog_name']}")  # Removed similarity score
        else:
            st.error(f"Error: {response.status_code} - {response.text}")
    except Exception as e:
        st.error(f"Request failed: {str(e)}")
