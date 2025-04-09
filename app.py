import streamlit as st
from sentence_transformers import SentenceTransformer
import chromadb
import re
import requests
from bs4 import BeautifulSoup

model = SentenceTransformer("all-MiniLM-L6-v2")
chroma_client = chromadb.PersistentClient(path="./shl_db")
collection = chroma_client.get_or_create_collection(name="shl_assessments")

def extract_text_from_url(url):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")
            return soup.get_text(separator=" ", strip=True)[:1000]
        else:
            return None
    except Exception as e:
        print(f"Error scraping URL: {e}")
        return None

def recommend_assessment(user_input, top_k=3):
    url_match = re.search(r'(https?://\S+)', user_input)
    if url_match:
        url = url_match.group(0)
        extracted_text = extract_text_from_url(url)
        if extracted_text:
            query_text = extracted_text
        else:
            return ["❌ Unable to extract job description from the link."]
    else:
        query_text = user_input

    user_embedding = model.encode(query_text).tolist()
    results = collection.query(query_embeddings=[user_embedding], n_results=top_k)

    recommendations = []
    for idx, metadata in enumerate(results["metadatas"][0]):
        recommendations.append(
            f"### {idx+1}. {metadata['Assessment name']}\n"
            f"- **Test Type**: {metadata['Test Type']}\n"
            f"- **Duration**: {metadata['Duration']}\n"
            f"- **Remote Testing**: {metadata['Remote Testing']}\n"
            f"- **Adaptive/IRT**: {metadata['Adaptive/IRT']}\n"
            f"- **URL**: [Link]({metadata['URL']})\n"
        )
    return recommendations if recommendations else ["No matching assessments found."]


st.set_page_config(page_title="SHL Assessment Recommender", layout="centered")

st.title("🧠 SHL Assessment Recommendation Engine")
st.markdown("Paste a job description, a LinkedIn job URL, or just describe what kind of role you want to assess.")

query = st.text_area("💬 Enter your query or job link", height=150)
top_k = st.slider("How many assessments to show?", min_value=1, max_value=10, value=3)

if st.button("🔍 Recommend Assessments"):
    if query.strip():
        with st.spinner("Finding the best matching SHL assessments..."):
            results = recommend_assessment(query, top_k=top_k)
            for r in results:
                st.markdown(r)
    else:
        st.warning("Please enter a query or URL first.")
