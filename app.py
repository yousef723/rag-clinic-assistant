import os
import streamlit as st
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from openai import OpenAI

# Load API keys
load_dotenv()
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "clinic-assistant"

# Page setup
st.title("🏥 Clinic FAQ Assistant")
st.write("Ask me anything about clinic policies.")

# Load models (cached so it doesn't reload)
@st.cache_resource
def load_everything():
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    pc = Pinecone(api_key=PINECONE_KEY)
    index = pc.Index(INDEX_NAME)
    client = OpenAI(api_key=OPENAI_KEY)
    return embedder, index, client

embedder, index, client = load_everything()

# Chat input
question = st.text_input("Your question:")

if question:
    with st.spinner("Thinking..."):
        # 1. Embed question
        q_vec = embedder.encode(question).tolist()
        
        # 2. Search Pinecone
        results = index.query(vector=q_vec, top_k=3, include_metadata=True)
        context = "\n".join([m["metadata"]["text"] for m in results["matches"]])
        
        # 3. Ask LLM
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Answer using ONLY the context below."},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
            ]
        )
        
        st.write("### Answer")
        st.write(response.choices[0].message.content)