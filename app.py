import streamlit as st
from groq import Groq
import os
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from pypdf import PdfReader

# ==============================
# 🔐 Load API Key (Streamlit Cloud Compatible)
# ==============================
api_key = os.environ.get("GROQ_API_KEY")

if not api_key:
    st.error("❌ GROQ_API_KEY not found in Streamlit Secrets.")
    st.stop()

client = Groq(api_key=api_key)

# ==============================
# ⚙ Page Setup
# ==============================
st.set_page_config(page_title="PDF RAG Assistant", page_icon="📄", layout="centered")

st.title("📄 PDF Knowledge RAG Assistant")
st.caption("Production-Ready Retrieval-Augmented Generation • Built by Umar Imam")

# ==============================
# 📂 Upload PDFs
# ==============================
uploaded_files = st.sidebar.file_uploader(
    "Upload PDF Documents",
    type="pdf",
    accept_multiple_files=True
)

# ==============================
# 🧠 Reset State if New File Uploaded
# ==============================
if uploaded_files:

    current_file_names = sorted([file.name for file in uploaded_files])

    if "uploaded_file_names" not in st.session_state or \
       st.session_state.uploaded_file_names != current_file_names:

        st.session_state.uploaded_file_names = current_file_names
        st.session_state.messages = []

        if "rag_data" in st.session_state:
            del st.session_state["rag_data"]

        st.toast("📄 New document loaded. Previous chat cleared.")

# ==============================
# 🧠 Load Embedding Model
# ==============================
@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedder = load_embedder()

# ==============================
# 🔄 Normalize Query
# ==============================
def normalize_query(query):
    if not query.lower().startswith(("what", "define", "explain", "how", "why")):
        return f"What is {query}?"
    return query

# ==============================
# 📄 Build FAISS Index
# ==============================
def build_faiss_index(files):

    chunks = []
    sources = []

    for file in files:
        reader = PdfReader(file)

        for page_num, page in enumerate(reader.pages):
            text = page.extract_text()

            if text:
                text = text.replace("\n", " ")

                sentences = text.split(". ")
                current_chunk = ""

                for sentence in sentences:
                    if len(current_chunk) + len(sentence) < 600:
                        current_chunk += sentence + ". "
                    else:
                        chunks.append(current_chunk.strip())
                        sources.append((file.name, page_num + 1))
                        current_chunk = sentence + ". "

                if current_chunk:
                    chunks.append(current_chunk.strip())
                    sources.append((file.name, page_num + 1))

    embeddings = embedder.encode(chunks, convert_to_numpy=True)

    faiss.normalize_L2(embeddings)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    return index, chunks, sources

# ==============================
# 🔎 Retrieve Context
# ==============================
def search_index(query, index, chunks, sources, top_k=6):

    query_embedding = embedder.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(query_embedding)

    distances, indices = index.search(query_embedding, top_k)

    context = ""
    cited_sources = []

    for idx in indices[0]:
        if idx < len(chunks):
            context += chunks[idx] + "\n\n"
            cited_sources.append(sources[idx])

    return context, cited_sources

# ==============================
# 💬 Chat History
# ==============================
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ==============================
# 🧠 Main RAG Logic
# ==============================
if uploaded_files:

    if "rag_data" not in st.session_state:
        with st.spinner("🔍 Building document index..."):
            index, chunks, sources = build_faiss_index(uploaded_files)
            st.session_state.rag_data = (index, chunks, sources)

    index, chunks, sources = st.session_state.rag_data

    prompt = st.chat_input("Ask a question about your documents...")

    if prompt:

        normalized_prompt = normalize_query(prompt)

        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):

            placeholder = st.empty()
            full_response = ""

            context, cited_sources = search_index(
                normalized_prompt, index, chunks, sources
            )

            if context.strip() == "":
                full_response = "Not found in document."
                placeholder.markdown(full_response)

            else:

                rag_prompt = f"""
You are a document-grounded AI assistant.

Rules:
- Answer ONLY using the context below.
- If not clearly present, say "Not found in document."
- Be concise (2–4 sentences).
- Do not hallucinate.

Context:
{context}

Question:
{normalized_prompt}

Answer:
"""

                response = client.chat.completions.create(
                    model="moonshotai/kimi-k2-instruct-0905",
                    messages=[{"role": "user", "content": rag_prompt}],
                    temperature=0.2,
                    max_tokens=800,
                    stream=True
                )

                for chunk in response:
                    if chunk.choices[0].delta.content:
                        token = chunk.choices[0].delta.content
                        full_response += token
                        placeholder.markdown(full_response)

                unique_sources = set(cited_sources)

                citation_text = "\n\n---\n### 📚 Sources:\n"
                for file, page in unique_sources:
                    citation_text += f"- **{file}** (Page {page})\n"

                full_response += citation_text
                placeholder.markdown(full_response)

        st.session_state.messages.append(
            {"role": "assistant", "content": full_response}
        )

else:
    st.info("⬅ Upload one or more PDF documents to start.")

# ==============================
# Footer
# ==============================
st.markdown("---")