import os

# 🔥 Set your Gemini API key here
os.environ["GOOGLE_API_KEY"] = "itsmysecretkey"

import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore

# -----------------------------
# Embeddings (LOCAL - HuggingFace)
# -----------------------------
embedder = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# -----------------------------
# Load existing Qdrant collections
# -----------------------------
html_qdrant = QdrantVectorStore.from_existing_collection(
    url="http://localhost:6333",
    collection_name="html_docs",
    embedding=embedder,
)

django_qdrant = QdrantVectorStore.from_existing_collection(
    url="http://localhost:6333",
    collection_name="django_docs",
    embedding=embedder,
)

sql_qdrant = QdrantVectorStore.from_existing_collection(
    url="http://localhost:6333",
    collection_name="sql_docs",
    embedding=embedder,
)

# -----------------------------
# Gemini LLM (Only for answering)
# -----------------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-001",
   
    temperature=0.2,
)

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("CHAI DOCS RAG ☕")

st.sidebar.header("About")
st.sidebar.text(
    "This is a CHAI Docs RAG system built using LangChain and Qdrant.\n\n"
    "Ask questions about HTML, Django, or SQL."
)

query = st.text_input("💬 Enter your query:")

if query:

    def handle_query(query):
        retrieved_html_docs = html_qdrant.similarity_search(query, k=3)
        retrieved_django_docs = django_qdrant.similarity_search(query, k=3)
        retrieved_sql_docs = sql_qdrant.similarity_search(query, k=3)

        context_html = "\n".join([doc.page_content for doc in retrieved_html_docs])
        context_django = "\n".join([doc.page_content for doc in retrieved_django_docs])
        context_sql = "\n".join([doc.page_content for doc in retrieved_sql_docs])

        prompt = f"""
You are an expert assistant answering questions based on documentation.

HTML Context:
{context_html}

Django Context:
{context_django}

SQL Context:
{context_sql}

Question: {query}

Answer in simple, clear explanation (200 words max).
"""

        response = llm.invoke(prompt)
        return response.content

    answer = handle_query(query)

    st.subheader("🧠 Answer:")
    st.write(answer)

else:
    st.text("💬 Enter a query to get started.")
