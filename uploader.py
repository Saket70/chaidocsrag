# =====================================
# FINAL STABLE uploader.py
# =====================================

import requests
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

# -------------------------------------
# Setup requests session
# -------------------------------------
requests.adapters.DEFAULT_RETRIES = 5
session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/123.0.0.0 Safari/537.36"
})

# -------------------------------------
# Load Documentation URLs
# -------------------------------------
html_loader = WebBaseLoader([
    "https://chaidocs.vercel.app/youtube/chai-aur-html/introduction/",
    "https://chaidocs.vercel.app/youtube/chai-aur-html/emmit-crash-course/",
    "https://chaidocs.vercel.app/youtube/chai-aur-html/html-tags/",
], session=session)

django_loader = WebBaseLoader([
    "https://chaidocs.vercel.app/youtube/chai-aur-django/getting-started/",
    "https://chaidocs.vercel.app/youtube/chai-aur-django/jinja-templates/",
    "https://chaidocs.vercel.app/youtube/chai-aur-django/tailwind/",
    "https://chaidocs.vercel.app/youtube/chai-aur-django/models/",
    "https://chaidocs.vercel.app/youtube/chai-aur-django/relationships-and-forms/",
], session=session)

sql_loader = WebBaseLoader([
    "https://chaidocs.vercel.app/youtube/chai-aur-sql/postgres/",
    "https://chaidocs.vercel.app/youtube/chai-aur-sql/normalization/",
    "https://chaidocs.vercel.app/youtube/chai-aur-sql/database-design-exercise/",
    "https://chaidocs.vercel.app/youtube/chai-aur-sql/joins-and-keys/",
], session=session)

# -------------------------------------
# Load Documents
# -------------------------------------
html_docs = html_loader.load()
django_docs = django_loader.load()
sql_docs = sql_loader.load()

print(f"✅ Loaded {len(html_docs)} HTML docs, {len(django_docs)} Django docs, {len(sql_docs)} SQL docs")

# -------------------------------------
# Split Documents
# -------------------------------------
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

html_splits = splitter.split_documents(html_docs)
django_splits = splitter.split_documents(django_docs)
sql_splits = splitter.split_documents(sql_docs)

print(f"✅ Split into {len(html_splits)} HTML chunks, {len(django_splits)} Django chunks, {len(sql_splits)} SQL chunks")

# -------------------------------------
# Local HuggingFace Embeddings
# -------------------------------------
embedder = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# -------------------------------------
# Connect to Qdrant
# -------------------------------------
client = QdrantClient(url="http://localhost:6333")

# Delete collections if exist (clean reset)
for name in ["html_docs", "django_docs", "sql_docs"]:
    try:
        client.delete_collection(collection_name=name)
        print(f"🗑 Deleted existing collection: {name}")
    except:
        pass

# -------------------------------------
# Upload to Qdrant (New Stable Method)
# -------------------------------------
QdrantVectorStore.from_documents(
    documents=html_splits,
    embedding=embedder,
    url="http://localhost:6333",
    collection_name="html_docs",
)

QdrantVectorStore.from_documents(
    documents=django_splits,
    embedding=embedder,
    url="http://localhost:6333",
    collection_name="django_docs",
)

QdrantVectorStore.from_documents(
    documents=sql_splits,
    embedding=embedder,
    url="http://localhost:6333",
    collection_name="sql_docs",
)

print("🎉 Successfully uploaded all collections!")
