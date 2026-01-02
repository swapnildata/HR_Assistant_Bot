import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
#from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

# -------------------------
# Config
# -------------------------
load_dotenv()

PDF_PATH = "HR-Policies-Manuals.pdf"
CHROMA_DIR = "chroma_hr_db"

# -------------------------
# Load PDF
# -------------------------
loader = PyPDFLoader(PDF_PATH)
documents = loader.load()

# -------------------------
# Split text
# -------------------------
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150
)
chunks = splitter.split_documents(documents)

# -------------------------
# Embeddings
# -------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# -------------------------
# Create Chroma DB
# -------------------------
vectordb = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=CHROMA_DIR
)

vectordb.persist()

print("✅ HR policy Chroma DB created successfully.")
print(f"📂 Stored at: {CHROMA_DIR}")
