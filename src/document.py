# src/document.py
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

# Define document paths
guide_pdf_path = "/content/Guide-to-Litigation-in-India.pdf"
corporate_pdf_path = "/content/Legal-Compliance-&-Corporate-Laws-by-ICAI.pdf"

# Load documents
loader_guide = PyMuPDFLoader(guide_pdf_path)
loader_corporate = PyMuPDFLoader(corporate_pdf_path)
docs_guide = loader_guide.load()
docs_corporate = loader_corporate.load()
all_docs = docs_guide + docs_corporate

# Chunking & Indexing:
text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
chunks = text_splitter.split_documents(all_docs)

embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
vector_store = FAISS.from_documents(chunks, embeddings)