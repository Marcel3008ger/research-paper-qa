import os
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

PAPERS_DIR = "papers"
INDEX_DIR = "faiss_index"

def ingest():
    docs = []
    for filename in os.listdir(PAPERS_DIR):
        if filename.endswith(".pdf"):
            path = os.path.join(PAPERS_DIR, filename)
            loader = PyMuPDFLoader(path)
            docs.extend(loader.load())
            print(f"Loaded: {filename}")

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    print(f"Total chunks: {len(chunks)}")

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(INDEX_DIR)
    print(f"Index saved to {INDEX_DIR}/")

if __name__ == "__main__":
    ingest()