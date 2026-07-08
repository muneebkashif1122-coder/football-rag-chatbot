"""
ingest.py
---------
Step 1 of the RAG pipeline: read your documents, split them into chunks,
turn each chunk into an embedding, and save them into a local FAISS vector index.

Run this ONCE whenever you add/change documents in data/sample_docs/:
    python ingest.py

To use your own documents instead of the sample ones:
    - Drop your .txt or .pdf files into data/sample_docs/ (or change DATA_DIR below)
    - Delete the old faiss_index/ folder
    - Re-run this script
"""

import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.vectorstores import FAISS

DATA_DIR = "data/sample_docs"
INDEX_DIR = "faiss_index"


def load_documents():
    """Loads all .txt and .pdf files from DATA_DIR."""
    docs = []

    txt_loader = DirectoryLoader(DATA_DIR, glob="**/*.txt", loader_cls=TextLoader)
    docs.extend(txt_loader.load())

    pdf_loader = DirectoryLoader(DATA_DIR, glob="**/*.pdf", loader_cls=PyPDFLoader)
    docs.extend(pdf_loader.load())

    if not docs:
        raise ValueError(
            f"No .txt or .pdf files found in '{DATA_DIR}'. Add some documents first."
        )

    return docs


def main():
    print(f"Loading documents from '{DATA_DIR}'...")
    documents = load_documents()
    print(f"Loaded {len(documents)} document(s).")

    print("Splitting documents into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,     # characters per chunk
        chunk_overlap=100,  # overlap so context isn't lost at chunk boundaries
    )
    chunks = splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks.")

    print("Loading embedding model (runs locally, no API key needed)...")
    embeddings = FastEmbedEmbeddings()

    print("Building FAISS vector index...")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    os.makedirs(INDEX_DIR, exist_ok=True)
    vectorstore.save_local(INDEX_DIR)
    print(f"Done! Vector index saved to '{INDEX_DIR}/'.")


if __name__ == "__main__":
    main()
