# RAG Chatbot Starter Project

A working Retrieval-Augmented Generation (RAG) chatbot: ask questions, get answers grounded in your own documents,
with sources shown for verification.

**Stack:** LangChain + FAISS (vector store) + HuggingFace sentence-transformers (embeddings, runs locally/free) +
Groq (LLM, free API) + Streamlit (UI).

## 1. Setup

```bash
# (recommended) create a virtual environment
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# install dependencies
pip install -r requirements.txt
```

## 2. Get a free Groq API key

1. Go to https://console.groq.com/keys
2. Sign up (free) and create an API key
3. Copy `.env.example` to `.env`
4. Paste your key in `.env`:
   ```
   GROQ_API_KEY=gsk_your_actual_key_here
   ```

## 3. Build the vector index

This reads everything in `data/sample_docs/`, splits it into chunks, embeds it, and saves a FAISS index.

```bash
python ingest.py
```

You only need to re-run this when you add/change documents.

## 4. Run the chatbot

```bash
streamlit run app.py
```

This opens a browser tab with a chat interface. Ask a question — try:
- "What is the FYP structure?"
- "How do I get an AI internship?"

Each answer shows the source chunks it was grounded in (expand "📄 Sources used for this answer").

## 5. Swap in your own documents

1. Delete the sample files in `data/sample_docs/` (or add alongside them)
2. Drop in your own `.txt` or `.pdf` files
3. Delete the `faiss_index/` folder
4. Re-run `python ingest.py`
5. Re-run `streamlit run app.py`

## Project structure

```
rag-project/
├── data/sample_docs/    # your source documents (.txt, .pdf)
├── faiss_index/         # generated vector index (created by ingest.py)
├── ingest.py            # builds the vector index from your documents
├── app.py                # Streamlit chat UI
├── requirements.txt
├── .env.example
└── README.md
```

## Next steps to make this portfolio-worthy

- [ ] Swap sample docs for a real, specific dataset you care about
- [ ] Add conversation memory (follow-up questions referencing earlier turns)
- [ ] Try different chunk sizes and compare retrieval quality
- [ ] Deploy for free on Streamlit Community Cloud or Hugging Face Spaces
- [ ] Write up what you learned: why this chunk size, what failed, what you'd improve
