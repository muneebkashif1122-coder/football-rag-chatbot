"""
app.py
------
Step 2 of the RAG pipeline: a simple chat UI.
User asks a question -> we retrieve the most relevant chunks from FAISS ->
we pass them + the question to a Groq-hosted LLM -> we show the answer + sources.

Run with:
    streamlit run app.py

Make sure you've already run `python ingest.py` at least once before this.
"""

import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
load_dotenv()

INDEX_DIR = "faiss_index"
GROQ_MODEL = "llama-3.3-70b-versatile"  # good free default on Groq; swap if it's deprecated later

PROMPT_TEMPLATE = """You are a helpful assistant answering questions using ONLY the context below.
If the answer isn't in the context, say you don't know based on the provided documents — do not make something up.

Context:
{context}

Question: {question}

Answer clearly and concisely:"""


@st.cache_resource
def load_vectorstore():
    embeddings = FastEmbedEmbeddings()
    return FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)


@st.cache_resource
def load_llm():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        st.error("GROQ_API_KEY not found. Add it to a .env file (see .env.example).")
        st.stop()
    return ChatGroq(model=GROQ_MODEL, api_key=api_key, temperature=0.2)


def main():
    st.set_page_config(page_title="Pitchside AI.", page_icon="⚽")
    st.title("⚽ Pitchside AI.")
    st.caption("Your AI expert on football rules, VAR regulations, and match analysis.")

    if not os.path.exists(INDEX_DIR):
        st.warning("No vector index found. Run `python ingest.py` first, then reload this page.")
        st.stop()

    vectorstore = load_vectorstore()
    llm = load_llm()
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

    if "history" not in st.session_state:
        st.session_state.history = []

    for msg in st.session_state.history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    question = st.chat_input("Ask a question about your documents...")

    if question:
        st.session_state.history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving relevant chunks and generating answer..."):
                # Step A: retrieve top-k relevant chunks
                retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
                retrieved_docs = retriever.invoke(question)
                context = "\n\n---\n\n".join(doc.page_content for doc in retrieved_docs)

                # Step B: generate answer grounded in retrieved context
                chain = prompt | llm
                response = chain.invoke({"context": context, "question": question})
                answer = response.content

                st.markdown(answer)

                # Show sources so the user can verify (this is a good talking point in interviews!)
                with st.expander("📄 Sources used for this answer"):
                    for i, doc in enumerate(retrieved_docs, 1):
                        source = doc.metadata.get("source", "unknown")
                        st.markdown(f"**Chunk {i}** — `{source}`")
                        st.text(doc.page_content[:300] + "...")

        st.session_state.history.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
