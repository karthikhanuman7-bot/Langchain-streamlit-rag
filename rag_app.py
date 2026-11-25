import os
import tempfile
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS

load_dotenv()  # delete these two lines if you set the key with `set` instead of .env

st.title("Multi-PDF RAG with Sources")

if not os.environ.get("OPENAI_API_KEY"):
    st.error("OPENAI_API_KEY is not set. Set it in the terminal and restart.")
    st.stop()


def build_index(files):
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    chunks = []
    for f in files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(f.getbuffer())
            path = tmp.name
        try:
            pages = PyPDFLoader(path).load()
        finally:
            try:
                os.remove(path)
            except PermissionError:
                pass  # Windows sometimes keeps the file locked; harmless
        for p in pages:
            p.metadata["source"] = f.name  # real file name, not the temp path
        chunks.extend(splitter.split_documents(pages))
    if not chunks:
        return None, 0  # scanned/image PDFs have no extractable text
    store = FAISS.from_documents(chunks, OpenAIEmbeddings())
    return store, len(chunks)


files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)

if files:
    signature = tuple(sorted((f.name, f.size) for f in files))
    if st.session_state.get("signature") != signature:
        with st.spinner("Indexing..."):
            store, n = build_index(files)
        if store is None:
            st.error("No readable text found. These may be scanned PDFs.")
            st.stop()
        st.session_state.store = store
        st.session_state.n_chunks = n
        st.session_state.signature = signature

    st.write("Uploaded: " + ", ".join(f.name for f in files))
    st.success(f"Indexed {st.session_state.n_chunks} chunks.")

    # TEMPORARY check, remove after verifying
    with st.expander("Debug: sample chunk metadata"):
        sample = st.session_state.store.similarity_search("leave", k=3)
        for d in sample:
            st.write(d.metadata)

        question = st.text_input("Ask a question about your documents")

    if question:
        retriever = st.session_state.store.as_retriever(
            search_type="mmr", search_kwargs={"k": 4, "fetch_k": 10}
        )
        docs = retriever.invoke(question)

        context = "\n\n".join(
            f"[Source: {d.metadata['source']}, page {d.metadata['page'] + 1}]\n{d.page_content}"
            for d in docs
        )

        prompt = f"""You answer questions using ONLY the context below.
Rules:
- If the context does not contain the answer, reply exactly: "I could not find this in the uploaded documents."
- If different sources give different answers, do NOT pick one. State that they
  disagree and say which file and page says what.
- Do not use outside knowledge.

Context:
{context}

Question: {question}
Answer:"""

        with st.spinner("Thinking..."):
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
            answer = llm.invoke(prompt).content

        st.subheader("Answer")
        st.write(answer)

        sources = sorted({(d.metadata["source"], d.metadata["page"] + 1) for d in docs})
        st.markdown("**Sources:** " + " | ".join(f"{s} - page {p}" for s, p in sources))

        with st.expander("Retrieved chunks (for checking)"):
            for d in docs:
                st.caption(f"{d.metadata['source']} - page {d.metadata['page'] + 1}")
                st.write(d.page_content)