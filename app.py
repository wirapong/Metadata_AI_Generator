import streamlit as st
import pandas as pd
from utils import (
    extract_text,
    chunk_text,
    build_embeddings,
    retrieve_context,
    generate_metadata,
    answer_query
)

st.set_page_config(page_title="AI Metadata Generator", layout="wide")
st.title("AI Metadata Generator + GraphRAG")

if "documents" not in st.session_state:
    st.session_state.documents = []

if "embeddings" not in st.session_state:
    st.session_state.embeddings = None

uploaded_files = st.file_uploader(
    "Upload Documents",
    accept_multiple_files=True,
    type=["pdf","docx","txt","csv","json","md","html","py"]
)

if uploaded_files:

    all_chunks = []

    for file in uploaded_files:
        text = extract_text(file)
        chunks = chunk_text(text)
        all_chunks.extend(chunks)

    embeddings = build_embeddings(all_chunks)

    st.session_state.documents = all_chunks
    st.session_state.embeddings = embeddings

    st.success("Documents indexed successfully")

    metadata = generate_metadata(text)

    st.subheader("Generated Metadata")
    st.json(metadata)

    df = pd.DataFrame([metadata])

    csv = df.to_csv(index=False).encode()

    st.download_button(
        "Download Metadata CSV",
        csv,
        "metadata.csv",
        "text/csv"
    )

st.divider()

query = st.text_input("Ask questions about your documents")

if query:

    context = retrieve_context(
        query,
        st.session_state.documents,
        st.session_state.embeddings
    )

    response = answer_query(query, context)

    st.subheader("Answer")
    st.write(response)