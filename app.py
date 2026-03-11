import io
import json
import streamlit as st
import pandas as pd
import streamlit.components.v1 as components

from utils import (
    extract_text_from_upload,
    build_corpus_from_uploads,
    build_faiss_index,
    semantic_search,
    generate_metadata_for_corpus,
    answer_query_with_context,
    build_knowledge_graph,
    graph_to_pyvis_html,
    corpus_dashboard_stats,
)
from ontology import generate_ontology_bundle
from exporters import export_rdf_turtle, export_jsonld

st.set_page_config(page_title="AI Metadata + Knowledge Graph Platform", layout="wide")
st.title("AI Metadata + Knowledge Graph Platform")
st.caption("Version 2: GraphRAG-style Retrieval + Knowledge Graph + Ontology + RDF/JSON-LD")

# ---------- Session state ----------
if "corpus_df" not in st.session_state:
    st.session_state.corpus_df = None
if "faiss_index" not in st.session_state:
    st.session_state.faiss_index = None
if "embedding_matrix" not in st.session_state:
    st.session_state.embedding_matrix = None
if "kg" not in st.session_state:
    st.session_state.kg = None
if "metadata_bundle" not in st.session_state:
    st.session_state.metadata_bundle = None
if "ontology_bundle" not in st.session_state:
    st.session_state.ontology_bundle = None

# ---------- Sidebar ----------
with st.sidebar:
    st.header("Configuration")
    embedding_model = st.selectbox(
        "Embedding model",
        ["text-embedding-3-small"],
        index=0,
    )
    answer_model = st.selectbox(
        "Answer / metadata model",
        ["gpt-4.1-mini"],
        index=0,
    )
    top_k = st.slider("Top-K retrieval", min_value=3, max_value=15, value=5, step=1)
    chunk_size = st.slider("Chunk size (tokens, approximate words)", 300, 1200, 700, 50)
    overlap = st.slider("Chunk overlap", 50, 300, 150, 25)
    st.divider()
    st.markdown(
        """
        **Research modules**
        - Multi-document corpus
        - Semantic search dashboard
        - GraphRAG-style retrieval
        - Knowledge Graph visualization
        - Auto ontology generation
        - RDF / JSON-LD export
        """
    )

# ---------- Upload ----------
st.subheader("1. Upload corpus")
uploaded_files = st.file_uploader(
    "Upload one or more files",
    type=["pdf", "docx", "txt", "md", "csv", "json", "html", "htm", "py"],
    accept_multiple_files=True,
)

col_a, col_b = st.columns([1, 1])

with col_a:
    process_clicked = st.button("Build Corpus + Index", type="primary", use_container_width=True)
with col_b:
    clear_clicked = st.button("Clear Session", use_container_width=True)

if clear_clicked:
    for k in list(st.session_state.keys()):
        del st.session_state[k]
    st.rerun()

if process_clicked:
    if not uploaded_files:
        st.warning("Please upload at least one document.")
    else:
        with st.spinner("Processing corpus, embeddings, metadata, ontology, and graph..."):
            corpus_df = build_corpus_from_uploads(
                uploaded_files=uploaded_files,
                embedding_model=embedding_model,
                chunk_size=chunk_size,
                overlap=overlap,
            )

            faiss_index, embedding_matrix = build_faiss_index(corpus_df["embedding"].tolist())
            kg = build_knowledge_graph(corpus_df)
            metadata_bundle = generate_metadata_for_corpus(
                corpus_df=corpus_df,
                model=answer_model,
            )
            ontology_bundle = generate_ontology_bundle(
                corpus_df=corpus_df,
                kg=kg,
                model=answer_model,
            )

            st.session_state.corpus_df = corpus_df
            st.session_state.faiss_index = faiss_index
            st.session_state.embedding_matrix = embedding_matrix
            st.session_state.kg = kg
            st.session_state.metadata_bundle = metadata_bundle
            st.session_state.ontology_bundle = ontology_bundle

        st.success("Corpus indexed successfully.")

# ---------- Show corpus ----------
if st.session_state.corpus_df is not None:
    corpus_df = st.session_state.corpus_df
    kg = st.session_state.kg
    metadata_bundle = st.session_state.metadata_bundle
    ontology_bundle = st.session_state.ontology_bundle

    st.subheader("2. Corpus overview")
    stats = corpus_dashboard_stats(corpus_df=corpus_df, kg=kg)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Documents", stats["documents"])
    c2.metric("Chunks", stats["chunks"])
    c3.metric("Entities", stats["entities"])
    c4.metric("Relations", stats["relations"])

    with st.expander("Corpus table", expanded=False):
        view_cols = ["doc_id", "file_name", "chunk_id", "text", "entities"]
        st.dataframe(corpus_df[view_cols], use_container_width=True, height=320)

    st.subheader("3. Generated metadata")
    st.json(metadata_bundle)

    meta_df = pd.DataFrame([metadata_bundle])
    meta_csv = meta_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download metadata CSV",
        data=meta_csv,
        file_name="metadata_bundle.csv",
        mime="text/csv",
    )

    st.subheader("4. Auto ontology")
    st.json(ontology_bundle)

    st.subheader("5. Knowledge Graph visualization")
    html = graph_to_pyvis_html(kg)
    components.html(html, height=620, scrolling=True)

    st.subheader("6. Semantic search dashboard")
    query = st.text_input("Ask a semantic question about the corpus")

    if query:
        results_df = semantic_search(
            query=query,
            corpus_df=corpus_df,
            faiss_index=st.session_state.faiss_index,
            embedding_model=embedding_model,
            top_k=top_k,
        )

        st.markdown("**Top retrieved chunks**")
        st.dataframe(
            results_df[["score", "file_name", "chunk_id", "entities", "text"]],
            use_container_width=True,
            height=320,
        )

        context = "\n\n".join(results_df["text"].tolist())
        answer = answer_query_with_context(
            query=query,
            context=context,
            model=answer_model,
        )

        st.markdown("**GraphRAG answer**")
        st.write(answer)

    st.subheader("7. Export linked data")
    rdf_ttl = export_rdf_turtle(
        corpus_df=corpus_df,
        metadata_bundle=metadata_bundle,
        ontology_bundle=ontology_bundle,
        kg=kg,
    )
    jsonld_text = export_jsonld(
        corpus_df=corpus_df,
        metadata_bundle=metadata_bundle,
        ontology_bundle=ontology_bundle,
        kg=kg,
    )

    st.download_button(
        "Download RDF Turtle",
        data=rdf_ttl.encode("utf-8"),
        file_name="corpus_graph.ttl",
        mime="text/turtle",
    )
    st.download_button(
        "Download JSON-LD",
        data=jsonld_text.encode("utf-8"),
        file_name="corpus_graph.jsonld",
        mime="application/ld+json",
    )
else:
    st.info("Upload files and click 'Build Corpus + Index' to start.")