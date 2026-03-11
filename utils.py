import json
import re
import uuid
from io import BytesIO
from typing import List, Tuple

import faiss
import networkx as nx
import numpy as np
import pandas as pd
import streamlit as st
from docx import Document
from openai import OpenAI
from pypdf import PdfReader
from pyvis.network import Network

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])


# ---------- File extraction ----------
def extract_text_from_upload(uploaded_file) -> str:
    file_name = uploaded_file.name.lower()

    if file_name.endswith(".pdf"):
        reader = PdfReader(uploaded_file)
        texts = []
        for page in reader.pages:
            page_text = page.extract_text() or ""
            texts.append(page_text)
        return "\n".join(texts).strip()

    if file_name.endswith(".docx"):
        doc = Document(uploaded_file)
        return "\n".join([p.text for p in doc.paragraphs]).strip()

    if file_name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
        return df.to_csv(index=False)

    if file_name.endswith(".json"):
        data = json.load(uploaded_file)
        return json.dumps(data, ensure_ascii=False, indent=2)

    raw = uploaded_file.read()
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("latin-1", errors="ignore")


# ---------- Chunking ----------
def chunk_text(text: str, chunk_size: int = 700, overlap: int = 150) -> List[str]:
    words = text.split()
    if not words:
        return []

    step = max(1, chunk_size - overlap)
    chunks = []

    for i in range(0, len(words), step):
        chunk_words = words[i:i + chunk_size]
        chunk = " ".join(chunk_words).strip()
        if chunk:
            chunks.append(chunk)

    return chunks


# ---------- Entity extraction ----------
def simple_entity_extraction(text: str) -> List[str]:
    """
    Rule-based fallback extractor:
    - capitalized multiword phrases
    - acronyms
    - year-like patterns
    """
    entities = set()

    multiword = re.findall(r"\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b", text)
    acronyms = re.findall(r"\b[A-Z]{2,}\b", text)
    years = re.findall(r"\b(18|19|20)\d{2}\b", text)

    for x in multiword:
        entities.add(x.strip())
    for x in acronyms:
        entities.add(x.strip())
    for x in years:
        entities.add(x.strip())

    return sorted(list(entities))[:30]


# ---------- Embeddings ----------
def embed_texts(texts: List[str], model: str = "text-embedding-3-small") -> List[List[float]]:
    response = client.embeddings.create(
        model=model,
        input=texts
    )
    return [item.embedding for item in response.data]


# ---------- Build corpus ----------
def build_corpus_from_uploads(uploaded_files, embedding_model: str, chunk_size: int, overlap: int) -> pd.DataFrame:
    rows = []

    for file_idx, uploaded_file in enumerate(uploaded_files):
        raw_text = extract_text_from_upload(uploaded_file)
        if not raw_text.strip():
            continue

        chunks = chunk_text(raw_text, chunk_size=chunk_size, overlap=overlap)

        if not chunks:
            continue

        embeddings = embed_texts(chunks, model=embedding_model)

        for chunk_idx, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            rows.append({
                "doc_id": f"doc_{file_idx+1}",
                "file_name": uploaded_file.name,
                "chunk_id": f"{uploaded_file.name}_chunk_{chunk_idx+1}",
                "text": chunk,
                "embedding": emb,
                "entities": simple_entity_extraction(chunk),
            })

    if not rows:
        return pd.DataFrame(columns=["doc_id", "file_name", "chunk_id", "text", "embedding", "entities"])

    return pd.DataFrame(rows)


# ---------- Vector DB ----------
def build_faiss_index(embeddings: List[List[float]]) -> Tuple[faiss.IndexFlatIP, np.ndarray]:
    matrix = np.array(embeddings).astype("float32")

    # cosine similarity via normalized inner product
    faiss.normalize_L2(matrix)
    index = faiss.IndexFlatIP(matrix.shape[1])
    index.add(matrix)

    return index, matrix


def semantic_search(query: str, corpus_df: pd.DataFrame, faiss_index, embedding_model: str, top_k: int = 5) -> pd.DataFrame:
    q = embed_texts([query], model=embedding_model)[0]
    q_vec = np.array([q]).astype("float32")
    faiss.normalize_L2(q_vec)

    scores, idxs = faiss_index.search(q_vec, top_k)
    idxs = idxs[0].tolist()
    scores = scores[0].tolist()

    rows = []
    for score, idx in zip(scores, idxs):
        if idx < 0 or idx >= len(corpus_df):
            continue
        row = corpus_df.iloc[idx].to_dict()
        row["score"] = round(float(score), 4)
        rows.append(row)

    return pd.DataFrame(rows)


# ---------- Metadata generation ----------
def generate_metadata_for_corpus(corpus_df: pd.DataFrame, model: str = "gpt-4.1-mini") -> dict:
    joined = "\n\n".join(corpus_df["text"].head(20).tolist())[:12000]

    prompt = f"""
You are an expert in digital humanities, metadata, and library science.

From the corpus below, generate a compact metadata bundle as valid JSON with keys:
title,
probable_authors,
document_types,
core_topics,
keywords,
named_entities,
summary,
research_domains,
suggested_subject_headings,
suggested_description,
language_guess

Return JSON only.

CORPUS:
{joined}
"""

    response = client.responses.create(
        model=model,
        input=prompt,
    )

    text = response.output[0].content[0].text
    try:
        return json.loads(text)
    except Exception:
        return {"raw_metadata_output": text}


# ---------- QA ----------
def answer_query_with_context(query: str, context: str, model: str = "gpt-4.1-mini") -> str:
    prompt = f"""
Answer the question using only the provided context.
If the answer is uncertain, say so clearly.
Write in a concise academic style.

Context:
{context}

Question:
{query}
"""

    response = client.responses.create(
        model=model,
        input=prompt,
    )
    return response.output[0].content[0].text


# ---------- Knowledge graph ----------
def build_knowledge_graph(corpus_df: pd.DataFrame) -> nx.Graph:
    G = nx.Graph()

    for _, row in corpus_df.iterrows():
        doc_node = row["chunk_id"]
        G.add_node(doc_node, label=row["chunk_id"], node_type="chunk", file_name=row["file_name"])

        for entity in row["entities"]:
            G.add_node(entity, label=entity, node_type="entity")
            G.add_edge(doc_node, entity, relation="mentions")

        # entity co-occurrence edges
        entities = row["entities"]
        for i in range(len(entities)):
            for j in range(i + 1, len(entities)):
                a, b = entities[i], entities[j]
                if G.has_edge(a, b):
                    if "weight" in G[a][b]:
                        G[a][b]["weight"] += 1
                    else:
                        G[a][b]["weight"] = 1
                else:
                    G.add_edge(a, b, relation="co_occurs", weight=1)

    return G


def graph_to_pyvis_html(G: nx.Graph) -> str:
    net = Network(height="600px", width="100%", bgcolor="#ffffff", font_color="#222222")
    net.barnes_hut()

    # limit rendering to keep Streamlit stable
    nodes = list(G.nodes(data=True))[:200]
    allowed = set([n for n, _ in nodes])

    for node, attrs in nodes:
        node_type = attrs.get("node_type", "entity")
        title = json.dumps(attrs, ensure_ascii=False, indent=2)

        if node_type == "chunk":
            color = "#f59e0b"
            size = 16
        else:
            color = "#3b82f6"
            size = 20

        net.add_node(node, label=str(node)[:50], title=title, color=color, size=size)

    for s, t, attrs in G.edges(data=True):
        if s in allowed and t in allowed:
            label = attrs.get("relation", "")
            width = attrs.get("weight", 1)
            net.add_edge(s, t, label=label, width=min(width, 8))

    return net.generate_html()


# ---------- Dashboard stats ----------
def corpus_dashboard_stats(corpus_df: pd.DataFrame, kg: nx.Graph) -> dict:
    entity_nodes = [n for n, d in kg.nodes(data=True) if d.get("node_type") == "entity"]
    relation_edges = len(kg.edges())

    return {
        "documents": int(corpus_df["doc_id"].nunique()),
        "chunks": int(len(corpus_df)),
        "entities": int(len(entity_nodes)),
        "relations": int(relation_edges),
    }