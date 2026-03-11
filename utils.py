import pandas as pd
import numpy as np
import networkx as nx
from pypdf import PdfReader
from docx import Document
from openai import OpenAI
import tiktoken
from sklearn.metrics.pairwise import cosine_similarity
import json

import streamlit as st

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])


def extract_text(file):

    name = file.name.lower()

    if name.endswith(".pdf"):
        reader = PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text

    if name.endswith(".docx"):
        doc = Document(file)
        return "\n".join([p.text for p in doc.paragraphs])

    if name.endswith(".csv"):
        df = pd.read_csv(file)
        return df.to_string()

    if name.endswith(".json"):
        data = json.load(file)
        return json.dumps(data)

    return file.read().decode("utf-8")


def chunk_text(text, size=800, overlap=200):

    tokens = text.split()

    chunks = []

    for i in range(0, len(tokens), size - overlap):

        chunk = tokens[i:i+size]

        chunks.append(" ".join(chunk))

    return chunks


def build_embeddings(chunks):

    vectors = []

    for chunk in chunks:

        emb = client.embeddings.create(
            model="text-embedding-3-small",
            input=chunk
        )

        vectors.append(emb.data[0].embedding)

    return np.array(vectors)


def retrieve_context(query, docs, embeddings, top_k=5):

    q_emb = client.embeddings.create(
        model="text-embedding-3-small",
        input=query
    )

    q_vec = np.array(q_emb.data[0].embedding).reshape(1,-1)

    sims = cosine_similarity(q_vec, embeddings)[0]

    idx = sims.argsort()[-top_k:][::-1]

    return "\n\n".join([docs[i] for i in idx])


def generate_metadata(text):

    prompt = f"""
    Extract metadata from the following document.

    Return JSON with:

    title
    authors
    keywords
    summary
    entities

    TEXT:
    {text[:4000]}
    """

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=prompt
    )

    try:
        return json.loads(response.output[0].content[0].text)

    except:
        return {"metadata_raw": response.output[0].content[0].text}


def answer_query(query, context):

    prompt = f"""
    Answer the question based on the context.

    Context:
    {context}

    Question:
    {query}
    """

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=prompt
    )

    return response.output[0].content[0].text