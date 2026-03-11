import json
import networkx as nx
import pandas as pd
import streamlit as st
from openai import OpenAI

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])


def generate_ontology_bundle(corpus_df: pd.DataFrame, kg: nx.Graph, model: str = "gpt-4.1-mini") -> dict:
    entities = []
    for _, row in corpus_df.iterrows():
        entities.extend(row["entities"])

    unique_entities = sorted(list(set(entities)))[:200]
    sample_text = "\n\n".join(corpus_df["text"].head(12).tolist())[:9000]

    prompt = f"""
You are a knowledge organization and ontology design assistant.

Based on this corpus and candidate entities, generate a lightweight ontology proposal in valid JSON.

Required keys:
classes
properties
entity_type_hints
top_concepts
broader_narrower
recommended_metadata_schema
notes_for_digital_humanities_use

Rules:
- Keep classes concise and reusable.
- Include classes suitable for humanities, archives, libraries, collections, institutions, persons, places, works, events, and themes when relevant.
- Return JSON only.

ENTITIES:
{json.dumps(unique_entities, ensure_ascii=False)}

CORPUS SAMPLE:
{sample_text}
"""

    response = client.responses.create(
        model=model,
        input=prompt,
    )

    raw = response.output[0].content[0].text
    try:
        return json.loads(raw)
    except Exception:
        return {"raw_ontology_output": raw}