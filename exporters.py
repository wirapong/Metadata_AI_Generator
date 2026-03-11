import json
from rdflib import Graph, Literal, Namespace, RDF, RDFS, URIRef


BASE = Namespace("http://example.org/kg/")
SCHEMA = Namespace("http://schema.org/")
DCTERMS = Namespace("http://purl.org/dc/terms/")


def _safe_uri(value: str) -> str:
    return (
        str(value)
        .strip()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
        .replace("#", "_")
    )


def export_rdf_turtle(corpus_df, metadata_bundle, ontology_bundle, kg) -> str:
    g = Graph()
    g.bind("base", BASE)
    g.bind("schema", SCHEMA)
    g.bind("dcterms", DCTERMS)

    corpus_uri = BASE["corpus/main"]
    g.add((corpus_uri, RDF.type, SCHEMA.Dataset))

    if isinstance(metadata_bundle, dict):
        if "title" in metadata_bundle:
            g.add((corpus_uri, DCTERMS.title, Literal(str(metadata_bundle["title"]))))
        if "summary" in metadata_bundle:
            g.add((corpus_uri, DCTERMS.description, Literal(str(metadata_bundle["summary"]))))

        for kw in metadata_bundle.get("keywords", []) if isinstance(metadata_bundle.get("keywords", []), list) else []:
            g.add((corpus_uri, SCHEMA.keywords, Literal(str(kw))))

    # classes from ontology
    if isinstance(ontology_bundle, dict):
        for cls in ontology_bundle.get("classes", []) if isinstance(ontology_bundle.get("classes", []), list) else []:
            cls_uri = BASE[f"class/{_safe_uri(cls)}"]
            g.add((cls_uri, RDF.type, RDFS.Class))
            g.add((cls_uri, RDFS.label, Literal(str(cls))))

    # documents and entities
    for _, row in corpus_df.iterrows():
        chunk_uri = BASE[f"chunk/{_safe_uri(row['chunk_id'])}"]
        g.add((chunk_uri, RDF.type, SCHEMA.CreativeWork))
        g.add((chunk_uri, DCTERMS.identifier, Literal(row["chunk_id"])))
        g.add((chunk_uri, DCTERMS.source, Literal(row["file_name"])))
        g.add((chunk_uri, DCTERMS.description, Literal(row["text"][:1200])))

        for entity in row["entities"]:
            ent_uri = BASE[f"entity/{_safe_uri(entity)}"]
            g.add((ent_uri, RDF.type, SCHEMA.Thing))
            g.add((ent_uri, RDFS.label, Literal(entity)))
            g.add((chunk_uri, SCHEMA.mentions, ent_uri))

    return g.serialize(format="turtle")


def export_jsonld(corpus_df, metadata_bundle, ontology_bundle, kg) -> str:
    graph_items = []

    graph_items.append({
        "@id": "http://example.org/kg/corpus/main",
        "@type": "Dataset",
        "name": metadata_bundle.get("title", "Corpus"),
        "description": metadata_bundle.get("summary", ""),
        "keywords": metadata_bundle.get("keywords", []),
    })

    for _, row in corpus_df.iterrows():
        chunk_id = f"http://example.org/kg/chunk/{_safe_uri(row['chunk_id'])}"
        mentions = [f"http://example.org/kg/entity/{_safe_uri(e)}" for e in row["entities"]]

        graph_items.append({
            "@id": chunk_id,
            "@type": "CreativeWork",
            "identifier": row["chunk_id"],
            "source": row["file_name"],
            "text": row["text"][:1200],
            "mentions": mentions,
        })

        for entity in row["entities"]:
            graph_items.append({
                "@id": f"http://example.org/kg/entity/{_safe_uri(entity)}",
                "@type": "Thing",
                "name": entity,
            })

    doc = {
        "@context": {
            "name": "http://schema.org/name",
            "description": "http://schema.org/description",
            "keywords": "http://schema.org/keywords",
            "identifier": "http://purl.org/dc/terms/identifier",
            "source": "http://purl.org/dc/terms/source",
            "text": "http://schema.org/text",
            "mentions": {"@id": "http://schema.org/mentions", "@type": "@id"},
            "CreativeWork": "http://schema.org/CreativeWork",
            "Thing": "http://schema.org/Thing",
            "Dataset": "http://schema.org/Dataset",
        },
        "@graph": graph_items,
    }

    return json.dumps(doc, ensure_ascii=False, indent=2)