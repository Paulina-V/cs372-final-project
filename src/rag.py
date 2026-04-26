"""
RAG pipeline over CMS Medicare fee schedule data.
Indexes fee schedule records into ChromaDB for retrieval.
"""

import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
from src.config import CHROMA_PERSIST_DIR, CMS_DATA_PATH, EMBEDDING_MODEL


def get_embedding_function():
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )


def build_index(csv_path: str = CMS_DATA_PATH) -> chromadb.Collection:
    """Build a ChromaDB index from the CMS fee schedule CSV."""
    df = pd.read_csv(csv_path)

    # Adapt column names based on actual CMS data format
    # Common columns: HCPCS, Description, Fee
    # We'll normalize in issue #2 once we see the real data
    documents = []
    metadatas = []
    ids = []

    for i, row in df.iterrows():
        doc = f"CPT/HCPCS Code: {row.get('HCPCS', row.get('code', 'N/A'))} | "
        doc += f"Description: {row.get('Description', row.get('description', 'N/A'))} | "
        doc += f"Medicare Fee: ${row.get('Fee', row.get('fee', 'N/A'))}"

        documents.append(doc)
        metadatas.append({
            "code": str(row.get('HCPCS', row.get('code', ''))),
            "fee": float(row.get('Fee', row.get('fee', 0))),
        })
        ids.append(f"cms_{i}")

    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)

    # Delete existing collection if it exists
    try:
        client.delete_collection("cms_fee_schedule")
    except Exception:
        pass

    collection = client.create_collection(
        name="cms_fee_schedule",
        embedding_function=get_embedding_function(),
    )

    # Insert in batches
    batch_size = 500
    for start in range(0, len(documents), batch_size):
        end = min(start + batch_size, len(documents))
        collection.add(
            documents=documents[start:end],
            metadatas=metadatas[start:end],
            ids=ids[start:end],
        )

    print(f"Indexed {len(documents)} records into ChromaDB.")
    return collection


def get_collection() -> chromadb.Collection:
    """Get the existing ChromaDB collection."""
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    return client.get_collection(
        name="cms_fee_schedule",
        embedding_function=get_embedding_function(),
    )


def query_rate(code: str, collection: chromadb.Collection = None) -> dict:
    """Look up the Medicare rate for a CPT/HCPCS code."""
    if collection is None:
        collection = get_collection()

    results = collection.query(
        query_texts=[f"CPT code {code}"],
        n_results=3,
        where={"code": code} if code else None,
    )

    if results["documents"] and results["documents"][0]:
        return {
            "found": True,
            "document": results["documents"][0][0],
            "metadata": results["metadatas"][0][0],
        }
    return {"found": False, "document": None, "metadata": None}


def query_similar(query: str, n_results: int = 5, collection: chromadb.Collection = None) -> list:
    """General semantic search over the fee schedule."""
    if collection is None:
        collection = get_collection()

    results = collection.query(query_texts=[query], n_results=n_results)
    return [
        {"document": doc, "metadata": meta}
        for doc, meta in zip(results["documents"][0], results["metadatas"][0])
    ]
