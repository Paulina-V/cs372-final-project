"""Indexed retrieval over CMS-style Medicare fee schedule data.

Supports two embedding strategies:
  1. Deterministic hash-based (default, offline, lightweight)
  2. Sentence-transformer semantic embeddings (higher retrieval quality)

Use RAG_EMBEDDING_TYPE=semantic in .env to enable semantic embeddings.
"""

import hashlib
import math
import os
import re
import warnings

import pandas as pd
import chromadb
from src.config import CHROMA_PERSIST_DIR, CMS_DATA_PATH


EMBEDDING_DIMENSION = 128
RAG_EMBEDDING_TYPE = os.getenv("RAG_EMBEDDING_TYPE", "hash")


class DeterministicEmbeddingFunction:
    """Small local embedding function for reliable offline Chroma indexing."""

    def __call__(self, input):
        documents = [input] if isinstance(input, str) else input
        return [_embed_text(document) for document in documents]

    def embed_query(self, input):
        return self(input)

    def embed_documents(self, input):
        return self(input)

    @staticmethod
    def name() -> str:
        return "deterministic-token-hash-v1"


class SemanticEmbeddingFunction:
    """Sentence-transformer embeddings for higher-quality semantic retrieval."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(model_name)
        self._model_name = model_name

    def __call__(self, input):
        documents = [input] if isinstance(input, str) else input
        embeddings = self._model.encode(documents, normalize_embeddings=True)
        return embeddings.tolist()

    def embed_query(self, input):
        return self(input)

    def embed_documents(self, input):
        return self(input)

    def name(self) -> str:
        return f"sentence-transformer-{self._model_name}"


def _embed_text(text: str) -> list[float]:
    """Convert text into a normalized hashed bag-of-words vector."""
    vector = [0.0] * EMBEDDING_DIMENSION
    tokens = re.findall(r"[a-z0-9]+", str(text).lower())

    for token in tokens:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        bucket = int.from_bytes(digest[:4], "big") % EMBEDDING_DIMENSION
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[bucket] += sign

    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector

    return [value / norm for value in vector]


def get_embedding_function(embedding_type: str | None = None):
    """Return the configured embedding function."""
    chosen = embedding_type or RAG_EMBEDDING_TYPE
    if chosen == "semantic":
        try:
            return SemanticEmbeddingFunction()
        except Exception as exc:
            if embedding_type == "semantic":
                raise
            warnings.warn(
                f"Semantic embeddings unavailable ({exc}); using deterministic hash embeddings instead.",
                RuntimeWarning,
                stacklevel=2,
            )
    return DeterministicEmbeddingFunction()


def _collection_name(embedding_type: str | None = None) -> str:
    chosen = embedding_type or RAG_EMBEDDING_TYPE
    if chosen == "semantic":
        return "cms_fee_schedule_semantic"
    return "cms_fee_schedule"


def build_index(csv_path: str = CMS_DATA_PATH, embedding_type: str | None = None) -> chromadb.Collection:
    """Build a ChromaDB index from the CMS fee schedule CSV."""
    ef = get_embedding_function(embedding_type)
    col_name = _collection_name(embedding_type)
    df = pd.read_csv(csv_path)

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

    try:
        client.delete_collection(col_name)
    except Exception:
        pass

    collection = client.create_collection(
        name=col_name,
        embedding_function=ef,
    )

    batch_size = 500
    for start in range(0, len(documents), batch_size):
        end = min(start + batch_size, len(documents))
        collection.add(
            documents=documents[start:end],
            metadatas=metadatas[start:end],
            ids=ids[start:end],
        )

    print(f"Indexed {len(documents)} records into ChromaDB ({ef.name() if hasattr(ef, 'name') and callable(ef.name) else type(ef).__name__}).")
    return collection


def get_collection(embedding_type: str | None = None) -> chromadb.Collection:
    """Get the existing ChromaDB collection, building the index if needed."""
    ef = get_embedding_function(embedding_type)
    col_name = _collection_name(embedding_type)
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    try:
        return client.get_collection(
            name=col_name,
            embedding_function=ef,
        )
    except Exception:
        return build_index(CMS_DATA_PATH, embedding_type)


def query_rate(code: str, collection: chromadb.Collection = None) -> dict:
    """Look up the benchmark rate for an exact CPT/HCPCS code."""
    if collection is None:
        collection = get_collection()

    results = collection.query(
        query_texts=[f"CPT code {code}"],
        n_results=1,
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
