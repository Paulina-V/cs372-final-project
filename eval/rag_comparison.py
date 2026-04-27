"""Compare hash-based vs semantic embedding retrieval quality.

Tests both embedding strategies on a set of code lookups and description-based
queries, measuring exact match rate and retrieval relevance.

AI-assisted portions of this file are documented in ATTRIBUTION.md.
"""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import CMS_DATA_PATH

RESULTS_PATH = ROOT / "eval" / "rag_comparison_results.json"

EXACT_CODE_QUERIES = [
    {"code": "99213", "expected_fee": 95.67},
    {"code": "99285", "expected_fee": 172.20},
    {"code": "80053", "expected_fee": 10.56},
    {"code": "85025", "expected_fee": 7.77},
    {"code": "71046", "expected_fee": 33.23},
    {"code": "93000", "expected_fee": 15.44},
    {"code": "96365", "expected_fee": 67.47},
    {"code": "99215", "expected_fee": 193.35},
    {"code": "36415", "expected_fee": 9.34},
    {"code": "00100", "expected_fee": 102.49},
]

DESCRIPTION_QUERIES = [
    {"query": "emergency department visit", "expected_codes": ["99281", "99282", "99283", "99284", "99285"]},
    {"query": "chest x-ray", "expected_codes": ["71045", "71046", "71047", "71048"]},
    {"query": "blood count", "expected_codes": ["85025", "85027", "85004"]},
    {"query": "metabolic panel", "expected_codes": ["80048", "80053", "80050"]},
    {"query": "office visit established patient", "expected_codes": ["99211", "99212", "99213", "99214", "99215"]},
]


def evaluate_embedding_type(embedding_type: str) -> dict:
    """Build index and evaluate retrieval quality for one embedding type."""
    from src.rag import build_index, get_collection, query_rate, query_similar

    print(f"\nBuilding index with {embedding_type} embeddings...")
    start = time.time()
    build_index(CMS_DATA_PATH, embedding_type=embedding_type)
    index_time = time.time() - start

    collection = get_collection(embedding_type=embedding_type)

    exact_results = []
    for test in EXACT_CODE_QUERIES:
        start = time.time()
        result = query_rate(test["code"], collection)
        latency = (time.time() - start) * 1000

        found = result["found"]
        fee_match = False
        if found and result["metadata"]:
            fee_match = abs(result["metadata"]["fee"] - test["expected_fee"]) < 0.01

        exact_results.append({
            "code": test["code"],
            "found": found,
            "fee_match": fee_match,
            "latency_ms": round(latency, 2),
        })

    desc_results = []
    for test in DESCRIPTION_QUERIES:
        start = time.time()
        results = query_similar(test["query"], n_results=5, collection=collection)
        latency = (time.time() - start) * 1000

        retrieved_codes = [r["metadata"]["code"] for r in results]
        hits = sum(1 for c in retrieved_codes if c in test["expected_codes"])
        recall_at_5 = hits / min(5, len(test["expected_codes"]))

        desc_results.append({
            "query": test["query"],
            "retrieved_codes": retrieved_codes,
            "expected_any_of": test["expected_codes"],
            "hits_at_5": hits,
            "recall_at_5": round(recall_at_5, 3),
            "latency_ms": round(latency, 2),
        })

    exact_accuracy = sum(1 for r in exact_results if r["fee_match"]) / len(exact_results)
    avg_recall = sum(r["recall_at_5"] for r in desc_results) / len(desc_results)

    return {
        "embedding_type": embedding_type,
        "index_build_time_s": round(index_time, 2),
        "exact_code_lookup": {
            "accuracy": round(exact_accuracy, 3),
            "results": exact_results,
        },
        "description_search": {
            "avg_recall_at_5": round(avg_recall, 3),
            "results": desc_results,
        },
    }


def run_comparison() -> dict:
    """Compare hash and semantic embeddings."""
    results = {}

    results["hash"] = evaluate_embedding_type("hash")

    try:
        results["semantic"] = evaluate_embedding_type("semantic")
    except Exception as exc:
        print(f"Semantic comparison unavailable; skipping semantic comparison: {exc}")
        results["semantic"] = {"error": str(exc)}

    output = {
        "comparison": results,
        "summary": {},
    }

    for name, data in results.items():
        if "error" not in data:
            output["summary"][name] = {
                "exact_lookup_accuracy": data["exact_code_lookup"]["accuracy"],
                "description_recall_at_5": data["description_search"]["avg_recall_at_5"],
                "index_build_time_s": data["index_build_time_s"],
            }

    RESULTS_PATH.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"\nComparison saved to {RESULTS_PATH}")

    print(f"\n{'Embedding':<15} {'Exact Acc':>12} {'Desc Recall@5':>15} {'Build Time':>12}")
    print("-" * 56)
    for name, s in output["summary"].items():
        print(f"{name:<15} {s['exact_lookup_accuracy']:>12.3f} {s['description_recall_at_5']:>15.3f} {s['index_build_time_s']:>10.2f}s")

    return output


if __name__ == "__main__":
    run_comparison()
