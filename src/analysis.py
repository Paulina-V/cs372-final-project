"""Compare billed amounts against benchmark rates and flag review signals.

AI-assisted portions of this file are documented in ATTRIBUTION.md.
"""

from src.rag import get_collection, query_rate
from src.config import OVERCHARGE_THRESHOLD


def compare_rates(line_items: list, embedding_type: str | None = None) -> tuple[list, str | None]:
    """Compare each line item's billed amount against the Medicare rate."""
    results = []
    lookup_warning = None

    try:
        collection = get_collection(embedding_type)
    except Exception as exc:
        if embedding_type == "semantic":
            lookup_warning = (
                f"Semantic embeddings unavailable ({exc}); fell back to hash embeddings."
            )
            collection = get_collection("hash")
            embedding_type = "hash"
        else:
            raise

    for item in line_items:
        code = item.get("cpt_code")
        billed = item.get("billed_amount", 0) or 0

        try:
            lookup = (
                query_rate(code, collection=collection, embedding_type=embedding_type)
                if code
                else {"found": False}
            )
        except Exception as exc:
            lookup = {
                "found": False,
                "document": None,
                "metadata": None,
                "error": f"Medicare rate lookup failed: {exc}",
            }

        medicare_rate = None
        ratio = None
        flag = None

        if lookup["found"] and lookup["metadata"]:
            medicare_rate = lookup["metadata"].get("fee", 0)
            if medicare_rate and medicare_rate > 0:
                ratio = billed / medicare_rate
                if ratio > OVERCHARGE_THRESHOLD:
                    flag = f"OVERCHARGE: Billed ${billed:.2f} is {ratio:.1f}x the Medicare rate of ${medicare_rate:.2f}"

        results.append({
            **item,
            "medicare_rate": medicare_rate,
            "ratio_to_medicare": ratio,
            "flag": flag,
            "rag_result": lookup.get("document"),
            "rag_error": lookup.get("error"),
        })

    return results, lookup_warning


def check_duplicates(line_items: list) -> list:
    """Flag duplicate CPT codes billed on the same date."""
    seen = {}
    flags = []

    for i, item in enumerate(line_items):
        key = (item.get("cpt_code"), item.get("date_of_service"))
        if key[0] and key in seen:
            flags.append({
                "type": "DUPLICATE",
                "message": f"CPT {key[0]} appears more than once on {key[1] or 'same date'}",
                "indices": [seen[key], i],
            })
        else:
            seen[key] = i

    return flags


HIGH_ACUITY_REVIEW_CODES = {
    "99215": "High-complexity established-patient visit (99215) should be supported by documentation.",
    "99285": "High-severity emergency department visit (99285) should be supported by acuity documentation.",
}


def check_upcoding(line_items: list) -> list:
    """Flag high-acuity CPT codes for documentation review."""
    codes = {item.get("cpt_code") for item in line_items}
    flags = []

    for code, message in HIGH_ACUITY_REVIEW_CODES.items():
        if code in codes:
            flags.append({
                "type": "HIGH_ACUITY_CODE_REVIEW",
                "message": message,
                "code": code,
            })

    return flags


def check_missing_rates(rated_items: list) -> list:
    """Flag extracted codes that could not be matched to a benchmark rate."""
    flags = []
    for item in rated_items:
        code = item.get("cpt_code")
        if code and not item.get("medicare_rate"):
            flags.append({
                "type": "RATE_NOT_FOUND",
                "message": f"No benchmark rate was found for CPT/HCPCS {code}; verify the code and any modifiers.",
                "code": code,
            })
    return flags


def run_all_checks(line_items: list, embedding_type: str | None = None) -> dict:
    """Run rate comparison and all rule-based checks."""
    rated_items, lookup_warning = compare_rates(line_items, embedding_type)
    duplicate_flags = check_duplicates(line_items)
    upcoding_flags = check_upcoding(line_items)
    missing_rate_flags = check_missing_rates(rated_items)

    overcharge_flags = [
        {"type": "OVERCHARGE", "message": item["flag"], "code": item.get("cpt_code")}
        for item in rated_items if item.get("flag")
    ]

    all_flags = overcharge_flags + duplicate_flags + upcoding_flags + missing_rate_flags

    return {
        "rated_items": rated_items,
        "flags": all_flags,
        "num_flags": len(all_flags),
        "total_billed": sum(i.get("billed_amount", 0) or 0 for i in line_items),
        "total_medicare": sum(i.get("medicare_rate", 0) or 0 for i in rated_items if i.get("medicare_rate")),
        "embedding_type": embedding_type or "hash",
        "embedding_warning": lookup_warning,
    }
