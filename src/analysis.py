"""
Compare billed amounts against Medicare rates and flag anomalies.
Includes rule-based checks for common billing errors.
"""

from src.rag import query_rate, query_similar
from src.config import OVERCHARGE_THRESHOLD


def compare_rates(line_items: list) -> list:
    """Compare each line item's billed amount against the Medicare rate."""
    results = []

    for item in line_items:
        code = item.get("cpt_code")
        billed = item.get("billed_amount", 0) or 0

        try:
            lookup = query_rate(code) if code else {"found": False}
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

    return results


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


UPCODING_PAIRS = [
    ("99215", "99214", "High-complexity visit (99215) is often upcoded from 99214"),
    ("99215", "99213", "High-complexity visit (99215) is often upcoded from 99213"),
    ("99285", "99284", "High-severity ER visit (99285) may be upcoded"),
    ("99285", "99283", "High-severity ER visit (99285) may be upcoded from 99283"),
]


def check_upcoding(line_items: list) -> list:
    """Flag potential upcoding based on known suspicious patterns."""
    codes = {item.get("cpt_code") for item in line_items}
    flags = []

    for high, low, msg in UPCODING_PAIRS:
        if high in codes:
            flags.append({
                "type": "POTENTIAL_UPCODING",
                "message": msg,
                "code": high,
            })

    return flags


def run_all_checks(line_items: list) -> dict:
    """Run rate comparison and all rule-based checks."""
    rated_items = compare_rates(line_items)
    duplicate_flags = check_duplicates(line_items)
    upcoding_flags = check_upcoding(line_items)

    overcharge_flags = [
        {"type": "OVERCHARGE", "message": item["flag"], "code": item.get("cpt_code")}
        for item in rated_items if item.get("flag")
    ]

    all_flags = overcharge_flags + duplicate_flags + upcoding_flags

    return {
        "rated_items": rated_items,
        "flags": all_flags,
        "num_flags": len(all_flags),
        "total_billed": sum(i.get("billed_amount", 0) or 0 for i in line_items),
        "total_medicare": sum(i.get("medicare_rate", 0) or 0 for i in rated_items if i.get("medicare_rate")),
    }
