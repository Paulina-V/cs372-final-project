"""Evaluate rule-based billing anomaly detection on synthetic bills.

This script is intentionally API-free so evaluation can run without a Duke GPT
key. It measures the deterministic part of the system: parsing clean bill text,
Medicare-rate comparison, duplicate detection, and upcoding heuristics.
"""

from __future__ import annotations

import csv
import json
import re
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CMS_CSV = ROOT / "data" / "cms_fee_schedule.csv"
RESULTS_PATH = ROOT / "eval" / "results.json"
OVERCHARGE_THRESHOLD = 2.0

UPCODING_CODES = {
    "99215": "POTENTIAL_UPCODING",
    "99285": "POTENTIAL_UPCODING",
}

EVAL_CASES = [
    {
        "name": "clean_office_visit",
        "text": """
PATIENT STATEMENT
Patient: Alex Johnson
Date of Service: 04/01/2026

SERVICE DESCRIPTION                  CPT CODE    AMOUNT
Office visit established patient      99213       $95.00
Chest X-Ray, 2 views                  71046       $35.00
""",
        "expected": [],
    },
    {
        "name": "overcharged_er_visit",
        "text": """
PATIENT STATEMENT
Patient: Jane Smith
Date of Service: 03/15/2026

SERVICE DESCRIPTION                  CPT CODE    AMOUNT
Emergency Dept Visit - High Severity  99285     $2,850.00
Chest X-Ray, 2 views                  71046       $385.00
""",
        "expected": ["OVERCHARGE", "POTENTIAL_UPCODING"],
    },
    {
        "name": "duplicate_charge",
        "text": """
PATIENT STATEMENT
Patient: Taylor Lee
Date of Service: 02/10/2026

SERVICE DESCRIPTION                  CPT CODE    AMOUNT
Complete Blood Count                  85025        $5.00
Complete Blood Count                  85025        $5.00
""",
        "expected": ["DUPLICATE"],
    },
    {
        "name": "upcoded_office_visit",
        "text": """
PATIENT STATEMENT
Patient: Morgan Davis
Date of Service: 01/20/2026

SERVICE DESCRIPTION                  CPT CODE    AMOUNT
Office visit established high         99215      $325.00
""",
        "expected": ["OVERCHARGE", "POTENTIAL_UPCODING"],
    },
    {
        "name": "mixed_multiple_flags",
        "text": """
PATIENT STATEMENT
Patient: Casey Brown
Date of Service: 05/05/2026

SERVICE DESCRIPTION                  CPT CODE    AMOUNT
Emergency Dept Visit - High Severity  99285     $2,850.00
Emergency Dept Visit - High Severity  99285     $2,850.00
ECG - 12 Lead                         93000      $340.00
""",
        "expected": ["OVERCHARGE", "DUPLICATE", "POTENTIAL_UPCODING"],
    },
]


def load_fee_schedule() -> dict[str, float]:
    """Load CPT/HCPCS Medicare benchmark fees from the local CSV."""
    with CMS_CSV.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        return {row["code"]: float(row["fee"]) for row in reader}


def parse_line_items(text: str) -> list[dict]:
    """Parse clean text bill rows into line items."""
    item_pattern = re.compile(r"^\s*(?P<description>.+?)\s+(?P<cpt_code>\d{5})\s+\$?(?P<amount>[\d,]+\.\d{2})\s*$")
    items = []
    for line in text.splitlines():
        match = item_pattern.match(line)
        if not match:
            continue
        items.append({
            "description": match.group("description").strip(),
            "cpt_code": match.group("cpt_code"),
            "billed_amount": float(match.group("amount").replace(",", "")),
        })
    return items


def predict_flags(items: list[dict], fees: dict[str, float]) -> set[str]:
    """Run deterministic anomaly checks and return flag types."""
    flags = set()
    seen_codes = set()

    for item in items:
        code = item["cpt_code"]
        billed = item["billed_amount"]
        fee = fees.get(code)

        if fee and billed / fee > OVERCHARGE_THRESHOLD:
            flags.add("OVERCHARGE")
        if code in seen_codes:
            flags.add("DUPLICATE")
        if code in UPCODING_CODES:
            flags.add(UPCODING_CODES[code])

        seen_codes.add(code)

    return flags


def precision_recall_f1(true_positive: int, false_positive: int, false_negative: int) -> dict[str, float]:
    """Compute simple multilabel classification metrics."""
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
    }


def run_evaluation() -> dict:
    """Evaluate all cases and write a JSON summary."""
    fees = load_fee_schedule()
    start = time.perf_counter()
    case_results = []
    true_positive = false_positive = false_negative = 0

    for case in EVAL_CASES:
        items = parse_line_items(case["text"])
        predicted = predict_flags(items, fees)
        expected = set(case["expected"])

        true_positive += len(predicted & expected)
        false_positive += len(predicted - expected)
        false_negative += len(expected - predicted)

        case_results.append({
            "name": case["name"],
            "num_line_items": len(items),
            "expected_flags": sorted(expected),
            "predicted_flags": sorted(predicted),
            "passed": predicted == expected,
        })

    elapsed = time.perf_counter() - start
    metrics = precision_recall_f1(true_positive, false_positive, false_negative)

    results = {
        "num_cases": len(EVAL_CASES),
        "num_passed": sum(1 for case in case_results if case["passed"]),
        "latency_seconds_total": round(elapsed, 4),
        "latency_ms_per_case": round((elapsed / len(EVAL_CASES)) * 1000, 2),
        "metrics": metrics,
        "cases": case_results,
    }

    RESULTS_PATH.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    return results


if __name__ == "__main__":
    print(json.dumps(run_evaluation(), indent=2))
