"""Evaluate production billing anomaly checks on synthetic bills.

This script is intentionally API-free so evaluation can run without a Duke GPT
key. It parses clean synthetic bills, builds the local Chroma index from the
CMS-style fee schedule CSV, and evaluates the same deterministic analysis path
used by the app.

AI-assisted portions of this file are documented in ATTRIBUTION.md.
"""

from __future__ import annotations

import csv
import json
import re
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.analysis import run_all_checks
from src.rag import build_index, query_rate
from src.config import CMS_DATA_PATH


RESULTS_PATH = ROOT / "eval" / "results.json"
CMS_CSV = ROOT / CMS_DATA_PATH

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
        "expected": ["OVERCHARGE", "HIGH_ACUITY_CODE_REVIEW"],
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
        "name": "high_acuity_office_review",
        "text": """
PATIENT STATEMENT
Patient: Morgan Davis
Date of Service: 01/20/2026

SERVICE DESCRIPTION                  CPT CODE    AMOUNT
Office visit established high         99215      $475.00
""",
        "expected": ["OVERCHARGE", "HIGH_ACUITY_CODE_REVIEW"],
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
        "expected": ["OVERCHARGE", "DUPLICATE", "HIGH_ACUITY_CODE_REVIEW"],
    },
    {
        "name": "unknown_code",
        "text": """
PATIENT STATEMENT
Patient: Riley Chen
Date of Service: 06/12/2026

SERVICE DESCRIPTION                  CPT CODE    AMOUNT
Experimental Wellness Service         99999      $300.00
""",
        "expected": ["RATE_NOT_FOUND"],
    },
    {
        "name": "lab_heavy_bill",
        "text": """
PATIENT STATEMENT
Patient: Drew Patel
Date of Service: 07/09/2026

SERVICE DESCRIPTION                  CPT CODE    AMOUNT
Comprehensive Metabolic Panel         80053       $14.49
Complete Blood Count                  85025        $8.42
Venipuncture                          36415        $3.00
""",
        "expected": [],
    },
    {
        "name": "lab_overcharge",
        "text": """
PATIENT STATEMENT
Patient: Sam Rivera
Date of Service: 07/10/2026

SERVICE DESCRIPTION                  CPT CODE    AMOUNT
Comprehensive Metabolic Panel         80053      $140.00
Complete Blood Count                  85025       $90.00
""",
        "expected": ["OVERCHARGE"],
    },
    {
        "name": "duplicate_different_dates_not_flagged",
        "text": """
PATIENT STATEMENT
Patient: Avery Kim

SERVICE DESCRIPTION                  CPT CODE    AMOUNT
Office visit established patient      99213       $95.00     04/01/2026
Office visit established patient      99213       $95.00     04/15/2026
""",
        "expected": [],
    },
    {
        "name": "anesthesia_reference_code",
        "text": """
PATIENT STATEMENT
Patient: Jordan Green
Date of Service: 08/18/2026

SERVICE DESCRIPTION                  CPT CODE    AMOUNT
Anesthesia for knee procedure          01400      $82.00
""",
        "expected": [],
    },
]


def parse_line_items(text: str) -> list[dict]:
    """Parse synthetic text bill rows into line items for API-free evaluation."""
    date_match = re.search(r"Date of Service:\s*(.+)", text, re.IGNORECASE)
    default_date = date_match.group(1).strip() if date_match else None
    item_pattern = re.compile(
        r"^\s*(?P<description>.+?)\s+"
        r"(?P<cpt_code>[A-Z]?\d{4,5}[A-Z]?)\s+"
        r"\$?(?P<amount>[\d,]+\.\d{2})"
        r"(?:\s+(?P<date>\d{1,2}/\d{1,2}/\d{4}))?\s*$"
    )

    items = []
    for line in text.splitlines():
        match = item_pattern.match(line)
        if not match:
            continue
        items.append({
            "description": match.group("description").strip(),
            "cpt_code": match.group("cpt_code"),
            "icd_codes": [],
            "billed_amount": float(match.group("amount").replace(",", "")),
            "date_of_service": match.group("date") or default_date,
        })
    return items


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


def run_index_parity_check(collection, sample_size: int = 20) -> dict:
    """Verify sampled Chroma lookups match the source CSV fees."""
    checked = mismatches = 0
    examples = []
    with CMS_CSV.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            if checked >= sample_size:
                break
            code = row["code"]
            expected_fee = round(float(row["fee"]), 2)
            lookup = query_rate(code, collection)
            actual_fee = None
            if lookup["found"] and lookup["metadata"]:
                actual_fee = round(float(lookup["metadata"].get("fee", 0)), 2)

            passed = actual_fee == expected_fee
            if not passed:
                mismatches += 1
            examples.append({
                "code": code,
                "expected_fee": expected_fee,
                "actual_fee": actual_fee,
                "passed": passed,
            })
            checked += 1

    return {
        "sample_size": checked,
        "num_mismatches": mismatches,
        "passed": mismatches == 0,
        "examples": examples,
    }


def run_evaluation() -> dict:
    """Evaluate synthetic cases and write a JSON summary."""
    collection = build_index(CMS_DATA_PATH)
    start = time.perf_counter()
    case_results = []
    true_positive = false_positive = false_negative = 0

    for case in EVAL_CASES:
        items = parse_line_items(case["text"])
        analysis = run_all_checks(items)
        predicted = {flag["type"] for flag in analysis["flags"]}
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
            "total_billed": round(analysis["total_billed"], 2),
            "total_medicare": round(analysis["total_medicare"], 2),
        })

    elapsed = time.perf_counter() - start
    metrics = precision_recall_f1(true_positive, false_positive, false_negative)
    parity = run_index_parity_check(collection)

    results = {
        "num_cases": len(EVAL_CASES),
        "num_passed": sum(1 for case in case_results if case["passed"]),
        "latency_seconds_total": round(elapsed, 4),
        "latency_ms_per_case": round((elapsed / len(EVAL_CASES)) * 1000, 2),
        "metrics": metrics,
        "index_parity_check": parity,
        "cases": case_results,
    }

    RESULTS_PATH.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    return results


if __name__ == "__main__":
    print(json.dumps(run_evaluation(), indent=2))
