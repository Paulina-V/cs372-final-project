"""Evaluate multiple prompt designs for bill extraction.

Tests three prompt strategies on synthetic bill text and compares extraction
quality (field coverage, code accuracy, structural validity).

AI-assisted portions of this file are documented in ATTRIBUTION.md.

Requires OPENAI_API_KEY to be set for live evaluation. Can also load
pre-computed results from eval/prompt_comparison_results.json.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

RESULTS_PATH = ROOT / "eval" / "prompt_comparison_results.json"


PROMPT_MINIMAL = """Extract billing codes and amounts from this medical bill as JSON.
Return: {"patient_name": "...", "provider_name": "...", "total_billed": 0.00, "line_items": [{"cpt_code": "...", "description": "...", "billed_amount": 0.00, "date_of_service": "..."}]}
If a value is missing, use null.

BILL TEXT:
"""

PROMPT_STRUCTURED = """You are a careful medical billing extraction system. Extract only information that is explicitly present in the bill text.

For each line item, extract:
- cpt_code: The CPT or HCPCS code exactly as written, such as "99213" or "J3490". Do not guess or infer missing codes.
- icd_codes: Any associated ICD-10 diagnosis codes explicitly shown, such as ["J06.9"].
- description: The service description
- billed_amount: The amount charged in dollars (as a number)
- date_of_service: The date of service if available

Also extract:
- patient_name: The patient's name if visible
- provider_name: The provider/hospital name
- total_billed: The total amount billed

Respond ONLY with valid JSON in this format:
{
  "patient_name": "...",
  "provider_name": "...",
  "total_billed": 0.00,
  "line_items": [
    {
      "cpt_code": "99213",
      "icd_codes": ["J06.9"],
      "description": "Office visit, established patient",
      "billed_amount": 250.00,
      "date_of_service": "2024-01-15"
    }
  ]
}

If you cannot find a value, use null. Do not include any text outside the JSON.
If the text is messy, preserve uncertain fields as null instead of hallucinating.

BILL TEXT:
"""

PROMPT_COT = """You are a careful medical billing extraction system. You will extract structured data from a medical bill.

Think step by step:
1. First, identify the patient name and provider/facility name at the top of the bill.
2. Find the total billed amount, often labeled "Total", "Amount Due", or "Patient Responsibility".
3. For each service line, identify the CPT/HCPCS code (a 5-character alphanumeric code like 99213 or J3490), the description, the billed amount, and the date of service.
4. For each line, also note any ICD-10 diagnosis codes if present (format like J06.9 or M79.3).
5. If any field is not clearly present, use null rather than guessing.

After your analysis, output ONLY valid JSON in this exact format:
{
  "patient_name": "...",
  "provider_name": "...",
  "total_billed": 0.00,
  "line_items": [
    {
      "cpt_code": "99213",
      "icd_codes": ["J06.9"],
      "description": "Office visit, established patient",
      "billed_amount": 250.00,
      "date_of_service": "2024-01-15"
    }
  ]
}

BILL TEXT:
"""

PROMPTS = {
    "minimal": PROMPT_MINIMAL,
    "structured": PROMPT_STRUCTURED,
    "chain_of_thought": PROMPT_COT,
}


TEST_BILLS = [
    {
        "name": "clean_er_bill",
        "text": """DUKE UNIVERSITY HOSPITAL
Patient: Jane Smith
Date of Service: 03/15/2026
Account: DUH-2026-4421

Emergency Dept Visit - High Severity    99285    $2,850.00
Chest X-Ray 2 Views                     71046    $425.00
Complete Blood Count                    85025    $185.00
Comprehensive Metabolic Panel           80053    $320.00

TOTAL: $3,780.00""",
        "expected_codes": ["99285", "71046", "85025", "80053"],
        "expected_total": 3780.00,
        "expected_items": 4,
    },
    {
        "name": "messy_bill_partial_codes",
        "text": """Memorial Regional Medical Center
Patient Name: Robert Johnson    DOB: 05/12/1965
Statement Date: 04/01/2026

Service Description                     Amount
-----------------------------------------------
Office Visit (est. patient, moderate)   $250.00
Blood Draw                              $45.00
Urinalysis                              $35.00

Subtotal:                               $330.00
Insurance Adjustment:                  -$120.00
Patient Responsibility:                 $210.00""",
        "expected_codes": [],
        "expected_total": 330.00,
        "expected_items": 3,
    },
    {
        "name": "complex_multi_date",
        "text": """ST. MARY'S HOSPITAL
Patient: Maria Garcia    Account: SM-88291

Date: 03/10/2026
  Emergency Dept Visit High Severity  99285  $3,200.00
  IV Infusion First Hour              96365  $890.00
  ECG 12-Lead                         93000  $450.00

Date: 03/11/2026
  Hospital Observation Care           99220  $1,800.00
  Comprehensive Metabolic Panel       80053  $320.00

TOTAL CHARGES: $6,660.00""",
        "expected_codes": ["99285", "96365", "93000", "99220", "80053"],
        "expected_total": 6660.00,
        "expected_items": 5,
    },
]


def evaluate_extraction(result: dict, expected: dict) -> dict:
    """Score an extraction result against expected values."""
    items = result.get("line_items", [])
    extracted_codes = [item.get("cpt_code") for item in items if item.get("cpt_code")]

    code_matches = sum(1 for c in expected["expected_codes"] if c in extracted_codes)
    code_precision = code_matches / len(extracted_codes) if extracted_codes else 0.0
    code_recall = code_matches / len(expected["expected_codes"]) if expected["expected_codes"] else 1.0

    item_count_correct = len(items) == expected["expected_items"]

    total = result.get("total_billed")
    total_correct = total is not None and abs(total - expected["expected_total"]) < 1.0

    valid_json = isinstance(result, dict) and "line_items" in result

    return {
        "valid_json": valid_json,
        "item_count": len(items),
        "item_count_expected": expected["expected_items"],
        "item_count_correct": item_count_correct,
        "code_precision": round(code_precision, 3),
        "code_recall": round(code_recall, 3),
        "total_correct": total_correct,
        "extracted_total": total,
        "has_patient_name": result.get("patient_name") is not None,
        "has_provider_name": result.get("provider_name") is not None,
    }


def run_live_comparison() -> dict:
    """Run all prompts against all test bills using the configured LLM."""
    from src.llm import chat_completion
    from src.code_extract import normalize_bill_data, _extract_json_object

    results = {}
    for prompt_name, prompt_template in PROMPTS.items():
        prompt_results = []
        for bill in TEST_BILLS:
            start = time.time()
            try:
                raw = chat_completion(
                    messages=[{"role": "user", "content": prompt_template + bill["text"]}],
                    max_tokens=2000,
                )
                parsed = json.loads(_extract_json_object(raw))
                normalized = normalize_bill_data(parsed)
                elapsed = time.time() - start
                scores = evaluate_extraction(normalized, bill)
                scores["latency_ms"] = round(elapsed * 1000)
                scores["success"] = True
            except Exception as exc:
                scores = {
                    "success": False,
                    "error": str(exc),
                    "latency_ms": round((time.time() - start) * 1000),
                }
            scores["bill_name"] = bill["name"]
            prompt_results.append(scores)

        successful = [r for r in prompt_results if r.get("success")]
        summary = {
            "avg_code_precision": round(sum(r["code_precision"] for r in successful) / len(successful), 3) if successful else 0,
            "avg_code_recall": round(sum(r["code_recall"] for r in successful) / len(successful), 3) if successful else 0,
            "avg_latency_ms": round(sum(r["latency_ms"] for r in successful) / len(successful)) if successful else 0,
            "item_count_accuracy": round(sum(1 for r in successful if r.get("item_count_correct")) / len(successful), 3) if successful else 0,
            "total_accuracy": round(sum(1 for r in successful if r.get("total_correct")) / len(successful), 3) if successful else 0,
            "success_rate": round(len(successful) / len(prompt_results), 3),
        }
        results[prompt_name] = {"summary": summary, "details": prompt_results}

    output = {"prompts_evaluated": list(PROMPTS.keys()), "results": results}
    RESULTS_PATH.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"Prompt comparison saved to {RESULTS_PATH}")
    return output


def print_comparison_table(results: dict):
    """Print a comparison table of prompt results."""
    print(f"\n{'Prompt Design':<20} {'Code Prec':>10} {'Code Rec':>10} {'Items OK':>10} {'Total OK':>10} {'Latency':>10}")
    print("-" * 72)
    for name, data in results["results"].items():
        s = data["summary"]
        print(f"{name:<20} {s['avg_code_precision']:>10.3f} {s['avg_code_recall']:>10.3f} {s['item_count_accuracy']:>10.3f} {s['total_accuracy']:>10.3f} {s['avg_latency_ms']:>8d}ms")


if __name__ == "__main__":
    if "--live" in sys.argv:
        results = run_live_comparison()
        print_comparison_table(results)
    else:
        if RESULTS_PATH.exists():
            results = json.loads(RESULTS_PATH.read_text())
            print_comparison_table(results)
        else:
            print("No pre-computed results found. Run with --live to evaluate prompts.")
            print("Requires OPENAI_API_KEY to be set.")
