"""
Extract CPT/ICD codes and line items from raw bill text using an LLM.
Returns structured JSON with billing details.
"""

import json
import re
from src.llm import chat_completion

EXTRACTION_PROMPT = """You are a medical billing expert. Analyze the following medical bill text and extract all line items.

For each line item, extract:
- cpt_code: The CPT or HCPCS code (e.g., "99213")
- icd_codes: Any associated ICD-10 diagnosis codes (e.g., ["J06.9"])
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

BILL TEXT:
"""


def extract_codes(bill_text: str) -> dict:
    """Use an LLM to extract structured billing data from raw text."""
    try:
        raw = chat_completion(
            messages=[{"role": "user", "content": EXTRACTION_PROMPT + bill_text}],
            max_tokens=2000,
        )
    except Exception as exc:
        fallback = extract_codes_with_regex(bill_text)
        fallback["warning"] = f"LLM extraction failed; used regex fallback instead: {exc}"
        return fallback

    # Clean up potential markdown fences
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1]
        if raw.endswith("```"):
            raw = raw[:-3]
        raw = raw.strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        fallback = extract_codes_with_regex(bill_text)
        fallback["warning"] = "LLM response was not valid JSON; used regex fallback instead."
        fallback["raw_response"] = raw
        return fallback


def extract_codes_with_regex(bill_text: str) -> dict:
    """Best-effort parser for clean text bills used when the LLM is unavailable."""
    patient_match = re.search(r"Patient:\s*(.+)", bill_text, re.IGNORECASE)
    provider_match = re.search(r"^(.+?HOSPITAL|.+?CLINIC|.+?MEDICAL CENTER)", bill_text, re.IGNORECASE | re.MULTILINE)
    date_match = re.search(r"Date of Service:\s*(.+)", bill_text, re.IGNORECASE)
    total_match = re.search(r"(?:SUBTOTAL|TOTAL|PATIENT RESPONSIBILITY):\s*\$?([\d,]+\.\d{2})", bill_text, re.IGNORECASE)

    line_items = []
    item_pattern = re.compile(r"^\s*(?P<description>.+?)\s+(?P<cpt_code>\d{5})\s+\$?(?P<amount>[\d,]+\.\d{2})\s*$")
    for line in bill_text.splitlines():
        match = item_pattern.match(line)
        if not match:
            continue

        line_items.append({
            "cpt_code": match.group("cpt_code"),
            "icd_codes": [],
            "description": match.group("description").strip(),
            "billed_amount": float(match.group("amount").replace(",", "")),
            "date_of_service": date_match.group(1).strip() if date_match else None,
        })

    return {
        "patient_name": patient_match.group(1).strip() if patient_match else None,
        "provider_name": provider_match.group(1).strip() if provider_match else None,
        "total_billed": float(total_match.group(1).replace(",", "")) if total_match else None,
        "line_items": line_items,
        "extraction_method": "regex_fallback",
    }
