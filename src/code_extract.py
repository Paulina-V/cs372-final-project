"""
Extract CPT/ICD codes and line items from raw bill text using an LLM.
Returns structured JSON with billing details.
"""

import json
import re
from src.llm import chat_completion, user_facing_model_error

EXTRACTION_PROMPT = """You are a careful medical billing extraction system. Extract only information that is explicitly present in the bill text.

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


def extract_codes(bill_text: str) -> dict:
    """Use an LLM to extract structured billing data from raw text."""
    try:
        raw = chat_completion(
            messages=[{"role": "user", "content": EXTRACTION_PROMPT + bill_text}],
            max_tokens=2000,
        )
    except Exception as exc:
        fallback = extract_codes_with_regex(bill_text)
        fallback["warning"] = f"LLM extraction failed; used regex fallback instead: {user_facing_model_error(exc)}"
        return fallback

    try:
        parsed = json.loads(_extract_json_object(raw))
    except json.JSONDecodeError:
        fallback = extract_codes_with_regex(bill_text)
        fallback["warning"] = "LLM response was not valid JSON; used regex fallback instead."
        fallback["raw_response"] = raw
        return fallback

    normalized = normalize_bill_data(parsed)
    if not normalized.get("line_items"):
        fallback = extract_codes_with_regex(bill_text)
        fallback["warning"] = "LLM extraction returned no line items; used regex fallback instead."
        fallback["raw_response"] = raw
        return fallback

    normalized["extraction_method"] = "llm"
    return normalized


def _extract_json_object(raw: str) -> str:
    """Extract a JSON object from a model response that may include fences."""
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start:end + 1]
    return text


def normalize_bill_data(data: dict) -> dict:
    """Normalize LLM output into the schema expected by downstream analysis."""
    line_items = data.get("line_items") if isinstance(data, dict) else []
    if not isinstance(line_items, list):
        line_items = []

    normalized_items = []
    warnings = []
    for index, item in enumerate(line_items):
        if not isinstance(item, dict):
            warnings.append(f"Skipped non-object line item at index {index}.")
            continue

        cpt_code = _clean_code(item.get("cpt_code"))
        billed_amount = _parse_amount(item.get("billed_amount"))
        description = item.get("description")
        date_of_service = item.get("date_of_service")
        icd_codes = item.get("icd_codes") or []

        if isinstance(icd_codes, str):
            icd_codes = [icd_codes]
        if not isinstance(icd_codes, list):
            icd_codes = []

        if billed_amount is None:
            warnings.append(f"Skipped line item {index + 1} because billed_amount was missing or invalid.")
            continue

        normalized_items.append({
            "cpt_code": cpt_code,
            "icd_codes": [str(code).strip() for code in icd_codes if str(code).strip()],
            "description": str(description).strip() if description else None,
            "billed_amount": billed_amount,
            "date_of_service": str(date_of_service).strip() if date_of_service else None,
        })

    result = {
        "patient_name": _clean_optional_text(data.get("patient_name")) if isinstance(data, dict) else None,
        "provider_name": _clean_optional_text(data.get("provider_name")) if isinstance(data, dict) else None,
        "total_billed": _parse_amount(data.get("total_billed")) if isinstance(data, dict) else None,
        "line_items": normalized_items,
    }
    if warnings:
        result["warning"] = " ".join(warnings)
    return result


def _clean_code(value) -> str | None:
    if value is None:
        return None
    code = str(value).strip().upper()
    if not code or code in {"N/A", "NULL", "NONE"}:
        return None
    return code


def _clean_optional_text(value) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"null", "none", "n/a"}:
        return None
    return text


def _parse_amount(value) -> float | None:
    if value is None:
        return None
    if isinstance(value, int | float):
        return float(value)
    text = str(value).strip()
    if not text or text.lower() in {"null", "none", "n/a"}:
        return None
    text = text.replace("$", "").replace(",", "")
    try:
        return float(text)
    except ValueError:
        return None


def extract_codes_with_regex(bill_text: str) -> dict:
    """Best-effort parser for clean text bills used when the LLM is unavailable."""
    patient_match = re.search(r"Patient:\s*(.+)", bill_text, re.IGNORECASE)
    client_match = re.search(r"Client\s*\n(.+)", bill_text, re.IGNORECASE)
    provider_match = re.search(r"^(.+?HOSPITAL|.+?CLINIC|.+?MEDICAL CENTER)", bill_text, re.IGNORECASE | re.MULTILINE)
    superbill_provider_match = re.search(r"Provider\s*\n(.+)", bill_text, re.IGNORECASE)
    date_match = re.search(r"Date of Service:\s*(.+)", bill_text, re.IGNORECASE)
    total_match = re.search(r"(?:SUBTOTAL|TOTAL FEES|TOTAL|PATIENT RESPONSIBILITY):?\s*\$?([\d,]+(?:\.\d{2})?)", bill_text, re.IGNORECASE)

    line_items = []
    item_pattern = re.compile(r"^\s*(?P<description>.+?)\s+(?P<cpt_code>[A-Z]?\d{4,5}[A-Z]?)\s+\$?(?P<amount>[\d,]+(?:\.\d{2})?)\s*$")
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

    line_items.extend(_parse_superbill_rows(bill_text))

    return {
        "patient_name": _first_match_text(patient_match, client_match),
        "provider_name": _first_match_text(provider_match, superbill_provider_match),
        "total_billed": float(total_match.group(1).replace(",", "")) if total_match else None,
        "line_items": line_items,
        "extraction_method": "regex_fallback",
    }


def _parse_superbill_rows(bill_text: str) -> list[dict]:
    """Parse superbill rows where CPT, modifier, and fee appear on separate lines."""
    lines = [line.strip() for line in bill_text.splitlines() if line.strip()]
    items = []
    row_start = re.compile(r"^(?P<date>\d{2}/\d{2}/\d{4})\s+(?P<pos>\d{2})\s+(?P<cpt_code>\d{5})(?:\s+-)?$")
    detail = re.compile(
        r"^(?P<dx>(?:\d+\s*,\s*)*\d+)\s+"
        r"(?P<description>.+?)\s+"
        r"(?P<units>\d+)\s+"
        r"\$(?P<fee>[\d,]+(?:\.\d{2})?)"
        r"(?:\s+\$(?P<paid>[\d,]+(?:\.\d{2})?))?\s*$"
    )

    index = 0
    while index < len(lines):
        start = row_start.match(lines[index])
        if not start:
            index += 1
            continue

        detail_line = None
        for lookahead in range(index + 1, min(index + 4, len(lines))):
            if detail.match(lines[lookahead]):
                detail_line = lines[lookahead]
                break

        if detail_line:
            detail_match = detail.match(detail_line)
            items.append({
                "cpt_code": start.group("cpt_code"),
                "icd_codes": [],
                "description": detail_match.group("description").strip(),
                "billed_amount": float(detail_match.group("fee").replace(",", "")),
                "date_of_service": start.group("date"),
            })
            index = lookahead + 1
        else:
            index += 1

    return items


def _first_match_text(*matches) -> str | None:
    """Return the first non-empty regex group from a list of optional matches."""
    for match in matches:
        if match:
            return match.group(1).strip()
    return None
