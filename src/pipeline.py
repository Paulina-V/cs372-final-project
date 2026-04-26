"""
Main pipeline: PDF → Extract → Codes → Analysis → Explanation
Chains all components together for end-to-end bill analysis.
"""

from src.pdf_extract import extract_text
from src.code_extract import extract_codes
from src.analysis import run_all_checks
from src.explain import generate_explanation, format_analysis_for_llm
from src.llm import user_facing_model_error
from src.risk_model import predict_bill_risk


def analyze_bill(file_path: str, zip_code: str | None = None) -> dict:
    """Run the full analysis pipeline on a medical bill."""
    # Step 1: Extract raw text from PDF/image
    print("Step 1: Extracting text from bill...")
    try:
        raw_text = extract_text(file_path)
    except Exception as exc:
        return {
            "error": f"Could not read uploaded file: {exc}",
            "raw_text": "",
        }

    if not raw_text or len(raw_text.strip()) < 10:
        return {
            "error": "Could not extract text from the uploaded file. Please ensure it's a readable PDF or image.",
            "raw_text": raw_text,
        }

    # Step 2: Extract CPT/ICD codes and line items via LLM
    print("Step 2: Extracting billing codes...")
    try:
        bill_data = extract_codes(raw_text)
    except Exception as exc:
        return {
            "error": f"Failed to parse bill: {exc}",
            "raw_text": raw_text,
        }

    if bill_data.get("error"):
        return {
            "error": f"Failed to parse bill: {bill_data['error']}",
            "raw_text": raw_text,
            "bill_data": bill_data,
        }

    line_items = bill_data.get("line_items", [])
    if not line_items:
        return {
            "error": "No line items found in the bill.",
            "raw_text": raw_text,
            "bill_data": bill_data,
        }

    # Step 3: Compare against Medicare rates + run anomaly checks
    print("Step 3: Analyzing charges...")
    analysis = run_all_checks(line_items)
    analysis["patient_name"] = bill_data.get("patient_name")
    analysis["provider_name"] = bill_data.get("provider_name")
    analysis["extraction_method"] = bill_data.get("extraction_method", "unknown")
    analysis["extraction_warning"] = bill_data.get("warning")
    analysis["zip_code"] = zip_code
    try:
        analysis["risk_model"] = predict_bill_risk(analysis, zip_code)
    except Exception as exc:
        analysis["risk_model_error"] = f"Risk model unavailable: {exc}"
    analysis["raw_text"] = raw_text

    # Step 4: Generate plain-English explanation
    print("Step 4: Generating explanation...")
    try:
        explanation = generate_explanation(analysis)
    except Exception as exc:
        model_error = user_facing_model_error(exc)
        explanation = (
            "The bill was analyzed, but the explanation model could not be reached.\n\n"
            f"Model error: {model_error}\n\n"
            "Here is the structured analysis:\n\n"
            f"{format_analysis_for_llm(analysis)}"
        )
    analysis["explanation"] = explanation

    return analysis
