"""
Agentic dispute letter generation workflow.
Collects necessary info and generates a formal dispute letter.
"""

from src.llm import chat_completion

DISPUTE_SYSTEM_PROMPT = """You are a medical billing dispute expert. Your job is to generate a professional, formal dispute letter that a patient can send to their healthcare provider or insurance company.

The letter should:
1. Be addressed to the billing department
2. Reference specific CPT codes and charges being disputed
3. Cite the Medicare rate as a benchmark for fair pricing
4. Reference specific issues found (overcharges, duplicates, upcoding)
5. Request an itemized bill review and adjustment
6. Be firm but professional
7. Include a deadline for response (30 days is standard)

Format the letter as a proper business letter with date, addresses, subject line, body, and signature block."""


COLLECT_INFO_PROMPT = """Based on the bill analysis, I need a few more details to generate your dispute letter. Please ask the user for:
1. Their full name (if not already on the bill)
2. Their address
3. Their account/patient ID number
4. Whether they want to dispute all flagged charges or specific ones

Respond with a friendly message asking for this information."""


def generate_dispute_letter(analysis: dict, patient_info: dict) -> str:
    """Generate a formal dispute letter based on the analysis and patient info."""
    from src.explain import format_analysis_for_llm
    context = format_analysis_for_llm(analysis)

    patient_block = "\n".join(f"{k}: {v}" for k, v in patient_info.items() if v)

    prompt = f"""Generate a formal dispute letter for this patient.

PATIENT INFO:
{patient_block}

BILL ANALYSIS:
{context}

Write the complete letter ready to print and send."""

    return chat_completion(
        messages=[
            {"role": "system", "content": DISPUTE_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        max_tokens=3000,
    )


def get_info_request(analysis: dict) -> str:
    """Generate a message asking the user for info needed to write the dispute letter."""
    flagged = [f for f in analysis.get("flags", [])]
    num_flags = len(flagged)

    msg = f"I found {num_flags} potential issue{'s' if num_flags != 1 else ''} with your bill. "
    msg += "To generate a dispute letter, I'll need a few details:\n\n"
    msg += "1. **Your full name** (as it appears on the bill)\n"
    msg += "2. **Your mailing address**\n"
    msg += "3. **Account or patient ID number** (from the bill)\n"
    msg += f"4. **Which charges to dispute** — all {num_flags} flagged items, or specific ones?\n\n"
    msg += "Just provide these and I'll draft the letter for you."

    return msg
