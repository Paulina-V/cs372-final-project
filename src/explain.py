"""
Generate plain-English explanations of medical bills and handle multi-turn conversation.
"""

from src.llm import chat_completion

SYSTEM_PROMPT = """You are a friendly, knowledgeable medical billing assistant. Your job is to help patients understand their medical bills in plain, non-technical language.

You have access to the patient's bill analysis, including:
- Extracted line items with CPT/HCPCS codes
- Medicare rate comparisons
- Flags for potential overcharges, duplicates, or upcoding

When explaining a bill:
1. Start with a brief, reassuring overview
2. Explain each line item in plain English (what the service was, what was charged, how it compares to Medicare rates)
3. Clearly highlight any flags or concerns
4. Suggest next steps the patient can take

Be empathetic and clear. Avoid jargon. If a charge looks suspicious, explain why without being alarmist — patients are already stressed about medical bills.

When the user asks follow-up questions, answer based on the bill data provided. If you don't know something, say so honestly."""


def generate_explanation(analysis: dict) -> str:
    """Generate a plain-English explanation of the bill analysis."""
    context = format_analysis_for_llm(analysis)

    return chat_completion(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Please explain this medical bill to me:\n\n{context}"}
        ],
        max_tokens=2000,
    )


def chat(messages: list, analysis: dict) -> str:
    """Handle multi-turn conversation about a bill."""
    context = format_analysis_for_llm(analysis)

    system = SYSTEM_PROMPT + f"\n\nHere is the patient's bill analysis:\n{context}"

    return chat_completion(
        messages=[{"role": "system", "content": system}] + messages,
        max_tokens=2000,
    )


def format_analysis_for_llm(analysis: dict) -> str:
    """Format the analysis dict into a readable string for the LLM."""
    lines = []
    lines.append(f"Total Billed: ${analysis.get('total_billed', 0):.2f}")
    lines.append(f"Total Medicare Equivalent: ${analysis.get('total_medicare', 0):.2f}")
    lines.append(f"Number of Flags: {analysis.get('num_flags', 0)}")
    lines.append("")

    lines.append("LINE ITEMS:")
    for item in analysis.get("rated_items", []):
        lines.append(f"  - {item.get('cpt_code', 'N/A')}: {item.get('description', 'No description')}")
        lines.append(f"    Billed: ${item.get('billed_amount', 0):.2f}")
        if item.get("medicare_rate"):
            lines.append(f"    Medicare Rate: ${item['medicare_rate']:.2f}")
            lines.append(f"    Ratio: {item.get('ratio_to_medicare', 0):.1f}x")
        if item.get("flag"):
            lines.append(f"    ⚠️ {item['flag']}")
        lines.append("")

    if analysis.get("flags"):
        lines.append("FLAGS:")
        for flag in analysis["flags"]:
            lines.append(f"  ⚠️ [{flag['type']}] {flag['message']}")

    return "\n".join(lines)
