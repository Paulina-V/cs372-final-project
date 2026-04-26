"""
Medical Billing Assistant — Gradio Web App
Upload a medical bill, get a plain-English explanation, and generate a dispute letter.

Production considerations: structured logging, in-memory rate limiting, and
graceful error handling for all user-facing operations.
"""

import logging
import os
import time

import gradio as gr
from src.pipeline import analyze_bill
from src.explain import chat
from src.dispute import generate_dispute_letter
from src.llm import user_facing_model_error
from src.risk_model import format_risk_summary


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("app.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("medical_billing_assistant")


class RateLimiter:
    """Simple in-memory rate limiter per session (IP not available in Gradio)."""

    def __init__(self, max_requests: int = 20, window_seconds: int = 60):
        self._max = max_requests
        self._window = window_seconds
        self._requests: list[float] = []

    def check(self) -> bool:
        now = time.time()
        self._requests = [t for t in self._requests if now - t < self._window]
        if len(self._requests) >= self._max:
            return False
        self._requests.append(now)
        return True

    @property
    def remaining(self) -> int:
        now = time.time()
        self._requests = [t for t in self._requests if now - t < self._window]
        return max(0, self._max - len(self._requests))


rate_limiter = RateLimiter(max_requests=20, window_seconds=60)


def process_bill(file, zip_code):
    """Handle bill upload and run the full pipeline."""
    if file is None:
        return "Please upload a medical bill (PDF, image, or text file).", [], {}

    if not rate_limiter.check():
        logger.warning("Rate limit exceeded for bill analysis request")
        return "Rate limit exceeded. Please wait a moment before trying again.", [], {}

    logger.info("Starting bill analysis for file: %s (ZIP: %s)", file.name, zip_code or "none")
    start = time.time()

    try:
        result = analyze_bill(file.name, zip_code=zip_code)
    except Exception as exc:
        logger.exception("Unexpected error during bill analysis")
        return f"Unexpected error while analyzing bill: {exc}", [], {}

    elapsed = time.time() - start
    logger.info(
        "Bill analysis complete: %d items, %d flags, %.2fs",
        len(result.get("rated_items", [])),
        result.get("num_flags", 0),
        elapsed,
    )

    if result.get("error"):
        logger.warning("Analysis returned error: %s", result["error"])
        return f"Error: {result['error']}", [], {}

    explanation = _format_analysis_output(result)

    history = [
        {"role": "assistant", "content": explanation}
    ]

    return explanation, history, result


def respond(message, history, analysis):
    """Handle follow-up questions in the chat."""
    history = history or []
    if not message or not message.strip():
        return history

    if not analysis:
        return history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": "Please upload a bill first using the upload tab."},
        ]

    if not rate_limiter.check():
        logger.warning("Rate limit exceeded for chat request")
        return history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": "Rate limit exceeded. Please wait a moment before trying again."},
        ]

    logger.info("Chat follow-up question: %s", message[:100])

    api_messages = []
    for msg in history:
        if isinstance(msg, dict) and msg.get("role") in {"user", "assistant"}:
            api_messages.append({"role": msg["role"], "content": msg.get("content", "")})
    api_messages.append({"role": "user", "content": message})

    try:
        response = chat(api_messages, analysis)
    except Exception as exc:
        logger.exception("Chat model call failed")
        response = f"I could not reach the chat model right now. Error: {user_facing_model_error(exc)}"

    history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": response},
    ]

    return history


def generate_dispute(name, address, account_id, dispute_all, analysis):
    """Generate a dispute letter from collected info."""
    if not analysis:
        return "Please upload and analyze a bill first."

    if not analysis.get("flags"):
        return "No issues were flagged in your bill; there's nothing to dispute."

    if not rate_limiter.check():
        logger.warning("Rate limit exceeded for dispute letter request")
        return "Rate limit exceeded. Please wait a moment before trying again."

    logger.info("Generating dispute letter for account: %s", account_id or "unknown")

    patient_info = {
        "patient_name": name or analysis.get("patient_name", ""),
        "address": address,
        "account_id": account_id,
        "provider_name": analysis.get("provider_name", ""),
        "dispute_scope": "all flagged charges" if dispute_all else "selected charges",
    }

    try:
        letter = generate_dispute_letter(analysis, patient_info)
    except Exception as exc:
        logger.exception("Dispute letter generation failed")
        return f"I could not generate the dispute letter because the model call failed: {user_facing_model_error(exc)}"

    logger.info("Dispute letter generated successfully")
    return letter


def _format_analysis_output(result: dict) -> str:
    """Add transparent pipeline status above the model explanation."""
    explanation = result.get("explanation", "No explanation generated.")
    matched = sum(1 for item in result.get("rated_items", []) if item.get("medicare_rate"))
    unmatched = sum(1 for item in result.get("rated_items", []) if not item.get("medicare_rate"))

    status_lines = [
        "### Pipeline Status",
        f"- Extraction method: `{result.get('extraction_method', 'unknown')}`",
        f"- Line items analyzed: `{len(result.get('rated_items', []))}`",
        f"- Fee schedule matches: `{matched}`",
        f"- Unmatched codes: `{unmatched}`",
        f"- Flags found: `{result.get('num_flags', 0)}`",
    ]

    if result.get("extraction_warning"):
        status_lines.append(f"- Extraction warning: {result['extraction_warning']}")
    if result.get("zip_code"):
        status_lines.append(f"- ZIP code context: `{result['zip_code']}`")
    if result.get("risk_model"):
        status_lines.append(f"- {format_risk_summary(result['risk_model'])}")
    if result.get("risk_model_error"):
        status_lines.append(f"- Risk model warning: {result['risk_model_error']}")

    return "\n".join(status_lines) + "\n\n---\n\n" + explanation


def build_app():
    with gr.Blocks(
        title="Medical Billing Assistant",
    ) as app:
        analysis_state = gr.State({})

        gr.Markdown(
            """
            # Medical Billing Assistant
            Upload your medical bill to get a plain-English explanation, identify potential overcharges,
            and generate a dispute letter.
            """
        )

        with gr.Tabs():
            with gr.Tab("Analyze Bill"):
                with gr.Row():
                    with gr.Column(scale=1):
                        file_input = gr.File(
                            label="Upload Medical Bill",
                            file_types=[".pdf", ".png", ".jpg", ".jpeg", ".webp", ".tiff", ".bmp", ".txt"],
                        )
                        zip_input = gr.Textbox(
                            label="Patient ZIP Code (optional)",
                            placeholder="e.g., 27708",
                        )
                        analyze_btn = gr.Button("Analyze Bill", variant="primary")

                    with gr.Column(scale=2):
                        explanation_output = gr.Markdown(label="Bill Explanation")

                chatbot = gr.Chatbot(label="Ask questions about your bill")
                msg_input = gr.Textbox(
                    placeholder="Ask a follow-up question (e.g., 'What is CPT 99213?' or 'Is the ER charge reasonable?')",
                    label="Your Question",
                )

                analyze_btn.click(
                    process_bill,
                    inputs=[file_input, zip_input],
                    outputs=[explanation_output, chatbot, analysis_state],
                )
                msg_input.submit(
                    respond,
                    inputs=[msg_input, chatbot, analysis_state],
                    outputs=[chatbot],
                ).then(lambda: "", outputs=[msg_input])

            with gr.Tab("Dispute Letter"):
                gr.Markdown("### Generate a Dispute Letter")
                gr.Markdown("Fill in your details below and we'll generate a formal dispute letter you can send.")

                with gr.Row():
                    name_input = gr.Textbox(label="Full Name")
                    account_input = gr.Textbox(label="Account / Patient ID")

                address_input = gr.Textbox(label="Mailing Address", lines=3)
                dispute_all = gr.Checkbox(label="Dispute all flagged charges", value=True)
                dispute_btn = gr.Button("Generate Dispute Letter", variant="primary")
                letter_output = gr.Markdown(label="Dispute Letter")

                dispute_btn.click(
                    generate_dispute,
                    inputs=[name_input, address_input, account_input, dispute_all, analysis_state],
                    outputs=[letter_output],
                )

            with gr.Tab("About"):
                gr.Markdown(
                    """
                    ### About This Tool

                    The Medical Billing Assistant helps patients understand and dispute their medical bills.

                    **How it works:**
                    1. Upload a PDF or image of your medical bill
                    2. Our system extracts the billing codes (CPT/HCPCS) and charges
                    3. ZIP code context is converted into a coarse regional feature for the trained risk model
                    4. Each charge is compared against Medicare reimbursement rates
                    5. Potential issues (overcharges, duplicates, high-acuity review signals) are flagged
                    6. A trained classifier predicts whether the bill is low, medium, or high risk
                    7. You get a plain-English explanation and can ask follow-up questions
                    8. If issues are found, you can generate a dispute letter

                    **Data sources:**
                    - CMS Medicare Physician Fee Schedule (public data from cms.gov)

                    **Disclaimer:** This tool is for educational purposes only and does not constitute
                    legal or financial advice. Always consult with a healthcare billing advocate or
                    attorney for serious billing disputes.

                    Built for CS 372: Introduction to Applied Machine Learning, Duke University, Spring 2026.
                    """
                )

    return app


if __name__ == "__main__":
    logger.info("Starting Medical Billing Assistant application")
    app = build_app()
    share = os.getenv("GRADIO_SHARE", "false").lower() == "true"
    server_name = os.getenv("GRADIO_SERVER_NAME", "127.0.0.1")
    server_port = int(os.getenv("GRADIO_SERVER_PORT", "7860"))
    app.launch(share=share, server_name=server_name, server_port=server_port, theme=gr.themes.Soft())
