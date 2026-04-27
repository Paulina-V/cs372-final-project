"""
Medical Billing Assistant — Gradio Web App
Upload a medical bill, get a plain-English explanation, and generate a dispute letter.

Production considerations: structured logging, in-memory rate limiting, and
graceful error handling for all user-facing operations.

AI-assisted portions of this file are documented in ATTRIBUTION.md.
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


def _logging_handlers() -> list[logging.Handler]:
    """Return logging handlers that are safe for local and hosted runs."""
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if os.getenv("ENABLE_FILE_LOGGING", "true").lower() == "true":
        try:
            handlers.append(logging.FileHandler("app.log", encoding="utf-8"))
        except OSError:
            # Hosted filesystems may be read-only; console logging is enough there.
            pass
    return handlers


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=_logging_handlers(),
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


def process_bill(file, zip_code, embedding_type):
    """Handle bill upload and run the full pipeline."""
    if file is None:
        return "Please upload a medical bill (PDF, image, or text file).", [], {}

    if not rate_limiter.check():
        logger.warning("Rate limit exceeded for bill analysis request")
        return "Rate limit exceeded. Please wait a moment before trying again.", [], {}

    logger.info(
        "Starting bill analysis for file: %s (ZIP: %s, embedding: %s)",
        file.name,
        zip_code or "none",
        embedding_type or "hash",
    )
    start = time.time()

    try:
        result = analyze_bill(file.name, zip_code=zip_code, embedding_type=embedding_type)
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
        f"- RAG embedding mode: `{result.get('embedding_type', 'hash')}`",
        f"- Line items analyzed: `{len(result.get('rated_items', []))}`",
        f"- Fee schedule matches: `{matched}`",
        f"- Unmatched codes: `{unmatched}`",
        f"- Flags found: `{result.get('num_flags', 0)}`",
    ]

    if result.get("embedding_warning"):
        status_lines.append(f"- Embedding warning: {result['embedding_warning']}")
    if result.get("extraction_warning"):
        status_lines.append(f"- Extraction warning: {result['extraction_warning']}")
    if result.get("zip_code"):
        status_lines.append(f"- ZIP code context: `{result['zip_code']}`")
    if result.get("risk_model"):
        status_lines.append(f"- {format_risk_summary(result['risk_model'])}")
    if result.get("risk_model_error"):
        status_lines.append(f"- Risk model warning: {result['risk_model_error']}")

    return "\n".join(status_lines) + "\n\n---\n\n" + explanation


def _chatbot(label: str):
    """Use message-style chat where supported while staying compatible with Gradio 6."""
    try:
        return gr.Chatbot(label=label, type="messages")
    except TypeError:
        return gr.Chatbot(label=label)


def _status_html(message: str) -> str:
    """Return a compact loading indicator for long-running model calls."""
    return f"""
    <style>
      @keyframes mba-spin {{
        0% {{ transform: rotate(0deg); }}
        100% {{ transform: rotate(360deg); }}
      }}
      .mba-status {{
        display: flex;
        align-items: center;
        gap: 0.6rem;
        padding: 0.75rem 0.9rem;
        margin: 0.5rem 0;
        border: 1px solid #dbeafe;
        border-radius: 0.75rem;
        background: #eff6ff;
        color: #1e3a8a;
        font-weight: 600;
      }}
      .mba-spinner {{
        width: 1rem;
        height: 1rem;
        border: 3px solid #bfdbfe;
        border-top-color: #2563eb;
        border-radius: 50%;
        animation: mba-spin 0.8s linear infinite;
      }}
    </style>
    <div class="mba-status">
      <span class="mba-spinner"></span>
      <span>{message}</span>
    </div>
    """


def show_analysis_status():
    """Show progress while the bill analysis pipeline is running."""
    return gr.update(
        value=_status_html("Analyzing your bill. This can take a moment while the model extracts codes and checks benchmarks..."),
        visible=True,
    )


def show_chat_status():
    """Show progress while the chat model is responding."""
    return gr.update(
        value=_status_html("Thinking through your question using the current bill analysis..."),
        visible=True,
    )


def show_dispute_status():
    """Show progress while the dispute letter is generated."""
    return gr.update(
        value=_status_html("Drafting your dispute letter..."),
        visible=True,
    )


def hide_status():
    """Hide a progress indicator after a long-running action finishes."""
    return gr.update(value="", visible=False)


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
                        embedding_input = gr.Radio(
                            choices=[
                                ("Hash embeddings (fast, reliable default)", "hash"),
                                ("Semantic embeddings (sentence-transformer)", "semantic"),
                            ],
                            value="hash",
                            label="RAG Embedding Mode",
                            info="Hash is fastest for the live demo. Semantic can improve description search but may take longer on first use.",
                        )
                        analyze_btn = gr.Button("Analyze Bill", variant="primary")

                    with gr.Column(scale=2):
                        analysis_status = gr.HTML(visible=False)
                        explanation_output = gr.Markdown(label="Bill Explanation")

                chatbot = _chatbot(label="Ask questions about your bill")
                chat_status = gr.HTML(visible=False)
                msg_input = gr.Textbox(
                    placeholder="Ask a follow-up question (e.g., 'What is CPT 99213?' or 'Is the ER charge reasonable?')",
                    label="Your Question",
                )

                analyze_btn.click(
                    show_analysis_status,
                    outputs=[analysis_status],
                    queue=False,
                ).then(
                    process_bill,
                    inputs=[file_input, zip_input, embedding_input],
                    outputs=[explanation_output, chatbot, analysis_state],
                ).then(
                    hide_status,
                    outputs=[analysis_status],
                    queue=False,
                )
                msg_input.submit(
                    show_chat_status,
                    outputs=[chat_status],
                    queue=False,
                ).then(
                    respond,
                    inputs=[msg_input, chatbot, analysis_state],
                    outputs=[chatbot],
                ).then(
                    hide_status,
                    outputs=[chat_status],
                    queue=False,
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
                dispute_status = gr.HTML(visible=False)
                letter_output = gr.Markdown(label="Dispute Letter")

                dispute_btn.click(
                    show_dispute_status,
                    outputs=[dispute_status],
                    queue=False,
                ).then(
                    generate_dispute,
                    inputs=[name_input, address_input, account_input, dispute_all, analysis_state],
                    outputs=[letter_output],
                ).then(
                    hide_status,
                    outputs=[dispute_status],
                    queue=False,
                )

            with gr.Tab("About"):
                gr.Markdown(
                    """
                    ### About This Tool

                    The Medical Billing Assistant helps patients understand and dispute their medical bills.

                    **How it works:**
                    1. Upload a PDF, image, or text version of your medical bill
                    2. Our system extracts the billing codes (CPT/HCPCS) and charges
                    3. Choose hash embeddings for speed or semantic embeddings for sentence-transformer retrieval
                    4. ZIP code context is converted into a coarse regional feature for the trained risk model
                    5. Each charge is compared against Medicare reimbursement rates
                    6. Potential issues (overcharges, duplicates, high-acuity review signals) are flagged
                    7. A trained classifier predicts whether the bill is low, medium, or high risk
                    8. You get a plain-English explanation and can ask follow-up questions
                    9. If issues are found, you can generate a dispute letter

                    **Data sources:**
                    - CMS-style Medicare Physician Fee Schedule, Clinical Laboratory Fee Schedule,
                      and anesthesia reference rates prepared for this educational project

                    **Disclaimer:** This tool is for educational purposes only and does not constitute
                    legal or financial advice. Always consult with a healthcare billing advocate or
                    attorney for serious billing disputes.

                    Built for CS 372: Introduction to Applied Machine Learning, Duke University, Spring 2026.
                    """
                )

    return app


def launch_app(app: gr.Blocks, share: bool, server_name: str, server_port: int) -> None:
    """Launch with a soft theme when the installed Gradio version supports it."""
    try:
        app.launch(
            share=share,
            server_name=server_name,
            server_port=server_port,
            theme=gr.themes.Soft(),
        )
    except TypeError as exc:
        if "theme" not in str(exc):
            raise
        app.launch(share=share, server_name=server_name, server_port=server_port)


if __name__ == "__main__":
    logger.info("Starting Medical Billing Assistant application")
    app = build_app()
    share = os.getenv("GRADIO_SHARE", "false").lower() == "true"
    server_name = os.getenv("GRADIO_SERVER_NAME", "127.0.0.1")
    server_port = int(os.getenv("GRADIO_SERVER_PORT", "7860"))
    launch_app(app, share=share, server_name=server_name, server_port=server_port)
