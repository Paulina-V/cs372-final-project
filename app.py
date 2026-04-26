"""
Medical Billing Assistant — Gradio Web App
Upload a medical bill, get a plain-English explanation, and generate a dispute letter.
"""

import os

import gradio as gr
from src.pipeline import analyze_bill
from src.explain import chat
from src.dispute import generate_dispute_letter

# Global state for current analysis
current_analysis = {}


def process_bill(file):
    """Handle bill upload and run the full pipeline."""
    global current_analysis

    if file is None:
        return "Please upload a medical bill (PDF, image, or text file).", []

    try:
        result = analyze_bill(file.name)
    except Exception as exc:
        return f"Unexpected error while analyzing bill: {exc}", []

    if result.get("error"):
        return f"Error: {result['error']}", []

    current_analysis = result
    explanation = result.get("explanation", "No explanation generated.")

    # Build initial chat history
    history = [
        {"role": "assistant", "content": explanation}
    ]

    return explanation, history


def respond(message, history):
    """Handle follow-up questions in the chat."""
    global current_analysis

    if not message or not message.strip():
        return history

    if not current_analysis:
        return history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": "Please upload a bill first using the upload tab."}
        ]

    # Convert history to API format
    api_messages = []
    for msg in history:
        api_messages.append({"role": msg["role"], "content": msg["content"]})
    api_messages.append({"role": "user", "content": message})

    try:
        response = chat(api_messages, current_analysis)
    except Exception as exc:
        response = f"I could not reach the chat model right now. Error: {exc}"

    history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": response}
    ]

    return history


def generate_dispute(name, address, account_id, dispute_all):
    """Generate a dispute letter from collected info."""
    global current_analysis

    if not current_analysis:
        return "Please upload and analyze a bill first."

    if not current_analysis.get("flags"):
        return "No issues were flagged in your bill; there's nothing to dispute."

    patient_info = {
        "patient_name": name or current_analysis.get("patient_name", ""),
        "address": address,
        "account_id": account_id,
        "provider_name": current_analysis.get("provider_name", ""),
        "dispute_scope": "all flagged charges" if dispute_all else "selected charges",
    }

    try:
        letter = generate_dispute_letter(current_analysis, patient_info)
    except Exception as exc:
        return f"I could not generate the dispute letter because the model call failed: {exc}"

    return letter


def build_app():
    with gr.Blocks(
        title="Medical Billing Assistant",
    ) as app:
        gr.Markdown(
            """
            # Medical Billing Assistant
            Upload your medical bill to get a plain-English explanation, identify potential overcharges,
            and generate a dispute letter.
            """
        )

        with gr.Tabs():
            # Tab 1: Bill Analysis
            with gr.Tab("Analyze Bill"):
                with gr.Row():
                    with gr.Column(scale=1):
                        file_input = gr.File(
                            label="Upload Medical Bill",
                            file_types=[".pdf", ".png", ".jpg", ".jpeg", ".txt"],
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
                    inputs=[file_input],
                    outputs=[explanation_output, chatbot],
                )
                msg_input.submit(
                    respond,
                    inputs=[msg_input, chatbot],
                    outputs=[chatbot],
                ).then(lambda: "", outputs=[msg_input])

            # Tab 2: Dispute Letter
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
                    inputs=[name_input, address_input, account_input, dispute_all],
                    outputs=[letter_output],
                )

            # Tab 3: About
            with gr.Tab("About"):
                gr.Markdown(
                    """
                    ### About This Tool

                    The Medical Billing Assistant helps patients understand and dispute their medical bills.

                    **How it works:**
                    1. Upload a PDF or image of your medical bill
                    2. Our system extracts the billing codes (CPT/HCPCS) and charges
                    3. Each charge is compared against Medicare reimbursement rates
                    4. Potential issues (overcharges, duplicates, upcoding) are flagged
                    5. You get a plain-English explanation and can ask follow-up questions
                    6. If issues are found, you can generate a dispute letter

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
    app = build_app()
    share = os.getenv("GRADIO_SHARE", "false").lower() == "true"
    app.launch(share=share, theme=gr.themes.Soft())
