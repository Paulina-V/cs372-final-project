# Attribution

## Project Author

Paulina Vargas.

## Data Sources

- `data/cms_fee_schedule.csv`: Project-formatted CMS-style fee schedule with CPT/HCPCS codes, descriptions, and Medicare benchmark fees derived from CMS Physician Fee Schedule, Clinical Laboratory Fee Schedule, and anesthesia reference data.
- `data/sample_bill.txt`: Synthetic medical bill used for demonstration and testing.

## Libraries And Tools

- Gradio for the web application interface.
- OpenAI Python SDK for Duke GPT / OpenAI-compatible language model calls.
- ChromaDB plus a deterministic local embedding function for retrieval over fee-schedule records.
- scikit-learn and joblib for the trained bill-risk classifier and model persistence.
- pypdf, Pillow, pytesseract, and pdf2image for document extraction and OCR support.
- pandas and numpy for data handling.
- python-dotenv for local environment configuration.

## AI Development Tool Use

AI coding assistance was used to accelerate implementation, debugging, and documentation. AI-generated or AI-assisted work included:

- Reviewing the project structure and identifying reliability gaps.
- Refactoring the LLM integration from Anthropic-specific calls to OpenAI-compatible calls.
- Adding setup hardening, including `.txt` bill support and safer error handling for missing API keys, missing Chroma indexes, and model-call failures.
- Creating and refining the deterministic evaluation harness in `eval/evaluate_rules.py`.
- Hardening LLM extraction with schema normalization, fallback metadata, and clearer prompts.
- Expanding the CMS-style fee-schedule conversion script and documenting data limitations.
- Adding the weakly supervised synthetic-data generator, feature engineering, trained risk classifier, and model evaluation scripts.
- Adding the Hugging Face Spaces deployment helper and compatibility fixes for hosted Gradio versions.
- Drafting documentation files including `README.md`, `SETUP.md`, and this attribution document.

Human judgment was used to choose the project goal, decide which rubric items to target, validate the design, review generated code, and keep the implementation aligned with the medical billing assistant use case.

## Known AI-Assisted Files

- `src/llm.py`
- `src/code_extract.py`
- `src/pipeline.py`
- `src/analysis.py`
- `src/features.py`
- `src/risk_model.py`
- `src/pdf_extract.py`
- `app.py`
- `eval/evaluate_rules.py`
- `eval/evaluate_risk_model.py`
- `eval/rag_comparison.py`
- `eval/prompt_comparison.py`
- `eval/error_analysis.py`
- `scripts/download_cms_data.py`
- `scripts/train_risk_model.py`
- `deploy_to_hf.py`
- `README.md`
- `SETUP.md`
- `ATTRIBUTION.md`

## Secret Handling

Real API keys should only be stored locally in `.env`, which is ignored by git. `.env.example` is a template and must not contain real credentials.
