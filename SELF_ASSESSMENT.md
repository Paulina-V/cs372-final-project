# Self-Assessment Draft

Use this as the basis for the Gradescope self-assessment. Select at most 15 Machine Learning items.

## Machine Learning Items To Claim

1. **Modular code design with reusable functions/classes (3 pts)**  
   Evidence: `src/pipeline.py`, `src/pdf_extract.py`, `src/code_extract.py`, `src/analysis.py`, `src/rag.py`, `src/features.py`, `src/risk_model.py`, `src/explain.py`, `src/dispute.py`.

2. **Implemented proper train/validation/test split with documented split ratios (3 pts)**  
   Evidence: `scripts/train_risk_model.py` uses a stratified 75/25 train/test split and cross-validation for tuning; results in `eval/risk_model_results.json`.

3. **Tracked and visualized training curves (3 pts)**  
   Evidence: `scripts/train_risk_model.py`; `eval/plots/learning_curve.png` and `eval/plots/training_curve_gb.png`.

4. **Created baseline model for comparison (3 pts)**  
   Evidence: `DummyClassifier` majority baseline in `scripts/train_risk_model.py`; baseline metrics in `eval/risk_model_results.json`.

5. **Applied regularization techniques to prevent overfitting (5 pts)**  
   Evidence: regularization comparison in `scripts/train_risk_model.py`; metrics and plot in `eval/risk_model_results.json` and `eval/plots/regularization_comparison.png`.

6. **Conducted systematic hyperparameter tuning (5 pts)**  
   Evidence: `RandomizedSearchCV` over 30 random-forest configurations with 5-fold CV in `scripts/train_risk_model.py`; results in `eval/risk_model_results.json`.

7. **Applied feature engineering (5 pts)**  
   Evidence: `src/features.py` creates ratio, flag-count, unmatched-code, CPT category, and ZIP-region features.

8. **Applied prompt engineering with evaluation of at least three prompt designs (3 pts)**  
   Evidence: `eval/prompt_comparison.py` compares minimal, structured, and chain-of-thought prompts; results in `eval/prompt_comparison_results.json`.

9. **Used sentence embeddings for semantic similarity or retrieval (5 pts)**  
   Evidence: `src/rag.py` supports sentence-transformer embeddings; `app.py` exposes hash vs. semantic retrieval in the deployed UI; comparison in `eval/rag_comparison.py` and `eval/rag_comparison_results.json`.

10. **Made API calls to a state-of-the-art model with meaningful integration (5 pts)**  
    Evidence: `src/llm.py`, `src/code_extract.py`, `src/explain.py`, and `src/dispute.py` use an OpenAI-compatible LLM for extraction, explanation, chat, and dispute drafting.

11. **Built multi-turn conversation system with context management (7 pts)**  
    Evidence: `app.py` stores chat history and `src/explain.py` injects bill-analysis context into follow-up chat.

12. **Developed a retrieval-augmented generation system with custom retrieval pipeline (10 pts)**  
    Evidence: `src/rag.py`, `src/analysis.py`, and `eval/rag_comparison.py`; custom CMS index, metadata-filtered code lookup, deployed embedding-mode selector, and embedding comparison.

13. **Built multi-stage ML pipeline (7 pts)**  
    Evidence: `src/pipeline.py` connects document extraction, LLM parsing, RAG benchmark lookup, deterministic checks, trained risk model, LLM explanation, chat, and dispute generation.

14. **Deployed model as functional web application with user interface (10 pts)**  
    Evidence: public Hugging Face Space linked in `README.md`; Gradio UI in `app.py`.

15. **Implemented production-grade deployment considerations (10 pts)**  
    Evidence: rate limiting, structured logging, hosted deployment support, system dependencies in `packages.txt`, and user-facing error handling in `app.py`; deployment helper in `deploy_to_hf.py`.

## Following Directions

- `SETUP.md` exists with installation, evaluation, and deployment steps.
- `ATTRIBUTION.md` exists with data, library, and AI-assistance attributions.
- `requirements.txt` is included.
- `README.md` includes What It Does, Quick Start, Hosted Demo, Evaluation, Video Links, Limitations, and Individual Contributions.
- Demo and technical walkthrough video links still need to be pasted into `README.md`.

## Project Cohesion

- The unified goal is patient-facing medical bill understanding and dispute support.
- The pipeline follows problem -> extraction -> benchmark comparison -> risk modeling -> explanation/dispute -> evaluation.
- Evaluation metrics directly measure the stated goals: flag precision/recall/F1, lookup parity, risk-model F1, prompt extraction quality, retrieval recall, latency, and error cases.
