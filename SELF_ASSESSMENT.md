# Self-Assessment

Project: Medical Billing Assistant  
Author: Paulina Vvedenskaya  
Submission note: The Machine Learning category allows at most 15 selected items. I am selecting exactly 15 items below. The theoretical total of these selected ML items is 84 points, but I understand the Machine Learning category is capped at 73 points.

## Machine Learning Items To Claim

1. **Modular code design with reusable functions/classes (3 pts)**

   Evidence: `src/pipeline.py`, `src/pdf_extract.py`, `src/code_extract.py`, `src/analysis.py`, `src/rag.py`, `src/features.py`, `src/risk_model.py`, `src/explain.py`, `src/dispute.py`, and `app.py`.

   Justification: The project is organized as a multi-module system rather than a monolithic script. Document extraction, LLM extraction, retrieval, rule-based analysis, feature engineering, risk prediction, explanation, chat, dispute-letter generation, and UI concerns are separated into focused files with reusable functions. `src/pipeline.py` acts as the orchestration layer that connects these modules.

2. **Implemented proper train/validation/test split with documented split ratios (3 pts)**

   Evidence: `scripts/train_risk_model.py` and `eval/risk_model_results.json`.

   Justification: The trained bill-risk classifier uses a documented stratified 75/25 train/test split. Hyperparameter search uses 5-fold cross-validation on the training data, separating model selection from held-out test evaluation. The result file records 1,350 training examples and 450 test examples.

3. **Tracked and visualized training curves (3 pts)**

   Evidence: `scripts/train_risk_model.py`, `eval/plots/learning_curve.png`, `eval/plots/training_curve_gb.png`, and `eval/risk_model_results.json`.

   Justification: The training script generates learning curves and a gradient boosting training curve to show how model performance changes with data size and training progress. These plots help check whether performance is limited by data, overfitting, or model capacity.

4. **Created baseline model for comparison (3 pts)**

   Evidence: `scripts/train_risk_model.py` and `eval/risk_model_results.json`.

   Justification: The project compares the trained models against a `DummyClassifier` majority-class baseline. This gives the risk-model results a meaningful reference point; the majority baseline macro F1 is 0.172, while the tuned random forest macro F1 is about 0.995.

5. **Applied regularization techniques to prevent overfitting (5 pts)**

   Evidence: `scripts/train_risk_model.py`, `eval/risk_model_results.json`, and `eval/plots/regularization_comparison.png`.

   Justification: The training pipeline compares regularized and less-regularized model settings, including logistic regression with and without L2 penalty, gradient boosting with and without early stopping, and random forest models with and without depth limits. The regularization comparison reports train/test behavior and helps justify model choices.

6. **Conducted systematic hyperparameter tuning (5 pts)**

   Evidence: `scripts/train_risk_model.py` and `eval/risk_model_results.json`.

   Justification: The project uses `RandomizedSearchCV` over 30 random-forest configurations with 5-fold cross-validation. The tuned random forest is selected based on validation performance and then evaluated on the held-out test set. The best CV macro F1 is recorded as 0.9985.

7. **Applied feature engineering (5 pts)**

   Evidence: `src/features.py`, `src/analysis.py`, and `eval/risk_model_results.json`.

   Justification: The risk model does not train directly on raw text or images. It uses engineered billing features such as total billed amount, total benchmark amount, bill-to-benchmark ratio, maximum and mean line-item ratios, counts of overcharge/duplicate/high-acuity/missing-rate flags, unmatched-code fraction, CPT/HCPCS category counts, and a coarse ZIP-region multiplier. These features are interpretable and directly connected to the billing-review task.

8. **Applied prompt engineering with evaluation of at least three prompt designs (3 pts)**

   Evidence: `eval/prompt_comparison.py` and `eval/prompt_comparison_results.json`.

   Justification: The project evaluates three extraction prompt designs: minimal, structured, and chain-of-thought. The prompt comparison measures code recall, code precision, item-count accuracy, total accuracy, success rate, and latency on synthetic prompt-evaluation bills. This makes prompt selection empirical rather than purely subjective.

9. **Used sentence embeddings for semantic similarity or retrieval (5 pts)**

   Evidence: `src/rag.py`, `app.py`, `eval/rag_comparison.py`, and `eval/rag_comparison_results.json`.

   Justification: The retrieval layer supports sentence-transformer embeddings in addition to deterministic hash embeddings. The deployed Gradio app exposes a RAG embedding-mode selector so users can choose hash or semantic retrieval. The evaluation compares both modes and shows that sentence-transformer embeddings improve description recall@5 from 0.167 to 0.250 while exact CPT/HCPCS lookup accuracy remains 1.000 for both modes due to metadata filtering.

10. **Made API calls to a state-of-the-art model with meaningful integration (5 pts)**

    Evidence: `src/llm.py`, `src/code_extract.py`, `src/explain.py`, `src/dispute.py`, and `app.py`.

    Justification: The project uses an OpenAI-compatible LLM for several meaningful workflow steps: extracting structured billing line items from messy bill text, explaining the analysis in plain English, answering follow-up questions, and drafting dispute letters. The model calls are integrated into the full application rather than being isolated demos.

11. **Built multi-turn conversation system with context management (7 pts)**

    Evidence: `app.py` and `src/explain.py`.

    Justification: The app stores chat history and passes the current structured bill analysis into follow-up chat. This allows the assistant to answer user questions in context, such as identifying the most concerning charge or explaining a CPT/HCPCS code from the uploaded bill, instead of responding as a stateless chatbot.

12. **Developed a retrieval-augmented generation system with custom retrieval pipeline (10 pts)**

    Evidence: `src/rag.py`, `src/analysis.py`, `app.py`, `eval/rag_comparison.py`, and `data/cms_fee_schedule.csv`.

    Justification: The project builds a custom ChromaDB index over a CMS-style fee schedule and uses CPT/HCPCS metadata filtering for exact benchmark lookup. The retrieved benchmark rates are used by deterministic billing checks, the trained risk model, and the LLM explanation/dispute workflow. This is a custom retrieval pipeline tied to the project domain rather than a generic document-chat wrapper.

13. **Built multi-stage ML pipeline (7 pts)**

    Evidence: `src/pipeline.py`, `app.py`, and the modules under `src/`.

    Justification: The end-to-end workflow connects multiple stages: file/text extraction, LLM structured extraction, RAG benchmark lookup, deterministic anomaly checks, engineered features, trained risk classification, LLM explanation, contextual chat, and dispute-letter generation. Each stage transforms the output of earlier stages into inputs for later stages.

14. **Deployed model as functional web application with user interface (10 pts)**

    Evidence: public Hugging Face Space linked in `README.md`, plus `app.py`.

    Justification: The project is deployed as a public Gradio web app with tabs for bill analysis, follow-up chat, and dispute-letter generation. The UI supports uploads, ZIP-code input, RAG embedding-mode selection, loading indicators, pipeline status, risk output, chat, and a dispute-letter form.

15. **Implemented production-grade deployment considerations (10 pts)**

    Evidence: `app.py`, `deploy_to_hf.py`, `packages.txt`, `requirements.txt`, and `SETUP.md`.

    Justification: The app includes structured logging, optional file logging for hosted environments, in-memory rate limiting, visible loading indicators during long-running model calls, user-facing error messages, model-call fallbacks, deployment packaging, and system dependencies for OCR. The Hugging Face deployment script skips local-only files and secrets while preserving runtime dependencies.

## Following Directions

- `README.md` includes the required project title, short project description, What It Does section, Quick Start section, Video Links section, Evaluation section, Limitations section, and Individual Contributions section.
- `SETUP.md` provides step-by-step installation, API configuration, OCR dependency, local run, evaluation, and deployment instructions.
- `ATTRIBUTION.md` documents data sources, libraries, system tools, AI development assistance, known AI-assisted files, and secret-handling expectations.
- `requirements.txt` lists Python dependencies, and `packages.txt` lists Hugging Face system dependencies for OCR.
- Demo and technical walkthrough video links are included in `README.md`.
- The project is submitted as a solo project, and the README's Individual Contributions section states the work was completed by Paulina Vvedenskaya.

## Project Cohesion And Motivation

The project has a single coherent goal: helping patients and advocates understand confusing medical bills and prepare careful dispute requests when a bill has review signals. Every major component supports that goal. The extraction layer turns bills into structured line items, the retrieval layer connects CPT/HCPCS codes to public benchmark rates, deterministic checks identify concrete review signals, the trained risk model summarizes bill-level concern, and the LLM components make the result understandable through explanations, chat, and dispute letters.

The evaluation strategy also matches the project goal. Rule evaluation measures whether overcharges, duplicates, missing rates, and high-acuity review signals are detected correctly. Risk-model evaluation measures whether engineered billing features can classify synthetic bills into low, medium, and high risk. RAG evaluation measures exact CPT/HCPCS lookup and semantic retrieval behavior. Prompt comparison measures whether the LLM extraction step works reliably across prompt designs.

The project is framed with appropriate limitations. It uses synthetic bills for evaluation, Medicare-style rates as public benchmarks rather than definitive fair prices, and weak supervision for the risk model. The app avoids claiming fraud or giving legal advice and presents flagged items as review signals.

## Overall Reflection

The strongest parts of the project are the integration of multiple ML techniques into one patient-facing workflow and the emphasis on evaluation. The project combines LLM extraction, custom RAG, deterministic checks, feature engineering, supervised risk classification, prompt comparison, error analysis, and deployment. The final application is not just a notebook or model artifact; it is a working web app with a live demo, user-facing explanations, chat, and dispute-letter generation.

The main limitations are that the evaluation bills are synthetic and the risk labels are weakly supervised rather than validated against real adjudicated billing disputes. If I extended this project, I would add more diverse bill formats, collaborate with billing experts to validate labels and thresholds, and expand the benchmark data to handle more modifiers, payer-specific rules, and patient-specific insurance context.
