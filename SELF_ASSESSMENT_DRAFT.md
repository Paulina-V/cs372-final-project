# Self-Assessment Draft

This draft lists strong rubric items to consider claiming. Before submission, copy the final version into the required Gradescope/self-assessment format and include no more than 15 Machine Learning selections.

## Machine Learning Rubric Items

1. Modular code design with reusable functions and classes rather than monolithic scripts (3 pts)  
   Evidence: `app.py`, `src/pipeline.py`, `src/pdf_extract.py`, `src/code_extract.py`, `src/rag.py`, `src/analysis.py`, `src/explain.py`, `src/dispute.py`.

2. Applied basic preprocessing appropriate to the modality (3 pts)  
   Evidence: `src/pdf_extract.py` extracts text from PDFs, images, and text files before downstream processing.

3. Applied feature engineering through local embeddings and structured billing fields (5 pts)  
   Evidence: `src/rag.py` embeds CPT/HCPCS fee-schedule records with a deterministic local vectorizer; `src/code_extract.py` converts unstructured bill text into validated structured line items; `src/features.py` creates numeric bill-risk features.

4. Other substantial ML contribution: custom retrieval pipeline over medical billing data (proposed 5 pts)  
   Evidence: `scripts/download_cms_data.py` constructs a 9,926-row CMS-style benchmark; `src/rag.py` builds local vector embeddings, stores fee-schedule records in ChromaDB, and supports exact CPT/HCPCS lookup plus similarity search without external model downloads.

5. Made API calls to a state-of-the-art model with meaningful integration beyond a single isolated call (5 pts)  
   Evidence: `src/code_extract.py`, `src/explain.py`, and `src/dispute.py` use Duke GPT/OpenAI-compatible calls for extraction, explanation/chat, and dispute-letter generation.

6. Built multi-turn conversation system with context management and history tracking (7 pts)  
   Evidence: `app.py` and `src/explain.py` pass the current bill analysis plus message history into follow-up chat.

7. Used retrieval-augmented generation / retrieval integration in the project (5 pts)  
   Evidence: `src/rag.py` builds and queries a ChromaDB index over Medicare fee-schedule data; `src/analysis.py` uses retrieved rates in anomaly checks.

8. Built multi-stage ML pipeline connecting outputs of one model/component to inputs of another (7 pts)  
   Evidence: `src/pipeline.py` chains extraction, structured LLM parsing, retrieval/rate comparison, anomaly checks, trained risk classification, explanation generation, chat, and dispute generation.

9. Deployed model as functional web application with user interface (10 pts)  
   Evidence: `app.py` implements a Gradio app with upload, ZIP input, risk scoring, explanation, chat, and dispute-letter flows.

10. Measured and reported inference time, throughput, or computational efficiency (3 pts)  
    Evidence: `eval/evaluate_rules.py` reports production-path deterministic-check latency in `eval/results.json`.

11. Used at least three distinct and appropriate evaluation metrics for the task (3 pts)  
    Evidence: `eval/evaluate_rules.py` reports precision, recall, and F1; `eval/risk_model_results.json` reports accuracy, macro F1, weighted F1, classification reports, and confusion matrices.

12. Conducted both qualitative and quantitative evaluation with thoughtful discussion (5 pts)  
    Evidence: `eval/evaluate_rules.py`, `eval/results.json`, `eval/risk_model_results.json`, index parity checks, ten synthetic rule-evaluation cases, 1,800 synthetic risk-model examples, baseline comparison, and the README evaluation/limitations sections.

13. Created baseline model for comparison (3 pts)  
    Evidence: `scripts/train_risk_model.py` compares a majority-class baseline against logistic regression and random forest classifiers in `eval/risk_model_results.json`.

14. Compared multiple model architectures or approaches quantitatively with controlled experimental setup (7 pts)  
    Evidence: `scripts/train_risk_model.py` evaluates majority baseline, logistic regression, and random forest on the same train/test split.

15. In `ATTRIBUTION.md`, provided a substantive account of how AI development tools were used (3 pts)  
    Evidence: `ATTRIBUTION.md`.

Optional if you need to swap items: Completed project individually without a partner (10 pts)  
    Evidence: README Individual Contributions section, if this was submitted as a solo project.

## Following Directions

- `SETUP.md` exists with installation and run instructions.
- `ATTRIBUTION.md` exists with sources and AI-tool usage.
- `requirements.txt` exists and reflects the OpenAI-compatible implementation.
- `README.md` includes What It Does, Quick Start, Video Links, Evaluation, and Individual Contributions sections.
- Demo and technical walkthrough links still need to be added before submission.

## Cohesion And Motivation

The project has a single goal: help patients understand and dispute medical bills. The components work together toward that goal: document extraction produces bill text, the LLM structures and validates it, retrieval supplies Medicare benchmark rates, deterministic checks flag possible billing issues, a trained classifier estimates bill risk, and generation produces explanations, chat responses, and dispute letters.

## Remaining Items Before Submission

- Add final video links to `README.md`.
- Confirm `.env` is local only and no real API key appears in committed files.
- Run `python scripts/build_index.py`.
- Run `python app.py` and record the full demo flow.
