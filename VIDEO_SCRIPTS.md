# Video Scripts

## Demo Video Script

Audience: non-specialist. Show the Hugging Face app, not the code.

### 0:00-0:20 Opening

Hi, my project is Medical Billing Assistant. It helps patients understand confusing medical bills by extracting line items, checking billing codes against public Medicare benchmark rates, flagging possible issues, and drafting a dispute letter when something looks wrong.

Medical bills are hard to interpret because they mix medical terminology, billing codes, insurance language, and large dollar amounts. This tool is designed as decision support, not legal or medical advice.

### 0:20-0:45 Open The Hosted App

I deployed the app publicly on Hugging Face Spaces. The interface has three main sections: analyzing a bill, asking follow-up questions, and generating a dispute letter.

I am going to upload the synthetic sample bill included with the project and enter a ZIP code for regional context.

### 0:45-1:30 Analyze A Bill

After clicking Analyze Bill, the app extracts text from the file, asks an OpenAI-compatible language model to structure the line items, compares CPT and HCPCS codes against the local CMS-style fee schedule, runs rule-based checks, and applies a trained risk classifier.

Here in the pipeline status, you can see the extraction method, number of line items, fee schedule matches, unmatched codes, number of flags, ZIP context, and the trained risk-model label.

For this sample bill, the tool identifies multiple review signals. It explains those signals in plain language, including which charges are much higher than Medicare benchmarks and which code appears more than once.

### 1:30-2:00 Chat Follow-Up

Now I will ask a follow-up question: "What charge looks most concerning?"

The assistant answers using the current bill analysis rather than starting from scratch. This lets a patient ask practical questions after the initial review.

### 2:00-2:40 Generate A Dispute Letter

Next, I will switch to the dispute letter tab. I enter the patient name, account number, and mailing address, then generate a letter for the flagged charges.

The generated letter is formal and careful. It references the account, the flagged billing issues, and the Medicare rates as public benchmarks, without claiming fraud or giving legal advice.

### 2:40-3:00 Closing

The goal is to make medical bill review more accessible: patients can see which line items deserve attention, ask follow-up questions, and leave with a concrete next step. The tool uses synthetic evaluation bills and public benchmark data, so it should be treated as educational decision support.

## Technical Walkthrough Script

Audience: technical grader. Show the repo, selected code, evaluation outputs, and deployed app link.

### 0:00-0:30 Project Overview

This project is an applied machine learning system for medical bill understanding and dispute support. The main entry point is `app.py`, which builds a Gradio web app with bill analysis, follow-up chat, and dispute-letter generation.

The end-to-end pipeline is in `src/pipeline.py`. It connects document extraction, LLM extraction, deterministic billing checks, RAG-style benchmark lookup, a trained risk classifier, LLM explanation, chat, and dispute generation.

### 0:30-1:20 Data And Extraction

Text ingestion lives in `src/pdf_extract.py`. It supports plain text bills, text PDFs, image OCR through Tesseract, and scanned PDF OCR through `pdf2image`.

Structured bill extraction is in `src/code_extract.py`. The LLM prompt asks for patient name, provider name, total billed, and line items with CPT or HCPCS code, ICD codes, description, amount, and date. The code normalizes the model output and falls back to a regex parser when the model call fails or the response is not valid JSON.

This is important because the rest of the pipeline depends on reliable structured line items.

### 1:20-2:20 RAG And Deterministic Billing Checks

The benchmark retrieval layer is `src/rag.py`. It builds a ChromaDB index over `data/cms_fee_schedule.csv`, which contains 9,926 CMS-style fee schedule rows. The default embedding is a deterministic hash embedding so the app can run reliably offline. The file also supports sentence-transformer embeddings for semantic comparison.

`src/analysis.py` performs the deterministic checks. It compares each billed amount against the Medicare benchmark, flags overcharges above the threshold, detects duplicate CPT or HCPCS codes on the same date, flags high-acuity codes for documentation review, and reports missing benchmark rates.

The retrieval comparison in `eval/rag_comparison.py` shows exact CPT lookup accuracy of 1.000 for both hash and semantic modes. Semantic embeddings improve description recall@5 from 0.167 to 0.250, but the hash mode is faster and more deployment-friendly.

### 2:20-3:30 Risk Model And Feature Engineering

The trained bill-risk classifier uses engineered features from `src/features.py`. Features include total billed, total benchmark, bill-to-benchmark ratio, max and mean line ratios, counts of different flags, unmatched-code fraction, procedure-category counts, and a coarse ZIP-region multiplier.

Training is in `scripts/train_risk_model.py`. It generates weakly supervised synthetic bills grounded in CMS-style rates, labels them using deterministic signals, and compares multiple models. The dataset uses a stratified 75/25 train/test split. The script includes a majority baseline, logistic regression, tuned random forest, and gradient boosting.

The script also runs RandomizedSearchCV across 30 random-forest configurations with 5-fold cross-validation, computes learning curves, compares regularization settings, and saves plots in `eval/plots/`.

The current best model is a tuned random forest with test macro F1 around 0.995, compared to a majority baseline macro F1 of 0.172.

### 3:30-4:20 Evaluation

The API-free deterministic evaluation is `eval/evaluate_rules.py`. It runs ten synthetic bills through the production `run_all_checks()` path and reports 10 out of 10 cases passing, with precision, recall, and F1 all equal to 1.000.

`eval/error_analysis.py` analyzes the risk classifier. It found 2 misclassifications out of 450 test examples, both near-threshold high-acuity cases predicted as low risk instead of medium risk. It also generates a confusion matrix and feature-distribution plots.

`eval/prompt_comparison.py` evaluates three extraction prompts: minimal, structured, and chain-of-thought. All three achieved 1.000 code recall, item-count accuracy, total accuracy, and success rate on the prompt-evaluation bills.

### 4:20-5:10 LLM Integration And Chat

`src/llm.py` wraps the OpenAI-compatible API client. `src/explain.py` turns the structured analysis into plain-English explanations and keeps the bill context available for follow-up chat. `src/dispute.py` generates a careful dispute letter using the analysis and patient details.

The app is multi-turn because `app.py` stores chat history and passes it back with the structured bill analysis. The assistant is instructed to be clear, empathetic, and careful not to claim fraud or give legal advice.

### 5:10-6:00 Deployment And Production Considerations

The app is deployed on Hugging Face Spaces. `deploy_to_hf.py` stages only the runtime files needed by the hosted app, skips local secrets and bulky supplemental folders, and keeps the Space package lightweight.

Production considerations are implemented in `app.py`: structured logging, optional file logging, in-memory rate limiting, clear user-facing errors, and graceful fallback messages if model calls fail. The app also reports pipeline status so users and graders can see what happened during extraction, matching, risk prediction, and explanation.

### 6:00-6:30 Closing

Overall, the project combines document extraction, language-model structured extraction, custom retrieval over public benchmark data, deterministic anomaly checks, a trained risk classifier, prompt evaluation, error analysis, and a deployed user interface. All components support the same goal: helping patients review and dispute confusing medical bills more effectively.
