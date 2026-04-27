# Video Scripts

Rubric timing from the course page:

- Demo video: 3-5 minutes, non-specialist audience, no code required.
- Technical walkthrough: 5-10 minutes, technical audience, code structure and ML contributions explained.

## Demo Video Script

Target length: about 4 minutes.

Audience: non-specialist. Show the Hugging Face app and a sample bill workflow, not the code.

Goal: convince a nontechnical viewer that the app solves a real patient pain point and that the workflow is understandable, useful, and appropriately cautious.

### Recording Checklist

- Open the hosted app before recording.
- Have `data/sample_bill.txt` ready to upload.
- Use ZIP code `27708`.
- Leave `RAG Embedding Mode` on hash for the fastest first demo run.
- Ask: `What charge looks most concerning?`
- Use patient name `Jane Smith`, account `DUH-2026-44521`, and address `2301 Erwin Road, Durham, NC 27710`.
- Keep the cursor movements slow and deliberate.
- Do not show `.env`, API keys, terminal logs, or source code in the demo video.

### Screen Plan

| Time | Screen | What to show |
| --- | --- | --- |
| 0:00-0:30 | Hosted app landing page | App title and three-tab layout. |
| 0:30-0:55 | Analyze Bill tab | File upload control, ZIP input, and RAG embedding mode selector. |
| 0:55-2:05 | Loading state and analysis output | Briefly show the analyzing spinner, then pipeline status, embedding mode, risk label, flags, and explanation summary. |
| 2:05-2:40 | Chat box | Ask one patient-style question and show the response. |
| 2:40-3:25 | Dispute Letter tab | Fill patient fields and show the generated letter. |
| 3:25-4:00 | Final analysis output or About tab | Restate usefulness and limitations. |

### 0:00-0:30 Opening

Hi, my project is Medical Billing Assistant. It helps patients understand confusing medical bills by extracting line items, checking CPT and HCPCS billing codes against public Medicare benchmark rates, flagging possible issues, and drafting a dispute letter when something looks wrong.

Medical bills are difficult because they combine medical terminology, insurance language, billing codes, abbreviations, and large dollar amounts. A patient may not know which charges are normal, which ones are duplicates, or which codes deserve a closer review. This tool is educational decision support. It is not legal, financial, or medical advice.

### 0:30-0:55 Open The Hosted App

I deployed the project as a public Hugging Face Space. The interface has three main parts: analyzing a bill, asking follow-up questions, and generating a dispute letter.

For this demo, I am using the synthetic sample bill included with the project. I will also enter a ZIP code. The ZIP code is used only as a coarse regional feature for the trained risk model. It does not prove that a bill is right or wrong. I am leaving the embedding mode on hash because it is the fastest and most reliable option for a live demo; the app also supports semantic embeddings for the retrieval comparison.

### 0:55-2:05 Analyze A Bill

Now I will upload the sample bill and click Analyze Bill. While the model is working, the app shows a loading indicator so the user knows the system is extracting codes, checking benchmarks, and generating the explanation.

Behind the scenes, the app extracts text from the file, then asks an OpenAI-compatible language model to turn the bill into structured line items. Each line item includes a service description, billed amount, date, and CPT or HCPCS code. CPT and HCPCS codes are standardized billing codes for medical services, procedures, supplies, and some drugs.

The app compares those codes against a local CMS-style Medicare fee schedule, runs rule-based checks, and applies a trained bill-risk classifier. In the pipeline status, you can see the extraction method, selected embedding mode, number of line items, fee schedule matches, unmatched codes, number of flags, ZIP context, and the trained risk-model label.

For this sample bill, the tool identifies several review signals: charges that are much higher than the Medicare benchmark, a repeated emergency department code, and one unmatched code. These are not proof of fraud, but they are useful reasons to ask the billing department for clarification.

If the explanation is long, I would not read all of it aloud. I would point out that the app gives both a quick pipeline summary and a plain-English explanation, so a patient can understand the bill without learning the billing-code system first.

### 2:05-2:40 Chat Follow-Up

Now I will ask: "What charge looks most concerning?"

The assistant answers using the current bill analysis. That matters because a patient may not know what to ask first. The chat turns the analysis into practical next steps, like which charge to inspect, what a code means, or what information to compare against the Explanation of Benefits.

### 2:40-3:25 Generate A Dispute Letter

Next, I will switch to the dispute letter tab. I enter the patient name, account number, and mailing address, then generate a letter for the flagged charges.

The letter is formal and careful. It references the account, the specific billing issues, and Medicare rates as public benchmarks, without claiming fraud or giving legal advice. This reduces friction for the patient because they do not have to start from a blank page.

### 3:25-4:00 Closing

The goal is to make medical bill review more accessible. Patients can see which line items deserve attention, ask follow-up questions, and leave with a concrete next step.

The project uses public benchmark data and synthetic evaluation bills, so it should not be treated as a final billing decision. But it shows how retrieval, language models, deterministic checks, and a trained classifier can work together to help patients review confusing bills.

### If You Need To Cut Time

- Skip the detailed CPT/HCPCS explanation and simply say: "These are standardized medical billing codes."
- Do not read the full generated dispute letter; show the greeting, subject line, and one paragraph.
- Keep the chat section to one question and one sentence of commentary.

### If You Have Extra Time

- Briefly show the About tab disclaimer.
- Mention that the bill is synthetic and included for a reproducible demo.
- Point out that Medicare rates are benchmarks, not final fair prices.
- If the app is already warmed up, briefly point to the semantic embedding option without rerunning the whole demo.

## Technical Walkthrough Script

Target length: about 7-8 minutes.

Audience: technical grader or ML engineer. Show the repo, selected code, evaluation outputs, and deployed app link.

Goal: make it easy for a grader to map the codebase to the rubric: modular design, LLM integration, RAG, trained classifier, evaluation, deployment, and production considerations.

### Recording Checklist

- Start from `README.md` and the hosted demo link.
- Show `app.py`, `src/pipeline.py`, `src/code_extract.py`, `src/rag.py`, `src/analysis.py`, `src/features.py`, `scripts/train_risk_model.py`, and the `eval/` results.
- Do not spend too long scrolling. Explain the architecture and point to evidence.
- Mention limitations clearly.
- Keep `.env` closed.
- Keep the browser or terminal zoomed enough that filenames and headings are readable.
- If using the terminal, use commands that print summaries, not long tracebacks or full JSON files.

### Screen Plan

| Time | Screen | Evidence to show |
| --- | --- | --- |
| 0:00-0:45 | `README.md` system overview and hosted demo | Problem statement, architecture diagram, Space link. |
| 0:45-1:45 | `src/pipeline.py`, `src/pdf_extract.py`, `src/code_extract.py` | End-to-end flow, OCR/text ingestion, LLM extraction, normalization, regex fallback. |
| 1:45-2:55 | `src/rag.py`, `src/analysis.py`, `app.py`, `eval/rag_comparison_results.json` | Chroma index, app embedding selector, exact code lookup, overcharge/duplicate/missing-rate checks. |
| 2:55-4:10 | `src/features.py`, `scripts/train_risk_model.py`, `eval/risk_model_results.json` | Feature engineering, synthetic weak supervision, train/test split, baselines, tuned models. |
| 4:10-5:15 | `eval/evaluate_rules.py`, `eval/error_analysis_results.json`, `eval/prompt_comparison_results.json` | Rule metrics, error analysis, prompt comparison. |
| 5:15-6:20 | `src/explain.py`, `src/dispute.py`, `app.py` | Chat context, prompt design, dispute generation, safety wording. |
| 6:20-7:25 | `deploy_to_hf.py`, hosted Space | Deployment helper, runtime trimming, rate limiting, logging, remote smoke test. |
| 7:25-8:00 | `README.md` limitations and rubric map | Limitations and final summary. |

### Key Metrics To Say Out Loud

- Deterministic rules: 10/10 synthetic cases passed, precision/recall/F1 all 1.000.
- Risk model: tuned random forest test macro F1 about 0.995; majority baseline macro F1 0.172.
- Error analysis: 2 misclassified examples out of 450 test examples.
- RAG comparison: exact CPT/HCPCS lookup accuracy 1.000 for both hash and semantic modes; semantic description recall@5 improved from 0.167 to 0.250.
- Prompt comparison: three prompt designs evaluated; all reached 1.000 code recall, item-count accuracy, total accuracy, and success rate on the prompt-eval bills.

### 0:00-0:45 Project Overview

This project is a machine learning system for medical bill understanding and dispute support. The main entry point is `app.py`, which builds a Gradio web app. Gradio is a Python framework for quickly creating interactive machine learning interfaces.

The main workflow is in `src/pipeline.py`. It connects document extraction, LLM-based bill parsing, deterministic billing checks, RAG-style benchmark lookup, a trained risk classifier, LLM explanation, follow-up chat, and dispute-letter generation.

The key design choice is that this is not just one model call. The system combines deterministic logic, retrieval over public benchmark data, supervised learning, and language-model generation.

This matters for reliability. The LLM is useful for messy text extraction and natural-language explanation, but benchmark matching and billing flags are handled by deterministic code where exactness matters.

### 0:45-1:45 Data And Extraction

Text ingestion lives in `src/pdf_extract.py`. It supports plain text bills, text PDFs, image OCR through Tesseract, and scanned PDF OCR through `pdf2image`. Tesseract is an open-source OCR engine that converts images of text into machine-readable text.

Structured extraction is in `src/code_extract.py`. The LLM prompt asks for patient name, provider name, total billed, and line items with CPT or HCPCS code, ICD codes, description, amount, and date.

The code then normalizes the model output. It cleans codes, parses dollar amounts, skips invalid line items, and falls back to a regex parser when the model call fails or the response is not valid JSON. This matters because the rest of the system depends on reliable structured line items.

A good point to show here is that extraction is schema-driven: downstream modules expect `cpt_code`, `description`, `billed_amount`, and `date_of_service`. The normalizer is what keeps the rest of the pipeline from being tightly coupled to whatever shape the LLM happens to return.

### 1:45-2:55 RAG And Deterministic Billing Checks

The retrieval layer is in `src/rag.py`. It builds a ChromaDB index over `data/cms_fee_schedule.csv`, which contains 9,926 CMS-style fee schedule rows derived from CMS Physician Fee Schedule, Clinical Laboratory Fee Schedule, and anesthesia reference data.

The default embedding is a deterministic hash embedding. That makes the app lightweight and reliable because it does not need to download a large embedding model to run. The project also supports sentence-transformer embeddings for semantic comparison, and the deployed app exposes both modes through the RAG Embedding Mode selector.

The two embedding modes have different tradeoffs. Hash embeddings are faster and easier to deploy. Sentence-transformer embeddings are more semantically meaningful and improved description recall@5 from 0.167 to 0.250 in `eval/rag_comparison_results.json`. For exact CPT lookup, both modes reached 1.000 accuracy because the app uses metadata filtering on the billing code. In the live app, the Pipeline Status reports which mode was selected.

The deterministic checks are in `src/analysis.py`. They compare billed charges to Medicare benchmarks, flag overcharges above the threshold, detect duplicate codes on the same date, flag high-acuity codes for documentation review, and report missing benchmark rates.

The important technical tradeoff is that I use Medicare rates as public reference points, not as absolute fair-price labels. That is why the UI and prompts describe flags as review signals, not proof of fraud.

### 2:55-4:10 Risk Model And Feature Engineering

The trained bill-risk classifier uses engineered features from `src/features.py`. Features include total billed, total benchmark, bill-to-benchmark ratio, max and mean line-item ratios, counts of different flag types, unmatched-code fraction, procedure-category counts, and a coarse ZIP-region multiplier.

Training is in `scripts/train_risk_model.py`. The script generates weakly supervised synthetic bills grounded in CMS-style rates, labels them using deterministic analysis signals, and compares multiple models. It uses a stratified 75/25 train/test split.

The models include a majority-class baseline, logistic regression, tuned random forest, and gradient boosting. The script runs RandomizedSearchCV across 30 random-forest configurations with 5-fold cross-validation, computes learning curves, compares regularization settings, and saves plots in `eval/plots/`.

The current best model is a tuned random forest with test macro F1 around 0.995, compared to a majority baseline macro F1 of 0.172.

The risk model is intentionally based on engineered billing features rather than raw bill images. That makes the model more interpretable: high ratios, duplicate flags, unmatched-code counts, and high-acuity-code signals directly explain why a bill looks risky.

### 4:10-5:15 Evaluation

The API-free deterministic evaluation is `eval/evaluate_rules.py`. It runs ten synthetic bills through the production `run_all_checks()` path. The current result is 10 out of 10 cases passing, with precision, recall, and F1 all equal to 1.000.

`eval/error_analysis.py` analyzes the risk classifier. It found 2 misclassifications out of 450 test examples, both near-threshold high-acuity cases predicted as low risk instead of medium risk. It also generates a confusion matrix and feature-distribution plots.

`eval/prompt_comparison.py` evaluates three extraction prompts: minimal, structured, and chain-of-thought. All three achieved 1.000 code recall, item-count accuracy, total accuracy, and success rate on the prompt-evaluation bills.

These evaluations are separated intentionally. The deterministic rule evaluation does not require an API key, so it is reproducible. The prompt comparison uses the live LLM because it measures extraction behavior.

This also lets the project distinguish between two kinds of performance: whether the billing logic works when given structured line items, and whether the LLM prompt can reliably extract those line items from messy bill text.

### 5:15-6:20 LLM Integration And Chat

`src/llm.py` wraps the OpenAI-compatible API client. `src/explain.py` turns structured analysis into a plain-English explanation and keeps the bill context available for follow-up chat. `src/dispute.py` generates a dispute letter using the analysis and patient details.

The app is multi-turn because `app.py` stores chat history and passes it back with the structured bill analysis. The assistant is prompted to be clear and empathetic, to explain flags as review signals, and to avoid claiming fraud or giving legal advice.

One tradeoff is using an API LLM instead of a fully local language model. The API is better for messy extraction and natural explanations. The regex fallback, deterministic checks, and structured pipeline keep the app transparent and partially useful even when model calls fail.

For safety, the prompts avoid definitive accusations. The explanation prompt says flags are review signals, and the dispute prompt asks for professional wording that references Medicare as a public benchmark rather than a legal price.

### 6:20-7:25 Deployment And Production Considerations

The app is deployed on Hugging Face Spaces. `deploy_to_hf.py` stages only the runtime files needed by the hosted app, skips local secrets and bulky supplemental folders, and trims optional evaluation dependencies for a lighter Space build. `packages.txt` installs Poppler and Tesseract for OCR, while `sentence-transformers` stays in the Space requirements so the semantic embedding toggle works online.

Production considerations are implemented in `app.py`: structured logging, optional file logging, in-memory rate limiting, loading indicators for long-running model calls, clear user-facing errors, and graceful fallback messages if model calls fail. The app also reports pipeline status so users and graders can see what happened during extraction, matching, risk prediction, and explanation.

For deployment, I added compatibility handling for different Gradio versions. The hosted Space was smoke tested remotely: sample analysis, hash embeddings, semantic embeddings, chat, and dispute-letter generation all passed.

### 7:25-8:00 Closing

Overall, the project combines document extraction, language-model structured extraction, custom retrieval over public benchmark data, deterministic anomaly checks, a trained risk classifier, prompt evaluation, error analysis, and a deployed user interface. All components support the same goal: helping patients review and dispute confusing medical bills more effectively.

The main limitations are that the evaluation bills are synthetic, Medicare is only a public benchmark, and the risk model is weakly supervised rather than clinically validated. Those limitations are documented in the README so the system is framed as decision support rather than a definitive billing judgment.

### If You Need To Cut Time

- Shorten the extraction section by showing only `src/pipeline.py` and `src/code_extract.py`.
- Mention only the three strongest evaluation results: rule F1, risk-model macro F1, and prompt comparison.
- Skip line-by-line discussion of `deploy_to_hf.py`; just show the hosted Space and mention deployment considerations.

### If You Have Extra Time

- Show `SELF_ASSESSMENT.md` to connect the project directly to rubric items.
- Show `eval/plots/learning_curve.png`, `eval/plots/feature_importance.png`, and `eval/plots/confusion_matrix.png`.
- Show the README limitations section and explain why those limitations matter in a healthcare-adjacent tool.
