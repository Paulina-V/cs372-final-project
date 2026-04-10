# CS 372 Final Project Plan
## Medical Billing Assistant
**Due:** April 26, 2026 | **Author** Paulina Vvedenskaya

---

## Project Overview

An AI-powered medical billing assistant that helps patients understand and dispute their medical bills. The system ingests bills (PDFs/images), extracts billing codes, compares charges against CMS fee schedules, explains discrepancies in plain language, and generates dispute letters via an agentic workflow.

---

## Rubric Goals

### Category 1: Machine Learning (target: 90–100 pts)
- RAG pipeline over CMS fee schedule data — 10 pts
- LLM API integration with multi-turn conversation — 10 pts
- PDF/image ingestion and OCR — 10 pts
- CPT/ICD code extraction (NLP) — 10 pts
- Fine-tuned model for billing anomaly classification (LoRA) — 10 pts
- Agentic dispute letter generation workflow — 10 pts
- Multi-stage pipeline connecting extraction, lookup, explanation, and action — 10 pts
- Evaluation and ablation study — 10 pts
- Deployed web app — 10 pts
- Exceptional item: reproduce results from a medical NLP paper OR RLHF for dispute generation — 10 pts

### Category 2: Following Directions (target: 14–15 pts)
- README with motivation, setup, and usage sections — 3 pts
- SETUP.md with reproducible environment instructions — 3 pts
- ATTRIBUTION.md documenting all AI tools used — 3 pts
- Demo video, user-facing walkthrough — 3 pts
- Technical walkthrough video — 3 pts

### Category 3: Project Cohesion & Motivation (target: 13–15 pts)
- All components serve the single goal of helping users understand and dispute medical bills, so this should be naturally strong
- Make sure the README clearly articulates the unified motivation

---

## Architecture Overview

The system is built as a pipeline that takes a raw medical bill and transforms it into actionable insight for the patient. When a user uploads a PDF or image of their bill, an OCR layer first extracts the raw text, which is then passed to an NLP model that identifies and parses the CPT and ICD billing codes present. Those codes are used to query a RAG system built over the CMS Medicare Physician Fee Schedule, retrieving the official allowed rates for each procedure. A combination of rule-based logic and a fine-tuned anomaly classifier then compares what was billed against what CMS allows, flagging discrepancies, duplicates, or suspicious charges. The results are handed off to an LLM, which explains the findings in plain language and supports follow-up questions in a multi-turn conversation interface. If the user wants to take action, an agentic workflow takes over, gathering the relevant details and drafting a formal dispute letter. 

---

## Data Sources

- CMS Medicare Physician Fee Schedule — RAG knowledge base, publicly available at cms.gov
- CMS Hospital Outpatient Prospective Payment — additional rate lookup, publicly available
- Sample medical bills — use synthetic or anonymized bills for testing and demo
- Labeled anomaly dataset — needed for fine-tuning; check Kaggle and MIMIC first, construct manually if unavailable

---

## Key Risks

- No labeled data for anomaly classification — pivot to rule-based flagging if needed; validate this in the first few days
- LLM hallucinations on billing codes — ground all outputs in retrieved CMS data and surface the source
- Videos taking longer than expected — start recording early, rough cuts are fine

---

## To Do

### Setup & Data
- Download CMS Physician Fee Schedule and inspect its structure
- Download CMS Hospital Outpatient fee schedule and inspect its structure
- Find or create 5–10 sample medical bills for testing (synthetic or anonymized)
- Search Kaggle and MIMIC for labeled billing anomaly datasets
- Set up project repo with folder structure and initial README skeleton
- Set up Python environment and pin dependencies in requirements.txt

### OCR & Code Extraction
- Extract sample bill pdf
- Write a preprocessing step to clean and normalize OCR output
- Write a regex + rule-based extractor for CPT codes from OCR text
- Write a regex + rule-based extractor for ICD codes from OCR text
- Manually verify extraction accuracy on 3–5 sample bills

### RAG Pipeline
- Chunk the CMS fee schedule into per-code records
- Choose and set up a vector store (e.g., FAISS or Chroma)
- Embed the CMS chunks and load them into the vector store
- Write a retrieval function: given a CPT code, return the CMS allowed rate and description
- Test retrieval on 10 sample codes and verify correctness

### Discrepancy Analysis
- Write a rule-based comparator: billed amount vs. CMS allowed rate, flag if over threshold
- Write a duplicate detection rule: flag repeated codes on the same bill
- If labeled data is available, fine-tune a classification model (LoRA) on billing anomalies
- If no labeled data, document the rule-based approach and move on
- Evaluate the classifier or rule-based system on held-out bills

### LLM Integration & Conversation
- Write a prompt template that takes extracted codes, CMS rates, and flags as input
- Test the prompt on 3 sample bills and evaluate explanation quality
- Add multi-turn conversation support: maintain message history across turns
- Test follow-up questions (e.g., "what does CPT 99213 mean?") and verify responses

### Agentic Dispute Letter Workflow
- Design the dispute letter template (patient info, billed vs. allowed, request for review)
- Write an agent that collects missing info from the user via conversation
- Write the final letter generation step using the LLM
- Test end-to-end: upload bill, get flags, request dispute letter, receive output

### Evaluation & Ablation
- Define evaluation metrics: code extraction accuracy, rate lookup accuracy, flagging precision/recall
- Build a small labeled eval set (10–20 bills with known ground truth)
- Run baseline evaluation (no RAG, no fine-tuning) and record results
- Run full system evaluation and record results
- Write up ablation: RAG vs. no RAG, fine-tuned vs. base model

### Deployment
- Choose deployment framework (Gradio, Streamlit, or Flask)
- Build a basic UI: file upload, chat interface, dispute letter download
- Deploy to a public URL (Hugging Face Spaces, Render, or similar)
- Verify the deployed app works end-to-end on a fresh browser

### Documentation & Videos
- Write the full README: motivation, architecture, setup instructions, usage, results
- Write SETUP.md with exact steps to reproduce the environment
- Write ATTRIBUTION.md listing all AI tools used and how
- Record demo video: user-facing walkthrough showing a bill being analyzed (~2–3 min)
- Record technical video: architecture and code walkthrough for graders
- Do a final review of all docs for completeness before submission
