# Setup Instructions

## 1. Create A Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

## 3. Configure Duke GPT / OpenAI-Compatible API Access

Create a local `.env` file:

```bash
cp .env.example .env
```

Edit `.env` with your real credentials:

```env
OPENAI_API_KEY=your-real-key
OPENAI_MODEL=your-duke-model-name
OPENAI_BASE_URL=your-duke-base-url-if-provided
```

If Duke GPT does not provide a custom base URL, remove `OPENAI_BASE_URL` or leave it commented out.

Do not put real keys in `.env.example`; `.env` is ignored by git.

## 4. Install Optional System OCR Tools

For image uploads and scanned PDFs, install Tesseract. On macOS with Homebrew:

```bash
brew install tesseract poppler
```

Text PDFs and `.txt` demo bills do not require Tesseract.

## 5. Optional: Refresh CMS Fee Schedule Data

The repository includes a generated `data/cms_fee_schedule.csv`. To rebuild it from CMS sources, run:

```bash
python scripts/download_cms_data.py
```

This downloads and converts CMS Physician Fee Schedule and Clinical Laboratory Fee Schedule files, then adds anesthesia reference rates. You do not need to run this before every demo unless you intentionally want to refresh the source CSV.

## 6. Build The Medicare Rate Index

```bash
python scripts/build_index.py
```

This creates a local `chroma_db/` directory. It is ignored by git because it is generated from `data/cms_fee_schedule.csv`.

## 7. Run The App

```bash
python app.py
```

To create a temporary public Gradio share link, run:

```bash
GRADIO_SHARE=true python app.py
```

Open the Gradio URL printed in the terminal. For the most reliable demo, upload:

```text
data/sample_bill.txt
```

## 8. Run Evaluation

```bash
python eval/evaluate_rules.py
python eval/evaluate_risk_model.py
```

The script rebuilds the local index, evaluates synthetic bills through the production deterministic analysis path, checks sampled index lookups against the CSV, and writes updated metrics to:

```text
eval/results.json
eval/risk_model_results.json
```

`evaluate_risk_model.py` trains and compares a majority baseline, logistic regression model, and random forest bill-risk classifier, then writes `models/risk_model.joblib`.

## Troubleshooting

- If the app says `OPENAI_API_KEY is not set`, check that your real key is in `.env`, not `.env.example` or `.env.s`.
- If Medicare rates are missing, run `python scripts/build_index.py`.
- If image OCR fails, install Tesseract and poppler or use the text demo bill.
- If package installation fails on Python 3.13, try Python 3.11 or 3.12 because some ML packages may lag behind the newest Python release.
