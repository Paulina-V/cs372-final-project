"""Train and run a supervised bill risk classifier."""

from __future__ import annotations

from pathlib import Path

import joblib

from src.features import FEATURE_NAMES, analysis_to_features, vectorize_features


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "risk_model.joblib"


def load_risk_model(path: Path = MODEL_PATH):
    """Load the trained risk model artifact."""
    if not path.exists():
        raise FileNotFoundError(
            f"Risk model not found at {path}. Run `python scripts/train_risk_model.py` first."
        )
    return joblib.load(path)


def predict_bill_risk(analysis: dict, zip_code: str | None = None, model=None) -> dict:
    """Predict LOW/MEDIUM/HIGH bill risk from engineered analysis features."""
    if model is None:
        model = load_risk_model()

    features = analysis_to_features(analysis, zip_code)
    vector = [vectorize_features(features)]
    label = model.predict(vector)[0]

    probabilities = {}
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(vector)[0]
        probabilities = {
            class_name: round(float(prob), 3)
            for class_name, prob in zip(model.classes_, proba)
        }

    return {
        "risk_label": label,
        "risk_probabilities": probabilities,
        "features": {name: round(float(features[name]), 4) for name in FEATURE_NAMES},
    }


def format_risk_summary(risk: dict) -> str:
    """Create a concise user-facing risk summary."""
    label = risk.get("risk_label", "UNKNOWN")
    probabilities = risk.get("risk_probabilities", {})
    confidence = probabilities.get(label)
    confidence_text = f" ({confidence:.0%} model confidence)" if confidence is not None else ""

    explanations = {
        "LOW_RISK": "The bill looks broadly consistent with the benchmark signals checked by the model.",
        "MEDIUM_RISK": "The model sees moderate review signals in the bill-level features, so it is worth checking the line items and benchmark comparisons.",
        "HIGH_RISK": "The bill has strong review signals, so requesting an itemized review or dispute may be worthwhile.",
    }

    return f"**Trained risk model:** `{label}`{confidence_text}. {explanations.get(label, '')}"
