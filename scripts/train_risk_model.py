"""Train a weakly supervised bill risk classifier on synthetic CMS-grounded bills."""

from __future__ import annotations

import csv
import json
import random
import sys
from pathlib import Path

import joblib
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.analysis import run_all_checks
from src.config import CMS_DATA_PATH
from src.features import FEATURE_NAMES, RISK_LABELS, analysis_to_features, vectorize_features, weak_label_from_analysis
from src.rag import build_index
from src.risk_model import MODEL_PATH


RESULTS_PATH = ROOT / "eval" / "risk_model_results.json"
RANDOM_SEED = 372


def load_fee_rows() -> list[dict]:
    """Load benchmark rows useful for synthetic bill generation."""
    rows = []
    with (ROOT / CMS_DATA_PATH).open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            fee = float(row["fee"])
            if fee <= 0:
                continue
            rows.append({
                "code": row["code"],
                "description": row["description"],
                "fee": fee,
                "category": code_category(row["code"]),
            })
    return rows


def code_category(code: str) -> str:
    """Coarse CPT/HCPCS category for synthetic case balancing."""
    if code in {"99281", "99282", "99283", "99284", "99285"}:
        return "emergency"
    if code.isdigit():
        value = int(code)
        if 100 <= value <= 1999:
            return "anesthesia"
        if 70000 <= value <= 79999:
            return "imaging"
        if 80000 <= value <= 89999 or value == 36415:
            return "lab"
        if 99000 <= value <= 99999:
            return "evaluation"
    return "other"


def generate_synthetic_dataset(n_samples: int = 1800, seed: int = RANDOM_SEED) -> tuple[list[list[float]], list[str], list[dict]]:
    """Generate weakly labeled feature vectors from synthetic bill analyses."""
    rng = random.Random(seed)
    rows = load_fee_rows()
    by_category: dict[str, list[dict]] = {}
    for row in rows:
        by_category.setdefault(row["category"], []).append(row)

    x_rows = []
    y_rows = []
    examples = []
    scenarios = [
        "normal",
        "moderate_overcharge",
        "severe_overcharge",
        "duplicate",
        "missing_rate",
        "mixed",
        "high_acuity",
        "lab_bill",
        "anesthesia_bill",
    ]

    for _ in range(n_samples):
        scenario = rng.choice(scenarios)
        zip_code = synthetic_zip(rng)
        items = synthesize_bill(scenario, rows, by_category, rng)
        analysis = run_all_checks(items)
        label = weak_label_from_analysis(analysis)
        features = analysis_to_features(analysis, zip_code)

        x_rows.append(vectorize_features(features))
        y_rows.append(label)
        examples.append({
            "scenario": scenario,
            "zip_code": zip_code,
            "label": label,
            "num_items": len(items),
            "num_flags": analysis["num_flags"],
            "max_line_ratio": round(features["max_line_ratio"], 2),
            "bill_to_benchmark_ratio": round(features["bill_to_benchmark_ratio"], 2),
        })

    return x_rows, y_rows, examples


def synthesize_bill(scenario: str, rows: list[dict], by_category: dict[str, list[dict]], rng: random.Random) -> list[dict]:
    """Create one synthetic structured bill."""
    if scenario == "lab_bill":
        pool = by_category.get("lab", rows)
        n_items = rng.randint(2, 5)
    elif scenario == "anesthesia_bill":
        pool = by_category.get("anesthesia", rows)
        n_items = rng.randint(1, 3)
    elif scenario == "high_acuity":
        pool = [row for row in rows if row["code"] in {"99215", "99285"}] or rows
        n_items = rng.randint(1, 3)
    else:
        pool = rows
        n_items = rng.randint(1, 7)

    selected = rng.sample(pool, k=min(n_items, len(pool)))
    items = []
    for row in selected:
        multiplier = scenario_multiplier(scenario, rng)
        items.append(make_item(row, row["fee"] * multiplier, rng))

    if scenario in {"duplicate", "mixed"} and items:
        duplicate = dict(items[0])
        duplicate["description"] = duplicate["description"] + " duplicate line"
        items.append(duplicate)

    if scenario in {"missing_rate", "mixed"}:
        items.append({
            "cpt_code": "ZZ999",
            "icd_codes": [],
            "description": "Unmapped patient convenience service",
            "billed_amount": round(rng.uniform(100, 900), 2),
            "date_of_service": "04/01/2026",
        })

    return items


def scenario_multiplier(scenario: str, rng: random.Random) -> float:
    """Return billed-to-benchmark multiplier for a synthetic scenario."""
    if scenario == "normal":
        return rng.uniform(0.75, 1.45)
    if scenario in {"lab_bill", "anesthesia_bill"}:
        return rng.uniform(0.8, 1.6)
    if scenario == "moderate_overcharge":
        return rng.uniform(2.1, 3.4)
    if scenario == "severe_overcharge":
        return rng.uniform(4.0, 10.0)
    if scenario == "mixed":
        return rng.choice([rng.uniform(0.8, 1.5), rng.uniform(2.5, 6.0)])
    if scenario == "high_acuity":
        return rng.uniform(0.9, 3.0)
    return rng.uniform(0.8, 1.8)


def make_item(row: dict, billed_amount: float, rng: random.Random) -> dict:
    """Convert a fee row into a synthetic line item."""
    return {
        "cpt_code": row["code"],
        "icd_codes": [],
        "description": row["description"],
        "billed_amount": round(max(billed_amount, 1.0), 2),
        "date_of_service": f"04/{rng.randint(1, 28):02d}/2026",
    }


def synthetic_zip(rng: random.Random) -> str:
    """Generate a simple synthetic ZIP code."""
    first_digit = rng.choice(list("0123456789"))
    return first_digit + "".join(rng.choice(list("0123456789")) for _ in range(4))


def train_and_evaluate() -> dict:
    """Train candidate models, compare to baseline, and save the best model."""
    build_index(CMS_DATA_PATH)
    x_rows, y_rows, examples = generate_synthetic_dataset()

    x_train, x_test, y_train, y_test = train_test_split(
        x_rows,
        y_rows,
        test_size=0.25,
        random_state=RANDOM_SEED,
        stratify=y_rows,
    )

    models = {
        "majority_baseline": DummyClassifier(strategy="most_frequent"),
        "logistic_regression": Pipeline([
            ("scale", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            random_state=RANDOM_SEED,
            class_weight="balanced",
            max_depth=10,
        ),
    }

    model_results = {}
    best_name = None
    best_score = -1.0
    best_model = None
    for name, model in models.items():
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        accuracy = accuracy_score(y_test, predictions)
        report = classification_report(
            y_test,
            predictions,
            labels=RISK_LABELS,
            output_dict=True,
            zero_division=0,
        )
        model_results[name] = {
            "accuracy": round(float(accuracy), 3),
            "macro_f1": round(float(report["macro avg"]["f1-score"]), 3),
            "weighted_f1": round(float(report["weighted avg"]["f1-score"]), 3),
            "classification_report": report,
            "confusion_matrix": confusion_matrix(y_test, predictions, labels=RISK_LABELS).tolist(),
        }
        if model_results[name]["macro_f1"] > best_score:
            best_name = name
            best_score = model_results[name]["macro_f1"]
            best_model = model

    assert best_model is not None
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)

    results = {
        "dataset": {
            "num_samples": len(x_rows),
            "num_train": len(x_train),
            "num_test": len(x_test),
            "feature_names": FEATURE_NAMES,
            "label_counts": {label: y_rows.count(label) for label in RISK_LABELS},
        },
        "best_model": best_name,
        "model_path": str(MODEL_PATH.relative_to(ROOT)),
        "models": model_results,
        "example_synthetic_cases": examples[:10],
    }
    RESULTS_PATH.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    return results


if __name__ == "__main__":
    print(json.dumps(train_and_evaluate(), indent=2))
