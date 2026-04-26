"""Error analysis and edge-case evaluation for the bill risk classifier.

Generates a report of misclassified examples, analyzes which scenarios and
feature ranges are most challenging, and tests edge-case distributions.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.train_risk_model import generate_synthetic_dataset, RANDOM_SEED
from src.features import FEATURE_NAMES, RISK_LABELS
from src.risk_model import load_risk_model

PLOTS_DIR = ROOT / "eval" / "plots"
RESULTS_PATH = ROOT / "eval" / "error_analysis_results.json"


def run_error_analysis():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    x_rows, y_rows, examples = generate_synthetic_dataset()
    x_arr = np.array(x_rows)
    y_arr = np.array(y_rows)

    x_train, x_test, y_train, y_test, ex_train, ex_test = train_test_split(
        x_arr, y_arr, examples,
        test_size=0.25,
        random_state=RANDOM_SEED,
        stratify=y_arr,
    )

    model = load_risk_model()
    predictions = model.predict(x_test)

    # --- Misclassified examples ---
    misclassified = []
    for i, (pred, true) in enumerate(zip(predictions, y_test)):
        if pred != true:
            features = {name: round(float(x_test[i][j]), 4) for j, name in enumerate(FEATURE_NAMES)}
            misclassified.append({
                "index": int(i),
                "predicted": pred,
                "actual": true,
                "scenario": ex_test[i]["scenario"],
                "features": features,
            })

    # --- Error by scenario ---
    scenario_errors: dict[str, dict] = {}
    for i, (pred, true) in enumerate(zip(predictions, y_test)):
        scenario = ex_test[i]["scenario"]
        if scenario not in scenario_errors:
            scenario_errors[scenario] = {"total": 0, "errors": 0}
        scenario_errors[scenario]["total"] += 1
        if pred != true:
            scenario_errors[scenario]["errors"] += 1

    for scenario in scenario_errors:
        s = scenario_errors[scenario]
        s["error_rate"] = round(s["errors"] / s["total"], 4) if s["total"] > 0 else 0.0

    # --- Edge case analysis: extreme feature values ---
    edge_cases = []
    for i in range(len(x_test)):
        bill_ratio = x_test[i][FEATURE_NAMES.index("bill_to_benchmark_ratio")]
        max_ratio = x_test[i][FEATURE_NAMES.index("max_line_ratio")]
        num_flags = x_test[i][FEATURE_NAMES.index("num_overcharge_flags")]

        is_edge = bill_ratio > 8.0 or max_ratio > 8.0 or (num_flags == 0 and y_test[i] != "LOW_RISK")
        if is_edge:
            edge_cases.append({
                "index": int(i),
                "predicted": predictions[i],
                "actual": y_test[i],
                "correct": predictions[i] == y_test[i],
                "bill_to_benchmark_ratio": round(float(bill_ratio), 2),
                "max_line_ratio": round(float(max_ratio), 2),
                "scenario": ex_test[i]["scenario"],
            })

    edge_accuracy = (
        sum(1 for e in edge_cases if e["correct"]) / len(edge_cases)
        if edge_cases else 1.0
    )

    # --- Confusion matrix plot ---
    fig, ax = plt.subplots(figsize=(7, 6))
    cm = confusion_matrix(y_test, predictions, labels=RISK_LABELS)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=RISK_LABELS)
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title("Confusion Matrix: Bill Risk Classifier")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    # --- Error distribution by feature ---
    if misclassified:
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        key_features = ["bill_to_benchmark_ratio", "max_line_ratio", "num_overcharge_flags"]
        for ax, feat in zip(axes, key_features):
            feat_idx = FEATURE_NAMES.index(feat)
            correct_vals = [x_test[i][feat_idx] for i in range(len(x_test)) if predictions[i] == y_test[i]]
            error_vals = [x_test[i][feat_idx] for i in range(len(x_test)) if predictions[i] != y_test[i]]
            ax.hist(correct_vals, bins=20, alpha=0.6, label="Correct", color="steelblue")
            if error_vals:
                ax.hist(error_vals, bins=20, alpha=0.8, label="Misclassified", color="tomato")
            ax.set_title(feat)
            ax.legend(fontsize=8)
        fig.suptitle("Feature Distribution: Correct vs Misclassified", fontsize=12)
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / "error_distribution.png", dpi=150)
        plt.close(fig)

    results = {
        "test_accuracy": round(float(accuracy_score(y_test, predictions)), 4),
        "num_test_examples": len(y_test),
        "num_misclassified": len(misclassified),
        "misclassified_examples": misclassified[:20],
        "error_by_scenario": scenario_errors,
        "edge_case_analysis": {
            "num_edge_cases": len(edge_cases),
            "edge_case_accuracy": round(edge_accuracy, 4),
            "examples": edge_cases[:10],
        },
        "plots": [
            "eval/plots/confusion_matrix.png",
            "eval/plots/error_distribution.png",
        ],
    }

    RESULTS_PATH.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"Error analysis complete: {len(misclassified)} misclassified out of {len(y_test)}")
    print(f"Results saved to {RESULTS_PATH}")
    return results


if __name__ == "__main__":
    print(json.dumps(run_error_analysis(), indent=2))
