"""Train a weakly supervised bill risk classifier on synthetic CMS-grounded bills.

Includes hyperparameter tuning via RandomizedSearchCV, learning curve analysis,
regularization comparison, and training curve visualization.

AI-assisted portions of this file are documented in ATTRIBUTION.md.
"""

from __future__ import annotations

import csv
import json
import random
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    learning_curve,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.analysis import HIGH_ACUITY_REVIEW_CODES
from src.config import CMS_DATA_PATH, OVERCHARGE_THRESHOLD
from src.features import FEATURE_NAMES, RISK_LABELS, analysis_to_features, vectorize_features, weak_label_from_analysis
from src.risk_model import MODEL_PATH


RESULTS_PATH = ROOT / "eval" / "risk_model_results.json"
PLOTS_DIR = ROOT / "eval" / "plots"
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
        analysis = analyze_synthetic_items(items)
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
        "benchmark_fee": row["fee"],
        "date_of_service": f"04/{rng.randint(1, 28):02d}/2026",
    }


def synthetic_zip(rng: random.Random) -> str:
    """Generate a simple synthetic ZIP code."""
    first_digit = rng.choice(list("0123456789"))
    return first_digit + "".join(rng.choice(list("0123456789")) for _ in range(4))


def analyze_synthetic_items(items: list[dict]) -> dict:
    """Fast analysis for generated bills using embedded benchmark fees."""
    rated_items = []
    flags = []
    seen = {}

    for index, item in enumerate(items):
        benchmark = item.get("benchmark_fee")
        billed = item.get("billed_amount", 0) or 0
        ratio = billed / benchmark if benchmark else None
        line_flag = None
        if ratio and ratio > OVERCHARGE_THRESHOLD:
            line_flag = f"OVERCHARGE: Billed ${billed:.2f} is {ratio:.1f}x the Medicare rate of ${benchmark:.2f}"
            flags.append({"type": "OVERCHARGE", "message": line_flag, "code": item.get("cpt_code")})

        rated_item = {
            key: value
            for key, value in item.items()
            if key != "benchmark_fee"
        }
        rated_item.update({
            "medicare_rate": benchmark,
            "ratio_to_medicare": ratio,
            "flag": line_flag,
            "rag_result": None,
            "rag_error": None,
        })
        rated_items.append(rated_item)

        key = (item.get("cpt_code"), item.get("date_of_service"))
        if key[0] and key in seen:
            flags.append({
                "type": "DUPLICATE",
                "message": f"CPT {key[0]} appears more than once on {key[1] or 'same date'}",
                "indices": [seen[key], index],
            })
        else:
            seen[key] = index

        code = item.get("cpt_code")
        if code in HIGH_ACUITY_REVIEW_CODES:
            flags.append({
                "type": "HIGH_ACUITY_CODE_REVIEW",
                "message": HIGH_ACUITY_REVIEW_CODES[code],
                "code": code,
            })
        if code and not benchmark:
            flags.append({
                "type": "RATE_NOT_FOUND",
                "message": f"No benchmark rate was found for CPT/HCPCS {code}; verify the code and any modifiers.",
                "code": code,
            })

    return {
        "rated_items": rated_items,
        "flags": flags,
        "num_flags": len(flags),
        "total_billed": sum(item.get("billed_amount", 0) or 0 for item in items),
        "total_medicare": sum(item.get("benchmark_fee", 0) or 0 for item in items if item.get("benchmark_fee")),
    }


def run_hyperparameter_search(x_train, y_train) -> dict:
    """Run RandomizedSearchCV over Random Forest hyperparameters."""
    param_distributions = {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [5, 10, 15, 20, None],
        "min_samples_split": [2, 5, 10, 20],
        "min_samples_leaf": [1, 2, 5, 10],
        "max_features": ["sqrt", "log2", None],
        "class_weight": ["balanced", "balanced_subsample"],
    }

    search = RandomizedSearchCV(
        RandomForestClassifier(random_state=RANDOM_SEED),
        param_distributions=param_distributions,
        n_iter=30,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED),
        scoring="f1_macro",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    search.fit(x_train, y_train)

    top_configs = []
    results_df = list(zip(search.cv_results_["params"], search.cv_results_["mean_test_score"], search.cv_results_["std_test_score"]))
    results_df.sort(key=lambda x: x[1], reverse=True)
    for params, mean_score, std_score in results_df[:5]:
        serializable_params = {k: (v if v is not None else "None") for k, v in params.items()}
        top_configs.append({
            "params": serializable_params,
            "mean_cv_f1": round(float(mean_score), 4),
            "std_cv_f1": round(float(std_score), 4),
        })

    return {
        "best_params": {k: (v if v is not None else "None") for k, v in search.best_params_.items()},
        "best_cv_f1": round(float(search.best_score_), 4),
        "n_configs_searched": len(search.cv_results_["params"]),
        "top_5_configs": top_configs,
        "best_model": search.best_estimator_,
    }


def compute_learning_curves(x_train, y_train) -> dict:
    """Compute learning curves showing train/val performance vs training size."""
    train_sizes_abs, train_scores, val_scores = learning_curve(
        RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            class_weight="balanced",
            random_state=RANDOM_SEED,
        ),
        x_train,
        y_train,
        train_sizes=np.linspace(0.1, 1.0, 10),
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED),
        scoring="f1_macro",
        n_jobs=-1,
    )

    return {
        "train_sizes": train_sizes_abs.tolist(),
        "train_scores_mean": np.mean(train_scores, axis=1).tolist(),
        "train_scores_std": np.std(train_scores, axis=1).tolist(),
        "val_scores_mean": np.mean(val_scores, axis=1).tolist(),
        "val_scores_std": np.std(val_scores, axis=1).tolist(),
    }


def compute_gb_staged_loss(x_train, y_train, x_test, y_test) -> dict:
    """Track GradientBoosting test loss across boosting iterations."""
    gb = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.1,
        n_iter_no_change=10,
        validation_fraction=0.15,
        random_state=RANDOM_SEED,
    )
    gb.fit(x_train, y_train)

    staged_train_loss = []
    staged_test_loss = []
    for i, (train_pred, test_pred) in enumerate(
        zip(gb.staged_predict(x_train), gb.staged_predict(x_test))
    ):
        staged_train_loss.append(1.0 - accuracy_score(y_train, train_pred))
        staged_test_loss.append(1.0 - accuracy_score(y_test, test_pred))

    return {
        "n_iterations": list(range(1, len(staged_train_loss) + 1)),
        "train_error": staged_train_loss,
        "test_error": staged_test_loss,
        "final_test_accuracy": round(1.0 - staged_test_loss[-1], 4),
    }


def compare_regularization(x_train, y_train, x_test, y_test) -> dict:
    """Compare models with and without regularization techniques.

    Demonstrates two rubric-listed regularization techniques:
      1. L2 penalty (Logistic Regression)
      2. Early stopping (Gradient Boosting via n_iter_no_change)
    Also compares Random Forest with varying capacity regularization.
    """
    configs = {
        "lr_no_regularization": Pipeline([
            ("scale", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000, penalty=None)),
        ]),
        "lr_l2_regularization": Pipeline([
            ("scale", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000, penalty="l2", C=1.0, class_weight="balanced")),
        ]),
        "gb_no_early_stopping": GradientBoostingClassifier(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.1,
            random_state=RANDOM_SEED,
        ),
        "gb_early_stopping": GradientBoostingClassifier(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.1,
            n_iter_no_change=10,
            validation_fraction=0.15,
            random_state=RANDOM_SEED,
        ),
        "rf_no_depth_limit": RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=RANDOM_SEED,
        ),
        "rf_regularized": RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=RANDOM_SEED,
        ),
    }

    results = {}
    for name, model in configs.items():
        model.fit(x_train, y_train)
        train_pred = model.predict(x_train)
        test_pred = model.predict(x_test)
        train_report = classification_report(y_train, train_pred, labels=RISK_LABELS, output_dict=True, zero_division=0)
        test_report = classification_report(y_test, test_pred, labels=RISK_LABELS, output_dict=True, zero_division=0)

        n_estimators_used = None
        if hasattr(model, "n_estimators_"):
            n_estimators_used = model.n_estimators_
        elif hasattr(model, "n_estimators"):
            n_estimators_used = model.n_estimators

        results[name] = {
            "train_accuracy": round(accuracy_score(y_train, train_pred), 4),
            "test_accuracy": round(accuracy_score(y_test, test_pred), 4),
            "train_macro_f1": round(float(train_report["macro avg"]["f1-score"]), 4),
            "test_macro_f1": round(float(test_report["macro avg"]["f1-score"]), 4),
            "train_test_f1_gap": round(
                float(train_report["macro avg"]["f1-score"]) - float(test_report["macro avg"]["f1-score"]), 4
            ),
        }
        if n_estimators_used is not None:
            results[name]["n_estimators_used"] = n_estimators_used

    return results


def plot_learning_curves(lc_data: dict, save_path: Path):
    """Plot and save learning curves."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    train_sizes = lc_data["train_sizes"]
    train_mean = np.array(lc_data["train_scores_mean"])
    train_std = np.array(lc_data["train_scores_std"])
    val_mean = np.array(lc_data["val_scores_mean"])
    val_std = np.array(lc_data["val_scores_std"])

    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color="blue")
    ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color="orange")
    ax.plot(train_sizes, train_mean, "o-", color="blue", label="Training F1 (macro)")
    ax.plot(train_sizes, val_mean, "o-", color="orange", label="Validation F1 (macro)")
    ax.set_xlabel("Training Set Size")
    ax.set_ylabel("Macro F1 Score")
    ax.set_title("Learning Curve: Random Forest Bill Risk Classifier")
    ax.legend(loc="lower right")
    ax.set_ylim(0.5, 1.05)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_staged_loss(staged_data: dict, save_path: Path):
    """Plot GradientBoosting train/test error over boosting iterations."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.plot(staged_data["n_iterations"], staged_data["train_error"], label="Train Error", color="blue")
    ax.plot(staged_data["n_iterations"], staged_data["test_error"], label="Test Error", color="orange")
    ax.set_xlabel("Boosting Iteration")
    ax.set_ylabel("Error Rate (1 - Accuracy)")
    ax.set_title("Training Curve: Gradient Boosting Staged Error")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_feature_importance(model, save_path: Path):
    """Plot feature importances from the trained model."""
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "best_estimator_"):
        importances = model.best_estimator_.feature_importances_
    else:
        return

    indices = np.argsort(importances)[::-1]
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.barh(
        [FEATURE_NAMES[i] for i in indices],
        [importances[i] for i in indices],
        color="steelblue",
    )
    ax.set_xlabel("Feature Importance (Gini)")
    ax.set_title("Random Forest Feature Importances")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_regularization_comparison(reg_results: dict, save_path: Path):
    """Plot bar chart comparing regularization configurations."""
    labels = []
    train_f1s = []
    test_f1s = []
    for name, metrics in reg_results.items():
        short = name.replace("_", " ").title()
        labels.append(short)
        train_f1s.append(metrics["train_macro_f1"])
        test_f1s.append(metrics["test_macro_f1"])

    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    ax.bar(x - width / 2, train_f1s, width, label="Train Macro F1", color="steelblue")
    ax.bar(x + width / 2, test_f1s, width, label="Test Macro F1", color="darkorange")
    ax.set_ylabel("Macro F1 Score")
    ax.set_title("Regularization Comparison: Train vs Test Macro F1")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.legend()
    ax.set_ylim(0.95, 1.005)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def train_and_evaluate() -> dict:
    """Train candidate models, compare to baseline, and save the best model."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    x_rows, y_rows, examples = generate_synthetic_dataset()
    x_arr = np.array(x_rows)
    y_arr = np.array(y_rows)

    x_train, x_test, y_train, y_test = train_test_split(
        x_arr,
        y_arr,
        test_size=0.25,
        random_state=RANDOM_SEED,
        stratify=y_arr,
    )

    # --- Hyperparameter tuning ---
    print("Running hyperparameter search (30 configs, 5-fold CV)...")
    hp_results = run_hyperparameter_search(x_train, y_train)
    best_rf = hp_results["best_model"]

    # --- Learning curves ---
    print("Computing learning curves...")
    lc_data = compute_learning_curves(x_train, y_train)
    plot_learning_curves(lc_data, PLOTS_DIR / "learning_curve.png")

    # --- Staged boosting loss ---
    print("Computing gradient boosting staged loss...")
    staged_data = compute_gb_staged_loss(x_train, y_train, x_test, y_test)
    plot_staged_loss(staged_data, PLOTS_DIR / "training_curve_gb.png")

    # --- Regularization comparison ---
    print("Comparing regularization configurations...")
    reg_results = compare_regularization(x_train, y_train, x_test, y_test)
    plot_regularization_comparison(reg_results, PLOTS_DIR / "regularization_comparison.png")

    # --- Model comparison ---
    print("Training and comparing models...")
    models = {
        "majority_baseline": DummyClassifier(strategy="most_frequent"),
        "logistic_regression": Pipeline([
            ("scale", StandardScaler()),
            ("model", LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                penalty="l2",
                C=1.0,
            )),
        ]),
        "random_forest_tuned": best_rf,
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.1,
            n_iter_no_change=10,
            validation_fraction=0.15,
            random_state=RANDOM_SEED,
        ),
    }

    model_results = {}
    best_name = None
    best_score = -1.0
    best_model = None
    for name, model in models.items():
        if name == "random_forest_tuned":
            predictions = model.predict(x_test)
        else:
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

    # --- Feature importance plot ---
    plot_feature_importance(best_rf, PLOTS_DIR / "feature_importance.png")

    # --- Save best model ---
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
        "hyperparameter_search": {
            "best_params": hp_results["best_params"],
            "best_cv_f1": hp_results["best_cv_f1"],
            "n_configs_searched": hp_results["n_configs_searched"],
            "top_5_configs": hp_results["top_5_configs"],
        },
        "learning_curve": {
            "train_sizes": lc_data["train_sizes"],
            "final_train_f1": round(lc_data["train_scores_mean"][-1], 4),
            "final_val_f1": round(lc_data["val_scores_mean"][-1], 4),
            "plot": "eval/plots/learning_curve.png",
        },
        "training_curve_gradient_boosting": {
            "final_test_accuracy": staged_data["final_test_accuracy"],
            "plot": "eval/plots/training_curve_gb.png",
        },
        "regularization_comparison": reg_results,
        "models": model_results,
        "example_synthetic_cases": examples[:10],
    }
    RESULTS_PATH.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    print(f"\nBest model: {best_name} (macro F1 = {best_score:.3f})")
    print(f"Saved to {MODEL_PATH}")
    print(f"Plots saved to {PLOTS_DIR}/")
    return results


if __name__ == "__main__":
    print(json.dumps(train_and_evaluate(), indent=2))
