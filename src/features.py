"""Feature engineering for trained bill risk classification.

AI-assisted portions of this file are documented in ATTRIBUTION.md.
"""

from __future__ import annotations


RISK_LABELS = ["LOW_RISK", "MEDIUM_RISK", "HIGH_RISK"]

ZIP_REGION_MULTIPLIERS = {
    "0": 1.12,  # Northeast
    "1": 1.10,
    "2": 1.02,  # Mid-Atlantic / Southeast mix
    "3": 0.95,
    "4": 0.93,  # Midwest
    "5": 0.92,
    "6": 0.96,
    "7": 0.94,  # South / central
    "8": 1.03,  # Mountain / Southwest
    "9": 1.15,  # West Coast / Alaska / Hawaii
}


FEATURE_NAMES = [
    "num_line_items",
    "total_billed",
    "total_benchmark",
    "bill_to_benchmark_ratio",
    "max_line_ratio",
    "mean_line_ratio",
    "num_overcharge_flags",
    "num_duplicate_flags",
    "num_high_acuity_flags",
    "num_missing_rate_flags",
    "num_unmatched_codes",
    "fraction_unmatched_codes",
    "lab_code_count",
    "imaging_code_count",
    "anesthesia_code_count",
    "emergency_code_count",
    "zip_region_multiplier",
]


def zip_region_multiplier(zip_code: str | None) -> float:
    """Return a coarse regional multiplier from the first ZIP digit."""
    if not zip_code:
        return 1.0
    first_digit = str(zip_code).strip()[:1]
    return ZIP_REGION_MULTIPLIERS.get(first_digit, 1.0)


def analysis_to_features(analysis: dict, zip_code: str | None = None) -> dict[str, float]:
    """Convert a bill analysis dict into numeric model features."""
    rated_items = analysis.get("rated_items", [])
    flags = analysis.get("flags", [])
    num_items = len(rated_items)
    ratios = [
        float(item["ratio_to_medicare"])
        for item in rated_items
        if item.get("ratio_to_medicare") is not None
    ]
    matched_items = [item for item in rated_items if item.get("medicare_rate")]
    unmatched_items = [item for item in rated_items if item.get("cpt_code") and not item.get("medicare_rate")]
    total_billed = float(analysis.get("total_billed", 0) or 0)
    total_benchmark = float(analysis.get("total_medicare", 0) or 0)
    flag_counts = _count_flags(flags)

    features = {
        "num_line_items": float(num_items),
        "total_billed": total_billed,
        "total_benchmark": total_benchmark,
        "bill_to_benchmark_ratio": total_billed / total_benchmark if total_benchmark else 0.0,
        "max_line_ratio": max(ratios) if ratios else 0.0,
        "mean_line_ratio": sum(ratios) / len(ratios) if ratios else 0.0,
        "num_overcharge_flags": float(flag_counts.get("OVERCHARGE", 0)),
        "num_duplicate_flags": float(flag_counts.get("DUPLICATE", 0)),
        "num_high_acuity_flags": float(flag_counts.get("HIGH_ACUITY_CODE_REVIEW", 0)),
        "num_missing_rate_flags": float(flag_counts.get("RATE_NOT_FOUND", 0)),
        "num_unmatched_codes": float(len(unmatched_items)),
        "fraction_unmatched_codes": len(unmatched_items) / num_items if num_items else 0.0,
        "lab_code_count": float(sum(1 for item in matched_items if _is_lab_code(item.get("cpt_code")))),
        "imaging_code_count": float(sum(1 for item in matched_items if _is_imaging_code(item.get("cpt_code")))),
        "anesthesia_code_count": float(sum(1 for item in matched_items if _is_anesthesia_code(item.get("cpt_code")))),
        "emergency_code_count": float(sum(1 for item in matched_items if _is_emergency_code(item.get("cpt_code")))),
        "zip_region_multiplier": zip_region_multiplier(zip_code),
    }
    return features


def vectorize_features(features: dict[str, float]) -> list[float]:
    """Return features in the stable order used by the trained model."""
    return [float(features.get(name, 0.0) or 0.0) for name in FEATURE_NAMES]


def weak_label_from_analysis(analysis: dict) -> str:
    """Assign a weak supervision label from deterministic analysis signals."""
    features = analysis_to_features(analysis)
    severe_signals = (
        features["num_overcharge_flags"]
        + features["num_duplicate_flags"]
        + features["num_missing_rate_flags"]
    )
    review_signals = severe_signals + 0.5 * features["num_high_acuity_flags"]

    if severe_signals >= 2 or features["max_line_ratio"] >= 5.0 or features["bill_to_benchmark_ratio"] >= 4.0:
        return "HIGH_RISK"
    if review_signals >= 1 or features["max_line_ratio"] >= 2.0 or features["bill_to_benchmark_ratio"] >= 2.0:
        return "MEDIUM_RISK"
    return "LOW_RISK"


def _count_flags(flags: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for flag in flags:
        flag_type = flag.get("type")
        if flag_type:
            counts[flag_type] = counts.get(flag_type, 0) + 1
    return counts


def _numeric_code(code: str | None) -> int | None:
    if not code:
        return None
    text = str(code).strip()
    if not text.isdigit():
        return None
    return int(text)


def _is_lab_code(code: str | None) -> bool:
    value = _numeric_code(code)
    return value is not None and (80000 <= value <= 89999 or value == 36415)


def _is_imaging_code(code: str | None) -> bool:
    value = _numeric_code(code)
    return value is not None and 70000 <= value <= 79999


def _is_anesthesia_code(code: str | None) -> bool:
    value = _numeric_code(code)
    return value is not None and 100 <= value <= 1999


def _is_emergency_code(code: str | None) -> bool:
    return str(code).strip() in {"99281", "99282", "99283", "99284", "99285"}
