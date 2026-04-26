"""Run and print the trained bill risk model evaluation."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.train_risk_model import train_and_evaluate


if __name__ == "__main__":
    print(json.dumps(train_and_evaluate(), indent=2))
