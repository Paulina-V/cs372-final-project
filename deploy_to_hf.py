"""Deploy the Medical Billing Assistant to HuggingFace Spaces.

Usage:
    1. hf auth login  (paste your HF write token)
    2. python deploy_to_hf.py --space-id YOUR_USERNAME/medical-billing-assistant
    3. Set OPENAI_API_KEY secret in the Space settings UI:
       https://huggingface.co/spaces/YOUR_USERNAME/medical-billing-assistant/settings
"""

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

try:
    from huggingface_hub import HfApi, SpaceHardware
except ImportError:
    print("Install huggingface_hub: pip install huggingface_hub")
    sys.exit(1)


ROOT = Path(__file__).resolve().parent

SPACE_README = """\
---
title: Medical Billing Assistant
emoji: 🏥
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: "5.33.0"
app_file: app.py
pinned: false
---

# Medical Billing Assistant

Upload a medical bill to get a plain-English explanation, identify potential overcharges,
and generate a dispute letter.

Built for CS 372: Introduction to Applied Machine Learning, Duke University, Spring 2026.
"""

SKIP_PATTERNS = [
    ".venv", "venv", "__pycache__", ".git", ".env", ".env.*",
    "chroma_db", "*.log", ".DS_Store", "*.pyc",
    "Supplements to Support Project Significance",
    "deploy_to_hf.py", "final-project-class-instructions.md",
    "SELF_ASSESSMENT_DRAFT.md", "project-planning.md",
    "Plan.md", "Data-Sources.md",
]


def should_skip(path: Path, root: Path) -> bool:
    rel = str(path.relative_to(root))
    name = path.name
    for pattern in SKIP_PATTERNS:
        if pattern.startswith("*"):
            if name.endswith(pattern[1:]):
                return True
        elif pattern in rel.split("/") or rel == pattern or name == pattern:
            return True
    return False


def copy_tree(src: Path, dst: Path):
    for item in sorted(src.iterdir()):
        if should_skip(item, ROOT):
            continue
        target = dst / item.name
        if item.is_dir():
            target.mkdir(exist_ok=True)
            copy_tree(item, target)
        else:
            shutil.copy2(item, target)


def main():
    parser = argparse.ArgumentParser(description="Deploy to HuggingFace Spaces")
    parser.add_argument("--space-id", required=True, help="HF Space ID, e.g. username/medical-billing-assistant")
    parser.add_argument("--private", action="store_true", help="Create as private Space")
    args = parser.parse_args()

    api = HfApi()

    print(f"Creating Space: {args.space_id} ...")
    try:
        api.create_repo(
            repo_id=args.space_id,
            repo_type="space",
            space_sdk="gradio",
            private=args.private,
            exist_ok=True,
        )
    except Exception as e:
        print(f"Warning creating space: {e}")

    with tempfile.TemporaryDirectory() as tmpdir:
        staging = Path(tmpdir) / "space"
        staging.mkdir()

        print("Copying files to staging directory ...")
        copy_tree(ROOT, staging)

        readme_path = staging / "README.md"
        original_readme = readme_path.read_text() if readme_path.exists() else ""
        readme_path.write_text(SPACE_README + "\n" + original_readme)

        print(f"Uploading to {args.space_id} ...")
        api.upload_folder(
            folder_path=str(staging),
            repo_id=args.space_id,
            repo_type="space",
        )

    print(f"\nDeployed! Visit: https://huggingface.co/spaces/{args.space_id}")
    print(f"\nIMPORTANT: Set your OPENAI_API_KEY secret at:")
    print(f"  https://huggingface.co/spaces/{args.space_id}/settings")
    print(f"  Add a secret named OPENAI_API_KEY with your API key value.")
    if "OPENAI_BASE_URL" in open(ROOT / ".env.example").read():
        print(f"  Also add OPENAI_BASE_URL and OPENAI_MODEL if using a custom endpoint.")
    print(f"  Also add GRADIO_SERVER_NAME=0.0.0.0 as a variable (not secret).")


if __name__ == "__main__":
    main()
