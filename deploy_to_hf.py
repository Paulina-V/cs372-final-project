"""Deploy the Medical Billing Assistant to Hugging Face Spaces.

Usage:
    1. pip install huggingface_hub
    2. hf auth login  (paste your HF write token)
    3. python deploy_to_hf.py --space-id YOUR_USERNAME/medical-billing-assistant
    4. Set OPENAI_API_KEY secret in the Space settings UI:
       https://huggingface.co/spaces/YOUR_USERNAME/medical-billing-assistant/settings
"""

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

try:
    from huggingface_hub import HfApi
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
    "data/bill images", "data/supplemental info",
    "docs", "notebooks", "videos",
    "deploy_to_hf.py", "final-project-class-instructions.md",
    "SELF_ASSESSMENT_DRAFT.md", "project-planning.md",
    "Plan.md", "Data-Sources.md",
]

SPACE_EXCLUDED_REQUIREMENTS = {
    "sentence-transformers",
    "matplotlib",
}


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
    """Copy deployable project files into the temporary Space folder."""
    for item in sorted(src.iterdir()):
        if should_skip(item, ROOT):
            continue
        target = dst / item.name
        if item.is_dir():
            target.mkdir(exist_ok=True)
            copy_tree(item, target)
        else:
            shutil.copy2(item, target)


def write_space_requirements(staging: Path):
    """Keep the hosted app lightweight by excluding optional eval-only packages."""
    req_path = staging / "requirements.txt"
    if not req_path.exists():
        return

    lines = []
    for line in req_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        package = stripped.split(">=")[0].split("==")[0].strip().lower()
        if package in SPACE_EXCLUDED_REQUIREMENTS:
            continue
        lines.append(line)
    req_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


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
        write_space_requirements(staging)

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
