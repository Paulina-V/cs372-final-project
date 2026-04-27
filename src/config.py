"""Central configuration loaded from environment variables."""

import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
CHROMA_PERSIST_DIR = "chroma_db"
CMS_DATA_PATH = "data/cms_fee_schedule.csv"
OVERCHARGE_THRESHOLD = 2.0  # flag if billed > 2x Medicare rate
