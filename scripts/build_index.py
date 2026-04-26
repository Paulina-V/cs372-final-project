"""Build the ChromaDB index from CMS fee schedule data."""

import sys
sys.path.insert(0, ".")

from src.rag import build_index
from src.config import CMS_DATA_PATH

if __name__ == "__main__":
    print(f"Building index from {CMS_DATA_PATH}...")
    collection = build_index(CMS_DATA_PATH)
    print("Done! Testing a query...")

    from src.rag import query_rate
    result = query_rate("99213", collection)
    print(f"Query for 99213: {result}")
