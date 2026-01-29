

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# CHROMA_PERSIST_DIR = os.getenv(
#     "CHROMA_PERSIST_DIR",
#     str(BASE_DIR / "data" / "chroma")  # safe local fallback
# )

CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", str(BASE_DIR / "data" / "chroma"))



COLLECTION_NAME = os.getenv("COLLECTION_NAME", "logs_titan_v2_1024")

AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
BEDROCK_EMBED_MODEL = os.getenv(
    "BEDROCK_EMBED_MODEL",
    "amazon.titan-embed-text-v2:0"
)
BEDROCK_LLM_MODEL = os.getenv(
    "BEDROCK_LLM_MODEL",
    "mistral.mistral-7b-instruct-v0:2"
)

MAX_CHARS = int(os.getenv("MAX_CHARS", "4000"))
OVERLAP = int(os.getenv("OVERLAP", "300"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "10"))

