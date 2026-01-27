# # config.py 
# from pathlib import Path

# # Absolute path so every script (uvicorn/streamlit) points to the same Chroma DB
# BASE_DIR = Path(__file__).resolve().parent
# CHROMA_DIR = str(BASE_DIR / "chroma_db")

# # Name includes model + dimension to avoid future mismatch
# COLLECTION_NAME = "logs_titan_1536"

# AWS_REGION = "us-east-1"
# BEDROCK_EMBED_MODEL = "amazon.titan-embed-text-v1"
# BEDROCK_LLM_MODEL = "mistral.mistral-7b-instruct-v0:2"


# import os
# from pathlib import Path

# BASE_DIR = Path(__file__).resolve().parent

# # ---- Persistent storage location (important for Docker/EC2/ECS) ----
# # In Docker/servers, you want this to point to a mounted volume like /data/chroma
# CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", str(BASE_DIR / "chroma_db"))

# # Collection name can also be env-controlled (handy across environments)
# COLLECTION_NAME = os.getenv("COLLECTION_NAME", "logs_titan_1536")

# # ---- AWS Bedrock settings ----
# AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
# BEDROCK_EMBED_MODEL = os.getenv("BEDROCK_EMBED_MODEL", "amazon.titan-embed-text-v2:0")
# BEDROCK_LLM_MODEL = os.getenv("BEDROCK_LLM_MODEL", "mistral.mistral-7b-instruct-v0:2")

# # ---- App tuning ----
# MAX_CHARS = int(os.getenv("MAX_CHARS", "4000"))
# OVERLAP = int(os.getenv("OVERLAP", "300"))
# BATCH_SIZE = int(os.getenv("BATCH_SIZE", "10"))


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

