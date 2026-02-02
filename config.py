# import os
# import sys
# from pathlib import Path
# from typing import List, Optional

# from dotenv import load_dotenv


# # Minimal supported version
# if sys.version_info < (3, 11):
#     raise RuntimeError("This application requires Python 3.11 or higher")


# BASE_DIR = Path(__file__).resolve().parent

# # Load .env (if present) from the project root
# load_dotenv(dotenv_path=BASE_DIR / ".env", override=False)


# class Settings:
#     """
#     Lightweight settings container (no pydantic) to avoid dependency conflicts on Windows.
#     Keeps attribute names compatible with your existing code: settings.XYZ
#     """

#     # Storage / Chroma
#     CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", str(BASE_DIR / "data" / "chroma"))
#     COLLECTION_NAME: str = os.getenv("COLLECTION_NAME", "logs_titan_v2_1024")

#     # Uploads
#     UPLOAD_DIR: Path = Path(os.getenv("UPLOAD_DIR", str(BASE_DIR / "uploads")))
#     MAX_FILE_SIZE_MB: int = int(os.getenv("MAX_FILE_SIZE_MB", "100"))
#     ALLOWED_FILE_TYPES: List[str] = ["log", "txt", "md", "json"]

#     # AWS / Bedrock
#     AWS_REGION: str = os.getenv("AWS_REGION", "us-east-1")

#     # For LOCAL dev you may set keys in .env (optional).
#     # For EC2, prefer IAM Role and keep these unset.
#     AWS_ACCESS_KEY_ID: Optional[str] = os.getenv("AWS_ACCESS_KEY_ID")
#     AWS_SECRET_ACCESS_KEY: Optional[str] = os.getenv("AWS_SECRET_ACCESS_KEY")
#     AWS_SESSION_TOKEN: Optional[str] = os.getenv("AWS_SESSION_TOKEN")

#     # BEDROCK_EMBED_MODEL: str = os.getenv("BEDROCK_EMBED_MODEL", "amazon.titan-embed-text-v2:0")
#     BEDROCK_EMBED_MODEL: str = os.getenv("BEDROCK_EMBED_MODEL", "amazon.titan-embed-text-v1")
#     BEDROCK_LLM_MODEL: str = os.getenv("BEDROCK_LLM_MODEL", "mistral.mistral-7b-instruct-v0:2")

#     # Chunking / batching
#     MAX_CHARS: int = int(os.getenv("MAX_CHARS", "4000"))
#     OVERLAP: int = int(os.getenv("OVERLAP", "300"))
#     BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", "10"))

#     # API server
#     HOST: str = os.getenv("HOST", "0.0.0.0")
#     PORT: int = int(os.getenv("PORT", "8000"))
#     LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

#     # Security
#     API_KEY: Optional[str] = os.getenv("API_KEY") or None

#     # UI settings (Streamlit)
#     UI_API_KEY: Optional[str] = os.getenv("UI_API_KEY") or None
#     API_BASE: str = os.getenv("API_BASE", "http://localhost:8000")
#     REQUEST_TIMEOUT: int = int(os.getenv("REQUEST_TIMEOUT", "30"))


# settings = Settings()

# # Create required directories
# try:
#     settings.UPLOAD_DIR.mkdir(exist_ok=True, parents=True)
#     Path(settings.CHROMA_PERSIST_DIR).mkdir(exist_ok=True, parents=True)
# except OSError as e:
#     print(f"Error creating required directories: {e}")
#     sys.exit(1)

import os
import sys
from pathlib import Path
from typing import List, Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


# Minimal supported version
if sys.version_info < (3, 11):
    raise RuntimeError("This application requires Python 3.11 or higher")


BASE_DIR = Path(__file__).resolve().parent


class Settings(BaseSettings):
    # Storage / Chroma
    CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", str(BASE_DIR / "data" / "chroma"))
    COLLECTION_NAME: str = os.getenv("COLLECTION_NAME", "logs_titan_v2_1024")

    # Uploads
    UPLOAD_DIR: Path = Path(os.getenv("UPLOAD_DIR", str(BASE_DIR / "uploads")))
    MAX_FILE_SIZE_MB: int = int(os.getenv("MAX_FILE_SIZE_MB", "100"))
    ALLOWED_FILE_TYPES: List[str] = ["log", "txt", "md", "json"]

    # AWS / Bedrock
    AWS_REGION: str = os.getenv("AWS_REGION", "us-east-1")

    # Local-only explicit keys (recommended: use AWS_PROFILE or aws sso login)
    AWS_ACCESS_KEY_ID: Optional[str] = os.getenv("AWS_ACCESS_KEY_ID")
    AWS_SECRET_ACCESS_KEY: Optional[str] = os.getenv("AWS_SECRET_ACCESS_KEY")
    AWS_SESSION_TOKEN: Optional[str] = os.getenv("AWS_SESSION_TOKEN")

    BEDROCK_EMBED_MODEL: str = os.getenv("BEDROCK_EMBED_MODEL", "amazon.titan-embed-text-v2:0")
    #BEDROCK_EMBED_MODEL: str = os.getenv("BEDROCK_EMBED_MODEL", "mazon.titan-embed-text-v1")
    BEDROCK_LLM_MODEL: str = os.getenv("BEDROCK_LLM_MODEL", "mistral.mistral-7b-instruct-v0:2")

    # Chunking / batching
    MAX_CHARS: int = int(os.getenv("MAX_CHARS", "4000"))
    OVERLAP: int = int(os.getenv("OVERLAP", "300"))
    BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", "10"))

    # API server
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    # Security (if API_KEY is set, you MUST send X-API-Key header)
    API_KEY: Optional[str] = os.getenv("API_KEY")

    # UI settings
    UI_API_KEY: Optional[str] = os.getenv("UI_API_KEY")
    API_BASE: str = os.getenv("API_BASE", "http://localhost:8000")
    REQUEST_TIMEOUT: int = int(os.getenv("REQUEST_TIMEOUT", "30"))

    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )


settings = Settings()

# Create required directories
settings.UPLOAD_DIR.mkdir(exist_ok=True, parents=True)
Path(settings.CHROMA_PERSIST_DIR).mkdir(exist_ok=True, parents=True)
