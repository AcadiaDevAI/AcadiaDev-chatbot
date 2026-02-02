# import sys
# from pathlib import Path
# from typing import List, Optional

# from pydantic_settings import BaseSettings, SettingsConfigDict


# # Minimal supported version
# if sys.version_info < (3, 11):
#     raise RuntimeError("This application requires Python 3.11 or higher")


# BASE_DIR = Path(__file__).resolve().parent


# class Settings(BaseSettings):
#     # Storage / Chroma
#     CHROMA_PERSIST_DIR: str = str(BASE_DIR / "data" / "chroma")
#     COLLECTION_NAME: str = "logs_titan_v2_1024"

#     # Uploads
#     UPLOAD_DIR: Path = BASE_DIR / "uploads"
#     MAX_FILE_SIZE_MB: int = 100
#     ALLOWED_FILE_TYPES: List[str] = ["log", "txt", "md", "json"]

#     # AWS / Bedrock
#     AWS_REGION: str = "us-east-1"

#     # Local-only explicit keys.
#     # For EC2, leave these unset and use an IAM role instead.
#     AWS_ACCESS_KEY_ID: Optional[str] = None
#     AWS_SECRET_ACCESS_KEY: Optional[str] = None
#     AWS_SESSION_TOKEN: Optional[str] = None

#     BEDROCK_EMBED_MODEL: str = "amazon.titan-embed-text-v2:0"
#     BEDROCK_LLM_MODEL: str = "mistral.mistral-7b-instruct-v0:2"

#     # Chunking / batching
#     MAX_CHARS: int = 4000
#     OVERLAP: int = 300
#     BATCH_SIZE: int = 10

#     # API server
#     HOST: str = "0.0.0.0"
#     PORT: int = 8000
#     LOG_LEVEL: str = "INFO"

#     # Security (if API_KEY is set, every request must send X-API-Key header)
#     API_KEY: Optional[str] = None

#     # UI settings
#     UI_API_KEY: Optional[str] = None
#     API_BASE: str = "http://localhost:8000"
#     REQUEST_TIMEOUT: int = 30

#     model_config = SettingsConfigDict(
#         env_file=str(BASE_DIR / ".env"),
#         env_file_encoding="utf-8",
#         case_sensitive=True,
#         extra="ignore",
#     )


# settings = Settings()

# # Create required directories
# settings.UPLOAD_DIR.mkdir(exist_ok=True, parents=True)
# Path(settings.CHROMA_PERSIST_DIR).mkdir(exist_ok=True, parents=True)


import sys
from pathlib import Path
from typing import List, Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


if sys.version_info < (3, 11):
    raise RuntimeError("This application requires Python 3.11 or higher")

BASE_DIR = Path(__file__).resolve().parent


class Settings(BaseSettings):
    CHROMA_PERSIST_DIR: str = str(BASE_DIR / "data" / "chroma")
    COLLECTION_NAME: str = "logs_titan_v2_1024"

    UPLOAD_DIR: Path = BASE_DIR / "uploads"
    MAX_FILE_SIZE_MB: int = 100
    ALLOWED_FILE_TYPES: List[str] = ["log", "txt", "md", "json"]

    AWS_REGION: str = "us-east-1"
    AWS_ACCESS_KEY_ID: Optional[str] = None
    AWS_SECRET_ACCESS_KEY: Optional[str] = None
    AWS_SESSION_TOKEN: Optional[str] = None

    BEDROCK_EMBED_MODEL: str = "amazon.titan-embed-text-v2:0"
    BEDROCK_LLM_MODEL: str = "mistral.mistral-7b-instruct-v0:2"

    MAX_CHARS: int = 4000
    OVERLAP: int = 300
    BATCH_SIZE: int = 10

    HOST: str = "0.0.0.0"
    PORT: int = 8000
    LOG_LEVEL: str = "INFO"

    API_KEY: Optional[str] = None

    UI_API_KEY: Optional[str] = None
    API_BASE: str = "http://localhost:8000"
    REQUEST_TIMEOUT: int = 30

    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )


settings = Settings()

# Backward-compatible exports (fixes ImportError in api.py)
AWS_REGION = settings.AWS_REGION
BEDROCK_EMBED_MODEL = settings.BEDROCK_EMBED_MODEL
BEDROCK_LLM_MODEL = settings.BEDROCK_LLM_MODEL
CHROMA_PERSIST_DIR = settings.CHROMA_PERSIST_DIR
COLLECTION_NAME = settings.COLLECTION_NAME

# Create required directories
settings.UPLOAD_DIR.mkdir(exist_ok=True, parents=True)
Path(settings.CHROMA_PERSIST_DIR).mkdir(exist_ok=True, parents=True)
