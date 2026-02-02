import os
from typing import Any

import chromadb

from config import settings


def get_collection() -> Any:
    """Return a persistent Chroma collection."""
    chroma_path = settings.CHROMA_PERSIST_DIR
    os.makedirs(chroma_path, exist_ok=True)

    client = chromadb.PersistentClient(
        path=chroma_path,
        settings=chromadb.Settings(
            anonymized_telemetry=False,
            allow_reset=True,
        ),
    )

    return client.get_or_create_collection(
        name=settings.COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
