
import chromadb
from config import CHROMA_PERSIST_DIR, COLLECTION_NAME

def get_collection():
    """
    Returns the persistent Chroma collection. 
    Using a persistent client prevents data loss between restarts.
    """
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    return client.get_or_create_collection(name=COLLECTION_NAME)