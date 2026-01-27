# """
# embed_fast.py – Batch embedding/indexing for parquet logs (Bedrock Titan)
# Uses ONE persistent Chroma DB + ONE collection (from config/vector_store).
# """
# import glob
# import os
# import uuid
# import json
# from pathlib import Path
# from typing import Any

# import boto3
# import pyarrow.parquet as pq

# from config import AWS_REGION, BEDROCK_EMBED_MODEL
# from vector_store import get_collection

# AWS_PROFILE = os.getenv("AWS_PROFILE")  # optional

# if AWS_PROFILE:
#     session = boto3.Session(profile_name=AWS_PROFILE, region_name=AWS_REGION)
# else:
#     session = boto3.Session(region_name=AWS_REGION)

# _bedrock = session.client("bedrock-runtime")


# def _invoke_bedrock(model_id: str, payload: dict) -> Any:
#     body = json.dumps(payload)
#     resp = _bedrock.invoke_model(modelId=model_id, contentType="application/json", accept="application/json", body=body)
#     raw = resp["body"].read().decode("utf-8")
#     try:
#         return json.loads(raw)
#     except Exception:
#         return raw


# def embed_one(text: str) -> list[float]:
#     out = _invoke_bedrock(BEDROCK_EMBED_MODEL, {"inputText": text})
#     if isinstance(out, dict):
#         if "embedding" in out:
#             return out["embedding"]
#         if "embeddings" in out and out["embeddings"]:
#             return out["embeddings"][0]
#     raise ValueError("Could not parse embedding from Bedrock response")


# def chunk(text: str, size: int = 350):
#     words = (text or "").split()
#     for i in range(0, len(words), size):
#         yield " ".join(words[i : i + size])


# MAX_CHROMA = 5000
# MOVE_DONE = True
# DONE_DIR = Path(r"data\done")
# if MOVE_DONE:
#     DONE_DIR.mkdir(exist_ok=True, parents=True)

# coll = get_collection()

# for pq_path in glob.glob(r"data\*.parquet"):
#     pq_path = Path(pq_path)
#     filename = pq_path.stem

#     if coll.get(where={"source": filename}, limit=1).get("ids"):
#         print(f"skip  {filename}  – already in collection")
#         if MOVE_DONE:
#             pq_path.rename(DONE_DIR / pq_path.name)
#         continue

#     tbl = pq.read_table(pq_path)
#     docs, meta, ids, embs = [], [], [], []

#     for batch in tbl.to_batches(50000):
#         for r in batch.to_pandas().itertuples(index=False):
#             text = getattr(r, "msg", "") or getattr(r, "text", "") or ""
#             for snip in chunk(text, 350):
#                 docs.append(snip)
#                 meta.append({"source": filename, "device": filename})
#                 ids.append(str(uuid.uuid4()))

#     for d in docs:
#         embs.append(embed_one(d))

#     for i in range(0, len(embs), MAX_CHROMA):
#         sl = slice(i, i + MAX_CHROMA)
#         coll.add(
#             embeddings=embs[sl],
#             documents=docs[sl],
#             metadatas=meta[sl],
#             ids=ids[sl],
#         )

#     print(f"embedded {filename} – vectors: {len(embs):,}")

#     if MOVE_DONE:
#         pq_path.rename(DONE_DIR / pq_path.name)

# print("Total chunks in Chroma:", coll.count())


import os
import glob
import uuid
import json
import boto3
import pyarrow.parquet as pq
from concurrent.futures import ThreadPoolExecutor
from botocore.config import Config
from config import AWS_REGION, BEDROCK_EMBED_MODEL, BATCH_SIZE
from vector_store import get_collection

# Optimized Boto3 Client for high-concurrency
boto_cfg = Config(max_pool_connections=50, retries={'max_attempts': 5})
_bedrock = boto3.client("bedrock-runtime", region_name=AWS_REGION, config=boto_cfg)

coll = get_collection()

def embed_one(text):
    """Single worker function for the thread pool."""
    try:
        body = json.dumps({"inputText": text[:4000]}) # Titan limit safety
        resp = _bedrock.invoke_model(modelId=BEDROCK_EMBED_MODEL, body=body)
        return json.loads(resp["body"].read().decode("utf-8"))["embedding"]
    except Exception as e:
        print(f"Embedding failed: {e}")
        return None

def process_parquet_file(pq_path):
    print(f"Processing {pq_path}...")
    table = pq.read_table(pq_path)
    df = table.to_pandas()
    
    # We use ThreadPoolExecutor to make 20 calls at once
    with ThreadPoolExecutor(max_workers=20) as executor:
        embeddings = list(executor.map(embed_one, df['raw_text'].tolist()))

    # Filter out any failed embeddings
    valid_data = [(txt, emb) for txt, emb in zip(df['raw_text'], embeddings) if emb is not None]
    
    if valid_data:
        texts, embs = zip(*valid_data)
        ids = [str(uuid.uuid4()) for _ in texts]
        metas = [{"source": os.path.basename(pq_path)} for _ in texts]
        
        # Batch add to Chroma (Chroma handles up to 5k at once)
        coll.add(ids=ids, embeddings=list(embs), metadatas=metas, documents=list(texts))
        print(f"  Successfully indexed {len(ids)} vectors.")

if __name__ == "__main__":
    for file in glob.glob("data/*.parquet"):
        process_parquet_file(file)