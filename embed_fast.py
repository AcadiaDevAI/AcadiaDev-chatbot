


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