#multiple files functionality

import os
import re
import time
import uuid
import json
import traceback
from pathlib import Path
from typing import List, Dict, Any, Optional, Iterator

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from pydantic import BaseModel
import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

from config import (
    AWS_REGION,
    BEDROCK_EMBED_MODEL,
    BEDROCK_LLM_MODEL,
    CHROMA_PERSIST_DIR,
    COLLECTION_NAME,
)
from vector_store import get_collection

app = FastAPI()

# ------------------------------------------------------------------------------
# Bedrock client configuration
# ------------------------------------------------------------------------------
AWS_PROFILE = os.getenv("AWS_PROFILE")

boto_cfg = Config(
    retries={"max_attempts": 10, "mode": "standard"},
    read_timeout=120,
    connect_timeout=30,
)

if AWS_PROFILE:
    session = boto3.Session(profile_name=AWS_PROFILE, region_name=AWS_REGION)
else:
    session = boto3.Session(region_name=AWS_REGION)

_bedrock = session.client("bedrock-runtime", config=boto_cfg)

# ------------------------------------------------------------------------------
# Tunables
# ------------------------------------------------------------------------------
# CHANGE #1: Reduced embed delay for faster indexing (was 0.10, now 0.05)
EMBED_DELAY_SECONDS = float(os.getenv("EMBED_DELAY_SECONDS", "0.05"))
MAX_THROTTLE_RETRIES = int(os.getenv("MAX_THROTTLE_RETRIES", "12"))
BACKOFF_BASE_SECONDS = float(os.getenv("BACKOFF_BASE_SECONDS", "0.5"))
BACKOFF_MAX_SECONDS = float(os.getenv("BACKOFF_MAX_SECONDS", "20"))

# Context / prompt controls
# CHANGE #2: Increased results to search across more files (was 6, now 10)
CHROMA_N_RESULTS = int(os.getenv("CHROMA_N_RESULTS", "10"))
MAX_TOTAL_CONTEXT_CHARS = int(os.getenv("MAX_TOTAL_CONTEXT_CHARS", "20000"))
# CHANGE #3: Allow more chunks for multi-file correlation (was 6, now 10)
MAX_CONTEXT_CHUNKS = int(os.getenv("MAX_CONTEXT_CHUNKS", "10"))
MAX_CHUNK_CHARS = int(os.getenv("MAX_CHUNK_CHARS", "1800"))

RETRY_MAX_TOTAL_CONTEXT_CHARS = int(os.getenv("RETRY_MAX_TOTAL_CONTEXT_CHARS", "9000"))
RETRY_MAX_CONTEXT_CHUNKS = int(os.getenv("RETRY_MAX_CONTEXT_CHUNKS", "3"))
RETRY_MAX_CHUNK_CHARS = int(os.getenv("RETRY_MAX_CHUNK_CHARS", "1200"))


def _is_throttle_error(err: Exception) -> bool:
    if not isinstance(err, ClientError):
        return False
    code = err.response.get("Error", {}).get("Code", "")
    return code in (
        "ThrottlingException",
        "TooManyRequestsException",
        "ProvisionedThroughputExceededException",
    )


def _invoke_bedrock(model_id: str, payload: dict) -> Any:
    body = json.dumps(payload)
    attempt = 0
    while True:
        try:
            resp = _bedrock.invoke_model(
                modelId=model_id,
                contentType="application/json",
                accept="application/json",
                body=body,
            )
            raw = resp["body"].read().decode("utf-8")
            try:
                return json.loads(raw)
            except Exception:
                return raw

        except ClientError as e:
            if _is_throttle_error(e) and attempt < MAX_THROTTLE_RETRIES:
                wait = min(BACKOFF_MAX_SECONDS, BACKOFF_BASE_SECONDS * (2 ** attempt))
                attempt += 1
                time.sleep(wait)
                continue
            raise


def bedrock_embed(text: str) -> List[float]:
    out = _invoke_bedrock(BEDROCK_EMBED_MODEL, {"inputText": text})
    if isinstance(out, dict):
        if "embedding" in out:
            return out["embedding"]
        if "embeddings" in out and out["embeddings"]:
            return out["embeddings"][0]
        if "outputs" in out and out["outputs"]:
            o = out["outputs"][0]
            if "embedding" in o:
                return o["embedding"]
            if "embeddings" in o and o["embeddings"]:
                return o["embeddings"][0]
    raise ValueError("Unrecognized embedding response")


def bedrock_generate(prompt: str, max_tokens: int = 512) -> str:
    out = _invoke_bedrock(BEDROCK_LLM_MODEL, {"prompt": prompt, "max_tokens": max_tokens})
    if isinstance(out, dict):
        if isinstance(out.get("output"), str):
            return out["output"]
        if isinstance(out.get("generated_text"), str):
            return out["generated_text"]
        if isinstance(out.get("outputs"), list) and out["outputs"]:
            o = out["outputs"][0]
            for k in ("text", "content", "generated_text", "output"):
                v = o.get(k)
                if isinstance(v, str) and v.strip():
                    return v
    if isinstance(out, str):
        return out
    raise ValueError("Unrecognized generation response")


# ------------------------------------------------------------------------------
# Chroma & Helpers
# ------------------------------------------------------------------------------
coll = get_collection()
UPLOAD_JOBS: Dict[str, Dict[str, Any]] = {}


class Q(BaseModel):
    q: str
    device: Optional[str] = None
    source: Optional[str] = None


# CHANGE #4: Updated system prompt to support multi-file correlation
SYS_PROMPT = (
    "You are a Cisco TAC engineer analyzing logs from multiple devices. "
    "Answer the question using the provided log fragments from various sources. "
    "When logs from different devices show related events, identify the correlation. "
    "Cite which device/source each piece of evidence comes from. "
    "If the logs do not contain enough evidence, say what is missing."
)


def _extract_filename(question: str) -> str | None:
    m = re.search(r"\b(\S+\.(?:log|txt|csv|json|md))\b", question, flags=re.I)
    return m.group(1) if m else None


# CHANGE #5: Enhanced context builder to show file sources clearly
def build_smart_context(
    documents: list[str],
    metadatas: list[dict],
    max_chars: int = MAX_TOTAL_CONTEXT_CHARS,
    max_doc_chars: int = MAX_CHUNK_CHARS,
    max_docs: int = MAX_CONTEXT_CHUNKS,
) -> str:
    """
    Combines top documents until character limit is reached.
    Now includes source information for multi-file correlation.
    """
    context_parts: List[str] = []
    current_length = 0

    for idx, doc in enumerate((documents or [])[:max_docs]):
        doc = (doc or "").strip()
        if not doc:
            continue

        # Get source info from metadata
        meta = metadatas[idx] if idx < len(metadatas) else {}
        source = meta.get("source", "unknown")
        device = meta.get("device", source)

        if len(doc) > max_doc_chars:
            doc = doc[:max_doc_chars] + "…"

        # Add source header for context
        doc_with_source = f"[Source: {source}]\n{doc}"
        add_len = len(doc_with_source) + 50

        if current_length + add_len > max_chars:
            break

        context_parts.append(doc_with_source)
        current_length += add_len

    return "\n\n---\n\n".join(context_parts)


# ------------------------------------------------------------------------------
# Core indexing helpers + endpoints
# ------------------------------------------------------------------------------
def iter_chunks_from_file(file_path: str, max_chars: int = 4000, overlap: int = 300) -> Iterator[str]:
    buf = ""
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            buf += line
            while len(buf) >= max_chars:
                chunk = buf[:max_chars].strip()
                if chunk:
                    yield chunk
                buf = buf[max_chars - overlap :]
    tail = buf.strip()
    if tail:
        yield tail


def clear_collection() -> None:
    """
    Deletes ALL data from the collection.
    Use the /clear endpoint to start fresh with no files.
    """
    try:
        all_data = coll.get(include=[])
        all_ids = all_data.get("ids", [])
        
        if all_ids:
            batch_size = 5000
            for i in range(0, len(all_ids), batch_size):
                batch = all_ids[i:i + batch_size]
                coll.delete(ids=batch)
            print(f"Cleared {len(all_ids)} vectors from collection")
    except Exception as e:
        print(f"Error clearing collection: {e}")
        raise


# CHANGE #6: New function to delete specific file
def delete_file_from_collection(filename: str) -> int:
    """
    Deletes all chunks from a specific file.
    Returns the number of chunks deleted.
    """
    try:
        results = coll.get(
            where={"source": filename},
            include=[]
        )
        ids_to_delete = results.get("ids", [])
        
        if ids_to_delete:
            batch_size = 5000
            for i in range(0, len(ids_to_delete), batch_size):
                batch = ids_to_delete[i:i + batch_size]
                coll.delete(ids=batch)
            print(f"Deleted {len(ids_to_delete)} vectors from {filename}")
            return len(ids_to_delete)
        return 0
    except Exception as e:
        print(f"Error deleting {filename}: {e}")
        raise


# CHANGE #7: REMOVED clear_collection() call - now keeps all files!
def index_file_job(job_id: str, file_path: str, filename: str) -> None:
    try:
        UPLOAD_JOBS[job_id]["status"] = "running"
        
        # Check if this file already exists and delete old version
        existing = coll.get(where={"source": filename}, limit=1).get("ids")
        if existing:
            print(f"File {filename} already exists, replacing...")
            delete_file_from_collection(filename)
        
        batch_ids: List[str] = []
        batch_metas: List[Dict[str, Any]] = []
        batch_docs: List[str] = []
        batch_embs: List[List[float]] = []
        total = 0

        for ch in iter_chunks_from_file(file_path):
            total += 1
            UPLOAD_JOBS[job_id]["processed_chunks"] = total

            emb = bedrock_embed(ch)
            if EMBED_DELAY_SECONDS > 0:
                time.sleep(EMBED_DELAY_SECONDS)

            batch_ids.append(f"{filename}-{job_id}-{total}")
            batch_metas.append({"source": filename, "device": filename})
            batch_docs.append(ch)
            batch_embs.append(emb)

            if len(batch_ids) >= 200:
                coll.add(ids=batch_ids, metadatas=batch_metas, documents=batch_docs, embeddings=batch_embs)
                batch_ids, batch_metas, batch_docs, batch_embs = [], [], [], []

        if batch_ids:
            coll.add(ids=batch_ids, metadatas=batch_metas, documents=batch_docs, embeddings=batch_embs)

        UPLOAD_JOBS[job_id]["status"] = "done"
        print(f"Successfully indexed {filename} with {total} chunks")

    except Exception as e:
        traceback.print_exc()
        UPLOAD_JOBS[job_id]["status"] = "failed"
        UPLOAD_JOBS[job_id]["message"] = str(e)


@app.get("/sources")
def sources():
    """
    CHANGE #8: Enhanced to show chunk counts per file
    """
    metas = coll.get(include=["metadatas"]).get("metadatas") or []
    
    # Count chunks per source
    source_counts = {}
    for m in metas:
        if isinstance(m, dict) and m.get("source"):
            source = m["source"]
            source_counts[source] = source_counts.get(source, 0) + 1
    
    files = sorted(source_counts.keys())
    return {
        "sources": files,
        "count": len(files),
        "details": source_counts,
        "total_chunks": coll.count()
    }


@app.post("/clear")
def clear_all_data():
    """
    CHANGE #9: Added warning about clearing all files
    """
    try:
        count = coll.count()
        clear_collection()
        return {
            "status": "success", 
            "message": f"Cleared all data from collection ({count} chunks removed)"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# CHANGE #10: New endpoint to delete specific file
@app.delete("/sources/{filename}")
def delete_source(filename: str):
    """
    Delete a specific file from the collection.
    Allows removing individual files while keeping others.
    """
    try:
        count = delete_file_from_collection(filename)
        if count == 0:
            raise HTTPException(status_code=404, detail=f"File '{filename}' not found")
        return {
            "status": "success",
            "file": filename,
            "deleted_chunks": count
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/upload_status/{job_id}")
def upload_status(job_id: str):
    job = UPLOAD_JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job_id not found")
    return job


@app.post("/upload")
async def upload_file(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    try:
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Empty file")

        uploads_dir = Path("uploads")
        uploads_dir.mkdir(parents=True, exist_ok=True)

        job_id = uuid.uuid4().hex
        safe_name = Path(file.filename).name
        file_path = str(uploads_dir / f"{job_id}__{safe_name}")

        Path(file_path).write_bytes(content)

        UPLOAD_JOBS[job_id] = {"status": "queued", "processed_chunks": 0, "file": safe_name}
        background_tasks.add_task(index_file_job, job_id, file_path, safe_name)

        return {"status": "accepted", "job_id": job_id, "file": safe_name}

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# CHANGE #11: Enhanced /ask endpoint for multi-file correlation
@app.post("/ask")
def ask(req: Q):
    start = time.time()
    try:
        emb = bedrock_embed(req.q)

        # CHANGE #12: Don't filter by source unless explicitly requested
        where: Dict[str, Any] = {}
        if req.device:
            where["device"] = req.device
        if req.source:
            where["source"] = req.source
        # Removed automatic filename extraction - search ALL files by default

        docs = coll.query(
            query_embeddings=[emb],
            n_results=CHROMA_N_RESULTS,
            where=where or None,
            include=["documents", "metadatas"],
        )

        if not docs.get("documents") or not docs["documents"][0]:
            return {
                "answer": "No relevant logs found.",
                "sources": [],
                "timing": {"duration_ms": int((time.time() - start) * 1000)},
            }

        # CHANGE #13: Pass metadatas to context builder for source labels
        context = build_smart_context(
            docs["documents"][0],
            docs["metadatas"][0],
            max_chars=MAX_TOTAL_CONTEXT_CHARS,
            max_doc_chars=MAX_CHUNK_CHARS,
            max_docs=MAX_CONTEXT_CHUNKS,
        )
        
        # CHANGE #14: Enhanced prompt to encourage correlation
        prompt = (
            f"{SYS_PROMPT}\n\n"
            f"Logs from multiple devices:\n{context}\n\n"
            f"Question: {req.q}\n"
            f"Answer (cite sources):"
        )

        try:
            ans = bedrock_generate(prompt)
        except Exception as e:
            msg = str(e)
            if "maximum context length" in msg or "ValidationException" in msg:
                context = build_smart_context(
                    docs["documents"][0],
                    docs["metadatas"][0],
                    max_chars=RETRY_MAX_TOTAL_CONTEXT_CHARS,
                    max_doc_chars=RETRY_MAX_CHUNK_CHARS,
                    max_docs=RETRY_MAX_CONTEXT_CHUNKS,
                )
                prompt = (
                    f"{SYS_PROMPT}\n\n"
                    f"Logs:\n{context}\n\n"
                    f"Q: {req.q}\nA:"
                )
                ans = bedrock_generate(prompt)
            else:
                raise

        # CHANGE #15: Return unique sources for the answer
        unique_sources = list(set(m.get("source", "unknown") for m in docs["metadatas"][0]))
        
        return {
            "answer": ans,
            "sources": unique_sources,
            "sources_count": len(unique_sources),
            "timing": {"duration_ms": int((time.time() - start) * 1000)},
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
