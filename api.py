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
EMBED_DELAY_SECONDS = float(os.getenv("EMBED_DELAY_SECONDS", "0.10"))
MAX_THROTTLE_RETRIES = int(os.getenv("MAX_THROTTLE_RETRIES", "12"))
BACKOFF_BASE_SECONDS = float(os.getenv("BACKOFF_BASE_SECONDS", "0.5"))
BACKOFF_MAX_SECONDS = float(os.getenv("BACKOFF_MAX_SECONDS", "20"))


def _is_throttle_error(err: Exception) -> bool:
    if not isinstance(err, ClientError):
        return False
    code = err.response.get("Error", {}).get("Code", "")
    return code in ("ThrottlingException", "TooManyRequestsException", "ProvisionedThroughputExceededException")


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
        if "embedding" in out: return out["embedding"]
        if "embeddings" in out and out["embeddings"]: return out["embeddings"][0]
        if "outputs" in out and out["outputs"]:
            o = out["outputs"][0]
            if "embedding" in o: return o["embedding"]
            if "embeddings" in o and o["embeddings"]: return o["embeddings"][0]
    raise ValueError("Unrecognized embedding response")


def bedrock_generate(prompt: str, max_tokens: int = 512) -> str:
    out = _invoke_bedrock(BEDROCK_LLM_MODEL, {"prompt": prompt, "max_tokens": max_tokens})
    if isinstance(out, dict):
        if isinstance(out.get("output"), str): return out["output"]
        if isinstance(out.get("generated_text"), str): return out["generated_text"]
        if isinstance(out.get("outputs"), list) and out["outputs"]:
            o = out["outputs"][0]
            for k in ("text", "content", "generated_text", "output"):
                v = o.get(k)
                if isinstance(v, str) and v.strip(): return v
    if isinstance(out, str): return out
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

SYS_PROMPT = (
    "You are a Cisco TAC engineer. "
    "Answer the question using ONLY the provided log fragments. "
    "If the logs do not contain enough evidence, say what is missing."
)

def _extract_filename(question: str) -> str | None:
    m = re.search(r"\b(\S+\.(?:log|txt|csv|json|md))\b", question, flags=re.I)
    return m.group(1) if m else None

def build_smart_context(documents: list[str], max_chars: int = 75000) -> str:
    """Combines documents until character limit is reached to avoid LLM overflow."""
    context_parts = []
    current_length = 0
    for doc in documents:
        # Add buffer of 50 chars for the separator
        if current_length + len(doc) + 50 > max_chars:
            break
        context_parts.append(doc)
        current_length += len(doc)
    return "\n\n---\n\n".join(context_parts)

# ------------------------------------------------------------------------------
# Core Endpoints
# ------------------------------------------------------------------------------

def iter_chunks_from_file(file_path: str, max_chars: int = 4000, overlap: int = 300):
    buf = ""
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            buf += line
            while len(buf) >= max_chars:
                chunk = buf[:max_chars].strip()
                if chunk: yield chunk
                buf = buf[max_chars - overlap :]
    tail = buf.strip()
    if tail: yield tail

def index_file_job(job_id: str, file_path: str, filename: str) -> None:
    try:
        UPLOAD_JOBS[job_id]["status"] = "running"
        batch_ids, batch_metas, batch_docs, batch_embs = [], [], [], []
        total = 0
        for ch in iter_chunks_from_file(file_path):
            total += 1
            UPLOAD_JOBS[job_id]["processed_chunks"] = total
            emb = bedrock_embed(ch)
            if EMBED_DELAY_SECONDS > 0: time.sleep(EMBED_DELAY_SECONDS)
            
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
    except Exception as e:
        traceback.print_exc()
        UPLOAD_JOBS[job_id]["status"] = "failed"
        UPLOAD_JOBS[job_id]["message"] = str(e)

@app.get("/sources")
def sources():
    metas = coll.get(include=["metadatas"]).get("metadatas") or []
    files = sorted({m.get("source") for m in metas if isinstance(m, dict) and m.get("source")})
    return {"sources": files, "count": len(files)}

@app.post("/upload")
async def upload_file(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    content = await file.read()
    uploads_dir = Path("uploads")
    uploads_dir.mkdir(parents=True, exist_ok=True)
    job_id = uuid.uuid4().hex
    safe_name = Path(file.filename).name
    file_path = str(uploads_dir / f"{job_id}__{safe_name}")
    Path(file_path).write_bytes(content)
    UPLOAD_JOBS[job_id] = {"status": "queued", "processed_chunks": 0}
    background_tasks.add_task(index_file_job, job_id, file_path, safe_name)
    return {"status": "accepted", "job_id": job_id}

@app.post("/ask")
def ask(req: Q):
    start = time.time()
    try:
        emb = bedrock_embed(req.q)
        where = {}
        if req.device: where["device"] = req.device
        if req.source: where["source"] = req.source
        elif _extract_filename(req.q): where["source"] = _extract_filename(req.q)

        docs = coll.query(
            query_embeddings=[emb],
            n_results=30,
            where=where or None,
            include=["documents", "metadatas"],
        )

        if not docs.get("documents") or not docs["documents"][0]:
            return {"answer": "No relevant logs found.", "sources": []}

        # --- SMART CONTEXT TRUNCATION ---
        context = build_smart_context(docs["documents"][0], max_chars=75000)
        prompt = f"{SYS_PROMPT}\n\nLogs:\n{context}\n\nQ: {req.q}\nA:"
        # --------------------------------

        ans = bedrock_generate(prompt)
        return {
            "answer": ans,
            "sources": docs["metadatas"][0][:len(context.split("---"))], # match sources to context used
            "timing": {"duration_ms": int((time.time() - start) * 1000)},
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))