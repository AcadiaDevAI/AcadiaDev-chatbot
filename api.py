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
    CHROMA_DIR,
    COLLECTION_NAME,
)
from vector_store import get_collection

app = FastAPI()

# ------------------------------------------------------------------------------
# Bedrock client configuration (increase retries; we also do our own backoff)
# ------------------------------------------------------------------------------
AWS_PROFILE = os.getenv("AWS_PROFILE")  # optional (local only)

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
# Tunables (you can adjust later)
# ------------------------------------------------------------------------------
# Add a small delay between embedding calls to reduce throttling
EMBED_DELAY_SECONDS = float(os.getenv("EMBED_DELAY_SECONDS", "0.10"))  # 100ms default

# Backoff controls for throttling
MAX_THROTTLE_RETRIES = int(os.getenv("MAX_THROTTLE_RETRIES", "12"))
BACKOFF_BASE_SECONDS = float(os.getenv("BACKOFF_BASE_SECONDS", "0.5"))  # start at 0.5s
BACKOFF_MAX_SECONDS = float(os.getenv("BACKOFF_MAX_SECONDS", "20"))     # cap wait


def _is_throttle_error(err: Exception) -> bool:
    if not isinstance(err, ClientError):
        return False
    code = err.response.get("Error", {}).get("Code", "")
    return code in ("ThrottlingException", "TooManyRequestsException", "ProvisionedThroughputExceededException")


def _invoke_bedrock(model_id: str, payload: dict) -> Any:
    """
    Invoke a Bedrock model with strong throttling protection.
    """
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
            # Handle throttling with exponential backoff
            if _is_throttle_error(e) and attempt < MAX_THROTTLE_RETRIES:
                wait = min(BACKOFF_MAX_SECONDS, BACKOFF_BASE_SECONDS * (2 ** attempt))
                attempt += 1
                time.sleep(wait)
                continue

            # Non-throttle error or exceeded retries
            raise


def bedrock_embed(text: str) -> List[float]:
    out = _invoke_bedrock(BEDROCK_EMBED_MODEL, {"inputText": text})

    if isinstance(out, dict):
        if "embedding" in out:
            return out["embedding"]
        if "embeddings" in out and isinstance(out["embeddings"], list) and out["embeddings"]:
            return out["embeddings"][0]
        if "outputs" in out and isinstance(out["outputs"], list) and out["outputs"]:
            o = out["outputs"][0]
            if "embedding" in o:
                return o["embedding"]
            if "embeddings" in o and o["embeddings"]:
                return o["embeddings"][0]

    raise ValueError("Unrecognized embedding response from Bedrock")


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

    raise ValueError("Unrecognized generation response from Bedrock")


# ------------------------------------------------------------------------------
# Chroma setup
# ------------------------------------------------------------------------------
coll = get_collection()

# ------------------------------------------------------------------------------
# Job tracking
# ------------------------------------------------------------------------------
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


# def iter_chunks_from_file(file_path: str, max_chars: int = 1500, overlap: int = 200) -> Iterator[str]:
#     """
#     Stream chunks from a file WITHOUT loading entire file into RAM.
#     """
#     buf = ""
#     with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
#         for line in f:
#             buf += line

#             while len(buf) >= max_chars:
#                 chunk = buf[:max_chars].strip()
#                 if chunk:
#                     yield chunk
#                 buf = buf[max_chars - overlap :]

#     tail = buf.strip()
#     if tail:
#         yield tail

def iter_chunks_from_file(file_path: str, max_chars: int = 4000, overlap: int = 300):

    """
    Stream chunks from a file WITHOUT loading entire file into RAM.
    Larger chunks = fewer Bedrock embedding calls = much faster indexing.
    """
    buf = ""
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            buf += line

            while len(buf) >= max_chars:
                chunk = buf[:max_chars].strip()
                if chunk:
                    yield chunk

                # keep overlap tail
                buf = buf[max_chars - overlap :]

    tail = buf.strip()
    if tail:
        yield tail



def index_file_job(job_id: str, file_path: str, filename: str) -> None:
    """
    Background indexing job with throttling-safe embedding.
    """
    try:
        UPLOAD_JOBS[job_id]["status"] = "running"
        UPLOAD_JOBS[job_id]["message"] = "Indexing started"
        UPLOAD_JOBS[job_id]["processed_chunks"] = 0

        batch_ids: List[str] = []
        batch_metas: List[Dict[str, Any]] = []
        batch_docs: List[str] = []
        batch_embs: List[List[float]] = []

        total = 0
        for ch in iter_chunks_from_file(file_path, max_chars=4000, overlap=300):

            total += 1
            UPLOAD_JOBS[job_id]["processed_chunks"] = total
            UPLOAD_JOBS[job_id]["message"] = f"Embedding chunk #{total}"

            emb = bedrock_embed(ch)

            # small delay to reduce throttling
            if EMBED_DELAY_SECONDS > 0:
                time.sleep(EMBED_DELAY_SECONDS)

            batch_ids.append(f"{filename}-{job_id}-{total}")
            batch_metas.append({"source": filename, "device": filename})
            batch_docs.append(ch)
            batch_embs.append(emb)

            # write to Chroma in batches
            if len(batch_ids) >= 200:
                coll.add(ids=batch_ids, metadatas=batch_metas, documents=batch_docs, embeddings=batch_embs)
                batch_ids, batch_metas, batch_docs, batch_embs = [], [], [], []

        if batch_ids:
            coll.add(ids=batch_ids, metadatas=batch_metas, documents=batch_docs, embeddings=batch_embs)

        UPLOAD_JOBS[job_id]["status"] = "done"
        UPLOAD_JOBS[job_id]["message"] = f"Indexing completed. Stored chunks: {total}"
        UPLOAD_JOBS[job_id]["total_chunks"] = total

    except Exception as e:
        traceback.print_exc()
        UPLOAD_JOBS[job_id]["status"] = "failed"
        UPLOAD_JOBS[job_id]["message"] = f"Indexing error: {str(e)}"


@app.get("/debug_paths")
def debug_paths():
    return {"CHROMA_DIR": CHROMA_DIR, "COLLECTION_NAME": COLLECTION_NAME}


@app.get("/stats")
def stats():
    return {"collection": coll.name, "count": coll.count()}


@app.get("/sources")
def sources():
    metas = coll.get(include=["metadatas"]).get("metadatas") or []
    files = sorted({m.get("source") for m in metas if isinstance(m, dict) and m.get("source")})
    return {"sources": files, "count": len(files)}


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

        UPLOAD_JOBS[job_id] = {
            "job_id": job_id,
            "file": safe_name,
            "status": "queued",
            "message": "Queued for indexing",
            "total_chunks": None,
            "processed_chunks": 0,
            "created_utc": time.time(),
        }

        background_tasks.add_task(index_file_job, job_id, file_path, safe_name)

        return {
            "status": "accepted",
            "job_id": job_id,
            "file": safe_name,
            "message": "Upload received. Indexing started in background.",
        }

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask")
def ask(req: Q):
    start = time.time()
    try:
        emb = bedrock_embed(req.q)

        where: Dict[str, Any] = {}
        if req.device:
            where["device"] = req.device
        if req.source:
            where["source"] = req.source
        if not req.source:
            fname = _extract_filename(req.q)
            if fname:
                where["source"] = fname

        docs = coll.query(
            query_embeddings=[emb],
            n_results=30,
            where=where or None,
            include=["documents", "metadatas"],
        )

        if not docs.get("documents") or not docs["documents"][0]:
            return {
                "answer": "No relevant logs found for this query.",
                "sources": [],
                "timing": {"duration_ms": int((time.time() - start) * 1000)},
            }

        context = "\n".join(docs["documents"][0])
        prompt = f"{SYS_PROMPT}\n\nLogs:\n{context}\n\nQ: {req.q}\nA:"
        ans = bedrock_generate(prompt)

        return {
            "answer": ans,
            "sources": docs["metadatas"][0],
            "timing": {"duration_ms": int((time.time() - start) * 1000)},
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
