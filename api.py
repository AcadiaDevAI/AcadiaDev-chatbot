import json
import logging
import hashlib
import sys
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, TypedDict

import boto3
from botocore.config import Config as BotoConfig
from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    File,
    Header,
    HTTPException,
    Query,
    Request,
    UploadFile,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from pydantic import BaseModel, ConfigDict, Field, field_validator

from config import settings
from vector_store import get_collection


# Minimal supported version
if sys.version_info < (3, 11):
    raise RuntimeError("This application requires Python 3.11 or higher")


# Logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("acadia-log-iq")


class JobInfo(TypedDict, total=False):
    job_id: str
    status: str
    processed_chunks: int
    total_chunks: int
    successful_chunks: int
    file: Optional[str]
    file_type: Optional[str]
    file_size_mb: float
    file_hash: str
    error: Optional[str]
    created_at: datetime
    completed_at: Optional[datetime]


UPLOAD_JOBS: Dict[str, JobInfo] = {}
coll = None  # set at startup


def _make_bedrock_client():
    """
    Create a Bedrock Runtime client.

    Credential resolution order (boto3 default chain):
    1) Env vars (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_SESSION_TOKEN)
    2) AWS_PROFILE / shared credentials (~/.aws/credentials or SSO config)
    3) EC2 Instance Role (IMDS)  <-- best for server
    """
    boto_cfg = BotoConfig(
        retries={"max_attempts": 10, "mode": "adaptive"},
        read_timeout=120,
        connect_timeout=30,
        tcp_keepalive=True,
    )

    kwargs = {
        "service_name": "bedrock-runtime",
        "region_name": settings.AWS_REGION,
        "config": boto_cfg,
    }

    # Only pass explicit credentials if provided (local dev)
    if settings.AWS_ACCESS_KEY_ID and settings.AWS_SECRET_ACCESS_KEY:
        kwargs["aws_access_key_id"] = settings.AWS_ACCESS_KEY_ID
        kwargs["aws_secret_access_key"] = settings.AWS_SECRET_ACCESS_KEY
        if settings.AWS_SESSION_TOKEN:
            kwargs["aws_session_token"] = settings.AWS_SESSION_TOKEN

    return boto3.client(**kwargs)


bedrock = _make_bedrock_client()


# Lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    global coll
    logger.info("Starting API…")
    coll = get_collection()
    logger.info("Chroma collection ready: %s", settings.COLLECTION_NAME)
    yield
    logger.info("Shutting down API…")


app = FastAPI(
    title="Acadia's Log IQ API",
    description="AI-powered log analysis and troubleshooting system",
    version="1.0.0",
    lifespan=lifespan,
)


# Rate limiting
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8501",
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Processing-Time"],
    max_age=600,
)


def verify_api_key(x_api_key: Optional[str] = Header(default=None, alias="X-API-Key")) -> bool:
    """
    If API_KEY is set in .env, every request must send:
      X-API-Key: <API_KEY>
    """
    if settings.API_KEY:
        if not x_api_key or x_api_key != settings.API_KEY:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or missing API key",
                headers={"WWW-Authenticate": "APIKey"},
            )
    return True


class Question(BaseModel):
    q: str = Field(min_length=1, max_length=1000)

    model_config = ConfigDict(extra="ignore")

    @field_validator("q")
    @classmethod
    def validate_q(cls, v: str) -> str:
        v = (v or "").strip()
        if not v:
            raise ValueError("Question cannot be empty")
        return v


class UploadResponse(BaseModel):
    job_id: str
    message: str
    file_hash: Optional[str] = None

    model_config = ConfigDict(extra="ignore")


class JobStatus(BaseModel):
    job_id: str
    status: str
    processed_chunks: int = 0
    total_chunks: Optional[int] = None
    successful_chunks: Optional[int] = None
    file: Optional[str] = None
    file_type: Optional[str] = None
    file_size_mb: Optional[float] = None
    file_hash: Optional[str] = None
    error: Optional[str] = None
    created_at: datetime
    completed_at: Optional[datetime] = None

    model_config = ConfigDict(extra="ignore")


class AnswerResponse(BaseModel):
    answer: str
    log_sources: List[str]
    kb_sources: List[str]
    confidence: float = Field(ge=0, le=1)
    processing_time_ms: Optional[int] = None

    model_config = ConfigDict(extra="ignore")


def safe_embed(text: str) -> Optional[List[float]]:
    if not text or not text.strip():
        return None
    try:
        truncated = text[: settings.MAX_CHARS]
        body = json.dumps({"inputText": truncated}).encode("utf-8")

        resp = bedrock.invoke_model(
            modelId=settings.BEDROCK_EMBED_MODEL,
            body=body,
            accept="application/json",
            contentType="application/json",
        )
        payload = json.loads(resp["body"].read().decode("utf-8"))
        emb = payload.get("embedding")
        if not isinstance(emb, list):
            return None
        return emb
    except Exception as e:
        logger.exception("Embedding failed: %s", e)
        return None


def safe_generate(prompt: str, max_tokens: int = 1024) -> str:
    try:
        body = json.dumps(
            {
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": 0.1,
                "top_p": 0.9,
            }
        ).encode("utf-8")

        resp = bedrock.invoke_model(
            modelId=settings.BEDROCK_LLM_MODEL,
            body=body,
            accept="application/json",
            contentType="application/json",
        )
        payload = json.loads(resp["body"].read().decode("utf-8"))

        # Try common shapes
        if isinstance(payload, dict):
            if "outputs" in payload and payload["outputs"]:
                return (payload["outputs"][0].get("text") or "").strip() or "No response generated."
            if "generation" in payload:
                return str(payload["generation"]).strip()
            if "outputText" in payload:
                return str(payload["outputText"]).strip()

        return "No response generated."
    except Exception as e:
        logger.exception("Generation failed: %s", e)
        return "I hit an error while generating a response. Please try again."


def calculate_file_hash(file_path: Path) -> str:
    sha = hashlib.sha256()
    with file_path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha.update(chunk)
    return sha.hexdigest()


def iter_line_chunks(file_path: Path, lines_per_chunk: int = 100):
    try:
        with file_path.open("r", encoding="utf-8", errors="ignore") as f:
            buf: List[str] = []
            buf_size = 0
            for line in f:
                buf.append(line)
                buf_size += len(line)
                if len(buf) >= lines_per_chunk or buf_size > 100_000:
                    yield "".join(buf)
                    buf.clear()
                    buf_size = 0
            if buf:
                yield "".join(buf)
    except Exception as e:
        logger.exception("Failed reading file %s: %s", file_path, e)
        yield ""


async def index_file_job(job_id: str, file_path: Path, filename: str, file_type: str):
    try:
        job = UPLOAD_JOBS.get(job_id)
        if not job:
            return

        job["status"] = "running"
        total_chunks = 0
        ok_chunks = 0

        batch_embeddings: List[List[float]] = []
        batch_documents: List[str] = []
        batch_metadatas: List[Dict] = []
        batch_ids: List[str] = []

        for chunk in iter_line_chunks(file_path):
            total_chunks += 1
            emb = safe_embed(chunk)
            if not emb:
                continue

            chunk_id = f"{job_id}-{ok_chunks}"
            ok_chunks += 1

            batch_ids.append(chunk_id)
            batch_embeddings.append(emb)
            batch_documents.append(chunk)
            batch_metadatas.append(
                {
                    "source": filename,
                    "file_type": file_type,
                    "job_id": job_id,
                    "chunk_index": ok_chunks - 1,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            )

            if ok_chunks % 10 == 0:
                job["processed_chunks"] = ok_chunks

            if len(batch_embeddings) >= settings.BATCH_SIZE:
                coll.add(
                    ids=batch_ids,
                    embeddings=batch_embeddings,
                    metadatas=batch_metadatas,
                    documents=batch_documents,
                )
                batch_ids, batch_embeddings, batch_metadatas, batch_documents = [], [], [], []

        if batch_embeddings:
            coll.add(
                ids=batch_ids,
                embeddings=batch_embeddings,
                metadatas=batch_metadatas,
                documents=batch_documents,
            )

        try:
            file_path.unlink(missing_ok=True)
        except Exception:
            pass

        job["status"] = "done"
        job["processed_chunks"] = ok_chunks
        job["total_chunks"] = total_chunks
        job["successful_chunks"] = ok_chunks
        job["completed_at"] = datetime.now(timezone.utc)

    except Exception as e:
        logger.exception("Index job failed %s: %s", job_id, e)
        if job_id in UPLOAD_JOBS:
            UPLOAD_JOBS[job_id]["status"] = "failed"
            UPLOAD_JOBS[job_id]["error"] = str(e)
            UPLOAD_JOBS[job_id]["completed_at"] = datetime.now(timezone.utc)
        try:
            file_path.unlink(missing_ok=True)
        except Exception:
            pass


@app.middleware("http")
async def log_requests(request: Request, call_next):
    import time

    start = time.perf_counter()
    response = await call_next(request)
    ms = (time.perf_counter() - start) * 1000.0
    response.headers["X-Processing-Time"] = f"{ms:.2f}ms"
    logger.info("%s %s -> %s (%.2fms)", request.method, request.url.path, response.status_code, ms)
    return response


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version,
        "services": {
            "vector_store": "healthy" if coll is not None else "uninitialized",
            "bedrock": "available",
        },
    }


@app.post("/upload", response_model=UploadResponse)
@limiter.limit("10/minute")
async def upload(
    request: Request,  # REQUIRED for slowapi
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    file_type: str = Query(default="log", pattern="^(log|kb)$"),
    _: bool = Depends(verify_api_key),
):
    file_ext = Path(file.filename).suffix[1:].lower() if file.filename else ""
    if not file_ext or file_ext not in settings.ALLOWED_FILE_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File type '{file_ext}' not allowed. Allowed: {settings.ALLOWED_FILE_TYPES}",
        )

    job_id = uuid.uuid4().hex
    safe_filename = f"{job_id}_{Path(file.filename).name}"
    file_path = settings.UPLOAD_DIR / safe_filename

    content = await file.read()
    size_mb = len(content) / (1024 * 1024)

    if size_mb > settings.MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File size ({size_mb:.1f}MB) exceeds limit of {settings.MAX_FILE_SIZE_MB}MB",
        )

    file_path.write_bytes(content)
    file_hash = calculate_file_hash(file_path)

    created_at = datetime.now(timezone.utc)
    UPLOAD_JOBS[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "processed_chunks": 0,
        "total_chunks": 0,
        "successful_chunks": 0,
        "file": file.filename,
        "file_type": file_type,
        "file_size_mb": size_mb,
        "file_hash": file_hash,
        "created_at": created_at,
    }

    background_tasks.add_task(index_file_job, job_id, file_path, file.filename, file_type)

    return UploadResponse(
        job_id=job_id,
        message="File uploaded successfully. Processing started.",
        file_hash=file_hash,
    )


@app.get("/upload_status/{job_id}", response_model=JobStatus)
async def upload_status(job_id: str, _: bool = Depends(verify_api_key)):
    job = UPLOAD_JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return JobStatus(**job)


@app.post("/ask", response_model=AnswerResponse)
@limiter.limit("30/minute")
async def ask(
    request: Request,  # REQUIRED for slowapi
    req: Question,
    _: bool = Depends(verify_api_key),
):
    import time

    start = time.perf_counter()

    if coll is None:
        raise HTTPException(status_code=503, detail="Vector store is not ready")

    q_emb = safe_embed(req.q)
    if not q_emb:
        raise HTTPException(status_code=500, detail="Failed to generate question embedding")

    log_results = coll.query(query_embeddings=[q_emb], n_results=5, where={"file_type": "log"})
    kb_results = coll.query(query_embeddings=[q_emb], n_results=3, where={"file_type": "kb"})

    log_docs = (log_results.get("documents") or [[]])[0]
    kb_docs = (kb_results.get("documents") or [[]])[0]

    log_context = "\n".join(log_docs) if log_docs else "No relevant log entries found."
    kb_context = "\n".join(kb_docs) if kb_docs else "No relevant knowledge base articles found."

    log_sources = sorted({m.get("source", "") for m in (log_results.get("metadatas") or [[]])[0] if m})
    kb_sources = sorted({m.get("source", "") for m in (kb_results.get("metadatas") or [[]])[0] if m})

    confidence = min(0.3 + (len(log_docs) * 0.1) + (len(kb_docs) * 0.15), 1.0)

    prompt = f"""You are an expert system administrator analyzing logs and knowledge base articles.

LOG ENTRIES:
{log_context}

KNOWLEDGE BASE ARTICLES:
{kb_context}

USER QUESTION: {req.q}

Provide a concise, accurate answer based on the information above.
- If the logs show errors, explain them clearly.
- If KB articles provide solutions, summarize them.
- If information is insufficient, say what additional details are needed.

ANSWER:"""

    answer = safe_generate(prompt)
    ms = int((time.perf_counter() - start) * 1000)

    return AnswerResponse(
        answer=answer,
        log_sources=[s for s in log_sources if s],
        kb_sources=[s for s in kb_sources if s],
        confidence=confidence,
        processing_time_ms=ms,
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "path": request.url.path,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception: %s", exc)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "path": request.url.path,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    )