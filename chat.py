import json
import os
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional, TypedDict
from urllib.parse import urljoin

import requests
import streamlit as st


if sys.version_info < (3, 11):
    st.error("âš ï¸ This application requires Python 3.11 or higher")
    st.stop()


st.set_page_config(
    page_title="Acadia's Log IQ",
    page_icon="ðŸ”",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_BASE = os.getenv("API_BASE", "http://localhost:8000")
API_KEY = os.getenv("UI_API_KEY", "")
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "30"))


class ChatMessage(TypedDict, total=False):
    role: str
    content: str
    timestamp: str
    sources: Dict[str, list]


class UploadedFile(TypedDict, total=False):
    name: str
    type: str
    size_mb: float
    uploaded_at: str


class JobStatus(TypedDict, total=False):
    job_id: str
    status: str
    processed_chunks: int
    total_chunks: Optional[int]
    successful_chunks: Optional[int]
    file: Optional[str]
    file_type: Optional[str]
    error: Optional[str]
    created_at: str


if "messages" not in st.session_state:
    st.session_state.messages = []  # list[ChatMessage]

if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = {"log": None, "kb": None}

if "jobs" not in st.session_state:
    st.session_state.jobs = {}

if "api_health" not in st.session_state:
    st.session_state.api_health = None


def make_request(method: str, endpoint: str, **kwargs) -> Optional[Dict[str, Any]]:
    try:
        headers = kwargs.pop("headers", {})
        if API_KEY:
            headers["X-API-Key"] = API_KEY

        headers["X-Request-ID"] = str(int(time.time() * 1000))

        url = urljoin(API_BASE, endpoint)

        if method.upper() == "GET":
            resp = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT, **kwargs)
        elif method.upper() == "POST":
            resp = requests.post(url, headers=headers, timeout=REQUEST_TIMEOUT, **kwargs)
        else:
            raise ValueError(f"Unsupported method: {method}")

        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.Timeout:
        st.error("â° Request timed out. The server is taking too long to respond.")
        return None
    except requests.exceptions.ConnectionError:
        st.error("ðŸ”Œ Connection error. Please check if the API server is running.")
        return None
    except requests.exceptions.HTTPError as e:
        code = e.response.status_code if e.response else "Unknown"
        st.error(f"ðŸš« HTTP Error: {code}")
        try:
            st.json(e.response.json())
        except Exception:
            pass
        return None
    except json.JSONDecodeError:
        st.error("ðŸ“„ Invalid response format from server.")
        return None
    except Exception as e:
        st.error(f"âŒ Unexpected error: {e}")
        return None


def poll_job(job_id: str, bar, text, timeout_s: int = 300) -> Optional[Dict[str, Any]]:
    start = time.time()
    while time.time() - start < timeout_s:
        status = make_request("GET", f"/upload_status/{job_id}")
        if not status:
            return None

        st.session_state.jobs[job_id] = status

        state = status.get("status", "unknown")
        processed = int(status.get("processed_chunks", 0) or 0)
        total = int(status.get("total_chunks", 0) or 0)

        if state == "running":
            progress = 0.0
            if total > 0:
                progress = min(processed / max(total, 1), 1.0)
            bar.progress(progress)
            text.text(f"Processingâ€¦ {processed}/{total if total else 'â€”'}")
        elif state == "done":
            bar.progress(1.0)
            text.text("âœ… Done")
            return status
        elif state == "failed":
            bar.progress(0.0)
            text.text("âŒ Failed")
            return status

        time.sleep(1)

    text.text("â³ Timeout")
    return {"status": "timeout", "error": "Processing timeout"}


def get_confidence_emoji(confidence: float) -> str:
    if confidence >= 0.8:
        return "ðŸ”¥"
    if confidence >= 0.6:
        return "âœ…"
    if confidence >= 0.4:
        return "âš ï¸"
    return "ðŸ’¡"


# Sidebar
with st.sidebar:
    st.title("Acadia's Log IQ")
    st.caption("AI-powered log analysis")

    st.markdown("---")
    st.header("📂 Data Ingestion")

    # ✅ Multiple logs
    log_files = st.file_uploader(
        "Upload System Logs (multiple allowed)",
        type=["log", "txt"],
        key="log_uploader",
        accept_multiple_files=True,
    )

    # ✅ Multiple KBs (you had accept_multiple_files=True, keeping it)
    kb_files = st.file_uploader(
        "Upload Knowledge Base (multiple allowed)",
        type=["txt", "md", "json"],
        key="kb_uploader",
        accept_multiple_files=True,
    )

    if st.button("🚀 Start Indexing", use_container_width=True, type="primary"):
        # Validation
        if not log_files:
            st.error("Please upload at least one log file.")
            st.stop()

        if not kb_files:
            st.error("Please upload at least one knowledge base file.")
            st.stop()

        st.info("📤 Uploading files…")

        try:
            # -------------------------------
            # 1) Upload ALL logs
            # -------------------------------
            log_job_ids = []
            for lf in log_files:
                resp = make_request(
                    "POST",
                    "/upload?file_type=log",
                    files={"file": (lf.name, lf.getvalue())},
                )
                if not resp:
                    st.stop()
                log_job_ids.append(resp["job_id"])

            # -------------------------------
            # 2) Upload ALL KB files
            # -------------------------------
            kb_job_ids = []
            for kf in kb_files:
                resp = make_request(
                    "POST",
                    "/upload?file_type=kb",
                    files={"file": (kf.name, kf.getvalue())},
                )
                if not resp:
                    st.stop()
                kb_job_ids.append(resp["job_id"])

            st.success("Uploads accepted. Processing…")

            # -------------------------------
            # 3) Progress UI - Logs
            # -------------------------------
            st.write("Log indexing:")
            log_statuses = []
            for i, job_id in enumerate(log_job_ids, start=1):
                st.write(f"• Log file {i}/{len(log_job_ids)} — job: {job_id[:8]}")
                bar = st.progress(0)
                txt = st.empty()
                status = poll_job(job_id, bar, txt)
                log_statuses.append(status)

            # -------------------------------
            # 4) Progress UI - KB
            # -------------------------------
            st.write("KB indexing:")
            kb_statuses = []
            for i, job_id in enumerate(kb_job_ids, start=1):
                st.write(f"• KB file {i}/{len(kb_job_ids)} — job: {job_id[:8]}")
                bar = st.progress(0)
                txt = st.empty()
                status = poll_job(job_id, bar, txt)
                kb_statuses.append(status)

            # -------------------------------
            # 5) Check failures
            # -------------------------------
            any_log_failed = any(s and s.get("status") == "failed" for s in log_statuses)
            any_kb_failed = any(s and s.get("status") == "failed" for s in kb_statuses)

            if any_log_failed:
                st.error("❌ One or more log files failed to index. Check API logs.")
            if any_kb_failed:
                st.error("❌ One or more KB files failed to index. Check API logs.")

            all_done = (
                log_statuses
                and kb_statuses
                and all(s and s.get("status") == "done" for s in log_statuses)
                and all(s and s.get("status") == "done" for s in kb_statuses)
            )

            if all_done:
                st.success("✅ Indexing completed! System is ready for queries.")

                now = datetime.now(timezone.utc).isoformat()

                # Store list of filenames in session state
                st.session_state.uploaded_files["log"] = {
                    "names": [f.name for f in log_files],
                    "type": "log",
                    "count": len(log_files),
                    "total_size_mb": sum(len(f.getvalue()) for f in log_files) / (1024 * 1024),
                    "uploaded_at": now,
                }

                st.session_state.uploaded_files["kb"] = {
                    "names": [f.name for f in kb_files],
                    "type": "kb",
                    "count": len(kb_files),
                    "total_size_mb": sum(len(f.getvalue()) for f in kb_files) / (1024 * 1024),
                    "uploaded_at": now,
                }

                st.rerun()

        except Exception as e:
            st.error(f"❌ Error during indexing: {e}")

    st.markdown("---")
    st.header("📊 System Status")

    if st.button("🔁 Check API Health", use_container_width=True):
        with st.spinner("Checking API…"):
            health = make_request("GET", "/health")
            if health:
                st.session_state.api_health = health
                st.success("✅ API is healthy")
                st.json(health)
            else:
                st.error("❌ API is not responding")



# Main page
st.title("ðŸ” Ask questions about your logs")

if st.session_state.uploaded_files.get("log") and st.session_state.uploaded_files.get("kb"):
    st.caption(
        f"Loaded: **{st.session_state.uploaded_files['log']['name']}** (logs) + "
        f"**{st.session_state.uploaded_files['kb']['name']}** (KB)"
    )
else:
    st.info("Upload and index both a log file and a KB file from the sidebar to start asking questions.")

# Chat history
for msg in st.session_state.messages:
    with st.chat_message(msg.get("role", "assistant")):
        st.markdown(msg.get("content", ""))
        sources = msg.get("sources")
        if sources:
            with st.expander("Sources"):
                st.json(sources)

# Chat input
user_q = st.chat_input("Type your questionâ€¦")
if user_q:
    st.session_state.messages.append(
        {"role": "user", "content": user_q, "timestamp": datetime.now(timezone.utc).isoformat()}
    )

    with st.chat_message("user"):
        st.markdown(user_q)

    with st.chat_message("assistant"):
        with st.spinner("Thinkingâ€¦"):
            resp = make_request("POST", "/ask", json={"q": user_q})
            if resp:
                confidence = float(resp.get("confidence", 0.0) or 0.0)
                emoji = get_confidence_emoji(confidence)

                answer = resp.get("answer", "")
                st.markdown(f"{emoji} {answer}")

                sources = {"logs": resp.get("log_sources", []), "kb": resp.get("kb_sources", [])}
                with st.expander("Sources"):
                    st.json(sources)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": f"{emoji} {answer}",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "sources": sources,
                    }
                )