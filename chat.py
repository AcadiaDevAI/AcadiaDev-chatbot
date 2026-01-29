
import os
import time
import requests
import streamlit as st

# ✅ Docker-friendly default:
# - In docker-compose, your backend service can be named "api" so the UI calls http://api:8000
# - Locally, set API_BASE=http://127.0.0.1:8000 in your .env
API_BASE = os.getenv("API_BASE", "http://api:8000")

st.set_page_config(
    page_title="Acadia's Log IQ",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar uploader button styling (kept from your version)
st.markdown(
    """
<style>
/* ✅ Sidebar file-uploader "Browse files" button */
[data-testid="stSidebar"] [data-testid="stFileUploader"] button {
    background-color: #4da6ff !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    padding: 0.5rem 1rem !important;
}

/* Hover */
[data-testid="stSidebar"] [data-testid="stFileUploader"] button:hover {
    background-color: #1e88ff !important;
    color: #ffffff !important;
}

/* Optional: remove default white box feel */
[data-testid="stSidebar"] [data-testid="stFileUploader"] section {
    background: transparent !important;
    border: 1px dashed rgba(77, 166, 255, 0.6) !important;
}
</style>
""",
    unsafe_allow_html=True
)

# --- App Header ---
st.title("📁 Acadia's Log IQ")
st.caption("High-speed AI-powered log analysis and vector indexing.")

# --- Helper Functions ---
def safe_get_json(url: str, timeout: int = 5):
    try:
        r = requests.get(url, timeout=timeout)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


def safe_post_json(url: str, payload: dict, timeout: int = 60):
    try:
        r = requests.post(url, json=payload, timeout=timeout)
        return r
    except Exception as e:
        return e  # return exception so we can display it


def safe_post_file(url: str, filename: str, data: bytes, timeout: int = 60):
    try:
        files = {"file": (filename, data)}
        r = requests.post(url, files=files, timeout=timeout)
        return r
    except Exception as e:
        return e


def fetch_sources():
    data = safe_get_json(f"{API_BASE}/sources", timeout=5)
    if not data:
        return []
    return data.get("sources", [])


def fetch_job_status(job_id: str):
    data = safe_get_json(f"{API_BASE}/upload_status/{job_id}", timeout=5)
    return data or {"status": "unknown", "message": "No response from server."}


# ------------------------------------------------------------------------------
# Session state defaults
# ------------------------------------------------------------------------------
if "active_job" not in st.session_state:
    st.session_state["active_job"] = None

if "active_source" not in st.session_state:
    st.session_state["active_source"] = None

# NEW: remember last finished job info so UI can show DONE even after clearing active_job
if "last_job_result" not in st.session_state:
    st.session_state["last_job_result"] = None

# NEW: small debounce to avoid repeated reruns too quickly
if "last_refresh_ts" not in st.session_state:
    st.session_state["last_refresh_ts"] = 0.0


# --- Sidebar: File Ingestion & Monitoring ---
with st.sidebar:
    st.header("1. Ingest Logs")

    # Show backend URL for debugging (optional, helpful during deployment)
    with st.expander("Backend connection"):
        st.write("API_BASE:", API_BASE)

    # uploaded_file = st.file_uploader("Upload .log or .txt file", type=["log", "txt", "md"])

    uploaded_file = st.file_uploader(
    "Upload .log or .txt file",
    type=["log", "txt", "md"],
    key="uploader"
)

# ✅ If user selects a new file, update active_source immediately
if uploaded_file is not None:
    st.session_state["active_source"] = uploaded_file.name


    col_a, col_b = st.columns(2)

    with col_a:
        start_clicked = st.button("Start Fast Indexing", use_container_width=True)

    with col_b:
        refresh_clicked = st.button("Refresh Status", use_container_width=True)

    # If user has already uploaded something earlier in this session, show it.
    if st.session_state.get("active_source"):
        st.caption(f"📌 Active file for Q&A: **{st.session_state['active_source']}**")

    # Show last completed job summary (so you ALWAYS see done)
    if st.session_state.get("last_job_result"):
        st.divider()
        st.subheader("Last Job Result")
        jr = st.session_state["last_job_result"]
        st.markdown(f"**File:** `{jr.get('file','')}`")
        st.markdown(f"**Status:** `{jr.get('status','')}`")
        st.caption(f"Processed chunks: {jr.get('processed_chunks', 0)}")
        if st.button("Clear Last Result", use_container_width=True):
            st.session_state["last_job_result"] = None
            st.rerun()

    if uploaded_file and start_clicked:
        # IMPORTANT: Set active source immediately so /ask uses the right file
        st.session_state["active_source"] = uploaded_file.name

        with st.spinner("Uploading to server..."):
            result = safe_post_file(
                f"{API_BASE}/upload",
                uploaded_file.name,
                uploaded_file.getvalue(),
                timeout=120
            )

            if isinstance(result, Exception):
                st.error(f"Connection error: {result}")
            else:
                if result.status_code == 200:
                    try:
                        job_id = result.json().get("job_id")
                        if job_id:
                            st.session_state["active_job"] = job_id
                            # Reset last job result on new upload (optional)
                            st.session_state["last_job_result"] = None
                            st.success(f"Job started: {job_id[:8]}")
                        else:
                            st.error("Upload succeeded but no job_id returned.")
                    except Exception:
                        st.error("Upload succeeded but response JSON was invalid.")
                else:
                    st.error(f"Upload failed: {result.status_code} — {result.text[:200]}")

    # Progress Tracking Section (safe approach: one poll per rerun)
    if st.session_state.get("active_job"):
        st.divider()
        st.subheader("Live Job Status")

        job_id = st.session_state["active_job"]
        res = fetch_job_status(job_id)

        status = res.get("status", "unknown")
        processed = int(res.get("processed_chunks", 0) or 0)
        total = int(res.get("total_chunks", 0) or 0)
        message = res.get("message", "")

        st.markdown(f"**Job:** `{job_id}`")
        st.markdown(f"**Status:** `{status}`")
        if st.session_state.get("active_source"):
            st.markdown(f"**File:** `{st.session_state['active_source']}`")

        if message:
            st.caption(message)

        # Some versions of the API may not return total_chunks; show processed anyway
        if total > 0:
            st.progress(min(processed / total, 1.0))
            st.caption(f"{processed}/{total} chunks processed")
        else:
            st.caption(f"Processed chunks: {processed}")

        # ✅ FIX: Do NOT immediately clear + rerun. Persist a "done" view.
        if status in ["done", "failed"]:
            # Save final result so it remains visible even after active_job is cleared
            st.session_state["last_job_result"] = {
                "status": status,
                "processed_chunks": processed,
                "total_chunks": total,
                "message": message,
                "file": st.session_state.get("active_source"),
                "job_id": job_id,
            }

            if status == "done":
                st.success("✅ Indexing completed successfully.")
                st.balloons()
            else:
                st.error("❌ Indexing failed. Check API logs for details.")

            # Clear the active job, but do NOT auto-rerun; let user decide
            st.session_state["active_job"] = None

            if st.button("Dismiss / Continue", use_container_width=True):
                st.rerun()

        # If user clicked refresh, rerun to fetch latest status (with light debounce)
        if refresh_clicked:
            now = time.time()
            if now - st.session_state["last_refresh_ts"] > 0.5:
                st.session_state["last_refresh_ts"] = now
                st.rerun()


# --- Main Interface: Chat & Retrieval ---
st.header("2. Search & Analyze")

# Ensure we query the ACTIVE uploaded file by default
target_source = st.session_state.get("active_source")

if not target_source:
    st.info("Upload a log file and click **Start Fast Indexing**. Then ask questions here.")

if st.session_state.get("active_source"):
    st.caption(
        f"🔎 Searching within: {st.session_state.get('active_source')}"
    )


col1, col2 = st.columns([3, 1])

with col1:
    query = st.chat_input("Ask about error codes, timestamps, or system patterns...")

if query:
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing log vectors..."):
            # payload = {
            #     "q": query,
            #     "source": target_source  # ✅ Force retrieval from active file
            # }
            payload = {"q": query, "source": st.session_state.get("active_source")}

            result = safe_post_json(f"{API_BASE}/ask", payload, timeout=120)

            if isinstance(result, Exception):
                st.error(f"Failed to reach backend: {result}")
            else:
                if result.status_code == 200:
                    try:
                        data = result.json()
                    except Exception:
                        st.error("Backend returned invalid JSON.")
                        data = {}

                    st.markdown(data.get("answer", "No answer provided."))

                    if data.get("sources"):
                        with st.expander("View Evidence (Log Fragments)"):
                            st.json(data["sources"])
                else:
                    st.error(f"AI service error: {result.status_code} — {result.text[:200]}")

# Footer
st.divider()
st.caption("Powered by AWS Bedrock (Titan Embeddings) & ChromaDB")

