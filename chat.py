import os
import time
import requests
import streamlit as st

# Docker-friendly default
API_BASE = os.getenv("API_BASE", "http://api:8000")

st.set_page_config(
    page_title="Acadia's Log IQ",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Styling for sidebar buttons
st.markdown(
    """
<style>
/* Sidebar file-uploader "Browse files" button */
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

# App Header
st.title("📁 Acadia's Log IQ")
st.caption("High-speed AI-powered log analysis and vector indexing.")


# Helper Functions
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
        return e


def safe_post_file(url: str, filename: str, data: bytes, timeout: int = 60):
    try:
        files = {"file": (filename, data)}
        r = requests.post(url, files=files, timeout=timeout)
        return r
    except Exception as e:
        return e


def fetch_sources():
    """Fetch all indexed sources from the API"""
    data = safe_get_json(f"{API_BASE}/sources", timeout=5)
    if not data:
        return []
    return data.get("sources", [])


def fetch_sources_detailed():
    """Fetch detailed source information including chunk counts"""
    data = safe_get_json(f"{API_BASE}/sources", timeout=5)
    return data or {"sources": [], "count": 0, "details": {}, "total_chunks": 0}


def fetch_job_status(job_id: str):
    data = safe_get_json(f"{API_BASE}/upload_status/{job_id}", timeout=5)
    return data or {"status": "unknown", "message": "No response from server."}


def delete_source(filename: str):
    """Delete a specific source file"""
    try:
        r = requests.delete(f"{API_BASE}/sources/{filename}", timeout=10)
        if r.status_code == 200:
            return r.json()
        return None
    except Exception:
        return None


def clear_all_sources():
    """Clear all indexed data"""
    try:
        r = requests.post(f"{API_BASE}/clear", timeout=10)
        if r.status_code == 200:
            return r.json()
        return None
    except Exception:
        return None


# Session state defaults
if "active_job" not in st.session_state:
    st.session_state["active_job"] = None

if "last_job_result" not in st.session_state:
    st.session_state["last_job_result"] = None

if "last_refresh_ts" not in st.session_state:
    st.session_state["last_refresh_ts"] = 0.0


# --- Sidebar: File Ingestion & Monitoring ---
with st.sidebar:
    st.header("1. Ingest Logs")

    with st.expander("Backend connection"):
        st.write("API_BASE:", API_BASE)

    uploaded_file = st.file_uploader(
        "Upload .log or .txt file",
        type=["log", "txt", "md"],
        key="uploader"
    )

    col_a, col_b = st.columns(2)

    with col_a:
        start_clicked = st.button("Start Fast Indexing", use_container_width=True)

    with col_b:
        refresh_clicked = st.button("Refresh Status", use_container_width=True)

    # CHANGE: Show current indexed files
    st.divider()
    st.subheader("📚 Indexed Files")
    sources_data = fetch_sources_detailed()
    sources = sources_data.get("sources", [])
    details = sources_data.get("details", {})
    total_chunks = sources_data.get("total_chunks", 0)

    if sources:
        st.caption(f"Total: {len(sources)} files ({total_chunks} chunks)")
        for src in sources:
            chunk_count = details.get(src, 0)
            col1, col2 = st.columns([3, 1])
            with col1:
                st.caption(f"📄 {src} ({chunk_count} chunks)")
            with col2:
                if st.button("❌", key=f"del_{src}", help=f"Delete {src}"):
                    with st.spinner(f"Deleting {src}..."):
                        result = delete_source(src)
                        if result:
                            st.success(f"Deleted {src}")
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error(f"Failed to delete {src}")
        
        # Clear all button
        if st.button("🗑️ Clear All Files", use_container_width=True, type="secondary"):
            with st.spinner("Clearing all files..."):
                result = clear_all_sources()
                if result:
                    st.success("All files cleared")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error("Failed to clear files")
    else:
        st.info("No files indexed yet")

    # Show last completed job summary
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

    # Handle file upload
    if uploaded_file and start_clicked:
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
                            st.session_state["last_job_result"] = None
                            st.success(f"Job started: {job_id[:8]}")
                        else:
                            st.error("Upload succeeded but no job_id returned.")
                    except Exception:
                        st.error("Upload succeeded but response JSON was invalid.")
                else:
                    st.error(f"Upload failed: {result.status_code} — {result.text[:200]}")

    # Progress Tracking Section
    if st.session_state.get("active_job"):
        st.divider()
        st.subheader("Live Job Status")

        job_id = st.session_state["active_job"]
        res = fetch_job_status(job_id)

        status = res.get("status", "unknown")
        processed = int(res.get("processed_chunks", 0) or 0)
        total = int(res.get("total_chunks", 0) or 0)
        message = res.get("message", "")
        file = res.get("file", "")

        st.markdown(f"**Job:** `{job_id}`")
        st.markdown(f"**Status:** `{status}`")
        if file:
            st.markdown(f"**File:** `{file}`")

        if message:
            st.caption(message)

        if total > 0:
            st.progress(min(processed / total, 1.0))
            st.caption(f"{processed}/{total} chunks processed")
        else:
            st.caption(f"Processed chunks: {processed}")

        if status in ["done", "failed"]:
            st.session_state["last_job_result"] = {
                "status": status,
                "processed_chunks": processed,
                "total_chunks": total,
                "message": message,
                "file": file,
                "job_id": job_id,
            }

            if status == "done":
                st.success("✅ Indexing completed successfully.")
                st.balloons()
            else:
                st.error("❌ Indexing failed. Check API logs for details.")

            st.session_state["active_job"] = None

            if st.button("Dismiss / Continue", use_container_width=True):
                st.rerun()

        if refresh_clicked:
            now = time.time()
            if now - st.session_state["last_refresh_ts"] > 0.5:
                st.session_state["last_refresh_ts"] = now
                st.rerun()


# --- Main Interface: Chat & Retrieval ---
st.header("2. Search & Analyze")

# CRITICAL CHANGE: Don't filter by source - search ALL files
sources_data = fetch_sources_detailed()
sources = sources_data.get("sources", [])
total_chunks = sources_data.get("total_chunks", 0)

if not sources:
    st.info("📤 Upload log files using the sidebar. Then ask questions here.")
else:
    # Show what files are being searched
    st.caption(f"🔎 Searching across **{len(sources)} file(s)** ({total_chunks} chunks total)")
    with st.expander("📂 Files being searched"):
        for src in sources:
            chunk_count = sources_data.get("details", {}).get(src, 0)
            st.caption(f"• {src} ({chunk_count} chunks)")

col1, col2 = st.columns([3, 1])

with col1:
    query = st.chat_input("Ask about error codes, timestamps, or system patterns...")

# CHANGE: Search examples
with col2:
    with st.expander("💡 Examples"):
        st.caption("Single file:")
        st.code("What errors in router1.log?")
        st.caption("Multi-file correlation:")
        st.code("What caused the network outage?")
        st.code("Show BGP issues across all devices")

if query:
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing log vectors..."):
            # CRITICAL CHANGE: Don't send "source" parameter
            # This allows API to search ALL files and correlate
            payload = {"q": query}
            
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

                    answer = data.get("answer", "No answer provided.")
                    st.markdown(answer)

                    # CHANGE: Show which files were used in the answer
                    sources_used = data.get("sources", [])
                    sources_count = data.get("sources_count", 0)
                    
                    if sources_used:
                        st.divider()
                        st.caption(f"📚 Sources used: {sources_count} file(s)")
                        cols = st.columns(min(len(sources_used), 4))
                        for idx, src in enumerate(sources_used):
                            with cols[idx % 4]:
                                st.caption(f"📄 {src}")

                    # Show evidence
                    if data.get("sources"):
                        with st.expander("View Evidence (Log Fragments)"):
                            st.json(data["sources"])
                else:
                    st.error(f"AI service error: {result.status_code} — {result.text[:200]}")

# Footer
st.divider()
st.caption("Powered by AWS Bedrock (Titan Embeddings) & ChromaDB | Multi-device correlation enabled")
