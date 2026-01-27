# import os
# import time
# import requests
# import streamlit as st
# from pathlib import Path

# # Load API base from environment or default to local
# API_BASE = os.getenv("API_BASE", "http://127.0.0.1:8000")

# st.set_page_config(
#     page_title="Acadia's Log IQ", 
#     layout="wide", 
#     initial_sidebar_state="expanded"
# )

# # Custom CSS for a professional "Log-focused" theme
# # st.markdown("""
# # <style>
# #     .stApp { background-color: #0e1117; color: #008000; }
# #     .stTextInput input { color: #008000; }
# #     .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #262730; color: white; }
# #     .status-text { font-size: 0.85rem; color: #008000; }
# # </style>
# # """, unsafe_allow_html=True)

# # st.markdown("""
# # <style>
# #     /* Global high-contrast settings */
# #     .stApp { 
# #         background-color: #000000; 
# #         color: #ffffff !important; 
# #     }

# #     /* SPECIFIC FIX: High-contrast Chat Messages */
# #     /* This targets the text generated in st.chat_message */
# #     [data-testid="stChatMessage"] p, 
# #     [data-testid="stChatMessage"] span,
# #     [data-testid="stChatMessage"] .stMarkdown {
# #         color: #ffffff !important;
# #         font-weight: 500 !important; /* Slightly bolder for better legibility */
# #         font-size: 1.05rem !important;
# #         line-height: 1.6 !important;
# #     }

# #     /* Add a subtle dark background to the assistant message for depth */
# #     [data-testid="stChatMessage"] {
# #         background-color: #1a1c23 !important;
# #         border-radius: 10px;
# #         margin-bottom: 10px;
# #         border: 1px solid #333;
# #     }

# #     /* Force all standard labels and headers to white */
# #     p, span, label, h1, h2, h3, .stCaption {
# #         color: #ffffff !important;
# #     }

# #     /* Input text box contrast */
# #     .stTextInput input {
# #         color: #ffffff !important;
# #         background-color: #262730 !important;
# #     }
# # </style>
# # """, unsafe_allow_html=True)


# st.markdown("""
# <style>
# /* ✅ Sidebar file-uploader "Browse files" button */
# [data-testid="stSidebar"] [data-testid="stFileUploader"] button {
#     background-color: #4da6ff !important;
#     color: #ffffff !important;
#     border: none !important;
#     border-radius: 8px !important;
#     font-weight: 600 !important;
#     padding: 0.5rem 1rem !important;
# }

# /* Hover */
# [data-testid="stSidebar"] [data-testid="stFileUploader"] button:hover {
#     background-color: #1e88ff !important;
#     color: #ffffff !important;
# }

# /* Optional: remove default white box feel */
# [data-testid="stSidebar"] [data-testid="stFileUploader"] section {
#     background: transparent !important;
#     border: 1px dashed rgba(77, 166, 255, 0.6) !important;
# }
# </style>


# """, unsafe_allow_html=True)


# # --- App Header ---
# st.title("📁 Acadia's Log IQ")
# st.caption("High-speed AI-powered log analysis and vector indexing.")

# # --- Helper Functions ---
# def fetch_sources():
#     try:
#         r = requests.get(f"{API_BASE}/sources", timeout=5)
#         return r.json().get("sources", []) if r.status_code == 200 else []
#     except:
#         return []

# # --- Sidebar: File Ingestion & Monitoring ---
# with st.sidebar:
#     st.header("1. Ingest Logs")
#     uploaded_file = st.file_uploader("Upload .log or .txt file", type=["log", "txt", "md"])

#     if uploaded_file and st.button("Start Fast Indexing"):
#         with st.spinner("Uploading to server..."):
#             files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
#             try:
#                 r = requests.post(f"{API_BASE}/upload", files=files, timeout=60)
#                 if r.status_code == 200:
#                     st.session_state["active_job"] = r.json()["job_id"]
#                     st.success(f"Job started: {st.session_state['active_job'][:8]}")
#                 else:
#                     st.error(f"Upload failed: {r.status_code}")
#             except Exception as e:
#                 st.error(f"Connection error: {e}")

#     # Progress Tracking Section (Polling)
#     if "active_job" in st.session_state:
#         st.divider()
#         st.subheader("Live Job Status")
#         job_id = st.session_state["active_job"]
        
#         status_placeholder = st.empty()
#         progress_bar = st.progress(0)
        
#         # Simple polling loop for UI feedback
#         while True:
#             try:
#                 res = requests.get(f"{API_BASE}/upload_status/{job_id}", timeout=2).json()
#                 status = res.get("status", "unknown")
#                 processed = res.get("processed_chunks", 0)
#                 total = res.get("total_chunks", 0)
                
#                 status_placeholder.markdown(f"**Status:** `{status}`\n\n{res.get('message', '')}")
                
#                 if total > 0:
#                     progress_bar.progress(min(processed / total, 1.0))
                
#                 if status in ["done", "failed"]:
#                     if status == "done": st.balloons()
#                     del st.session_state["active_job"]
#                     st.rerun() # Refresh to update source list
#                     break
                    
#                 time.sleep(1) # Wait 1 second before polling again
#             except:
#                 break

# # --- Main Interface: Chat & Retrieval ---
# st.header("2. Search & Analyze")

# # sources = fetch_sources()
# col1, col2 = st.columns([3, 1])

# # with col2:
# #     selected_source = st.selectbox(
# #         "Filter Source", 
# #         ["All Uploads"] + sources, 
# #         index=0
# #     )
# #  target_source = None if selected_source == "All Uploads" else selected_source
# target_source = None 

# with col1:
#  query = st.chat_input("Ask about error codes, timestamps, or system patterns...")



# if query:
#     # Display user message
#     with st.chat_message("user"):
#         st.markdown(query)

#     # Fetch and display AI response
#     with st.chat_message("assistant"):
#         with st.spinner("Analyzing log vectors..."):
#             payload = {"q": query, "source": target_source}
#             try:
#                 response = requests.post(f"{API_BASE}/ask", json=payload, timeout=60)
#                 if response.status_code == 200:
#                     data = response.json()
#                     st.markdown(data.get("answer", "No answer provided."))
                    
#                     if data.get("sources"):
#                         with st.expander("View Evidence (Log Fragments)"):
#                             st.json(data["sources"])
#                 else:
#                     st.error("AI service is currently unavailable.")
#             except Exception as e:
#                 st.error(f"Failed to reach backend: {e}")

# # Footer
# st.divider()
# st.caption("Powered by AWS Bedrock (Titan Embeddings) & ChromaDB")


# ----------------------------------------------------------------------

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


# --- Sidebar: File Ingestion & Monitoring ---
with st.sidebar:
    st.header("1. Ingest Logs")

    # Show backend URL for debugging (optional, helpful during deployment)
    with st.expander("Backend connection"):
        st.write("API_BASE:", API_BASE)

    uploaded_file = st.file_uploader("Upload .log or .txt file", type=["log", "txt", "md"])

    col_a, col_b = st.columns(2)

    with col_a:
        start_clicked = st.button("Start Fast Indexing", use_container_width=True)

    with col_b:
        refresh_clicked = st.button("Refresh Status", use_container_width=True)

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
                            st.success(f"Job started: {job_id[:8]}")
                        else:
                            st.error("Upload succeeded but no job_id returned.")
                    except Exception:
                        st.error("Upload succeeded but response JSON was invalid.")
                else:
                    st.error(f"Upload failed: {result.status_code} — {result.text[:200]}")

    # Progress Tracking Section (safe approach: one poll per rerun)
    if "active_job" in st.session_state:
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
        if message:
            st.caption(message)

        if total > 0:
            st.progress(min(processed / total, 1.0))
            st.caption(f"{processed}/{total} chunks processed")

        if status in ["done", "failed"]:
            if status == "done":
                st.balloons()
            st.success("Job finished." if status == "done" else "Job failed.")
            del st.session_state["active_job"]
            # refresh UI / sources
            st.rerun()

        # If user clicked refresh, rerun to fetch latest status
        if refresh_clicked:
            st.rerun()

        # Optional note: you can add auto-refresh later using st_autorefresh


# --- Main Interface: Chat & Retrieval ---
st.header("2. Search & Analyze")

# If you want source filtering later, you can re-enable these:
# sources = fetch_sources()
target_source = None

col1, col2 = st.columns([3, 1])

with col1:
    query = st.chat_input("Ask about error codes, timestamps, or system patterns...")

if query:
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing log vectors..."):
            payload = {"q": query, "source": target_source}
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
