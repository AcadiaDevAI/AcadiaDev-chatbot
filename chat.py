import json
import os
import sys
import time
import hashlib
import logging
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, Optional, List, TypedDict
from urllib.parse import urljoin

import requests
import streamlit as st

# ============================================================================
# 1. PAGE CONFIGURATION (MUST BE FIRST STREAMLIT COMMAND)
# ============================================================================
st.set_page_config(
    page_title="Acadia's Log IQ",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================================
# 2. CONFIGURATION
# ============================================================================
# File type allowlists - MATCHING API EXPECTATIONS
LOG_ALLOWED_TYPES = [".log", ".txt"]  # Only logs/txt for logs
KB_ALLOWED_TYPES = [".txt", ".md", ".json"]  # More formats for KB

# Security limits
SECURITY_CONFIG = {
    "MAX_FILE_SIZE_MB": 100,
    "MAX_QUESTION_LENGTH": 1000,
    "RATE_LIMIT_UPLOADS": 10,      # per 5 minutes (client-side UX only)
    "RATE_LIMIT_QUERIES": 30,      # per minute (client-side UX only)
    "SESSION_TIMEOUT_MINUTES": 120,
}

# Timeouts (in seconds) - PRACTICAL VALUES
TIMEOUT_CONFIG = {
    "default": 60,
    "upload": 300,   # 5 minutes for file uploads
    "status": 30,    # 30 seconds for status checks
    "ask": 120,      # 2 minutes for AI responses
}

# API Configuration
API_BASE = os.getenv("API_BASE", "http://localhost:8000")
API_KEY = os.getenv("UI_API_KEY", "")

# ============================================================================
# 3. LOGGING SETUP (STDOUT FOR DOCKER)
# ============================================================================
def setup_logging():
    """Setup logging to stdout (captured by Docker)"""
    logger = logging.getLogger("acadia_ui")
    logger.setLevel(logging.INFO)
    
    # Remove any existing handlers
    logger.handlers.clear()
    
    # Add stdout handler
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    # Also log to file if LOG_DIR is set (for non-Docker deployments)
    log_dir = os.getenv("LOG_DIR")
    if log_dir and os.path.exists(log_dir):
        file_handler = logging.FileHandler(f"{log_dir}/ui.log")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

logger = setup_logging()

# ============================================================================
# 4. TYPE DEFINITIONS
# ============================================================================
class ChatMessage(TypedDict, total=False):
    role: str
    content: str
    timestamp: str
    sources: Dict[str, list]


class UploadedFileInfo(TypedDict, total=False):
    names: List[str]
    type: str
    count: int
    total_size_mb: float
    uploaded_at: str
    file_hashes: List[str]


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


# ============================================================================
# 5. SESSION INITIALIZATION
# ============================================================================
def initialize_session():
    """Initialize and manage session state"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = {"log": None, "kb": None}
    
    if "uploaded_hashes" not in st.session_state:
        st.session_state.uploaded_hashes = set()  # For deduplication
    
    if "jobs" not in st.session_state:
        st.session_state.jobs = {}
    
    if "security" not in st.session_state:
        st.session_state.security = {
            "session_start": datetime.now(),
            "last_activity": datetime.now(),
            "upload_count": 0,
            "query_count": 0
        }
    
    # Check session timeout
    now = datetime.now()
    last_activity = st.session_state.security["last_activity"]
    timeout_minutes = SECURITY_CONFIG["SESSION_TIMEOUT_MINUTES"]
    
    if (now - last_activity).seconds > timeout_minutes * 60:
        logger.warning("Session expired due to inactivity")
        st.warning("🕒 Session expired. Please refresh the page.")
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
    
    # Update last activity
    st.session_state.security["last_activity"] = now

initialize_session()

# ============================================================================
# 6. SECURITY FUNCTIONS (PRACTICAL)
# ============================================================================
def validate_file_upload(
    file_data: bytes, 
    filename: str, 
    expected_type: str
) -> Dict[str, Any]:
    """
    Validate file upload with PRACTICAL security checks.
    Returns: {"valid": bool, "reason": str, "hash": str, "size_mb": float}
    """
    result = {
        "valid": False, 
        "reason": "", 
        "hash": "",
        "size_mb": len(file_data) / (1024 * 1024)
    }
    
    # 1. Check file size
    max_bytes = SECURITY_CONFIG["MAX_FILE_SIZE_MB"] * 1024 * 1024
    if len(file_data) > max_bytes:
        result["reason"] = f"File exceeds {SECURITY_CONFIG['MAX_FILE_SIZE_MB']}MB limit"
        logger.warning(f"File too large: {filename} ({result['size_mb']:.1f}MB)")
        return result
    
    # 2. Check file extension matches allowed types
    import os
    ext = os.path.splitext(filename)[1].lower()
    
    allowed_exts = LOG_ALLOWED_TYPES if expected_type == "log" else KB_ALLOWED_TYPES
    if ext not in allowed_exts:
        result["reason"] = f"File type {ext} not allowed for {expected_type}"
        logger.warning(f"Invalid file type: {filename} ({ext}) for {expected_type}")
        return result
    
    # 3. Calculate file hash (for deduplication)
    file_hash = hashlib.sha256(file_data).hexdigest()
    result["hash"] = file_hash
    
    # 4. Basic content validation (PRACTICAL - not too restrictive)
    try:
        # Try to decode as UTF-8
        sample = file_data[:5000].decode('utf-8', errors='replace')
        
        # Check for null bytes (indication of binary file)
        if '\x00' in sample:
            result["reason"] = "File appears to be binary, not text"
            logger.warning(f"Binary file detected: {filename}")
            return result
        
        # FIXED: Check for dangerous patterns (case-insensitive)
        dangerous_patterns = [
            b'<script', b'javascript:', b'vbscript:', 
            b'onload=', b'onerror=', b'onclick=',
        ]
        
        # Convert sample to lowercase for case-insensitive matching
        sample_lower = sample.lower().encode('utf-8')
        
        for pattern in dangerous_patterns:
            if pattern in sample_lower:
                result["reason"] = "File contains potentially dangerous content"
                logger.warning(f"Dangerous pattern in file: {filename}")
                return result
        
    except UnicodeDecodeError:
        # Some log files might be in different encodings
        # We'll be more permissive and let the API handle it
        logger.info(f"File {filename} is not UTF-8, allowing anyway")
        # Don't fail here - let backend handle encoding issues
    
    # 5. Log successful validation
    logger.info(
        f"File validated: {filename}, "
        f"type: {expected_type}, size: {result['size_mb']:.1f}MB, "
        f"hash: {file_hash[:16]}..."
    )
    
    result["valid"] = True
    return result


def sanitize_user_input(text: str) -> Optional[str]:
    """
    SANE input sanitization - blocks only truly dangerous content
    """
    if not text or not isinstance(text, str):
        return None
    
    # Trim and limit length
    text = text.strip()
    if len(text) > SECURITY_CONFIG["MAX_QUESTION_LENGTH"]:
        text = text[:SECURITY_CONFIG["MAX_QUESTION_LENGTH"]]
        logger.info(f"Input truncated to {len(text)} chars")
    
    # Only block truly dangerous patterns
    dangerous_patterns = [
        "<script", "javascript:", "data:", "vbscript:", 
        "onload=", "onerror=", "onclick=",
    ]
    
    text_lower = text.lower()
    for pattern in dangerous_patterns:
        if pattern in text_lower:
            logger.warning(f"Dangerous pattern in user input: {pattern}")
            return None  # Block dangerous content
    
    return text


def sanitize_filename_for_display(filename: str) -> str:
    """
    Light filename sanitization for display only
    """
    import os
    import re
    
    # Get basename
    name = os.path.basename(filename)
    
    # Remove path traversal attempts
    name = name.replace('..', '').replace('/', '').replace('\\', '')
    
    # Truncate very long names
    if len(name) > 50:
        name = name[:47] + "..."
    
    return name


# ============================================================================
# 7. HTTP CLIENT
# ============================================================================
def make_secure_request(
    method: str, 
    endpoint: str, 
    **kwargs
) -> Optional[Dict[str, Any]]:
    """
    Make HTTP request with proper error handling and logging
    """
    try:
        headers = kwargs.pop("headers", {})
        
        # Add API key if configured
        if API_KEY:
            headers["X-API-Key"] = API_KEY
        
        # Add request ID
        headers["X-Request-ID"] = str(int(time.time() * 1000))
        
        url = urljoin(API_BASE, endpoint)
        
        # Get timeout from kwargs or use default
        timeout = kwargs.pop("timeout", TIMEOUT_CONFIG["default"])
        
        # Log request (without sensitive data)
        logger.debug(f"Request: {method} {endpoint}")
        
        # Make request
        if method.upper() == "GET":
            response = requests.get(url, headers=headers, timeout=timeout, **kwargs)
        elif method.upper() == "POST":
            response = requests.post(url, headers=headers, timeout=timeout, **kwargs)
        else:
            raise ValueError(f"Unsupported method: {method}")
        
        # Log response status
        logger.debug(f"Response: {response.status_code} for {endpoint}")
        
        # Check for HTTP errors
        response.raise_for_status()
        
        # Try to parse JSON
        try:
            return response.json()
        except json.JSONDecodeError as json_err:
            # API returned non-JSON (could be HTML error page, proxy error, etc.)
            logger.error(f"Non-JSON response from {endpoint}: {response.text[:200]}")
            
            # For debugging, we might want to see the response
            if os.getenv("DEBUG_MODE") == "true":
                st.error(f"Server returned non-JSON: {response.text[:500]}")
            else:
                st.error("Server returned an invalid response format")
            
            return None
        
    except requests.exceptions.Timeout:
        logger.error(f"Timeout for {method} {endpoint}")
        st.error(f"⏰ Request timed out after {timeout}s")
        return None
        
    except requests.exceptions.ConnectionError:
        logger.error(f"Connection error for {endpoint}")
        st.error("🔌 Cannot connect to the server. Is it running?")
        st.info(f"Trying to reach: {API_BASE}")
        return None
        
    except requests.exceptions.HTTPError as e:
        status_code = e.response.status_code if e.response else "Unknown"
        logger.error(f"HTTP {status_code} for {endpoint}")
        
        # User-friendly error messages
        error_messages = {
            400: "❌ Bad request (check your input)",
            401: "🔒 Authentication required",
            403: "⛔ Access forbidden",
            413: "📁 File too large",
            422: "📝 Invalid input format",
            429: "⚠️ Too many requests - please slow down",
            500: "🔧 Server error - please try again",
            502: "🔌 Bad gateway - server unavailable",
            503: "🛠️ Service temporarily unavailable",
            504: "⏰ Gateway timeout",
        }
        
        message = error_messages.get(status_code, f"HTTP Error {status_code}")
        st.error(message)
        
        # For client errors (4xx), show a bit more detail if in debug mode
        if 400 <= status_code < 500 and os.getenv("DEBUG_MODE") == "true":
            try:
                error_detail = e.response.json()
                with st.expander("Debug Details"):
                    st.json(error_detail)
            except:
                pass
        
        return None
        
    except Exception as e:
        logger.error(f"Unexpected error for {endpoint}: {str(e)}")
        st.error("❌ An unexpected error occurred")
        return None


# ============================================================================
# 8. UPLOAD AND POLL LOGIC (FIXED)
# ============================================================================
def poll_job_status(
    job_id: str, 
    progress_bar, 
    status_text,
    timeout_minutes: int = 30
) -> Optional[Dict[str, Any]]:
    """
    Poll job status until completion
    IMPROVED: Shows progress and allows UI to update
    """
    import time
    start_time = time.time()
    timeout_seconds = timeout_minutes * 60
    last_update = start_time
    
    while time.time() - start_time < timeout_seconds:
        # Get status
        status = make_secure_request(
            "GET", 
            f"/upload_status/{job_id}",
            timeout=TIMEOUT_CONFIG["status"]
        )
        
        if not status:
            return None
        
        # Update session state
        st.session_state.jobs[job_id] = status
        
        # Update UI
        current_status = status.get("status", "unknown")
        processed = status.get("processed_chunks", 0) or 0
        total = status.get("total_chunks", 0) or 0
        
        if current_status == "running":
            # Update progress
            if total > 0:
                progress = min(processed / total, 1.0)
                progress_bar.progress(progress)
                status_text.text(f"Processing: {processed}/{total} chunks")
            else:
                # Indeterminate progress - bounce between 0.3 and 0.7
                elapsed = time.time() - start_time
                bounce_progress = 0.3 + 0.4 * (abs((elapsed % 2) - 1))
                progress_bar.progress(bounce_progress)
                status_text.text("Processing...")
        
        elif current_status == "done":
            progress_bar.progress(1.0)
            status_text.text("✅ Done")
            logger.info(f"Job {job_id} completed successfully")
            return status
        
        elif current_status == "failed":
            progress_bar.progress(0.0)
            error_msg = status.get("error", "Unknown error")
            status_text.text(f"❌ Failed: {error_msg[:50]}...")
            logger.error(f"Job {job_id} failed: {error_msg}")
            return status
        
        # Wait before polling again, but allow UI to update
        # FIXED: Better UX with shorter sleeps
        time.sleep(1.5)
    
    # Timeout
    status_text.text("⏰ Timeout")
    logger.warning(f"Job {job_id} polling timeout")
    return {"status": "timeout", "error": "Processing timeout"}


def upload_files(
    files: List[Any], 
    file_type: str
) -> List[str]:
    """
    Upload files and return job IDs
    FIXED: Returns only job_ids (not empty statuses)
    """
    job_ids = []
    
    for i, file_obj in enumerate(files, 1):
        # Show immediate feedback
        file_status = st.empty()
        file_status.info(f"Validating {file_obj.name}...")
        
        # Validate file
        validation = validate_file_upload(
            file_obj.getvalue(), 
            file_obj.name, 
            file_type
        )
        
        if not validation["valid"]:
            file_status.warning(f"Skipping {file_obj.name}: {validation['reason']}")
            time.sleep(0.5)  # Brief pause so user can see the message
            file_status.empty()
            continue
        
        # FIXED: Duplicate check only here (not in validate_file_upload)
        if validation["hash"] in st.session_state.uploaded_hashes:
            file_status.warning(f"Skipping duplicate file: {file_obj.name}")
            time.sleep(0.5)
            file_status.empty()
            continue
        
        # Upload file
        file_status.info(f"Uploading {file_obj.name} ({i}/{len(files)})...")
        
        response = make_secure_request(
            "POST",
            f"/upload?file_type={file_type}",
            files={"file": (file_obj.name, file_obj.getvalue())},
            timeout=TIMEOUT_CONFIG["upload"]
        )
        
        file_status.empty()  # Clear status
        
        if not response:
            st.error(f"Failed to upload {file_obj.name}")
            continue
        
        job_id = response.get("job_id")
        if not job_id:
            st.error(f"No job ID returned for {file_obj.name}")
            continue
        
        # Add hash to deduplication set
        st.session_state.uploaded_hashes.add(validation["hash"])
        
        job_ids.append(job_id)
        logger.info(f"Uploaded {file_obj.name} as job {job_id}")
    
    return job_ids


# ============================================================================
# 9. UI COMPONENTS
# ============================================================================
def render_sidebar():
    """Render the sidebar with file uploads"""
    with st.sidebar:
        st.title("📊 Acadia's Log IQ")
        st.caption("AI-powered log analysis")
        
        st.markdown("---")
        st.header("📂 Upload Files")
        
        # Log files upload (restricted types)
        log_files = st.file_uploader(
            "Upload System Logs",
            type=[ext.replace(".", "") for ext in LOG_ALLOWED_TYPES],
            accept_multiple_files=True,
            help=f"Allowed: {', '.join(LOG_ALLOWED_TYPES)}. Max {SECURITY_CONFIG['MAX_FILE_SIZE_MB']}MB per file."
        )
        
        # KB files upload (different allowed types)
        kb_files = st.file_uploader(
            "Upload Knowledge Base",
            type=[ext.replace(".", "") for ext in KB_ALLOWED_TYPES],
            accept_multiple_files=True,
            help=f"Allowed: {', '.join(KB_ALLOWED_TYPES)}. Max {SECURITY_CONFIG['MAX_FILE_SIZE_MB']}MB per file."
        )
        
        # Indexing button
        if st.button("🚀 Start Indexing", use_container_width=True, type="primary", key="start_indexing"):
            if not log_files:
                st.error("Please upload at least one log file.")
                st.stop()
            
            if not kb_files:
                st.error("Please upload at least one knowledge base file.")
                st.stop()
            
            # Update session activity
            st.session_state.security["upload_count"] += 1
            
            # Start processing
            process_uploads(log_files, kb_files)
        
        st.markdown("---")
        st.header("📈 System Status")
        
        # Health check
        if st.button("🩺 Check API Health", use_container_width=True, key="health_check"):
            with st.spinner("Checking..."):
                health = make_secure_request("GET", "/health", timeout=10)
                if health:
                    st.success("✅ API is healthy")
                    with st.expander("Details"):
                        st.json(health)
                else:
                    st.error("❌ API is not responding")
        
        # Session info
        st.markdown("---")
        st.header("ℹ️ Session Info")
        
        session_age = datetime.now() - st.session_state.security["session_start"]
        st.caption(f"Session active: {session_age.seconds // 60}m ago")
        st.caption(f"Uploads: {st.session_state.security.get('upload_count', 0)}")
        st.caption(f"Queries: {st.session_state.security.get('query_count', 0)}")
        st.caption(f"Unique files: {len(st.session_state.uploaded_hashes)}")
        
        # Clear button
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Chat", help="Clear chat history", use_container_width=True):
                st.session_state.messages = []
                st.success("Chat history cleared!")
                st.rerun()
        with col2:
            if st.button("📁 Files", help="Clear uploaded files", use_container_width=True):
                st.session_state.uploaded_files = {"log": None, "kb": None}
                st.session_state.uploaded_hashes = set()
                st.success("Uploaded files cleared!")
                st.rerun()


def process_uploads(log_files, kb_files):
    """Handle the complete upload and indexing process"""
    import time
    
    # Create a container for the entire process
    with st.container():
        st.info(f"📤 Processing {len(log_files)} log files and {len(kb_files)} KB files...")
        
        # Create a progress container
        progress_container = st.container()
        
        with progress_container:
            # FIXED: Better UX - show overall progress
            overall_progress = st.progress(0)
            overall_status = st.empty()
            
            total_steps = len(log_files) + len(kb_files) + 2  # +2 for processing steps
            current_step = 0
            
            # Step 1: Upload log files
            overall_status.text("📄 Uploading log files...")
            log_job_ids = upload_files(log_files, "log")
            current_step += len(log_files)
            overall_progress.progress(current_step / total_steps)
            
            if not log_job_ids:
                st.error("No log files were successfully uploaded")
                return
            
            # Brief pause for UX
            time.sleep(0.5)
            
            # Step 2: Upload KB files
            overall_status.text("📚 Uploading knowledge base files...")
            kb_job_ids = upload_files(kb_files, "kb")
            current_step += len(kb_files)
            overall_progress.progress(current_step / total_steps)
            
            if not kb_job_ids:
                st.error("No KB files were successfully uploaded")
                return
            
            # Step 3: Process jobs
            all_jobs = log_job_ids + kb_job_ids
            current_step += 1
            overall_progress.progress(current_step / total_steps)
            
            overall_status.text(f"🔄 Processing {len(all_jobs)} jobs...")
            
            # Create individual progress bars for each job
            st.subheader("Individual Job Progress")
            job_widgets = {}
            
            for i, job_id in enumerate(all_jobs, 1):
                col1, col2, col3 = st.columns([1, 3, 1])
                with col1:
                    st.caption(f"Job {i}")
                with col2:
                    progress_bar = st.progress(0)
                with col3:
                    status_text = st.empty()
                    status_text.text("⏳ Waiting...")
                
                job_widgets[job_id] = (progress_bar, status_text)
            
            # Poll all jobs (sequentially but with better UX)
            successful_jobs = []
            failed_jobs = []
            
            for job_id in all_jobs:
                progress_bar, status_text = job_widgets[job_id]
                status = poll_job_status(job_id, progress_bar, status_text)
                
                if status:
                    if status.get("status") == "done":
                        successful_jobs.append(job_id)
                    elif status.get("status") == "failed":
                        failed_jobs.append(job_id)
            
            # Final step
            current_step += 1
            overall_progress.progress(1.0)
            
            # Show summary
            overall_status.empty()
            
            if successful_jobs:
                success_rate = len(successful_jobs) / len(all_jobs) * 100
                st.success(f"✅ {len(successful_jobs)}/{len(all_jobs)} jobs completed ({success_rate:.0f}%)")
                
                # Update session state with uploaded files info
                now = datetime.now(timezone.utc).isoformat()
                
                st.session_state.uploaded_files["log"] = {
                    "names": [sanitize_filename_for_display(f.name) for f in log_files],
                    "type": "log",
                    "count": len(log_files),
                    "total_size_mb": sum(len(f.getvalue()) for f in log_files) / (1024 * 1024),
                    "uploaded_at": now
                }
                
                st.session_state.uploaded_files["kb"] = {
                    "names": [sanitize_filename_for_display(f.name) for f in kb_files],
                    "type": "kb",
                    "count": len(kb_files),
                    "total_size_mb": sum(len(f.getvalue()) for f in kb_files) / (1024 * 1024),
                    "uploaded_at": now
                }
                
                # Show celebration for good completion rate
                if success_rate > 80:
                    st.balloons()
            
            if failed_jobs:
                st.error(f"❌ {len(failed_jobs)} jobs failed. Check server logs.")
            
            # Brief pause before continuing
            time.sleep(1)


def render_chat_interface():
    """Render the main chat interface"""
    st.title("🔍 Ask Questions About Your Logs")
    
    # Show uploaded files status
    log_info = st.session_state.uploaded_files.get("log")
    kb_info = st.session_state.uploaded_files.get("kb")
    
    if log_info and kb_info:
        col1, col2, col3 = st.columns([2, 2, 1])
        with col1:
            st.info(f"**Logs:** {log_info['count']} files ({log_info['total_size_mb']:.1f}MB)")
        with col2:
            st.info(f"**KB:** {kb_info['count']} files ({kb_info['total_size_mb']:.1f}MB)")
        with col3:
            if st.button("🔄 Refresh", use_container_width=True):
                st.rerun()
    else:
        st.warning("📁 Upload logs and knowledge base files from the sidebar to begin")
    
    # Chat history
    for msg in st.session_state.messages:
        # FIXED: Validate role to prevent UI breakage
        role = msg.get("role", "assistant")
        if role not in ("user", "assistant"):
            role = "assistant"
        
        with st.chat_message(role):
            st.markdown(msg.get("content", ""))
            
            # Show sources if available
            sources = msg.get("sources")
            if sources and isinstance(sources, dict):
                with st.expander("📚 Sources"):
                    if sources.get("logs"):
                        st.write("**Log Sources:**")
                        for src in sources["logs"][:5]:  # Limit to 5
                            st.caption(f"• {src}")
                    
                    if sources.get("kb"):
                        st.write("**KB Sources:**")
                        for src in sources["kb"][:5]:  # Limit to 5
                            st.caption(f"• {src}")
    
    # Chat input
    user_input = st.chat_input("Ask a question about your logs...")
    
    if user_input:
        # Update session activity
        st.session_state.security["query_count"] += 1
        
        # Sanitize input (light check only)
        sanitized_input = sanitize_user_input(user_input)
        if not sanitized_input:
            st.error("❌ Input contains potentially dangerous content")
            return
        
        # Add to chat history
        st.session_state.messages.append({
            "role": "user",
            "content": sanitized_input,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(sanitized_input)
        
        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("🤔 Analyzing logs..."):
                response = make_secure_request(
                    "POST",
                    "/ask",
                    json={"q": sanitized_input},
                    timeout=TIMEOUT_CONFIG["ask"]
                )
                
                if response:
                    answer = response.get("answer", "No response received")
                    confidence = response.get("confidence", 0.0)
                    
                    # Add confidence indicator
                    if confidence > 0.8:
                        confidence_emoji = "🔥"
                    elif confidence > 0.6:
                        confidence_emoji = "✅"
                    elif confidence > 0.4:
                        confidence_emoji = "⚠️"
                    else:
                        confidence_emoji = "💡"
                    
                    displayed_answer = f"{confidence_emoji} {answer}"
                    st.markdown(displayed_answer)
                    
                    # Store in history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": displayed_answer,
                        "sources": {
                            "logs": response.get("log_sources", []),
                            "kb": response.get("kb_sources", [])
                        },
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    })
                    
                    logger.info(f"Question answered: '{sanitized_input[:50]}...'")
                else:
                    st.error("❌ Could not get a response from the server")


def render_footer():
    """Render the footer with security info"""
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.caption("🔐 **Security**")
        st.caption("• Client-side validation")
        st.caption("• Rate limiting (UX)")
        st.caption("• Input sanitization")
        st.caption(f"• Duplicate detection")
    
    with col2:
        st.caption("📊 **Stats**")
        if st.session_state.uploaded_files.get("log"):
            st.caption(f"• Logs: {st.session_state.uploaded_files['log']['count']} files")
        if st.session_state.uploaded_files.get("kb"):
            st.caption(f"• KB: {st.session_state.uploaded_files['kb']['count']} files")
        st.caption(f"• Messages: {len(st.session_state.messages)}")
        st.caption(f"• Unique files: {len(st.session_state.uploaded_hashes)}")
    
    with col3:
        st.caption("🔧 **Technical**")
        st.caption(f"• API: {API_BASE}")
        if API_KEY:
            st.caption("• Auth: 🔑 Enabled")
        else:
            st.caption("• Auth: ⚠️ Disabled")
        st.caption(f"• Timeout: {TIMEOUT_CONFIG['default']}s")
        st.caption(f"• Session: {SECURITY_CONFIG['SESSION_TIMEOUT_MINUTES']}m")


# ============================================================================
# 10. MAIN APPLICATION
# ============================================================================
def main():
    """Main application entry point"""
    try:
        # Log startup
        logger.info("Starting Acadia Log IQ UI")
        logger.info(f"Python: {sys.version}")
        logger.info(f"API Base: {API_BASE}")
        
        # Render UI
        render_sidebar()
        render_chat_interface()
        render_footer()
        
    except Exception as e:
        logger.error(f"Application error: {e}", exc_info=True)
        st.error("❌ Application error. Check logs for details.")


if __name__ == "__main__":
    main()