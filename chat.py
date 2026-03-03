import json
import os
import sys
import time
import hashlib
import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional, List, TypedDict
from urllib.parse import urljoin

import requests
import streamlit as st

st.set_page_config(page_title="Acadia's Log IQ", page_icon="🔍", layout="wide", initial_sidebar_state="expanded")

# ============================================================================
# CONFIG
# ============================================================================
LOG_ALLOWED_TYPES = [".log", ".txt", ".pdf", ".docx"]
KB_ALLOWED_TYPES = [".txt", ".md", ".json", ".pdf", ".docx"]
BINARY_FILE_TYPES = [".pdf", ".docx"]

SECURITY_CONFIG = {"MAX_FILE_SIZE_MB": 100, "MAX_QUESTION_LENGTH": 1000, "SESSION_TIMEOUT_MINUTES": 120}
TIMEOUT_CONFIG = {"default": 60, "upload": 300, "status": 30, "ask": 180}

API_BASE = os.getenv("API_BASE", "http://localhost:8000")
API_KEY = os.getenv("UI_API_KEY", "")

# ============================================================================
# LOGGING
# ============================================================================
def setup_logging():
    lg = logging.getLogger("acadia_ui")
    lg.setLevel(logging.INFO); lg.handlers.clear()
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    lg.addHandler(h)
    ld = os.getenv("LOG_DIR")
    if ld and os.path.exists(ld):
        fh = logging.FileHandler(f"{ld}/ui.log"); fh.setFormatter(h.formatter); lg.addHandler(fh)
    return lg

logger = setup_logging()

# ============================================================================
# SESSION
# ============================================================================
def init_session():
    for k, v in [("messages", []), ("uploaded_files", {"log": None, "kb": None}),
                  ("uploaded_hashes", set()), ("jobs", {}),
                  ("security", {"session_start": datetime.now(), "last_activity": datetime.now(),
                                "upload_count": 0, "query_count": 0})]:
        if k not in st.session_state: st.session_state[k] = v
    now = datetime.now()
    if (now - st.session_state.security["last_activity"]).seconds > SECURITY_CONFIG["SESSION_TIMEOUT_MINUTES"] * 60:
        st.warning("🕒 Session expired."); [st.session_state.pop(k) for k in list(st.session_state.keys())]; st.rerun()
    st.session_state.security["last_activity"] = now

init_session()

# ============================================================================
# VALIDATION
# ============================================================================
def validate_file(data, name, ftype):
    r = {"valid": False, "reason": "", "hash": "", "size_mb": len(data) / (1024*1024)}
    if len(data) > SECURITY_CONFIG["MAX_FILE_SIZE_MB"] * 1024 * 1024:
        r["reason"] = f"Exceeds {SECURITY_CONFIG['MAX_FILE_SIZE_MB']}MB"; return r
    ext = os.path.splitext(name)[1].lower()
    allowed = LOG_ALLOWED_TYPES if ftype == "log" else KB_ALLOWED_TYPES
    if ext not in allowed:
        r["reason"] = f"{ext} not allowed for {ftype}"; return r
    r["hash"] = hashlib.sha256(data).hexdigest()
    if ext in BINARY_FILE_TYPES:
        if ext == ".pdf" and not data[:5].startswith(b'%PDF'): r["reason"] = "Invalid PDF"; return r
        if ext == ".docx" and data[:4] != b'PK\x03\x04': r["reason"] = "Invalid DOCX"; return r
        r["valid"] = True; return r
    try:
        s = data[:5000].decode('utf-8', errors='replace')
        if '\x00' in s: r["reason"] = "Binary detected"; return r
        for p in [b'<script', b'javascript:', b'vbscript:', b'onload=', b'onerror=']:
            if p in s.lower().encode('utf-8'): r["reason"] = "Dangerous content"; return r
    except UnicodeDecodeError: pass
    r["valid"] = True; return r

def sanitize_input(text):
    if not text or not isinstance(text, str): return None
    text = text.strip()[:SECURITY_CONFIG["MAX_QUESTION_LENGTH"]]
    for p in ["<script", "javascript:", "data:", "vbscript:", "onload=", "onerror="]:
        if p in text.lower(): return None
    return text

def safe_name(n):
    n = os.path.basename(n).replace('..','').replace('/','').replace('\\','')
    return n[:47] + "..." if len(n) > 50 else n

# ============================================================================
# HTTP
# ============================================================================
def api_call(method, endpoint, **kw):
    try:
        headers = kw.pop("headers", {})
        if API_KEY: headers["X-API-Key"] = API_KEY
        url = urljoin(API_BASE, endpoint)
        timeout = kw.pop("timeout", TIMEOUT_CONFIG["default"])
        resp = (requests.get if method == "GET" else requests.post)(url, headers=headers, timeout=timeout, **kw)
        resp.raise_for_status()
        try: return resp.json()
        except json.JSONDecodeError: st.error("Invalid server response"); return None
    except requests.exceptions.Timeout: st.error(f"⏰ Timeout ({timeout}s)")
    except requests.exceptions.ConnectionError: st.error("🔌 Can't connect"); st.info(f"→ {API_BASE}")
    except requests.exceptions.HTTPError as e:
        c = e.response.status_code if e.response else 0
        m = {400:"❌ Bad request",401:"🔒 Auth",429:"⚠️ Rate limit",500:"🔧 Server error"}
        st.error(m.get(c, f"HTTP {c}"))
    except Exception as e: st.error(f"❌ {e}")
    return None

# ============================================================================
# UPLOAD & POLL
# ============================================================================
def poll_job(jid, pbar, stxt, tmin=30):
    t0 = time.time()
    while time.time() - t0 < tmin * 60:
        r = api_call("GET", f"/upload_status/{jid}", timeout=TIMEOUT_CONFIG["status"])
        if not r: return None
        st.session_state.jobs[jid] = r
        s, p, t = r.get("status","?"), r.get("processed_chunks",0) or 0, r.get("total_chunks",0) or 0
        if s == "running":
            pbar.progress(min(p/t,1) if t > 0 else 0.3+0.4*abs(((time.time()-t0)%2)-1))
            stxt.text(f"Processing: {p}/{t}" if t > 0 else "Processing...")
        elif s == "done": pbar.progress(1.0); stxt.text("✅ Done"); return r
        elif s == "failed": pbar.progress(0); stxt.text(f"❌ {r.get('error','')[:40]}"); return r
        time.sleep(1.5)
    stxt.text("⏰ Timeout"); return {"status": "timeout"}

def upload_files(files, ftype):
    ids = []
    for i, f in enumerate(files, 1):
        msg = st.empty(); msg.info(f"Validating {f.name}...")
        v = validate_file(f.getvalue(), f.name, ftype)
        if not v["valid"]: msg.warning(f"Skip {f.name}: {v['reason']}"); time.sleep(0.5); msg.empty(); continue
        if v["hash"] in st.session_state.uploaded_hashes: msg.warning(f"Dup: {f.name}"); time.sleep(0.5); msg.empty(); continue
        msg.info(f"Uploading {f.name} ({i}/{len(files)})...")
        r = api_call("POST", f"/upload?file_type={ftype}", files={"file": (f.name, f.getvalue())}, timeout=TIMEOUT_CONFIG["upload"])
        msg.empty()
        if not r: st.error(f"Failed: {f.name}"); continue
        jid = r.get("job_id")
        if not jid: st.error(f"No job ID: {f.name}"); continue
        st.session_state.uploaded_hashes.add(v["hash"]); ids.append(jid)
    return ids

# ============================================================================
# UI
# ============================================================================
def render_sidebar():
    with st.sidebar:
        st.title("📊 Acadia's Log IQ")
        st.caption("Hybrid Search • BM25 + Vector • Re-ranking")
        st.markdown("---")
        st.header("📂 Upload Files")

        logs = st.file_uploader("Upload System Logs",
            type=[e.replace(".","") for e in LOG_ALLOWED_TYPES], accept_multiple_files=True,
            help=f"Allowed: {', '.join(LOG_ALLOWED_TYPES)}. Max {SECURITY_CONFIG['MAX_FILE_SIZE_MB']}MB each.")
        kbs = st.file_uploader("Upload Knowledge Base",
            type=[e.replace(".","") for e in KB_ALLOWED_TYPES], accept_multiple_files=True,
            help=f"Allowed: {', '.join(KB_ALLOWED_TYPES)}. Max {SECURITY_CONFIG['MAX_FILE_SIZE_MB']}MB each.")

        if st.button("🚀 Start Indexing", use_container_width=True, type="primary"):
            if not logs: st.error("Upload log files first"); st.stop()
            if not kbs: st.error("Upload KB files first"); st.stop()
            st.session_state.security["upload_count"] += 1
            do_uploads(logs, kbs)

        st.markdown("---")
        if st.button("🩺 Health Check", use_container_width=True):
            with st.spinner("..."):
                h = api_call("GET", "/health", timeout=10)
                if h: st.success("✅ Healthy"); st.expander("Details").json(h)
                else: st.error("❌ Down")

        st.markdown("---")
        age = datetime.now() - st.session_state.security["session_start"]
        st.caption(f"⏱ {age.seconds//60}m | 📤 {st.session_state.security['upload_count']} | "
                   f"❓ {st.session_state.security['query_count']} | 📁 {len(st.session_state.uploaded_hashes)}")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("🗑️ Chat", use_container_width=True): st.session_state.messages = []; st.rerun()
        with c2:
            if st.button("📁 Reset", use_container_width=True):
                st.session_state.uploaded_files = {"log": None, "kb": None}
                st.session_state.uploaded_hashes = set(); st.rerun()


def do_uploads(logs, kbs):
    with st.container():
        st.info(f"📤 {len(logs)} logs + {len(kbs)} KB files...")
        prog = st.progress(0); stat = st.empty()
        total = len(logs) + len(kbs) + 2; step = 0

        stat.text("📄 Uploading logs..."); lid = upload_files(logs, "log")
        step += len(logs); prog.progress(step/total)
        if not lid: st.error("No logs uploaded"); return

        stat.text("📚 Uploading KB..."); kid = upload_files(kbs, "kb")
        step += len(kbs); prog.progress(step/total)
        if not kid: st.error("No KB uploaded"); return

        all_j = lid + kid; step += 1; prog.progress(step/total)
        stat.text(f"🔄 Processing {len(all_j)} jobs...")

        st.subheader("Job Progress")
        w = {}
        for i, j in enumerate(all_j, 1):
            c1, c2, c3 = st.columns([1, 3, 1])
            with c1: st.caption(f"Job {i}")
            with c2: pb = st.progress(0)
            with c3: sx = st.empty(); sx.text("⏳")
            w[j] = (pb, sx)

        ok, fail = [], []
        for j in all_j:
            r = poll_job(j, *w[j])
            if r and r.get("status") == "done": ok.append(j)
            elif r and r.get("status") == "failed": fail.append(j)

        prog.progress(1.0); stat.empty()
        if ok:
            rate = len(ok)/len(all_j)*100
            st.success(f"✅ {len(ok)}/{len(all_j)} ({rate:.0f}%)")
            now = datetime.now(timezone.utc).isoformat()
            st.session_state.uploaded_files["log"] = {
                "names": [safe_name(f.name) for f in logs], "count": len(logs),
                "total_size_mb": sum(len(f.getvalue()) for f in logs)/(1024*1024), "uploaded_at": now}
            st.session_state.uploaded_files["kb"] = {
                "names": [safe_name(f.name) for f in kbs], "count": len(kbs),
                "total_size_mb": sum(len(f.getvalue()) for f in kbs)/(1024*1024), "uploaded_at": now}
            if rate > 80: st.balloons()
        if fail: st.error(f"❌ {len(fail)} failed")


def render_chat():
    st.title("🔍 Ask Questions About Your Logs")
    li, ki = st.session_state.uploaded_files.get("log"), st.session_state.uploaded_files.get("kb")
    if li and ki:
        c1, c2, c3 = st.columns([2, 2, 1])
        with c1: st.info(f"**Logs:** {li['count']} files ({li['total_size_mb']:.1f}MB)")
        with c2: st.info(f"**KB:** {ki['count']} files ({ki['total_size_mb']:.1f}MB)")
        with c3:
            if st.button("🔄", use_container_width=True): st.rerun()
    else:
        st.warning("📁 Upload logs + KB from sidebar to begin")

    for m in st.session_state.messages:
        role = m.get("role", "assistant")
        if role not in ("user", "assistant"): role = "assistant"
        with st.chat_message(role):
            st.markdown(m.get("content", ""))
            src = m.get("sources")
            if src and isinstance(src, dict):
                with st.expander("📚 Sources"):
                    for k, lbl in [("logs", "Log"), ("kb", "KB")]:
                        for s in (src.get(k) or [])[:5]: st.caption(f"• {s}")

    inp = st.chat_input("Ask about your logs...")
    if inp:
        st.session_state.security["query_count"] += 1
        clean = sanitize_input(inp)
        if not clean: st.error("❌ Invalid input"); return

        st.session_state.messages.append({"role": "user", "content": clean,
                                           "timestamp": datetime.now(timezone.utc).isoformat()})
        with st.chat_message("user"): st.markdown(clean)

        with st.chat_message("assistant"):
            with st.spinner("🤔 Hybrid search + re-ranking..."):
                resp = api_call("POST", "/ask", json={"q": clean}, timeout=TIMEOUT_CONFIG["ask"])

            if resp:
                ans = resp.get("answer", "No response")
                conf = resp.get("confidence", 0)
                emoji = "🔥" if conf > 0.8 else "✅" if conf > 0.6 else "⚠️" if conf > 0.4 else "💡"
                shown = f"{emoji} {ans}"
                st.markdown(shown)

                ctx = resp.get("context_stats")
                if ctx:
                    with st.expander("📊 Search Pipeline Stats"):
                        st.caption(f"**Mode:** {ctx.get('search_mode', 'N/A')}")
                        c1, c2, c3, c4 = st.columns(4)
                        with c1:
                            st.metric("Log candidates", ctx.get("log_candidates", 0))
                            st.metric("→ After re-rank", ctx.get("log_after_rerank", 0))
                        with c2:
                            st.metric("KB candidates", ctx.get("kb_candidates", 0))
                            st.metric("→ After re-rank", ctx.get("kb_after_rerank", 0))
                        with c3:
                            st.metric("Log context", f"{ctx.get('log_context_chars',0):,} ch")
                            st.metric("KB context", f"{ctx.get('kb_context_chars',0):,} ch")
                        with c4:
                            st.metric("Prompt", f"~{ctx.get('prompt_tokens',0):,} tok")
                            st.metric("Headroom", f"~{ctx.get('headroom',0):,} tok")

                st.session_state.messages.append({
                    "role": "assistant", "content": shown,
                    "sources": {"logs": resp.get("log_sources",[]), "kb": resp.get("kb_sources",[])},
                    "timestamp": datetime.now(timezone.utc).isoformat()})
            else:
                st.error("❌ No response")


def render_footer():
    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    with c1: st.caption("🔐 Validation • Rate limiting • Dedup")
    with c2: st.caption("🔍 Hybrid: Vector + BM25 + LLM Re-rank")
    with c3: st.caption(f"⚙️ {API_BASE} | Auth: {'🔑' if API_KEY else '⚠️'}")

def main():
    try:
        render_sidebar(); render_chat(); render_footer()
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True); st.error("❌ App error")

if __name__ == "__main__": main()
