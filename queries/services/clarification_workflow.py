"""Persistent WhatsApp clarification workflow for PaddyMitra.

Scenario prompts are sourced from the client's clarification spreadsheet.
State is separate from ordinary chat history and stored as JSON per phone number.
"""
from __future__ import annotations

import difflib
try:
    import fcntl  # Linux/Unix process-level lock
except ImportError:  # pragma: no cover - Windows development fallback
    fcntl = None
import threading
import json
import logging
import os
import re
import tempfile
import time
from urllib.parse import unquote_plus
from pathlib import Path
from django.conf import settings

log = logging.getLogger("irri")
_LOCAL_LOCKS: dict[str, threading.RLock] = {}
_LOCAL_LOCKS_GUARD = threading.Lock()
_SCENARIOS_PATH = Path(__file__).with_name("clarification_scenarios.json")
_STOP_WORDS = {
    "a", "an", "the", "i", "my", "me", "we", "our", "you", "your", "do", "does", "did",
    "is", "are", "was", "were", "be", "been", "being", "can", "could", "should", "would",
    "will", "shall", "what", "which", "when", "where", "why", "who", "how", "much", "many",
    "for", "to", "of", "in", "on", "at", "if", "have", "has", "had", "already", "please",
    "need", "required", "apply", "applied", "use", "using", "do", "i", "my", "field", "crop",
    # Generic quantity words should not outweigh the actual crop/topic terms.
    "quantity", "amount", "dose", "rate", "requirement", "required", "tell", "farm", "farmer",
}
_CANCEL_RE = re.compile(r"^\s*(cancel|stop|forget it|leave it|never mind|nevermind|start over)\s*[.!]*$", re.I)
_NEW_QUESTION_RE = re.compile(
    r"^\s*(what|how|which|when|where|why|who|can i|should i|is it|tell me|explain)\b.*\??\s*$",
    re.I,
)


def _state_dir() -> Path:
    configured = getattr(settings, "WHATSAPP_CLARIFICATION_DIR", None)
    path = Path(configured) if configured else Path(getattr(settings, "BASE_DIR", Path.cwd())) / "whatsapp_clarifications"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _phone_key(phone: str) -> str:
    digits = re.sub(r"\D", "", str(phone or ""))
    if not digits:
        return ""
    # Hash the normalized number so filenames do not expose phone numbers.
    import hashlib
    return hashlib.sha256(digits.encode("utf-8")).hexdigest()


def _paths(phone: str):
    key = _phone_key(phone)
    if not key:
        return None, None
    root = _state_dir()
    return root / f"{key}.json", root / f"{key}.lock"


def _with_lock(phone: str):
    path, lock_path = _paths(phone)
    if path is None:
        return None, None, None
    if fcntl is not None:
        lock_file = open(lock_path, "a+b")
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        return path, lock_file, lock_file
    # Windows fallback protects threads in this process. Production Linux uses flock.
    with _LOCAL_LOCKS_GUARD:
        lock = _LOCAL_LOCKS.setdefault(str(lock_path), threading.RLock())
    lock.acquire()
    return path, lock, lock


def _read(path: Path):
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except FileNotFoundError:
        return None
    except Exception as exc:
        log.warning("Could not read WhatsApp clarification state %s: %s", path, exc)
        return None


def _write_atomic(path: Path, data: dict):
    tmp_name = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
            tmp_name = f.name
        os.replace(tmp_name, path)
    finally:
        if tmp_name and os.path.exists(tmp_name):
            try:
                os.unlink(tmp_name)
            except OSError:
                pass


def _tokens(text: str) -> set[str]:
    tokens = {t for t in re.findall(r"[a-z0-9]+", (text or "").lower())
              if t not in _STOP_WORDS and len(t) > 1}
    # Normalize common wording variants without making unrelated scenarios match.
    aliases = {"rice": "paddy", "fertiliser": "fertilizer", "fertilisers": "fertilizers"}
    return {aliases.get(t, t) for t in tokens}


def _load_scenarios():
    try:
        with _SCENARIOS_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception as exc:
        log.warning("Could not load clarification scenarios from %s: %s", _SCENARIOS_PATH, exc)
        return []


def match_scenario(query: str):
    """Return the best spreadsheet scenario only when similarity is convincing."""
    raw_query = query or ""
    if re.search(r"%[0-9A-Fa-f]{2}", raw_query):
        raw_query = unquote_plus(raw_query)
    q = " ".join(raw_query.lower().split())
    q_tokens = _tokens(q)
    scenarios = _load_scenarios()
    # Preserve exact matches even when the question has only two informative
    # tokens (e.g. "Can I mix urea with MOP?"). For short, non-exact queries,
    # there is too little evidence to distinguish related fertilizer scenarios.
    def normalized_phrase(value: str) -> str:
        return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()
    q_phrase = normalized_phrase(raw_query)
    for scenario in scenarios:
        initial = scenario.get("initial_question", "")
        if q_phrase == normalized_phrase(initial):
            return {**scenario, "match_score": 1.0}
    if len(q_tokens) < 3:
        return None
    best, best_score = None, 0.0
    for scenario in scenarios:
        initial = scenario.get("initial_question", "")
        s_tokens = _tokens(initial)
        if not s_tokens:
            continue
        overlap = len(q_tokens & s_tokens)
        query_coverage = overlap / max(1, len(q_tokens))
        jaccard = overlap / max(1, len(q_tokens | s_tokens))
        # Token evidence is more reliable than raw character similarity here:
        # generic strings such as "what ... fertilizer ... should I" can look
        # textually similar while referring to different scenarios (e.g. green gram).
        score = 0.72 * query_coverage + 0.28 * jaccard
        if score > best_score:
            best, best_score = scenario, score
    # Avoid starting a clarification session for unrelated questions.
    if best and best_score >= 0.68:
        return {**best, "match_score": round(best_score, 3)}
    return None


def handle_message(phone: str, query_en: str, *, timeout_minutes: int = 45) -> dict:
    """Actions: continue, ask, or complete. `query_en` must already be English."""
    if not phone or not (query_en or "").strip():
        return {"action": "continue"}
    path, lock_file, _ = _with_lock(phone)
    if path is None:
        return {"action": "continue"}
    try:
        state = _read(path)
        now = time.time()
        if state and now - float(state.get("updated_at", state.get("created_at", 0))) > timeout_minutes * 60:
            state = None
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass

        text = query_en.strip()
        if re.search(r"%[0-9A-Fa-f]{2}", text):
            text = unquote_plus(text).strip()
        if state and _CANCEL_RE.match(text):
            path.unlink(missing_ok=True)
            return {"action": "continue", "cancelled_clarification": True}

        if state:
            # A clearly new question starts a new turn rather than being swallowed
            # as an answer to the pending clarification.
            if _NEW_QUESTION_RE.match(text) and "?" in text:
                path.unlink(missing_ok=True)
                state = None
            else:
                answer = text
                original = str(state.get("original_question", "")).strip()
                scenario = state.get("scenario", {})
                path.unlink(missing_ok=True)
                combined = (
                    "Farmer's original question: " + original + "\n"
                    "Clarifying question asked: " + str(scenario.get("follow_up_question", "")).strip() + "\n"
                    "Farmer's clarification: " + answer + "\n"
                    "Answer the original question using the clarification where relevant. "
                    "Do not assume details the farmer did not provide."
                )
                return {
                    "action": "complete",
                    "original_question": original,
                    "clarification_answer": answer,
                    "combined_query": combined,
                    "scenario_question": scenario.get("follow_up_question", ""),
                    "scenario_initial_question": scenario.get("initial_question", ""),
                }

        scenario = match_scenario(text)
        if not scenario:
            return {"action": "continue"}
        state = {
            "version": 1,
            "status": "awaiting_farmer_clarification",
            "original_question": text,
            "scenario": scenario,
            "created_at": now,
            "updated_at": now,
        }
        _write_atomic(path, state)
        return {
            "action": "ask",
            "original_question": text,
            "question": scenario["follow_up_question"],
            "scenario_initial_question": scenario["initial_question"],
            "match_score": scenario["match_score"],
        }
    finally:
        try:
            if fcntl is not None:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
                lock_file.close()
            else:
                lock_file.release()
        except Exception:
            pass


def clear_session(phone: str) -> None:
    path, lock_file, _ = _with_lock(phone)
    if path is None:
        return
    try:
        path.unlink(missing_ok=True)
    finally:
        try:
            if fcntl is not None:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
                lock_file.close()
            else:
                lock_file.release()
        except Exception:
            pass
