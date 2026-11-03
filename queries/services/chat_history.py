"""
queries/services/chat_history.py
================================
Per-phone-number chat history persistence using JSON files.

Features:
- Stores the latest N (default: 25) conversation turns for each phone number.
- JSON serialization (safe, no arbitrary code execution on load).
- Sanitizes phone numbers for file naming (e.g. +919876543210 -> 919876543210.json).
- Concurrency-safe: file lock around read-modify-write, atomic rename.
- Backward-compatible: migrates legacy .pkl files to .json on first access.
- Formats past conversation history as context for the LLM prompt.
"""
from __future__ import annotations

import json
import logging
import os
import pickle
import re
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Optional

try:
    import fcntl
except ImportError:
    fcntl = None  # type: ignore[assignment]
    import msvcrt

from django.conf import settings

log = logging.getLogger("irri")

_DEFAULT_DIR = Path(getattr(settings, "BASE_DIR", Path("."))) / "chat_history"


def _history_dir() -> Path:
    base = getattr(settings, "WHATSAPP_CHAT_HISTORY_DIR", None)
    path = Path(base) if base else _DEFAULT_DIR
    try:
        path.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        log.warning("Could not create chat history directory %s: %s", path, exc)
    return path


def _sanitize_phone(phone_number: str) -> str:
    """Normalize phone number to digits only for safe filename creation."""
    digits = re.sub(r"\D", "", str(phone_number or ""))
    if not digits:
        log.warning("chat_history: empty phone number after sanitization")
        return ""
    return digits


def _json_path(phone_number: str) -> Path:
    sanitized = _sanitize_phone(phone_number)
    if not sanitized:
        return Path(os.devnull)
    return _history_dir() / f"{sanitized}.json"


def _pkl_path(phone_number: str) -> Path:
    sanitized = _sanitize_phone(phone_number)
    if not sanitized:
        return Path(os.devnull)
    return _history_dir() / f"{sanitized}.pkl"


def _lock_path(phone_number: str) -> Path:
    sanitized = _sanitize_phone(phone_number)
    return _history_dir() / f".{sanitized}.lock"


@contextmanager
def _file_lock(phone_number: str):
    """Per-phone file lock around read-modify-write."""
    lp = _lock_path(phone_number)
    fd = None
    try:
        fd = open(lp, "w")
        if fcntl is not None:
            fcntl.flock(fd, fcntl.LOCK_EX)
        else:
            msvcrt.locking(fd.fileno(), msvcrt.LK_LOCK, 1)
        yield
    finally:
        if fd is not None:
            try:
                if fcntl is not None:
                    fcntl.flock(fd, fcntl.LOCK_UN)
                else:
                    msvcrt.locking(fd.fileno(), msvcrt.LK_UNLCK, 1)
                fd.close()
            except Exception:
                pass


def _validate_turn(turn) -> bool:
    """A valid turn is a dict with at least one of 'user' or 'assistant'."""
    return (isinstance(turn, dict)
            and (turn.get("user") or turn.get("assistant")))


def _sanitize_history(raw: list) -> list:
    """Keep only valid turns from a loaded history."""
    return [t for t in raw if _validate_turn(t)]


def _migrate_pkl(phone_number: str) -> list:
    """One-time migration: read .pkl, write .json, delete .pkl.

    The pickle is loaded inside a try/except that catches everything —
    a corrupted or malicious .pkl is discarded, not executed blindly.
    """
    pkl = _pkl_path(phone_number)
    if not pkl.is_file():
        return []
    try:
        with open(pkl, "rb") as f:
            data = pickle.load(f)
        if not isinstance(data, list):
            log.warning("chat_history: pkl for %s had non-list root; discarding",
                        _sanitize_phone(phone_number))
            data = []
        history = _sanitize_history(data)
        _write_json(phone_number, history)
        pkl.unlink(missing_ok=True)
        log.info("chat_history: migrated %s turns from pkl to json for %s",
                 len(history), _sanitize_phone(phone_number))
        return history
    except Exception as exc:
        log.warning("chat_history: pkl migration failed for %s: %s",
                    _sanitize_phone(phone_number), exc)
        # Don't delete the pkl on failure — let it be retried.
        return []


def _read_json(phone_number: str) -> list:
    """Read history from JSON, falling back to pkl migration."""
    jp = _json_path(phone_number)
    if jp.is_file():
        try:
            with open(jp, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return _sanitize_history(data)
        except Exception as exc:
            log.warning("chat_history: failed to read %s: %s", jp, exc)
        return []
    # No .json — try migrating from .pkl
    return _migrate_pkl(phone_number)


def _write_json(phone_number: str, history: list) -> None:
    """Atomic write: tmpfile + rename."""
    jp = _json_path(phone_number)
    h_dir = _history_dir()
    try:
        with tempfile.NamedTemporaryFile(
            "w", dir=h_dir, suffix=".tmp", delete=False, encoding="utf-8",
        ) as tmp:
            json.dump(history, tmp, ensure_ascii=False)
            tmp_path = Path(tmp.name)
        tmp_path.replace(jp)
    except Exception as exc:
        log.warning("chat_history: failed to write %s: %s", jp, exc)
        try:
            if "tmp_path" in locals() and tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass


def _clamp_max_turns(max_turns: int | None) -> int:
    if max_turns is None:
        max_turns = getattr(settings, "WHATSAPP_CHAT_HISTORY_MAX_TURNS", 25)
    return max(int(max_turns), 1)


def load_chat_history(
    phone_number: str, max_turns: Optional[int] = None,
) -> List[Dict[str, str]]:
    """Load recent chat history for a phone number.

    Returns a list of message dicts:
        [{"user": "...", "assistant": "..."}, ...]
    """
    if not phone_number or not _sanitize_phone(phone_number):
        return []
    max_turns = _clamp_max_turns(max_turns)
    with _file_lock(phone_number):
        history = _read_json(phone_number)
    return history[-max_turns:]


def save_chat_turn(
    phone_number: str,
    user_text: str,
    assistant_text: str,
    max_turns: Optional[int] = None,
) -> None:
    """Append a new conversation turn and keep the last max_turns items.

    The entire read-modify-write is locked per phone number so concurrent
    requests for the same farmer cannot overwrite each other's turns.
    """
    if not phone_number or not _sanitize_phone(phone_number):
        return
    if not (user_text or assistant_text):
        return
    max_turns = _clamp_max_turns(max_turns)

    with _file_lock(phone_number):
        history = _read_json(phone_number)
        history.append({
            "user": (user_text or "").strip(),
            "assistant": (assistant_text or "").strip(),
        })
        _write_json(phone_number, history[-max_turns:])


def format_history_for_prompt(history: List[Dict[str, str]]) -> str:
    """Format past conversation turns into a prompt section for the LLM."""
    if not history:
        return ""

    lines = [
        "PRIOR CONVERSATION HISTORY WITH THIS FARMER (for follow-up context; "
        "use this to understand pronouns, crop references, or follow-up questions, "
        "but ensure advice remains strictly grounded in the CONTEXT below):"
    ]
    for turn in history:
        u = (turn.get("user") or "").strip()
        a = (turn.get("assistant") or "").strip()
        if u:
            lines.append(f"Farmer: {u}")
        if a:
            lines.append(f"PaddyMitra: {a}")

    return "\n" + "\n".join(lines) + "\n"
