"""
Hook logic for MemPalace — Python implementation of session-start, stop, and precompact hooks.

Reads JSON from stdin, outputs JSON to stdout.
Supported hooks: session-start, stop, precompact
Supported harnesses: claude-code, codex (extensible to cursor, gemini, etc.)
"""

import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

SAVE_INTERVAL = 15
STATE_DIR = Path.home() / ".mempalace" / "hook_state"
PALACE_ROOT = Path.home() / ".mempalace"


def _detached_popen_kwargs() -> dict:
    """Kwargs that fully detach a Popen child so the hook process can exit.

    Without these, Windows holds the parent open until the child closes the
    inherited stdout/stderr handles — manifesting as "Stop hook hangs" at
    session end (#1268). On POSIX the parent can already exit (orphan
    reparents to init), but ``start_new_session`` makes the boundary
    explicit so signals to the hook don't propagate to the background mine.
    """
    kwargs: dict = {"stdin": subprocess.DEVNULL, "close_fds": True}
    if os.name == "nt":
        flags = 0
        for name in ("DETACHED_PROCESS", "CREATE_NEW_PROCESS_GROUP", "CREATE_BREAKAWAY_FROM_JOB"):
            flags |= getattr(subprocess, name, 0)
        if flags:
            kwargs["creationflags"] = flags
    else:
        kwargs["start_new_session"] = True
    return kwargs


def _palace_root_exists() -> bool:
    """User-removable kill-switch.

    If ~/.mempalace/ does not exist, the user has explicitly cleared it.
    All hook side effects (logging, state dir creation, mining, ingestion)
    must respect this and short-circuit BEFORE touching disk — including
    before logging the short-circuit itself.

    Uses ``is_dir()`` rather than ``exists()`` so a stray regular file at
    ``~/.mempalace`` (or a broken symlink) is treated as absent — otherwise
    the kill-switch would be bypassed and ``STATE_DIR.mkdir()`` would later
    crash on ``NotADirectoryError``.
    """
    return PALACE_ROOT.is_dir()


def _mempalace_python() -> str:
    """Return the python interpreter that has mempalace installed.

    When hooks are invoked by Claude Code, sys.executable may be the system
    python which lacks chromadb and other deps.  Resolution order:
    1. MEMPALACE_PYTHON env var (explicit override)
    2. Venv python from package install path
    3. Editable install: venv/ sibling to mempalace/
    4. sys.executable fallback
    """
    # Honor explicit override (used by shell hook wrappers)
    env_python = os.environ.get("MEMPALACE_PYTHON", "")
    if env_python and os.path.isfile(env_python) and os.access(env_python, os.X_OK):
        return env_python
    # This file lives at <venv>/lib/pythonX.Y/site-packages/mempalace/hooks_cli.py
    # or <project>/mempalace/hooks_cli.py (editable install).
    venv_bin = Path(__file__).resolve().parents[3] / "bin" / "python"
    if venv_bin.is_file():
        return str(venv_bin)
    # Editable install: assumes project root has a venv/ sibling to mempalace/
    project_venv = Path(__file__).resolve().parents[1] / "venv" / "bin" / "python"
    if project_venv.is_file():
        return str(project_venv)
    return sys.executable


_RECENT_MSG_COUNT = 30  # how many recent user messages to summarize

STOP_BLOCK_REASON = (
    "AUTO-SAVE checkpoint (MemPalace). Save this session's key content:\n"
    "1. mempalace_diary_write — session summary (what was discussed, "
    "key decisions, current state of work)\n"
    "2. mempalace_add_drawer — verbatim quotes, decisions, code snippets "
    "(place in appropriate wing and room)\n"
    "3. mempalace_kg_add — entity relationships (optional)\n"
    "For THIS save, use MemPalace MCP tools only (not auto-memory .md files). "
    "Use verbatim quotes where possible. Continue conversation after saving."
)

PRECOMPACT_BLOCK_REASON = (
    "COMPACTION IMMINENT (MemPalace). Save ALL session content before context is lost:\n"
    "1. mempalace_diary_write — thorough session summary\n"
    "2. mempalace_add_drawer — ALL verbatim quotes, decisions, code, context "
    "(place each in appropriate wing and room)\n"
    "3. mempalace_kg_add — entity relationships (optional)\n"
    "For THIS save, use MemPalace MCP tools only (not auto-memory .md files). "
    "Be thorough — after compaction this is all that survives. "
    "Save everything to MemPalace, then allow compaction to proceed."
)


def _sanitize_session_id(session_id: str) -> str:
    """Only allow alnum, dash, underscore to prevent path traversal."""
    sanitized = re.sub(r"[^a-zA-Z0-9_-]", "", session_id)
    return sanitized or "unknown"


def _validate_transcript_path(transcript_path: str) -> Path:
    """Validate and resolve a transcript path, rejecting paths outside expected roots.

    Returns a resolved Path if valid, or None if the path should be rejected.
    Accepted paths must:
    - Have a .jsonl or .json extension
    - Not contain '..' after resolution (path traversal prevention)
    """
    if not transcript_path:
        return None
    path = Path(transcript_path).expanduser().resolve()
    if path.suffix not in (".jsonl", ".json"):
        return None
    # Reject if the original input contained '..' traversal components
    if ".." in Path(transcript_path).parts:
        return None
    return path


def _count_human_messages(transcript_path: str) -> int:
    """Count human messages in a JSONL transcript, skipping command-messages."""
    path = _validate_transcript_path(transcript_path)
    if path is None:
        if transcript_path:
            _log(f"WARNING: transcript_path rejected by validator: {transcript_path!r}")
        return 0
    if not path.is_file():
        return 0
    count = 0
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    msg = entry.get("message", {})
                    if isinstance(msg, dict) and msg.get("role") == "user":
                        content = msg.get("content", "")
                        if isinstance(content, str):
                            if "<command-message>" in content:
                                continue
                        elif isinstance(content, list):
                            text = " ".join(
                                b.get("text", "") for b in content if isinstance(b, dict)
                            )
                            if "<command-message>" in text:
                                continue
                        count += 1
                    # Also handle Codex CLI transcript format
                    # {"type": "event_msg", "payload": {"type": "user_message", "message": "..."}}
                    elif entry.get("type") == "event_msg":
                        payload = entry.get("payload", {})
                        if isinstance(payload, dict) and payload.get("type") == "user_message":
                            msg_text = payload.get("message", "")
                            if isinstance(msg_text, str) and "<command-message>" not in msg_text:
                                count += 1
                except (json.JSONDecodeError, AttributeError):
                    pass
    except OSError:
        return 0
    return count


_state_dir_initialized = False


def _log(message: str):
    """Append to hook state log file."""
    if not _palace_root_exists():
        return  # User removed the palace; do not recreate by logging
    global _state_dir_initialized
    try:
        if not _state_dir_initialized:
            STATE_DIR.mkdir(parents=True, exist_ok=True)
            try:
                STATE_DIR.chmod(0o700)
            except (OSError, NotImplementedError):
                pass
            _state_dir_initialized = True
        log_path = STATE_DIR / "hook.log"
        is_new = not log_path.exists()
        timestamp = datetime.now().strftime("%H:%M:%S")
        with open(log_path, "a") as f:
            f.write(f"[{timestamp}] {message}\n")
        if is_new:
            try:
                log_path.chmod(0o600)
            except (OSError, NotImplementedError):
                pass
    except OSError:
        pass


def _output(data: dict):
    """Print JSON to stdout without importing modules that may redirect streams.

    If mempalace.mcp_server is already loaded, reuse its saved real stdout fd.
    Otherwise, write directly to fd 1 so hook responses still go to stdout even
    if sys.stdout has been redirected elsewhere.
    """
    payload = (json.dumps(data, indent=2, ensure_ascii=False) + "\n").encode("utf-8")

    real_stdout_fd: int | None = None
    mcp_mod = sys.modules.get("mempalace.mcp_server") or sys.modules.get(
        f"{__package__}.mcp_server" if __package__ else "mcp_server"
    )
    if mcp_mod is not None:
        real_stdout_fd = getattr(mcp_mod, "_REAL_STDOUT_FD", None)

    fd = real_stdout_fd if real_stdout_fd is not None else 1
    offset = 0
    try:
        while offset < len(payload):
            try:
                offset += os.write(fd, payload[offset:])
            except InterruptedError:
                continue
        return
    except OSError:
        pass

    sys.stdout.buffer.write(payload)
    sys.stdout.buffer.flush()


def _get_mine_targets() -> list[tuple[str, str]]:
    """Return the list of ``(dir, mode)`` targets for auto-ingest.

    MEMPAL_DIR (when set and resolvable) contributes a ``"projects"``
    target. Transcript ingestion is handled separately by
    ``_ingest_transcript`` — emitting it here too would double-mine the
    same JSONL into a different wing on every hook fire (#1231 review).

    An empty list means no MEMPAL_DIR ingest should run.
    """
    targets: list[tuple[str, str]] = []
    mempal_dir = os.environ.get("MEMPAL_DIR", "")
    if mempal_dir:
        resolved = Path(mempal_dir).expanduser().resolve()
        if resolved.is_dir():
            targets.append((str(resolved), "projects"))
    return targets


# Per-target PID guard.
#
# Hook fires ingest mines in the background. If a previous fire's child is
# still running for the *same* target (same source dir, mode, wing), the new
# fire should skip rather than pile up — multiple concurrent mines against the
# same source corrupt the HNSW index and exhaust disk via duplicate upserts
# (#1212, #1206). But mines targeting *different* sources / modes must remain
# independent so the user can have e.g. project-mining and transcript-ingest
# running in parallel.
#
# The single ``mine.pid`` global file used previously failed both ways: the
# guard was rebuilt every spawn (so two near-simultaneous fires both passed
# the check before either wrote), and the file was unconditionally overwritten
# (so the second spawn lost the first PID, orphaning it). The replacement is
# a directory of per-target slots, claimed via ``O_CREAT | O_EXCL`` so the
# claim is atomic and per-target.
_MINE_PID_DIR = STATE_DIR / "mine_pids"

# The per-process PID file path is communicated to the mine subprocess via
# this env var so the child's cleanup hook (in miner.py) can remove its
# own slot on exit without scanning the whole directory.
_MINE_PID_FILE_ENV = "MEMPALACE_MINE_PID_FILE"


def _pid_file_for_cmd(cmd: list[str]) -> Path:
    """Return the per-target PID file path for a mine subcommand.

    The key is derived from the mine arguments (everything after ``mine``)
    so different (dir, mode, wing) combinations get independent slots.
    Two fires with the same arguments collapse to the same slot — which is
    exactly the dedup we want.
    """
    try:
        idx = cmd.index("mine")
        key = " ".join(cmd[idx:])
    except ValueError:
        key = " ".join(cmd)
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
    return _MINE_PID_DIR / f"mine_{digest}.pid"


def _pid_alive(pid: int) -> bool:
    """Cross-platform existence check for a PID.

    On POSIX, ``os.kill(pid, 0)`` is the well-known no-op existence probe.
    On Windows, ``os.kill`` maps to ``TerminateProcess(handle, sig)`` and
    would *terminate* the target process with exit code ``sig`` — using
    it here would kill our own mine child (or worse, the caller itself).
    Use ``OpenProcess`` + ``GetExitCodeProcess`` via ctypes instead.
    """
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        STILL_ACTIVE = 259
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            return False
        try:
            code = wintypes.DWORD()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                return False
            return code.value == STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ValueError):
        return False


def _mine_already_running(cmd: list[str]) -> bool:
    """Return True if a previous mine for ``cmd``'s target is still alive."""
    pid_file = _pid_file_for_cmd(cmd)
    try:
        recorded = pid_file.read_text().strip()
    except OSError:
        return False
    if not recorded.isdigit():
        return False
    return _pid_alive(int(recorded))


def _claim_mine_slot(cmd: list[str]) -> Optional[Path]:
    """Atomically reserve the per-target PID slot for ``cmd``.

    Returns the slot path on success, or ``None`` if the target is
    already being mined by a live process. The reservation is done via
    ``O_CREAT | O_EXCL`` so two simultaneous hook fires can never both
    pass the check; one wins, the other returns None.

    A stale slot (file exists but the recorded PID is dead) is reclaimed
    transparently — orphan miners that crashed without cleanup do not
    block future hook fires forever.
    """
    pid_file = _pid_file_for_cmd(cmd)
    pid_file.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(str(pid_file), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        os.close(fd)
        return pid_file
    except FileExistsError:
        pass
    # Slot exists. If the holder is alive, defer.
    if _mine_already_running(cmd):
        return None
    # Stale entry; reclaim. The unlink+create is racy against another hook
    # firing right now, but the second create's O_EXCL will fail and that
    # caller will see the live PID via the next round.
    try:
        pid_file.unlink()
    except FileNotFoundError:
        pass
    except OSError:
        return None
    try:
        fd = os.open(str(pid_file), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        os.close(fd)
        return pid_file
    except FileExistsError:
        return None


def _spawn_mine(cmd: list) -> None:
    """Spawn a mine subprocess if no live mine is already targeting it.

    The PID slot is claimed atomically *before* the spawn, so two near-
    simultaneous hook fires can't both proceed — the second sees the
    claimed slot and silently skips. The spawned process inherits a
    ``MEMPALACE_MINE_PID_FILE`` env var so its cleanup hook can remove
    the slot on exit without scanning the directory.
    """
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    log_path = STATE_DIR / "hook.log"
    pid_file = _claim_mine_slot(cmd)
    if pid_file is None:
        _log(f"Skipping mine: target already running ({' '.join(cmd[-3:])})")
        return
    child_env = os.environ.copy()
    child_env[_MINE_PID_FILE_ENV] = str(pid_file)
    with open(log_path, "a") as log_f:
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=log_f,
                stderr=log_f,
                env=child_env,
                **_detached_popen_kwargs(),
            )
        except OSError:
            # Spawn failed; release the slot we just claimed so the next
            # hook fire can try again rather than skipping forever.
            try:
                pid_file.unlink()
            except OSError:
                pass
            raise
    try:
        pid_file.write_text(str(proc.pid))
    except OSError:
        pass


def _maybe_auto_ingest():
    """Background-mine MEMPAL_DIR (project files) if set.

    Transcript convos are ingested separately via ``_ingest_transcript``
    in the hook handlers — this function does not handle them, to avoid
    asymmetric interpreter handling and PID-file overwrite when both
    targets fire from a single hook call (#1231 review).

    Per-target dedup is done by ``_spawn_mine`` itself: each (dir, mode)
    target gets its own PID slot, so distinct targets never block each
    other but a re-fire of the same target while the previous one is
    still running is silently skipped.
    """
    targets = _get_mine_targets()
    if not targets:
        return
    for mine_dir, mode in targets:
        try:
            _spawn_mine([_mempalace_python(), "-m", "mempalace", "mine", mine_dir, "--mode", mode])
        except OSError:
            pass


def _mine_sync():
    """Synchronously mine MEMPAL_DIR (precompact path).

    Transcript convos are ingested separately via ``_ingest_transcript``
    in ``hook_precompact`` — keeping them out of this function avoids
    timeout stacking against the harness 30s ceiling (#1231 review).
    """
    targets = _get_mine_targets()
    if not targets:
        return
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    log_path = STATE_DIR / "hook.log"
    for mine_dir, mode in targets:
        try:
            with open(log_path, "a") as log_f:
                subprocess.run(
                    [
                        _mempalace_python(),
                        "-m",
                        "mempalace",
                        "mine",
                        mine_dir,
                        "--mode",
                        mode,
                    ],
                    stdout=log_f,
                    stderr=log_f,
                    timeout=60,
                )
        except (OSError, subprocess.TimeoutExpired):
            pass


def _desktop_toast(body: str, title: str = "MemPalace"):
    """Send a desktop notification via notify-send. Fails silently."""
    try:
        subprocess.Popen(
            ["notify-send", "--app-name=MemPalace", "--icon=brain", title, body],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            **_detached_popen_kwargs(),
        )
    except OSError:
        pass


def _extract_recent_messages(transcript_path: str, count: int = _RECENT_MSG_COUNT) -> list[str]:
    """Extract the last N user messages from a JSONL transcript."""
    path = Path(transcript_path).expanduser()
    if not path.is_file():
        return []
    messages = []
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    # Claude Code format
                    msg = entry.get("message") or entry.get("event_message") or {}
                    if isinstance(msg, dict) and msg.get("role") == "user":
                        content = msg.get("content", "")
                        if isinstance(content, list):
                            content = " ".join(
                                b.get("text", "") for b in content if isinstance(b, dict)
                            )
                        if not isinstance(content, str) or not content.strip():
                            continue
                        if "<command-message>" in content or "<system-reminder>" in content:
                            continue
                        messages.append(content.strip()[:200])
                    # Codex CLI format
                    elif entry.get("type") == "event_msg":
                        payload = entry.get("payload", {})
                        if isinstance(payload, dict) and payload.get("type") == "user_message":
                            text = payload.get("message", "")
                            if isinstance(text, str) and text.strip():
                                if "<command-message>" not in text:
                                    messages.append(text.strip()[:200])
                except (json.JSONDecodeError, AttributeError):
                    pass
    except OSError:
        return []
    return messages[-count:]


_THEME_STOPWORDS = frozenset(
    "the a an and or but in on at to for of is it i me my you your we our "
    "this that with from by was were be been are not no yes can do did dont "
    "will would should could have has had lets let just also like so if then "
    "ok okay sure yeah hey hi here there what when where how why which some "
    "all any each every about into out up down over after before between "
    "get got make made need want use used using check look see run try "
    "know think right now still already really very much more most too "
    "file files code one two new first last next thing things way well".split()
)


# ── Hook LLM scaffolding ──────────────────────────────────────────────────
# A single lazy provider instance shared across hook fires within a process,
# so we pay Ollama warmup at most once. Process-level caching is enough —
# hooks are short-lived (<60s) and we don't share state across fires.
#
# ``_get_hook_llm`` returns ``None`` when:
#   - no hook flag is on (caller does the cheap deterministic path)
#   - provider build fails (unknown name, missing endpoint, etc.)
#   - ``check_available()`` reports the model is missing or Ollama is down
# Returning ``None`` is the contract callers rely on for fallback to
# deterministic logic. We never raise out of a hook.

_HOOK_LLM_CACHE: dict = {}


def _get_hook_llm(timeout: int | None = None):
    """Return a configured LLM provider for hook use, or None on any failure.

    Cached per (provider, model, timeout) within the process. Probing
    ``check_available`` once on first build keeps the per-fire latency low —
    subsequent calls skip the probe and go straight to ``classify``.

    Pass ``timeout`` to override the default ``hook_llm_timeout_s``. Used by
    ``_kg_extract_from_transcript`` which needs a longer budget than the
    Stop-hook themes/diary calls. Each unique timeout gets its own cache
    entry, so an unavailable Ollama at one timeout doesn't poison the other.
    """
    try:
        from .config import MempalaceConfig
    except Exception as exc:
        _log(f"hook_llm: config import failed: {exc}")
        return None

    try:
        cfg = MempalaceConfig()
        provider_name = cfg.hook_llm_provider
        model = cfg.hook_llm_model
        timeout = timeout if timeout is not None else cfg.hook_llm_timeout_s
    except Exception as exc:
        _log(f"hook_llm: config read failed: {exc}")
        return None

    cache_key = (provider_name, model, timeout)
    cached = _HOOK_LLM_CACHE.get(cache_key)
    if cached is not None:
        # Sentinel: a previously-failed probe. Don't retry within the process —
        # Ollama isn't going to come up mid-hook, and retrying just adds
        # 5s of urlopen timeout to every Stop fire.
        if cached == "__unavailable__":
            return None
        return cached

    try:
        from .llm_client import get_provider
    except Exception as exc:
        _log(f"hook_llm: llm_client import failed: {exc}")
        _HOOK_LLM_CACHE[cache_key] = "__unavailable__"
        return None

    try:
        provider = get_provider(provider_name, model=model, timeout=timeout)
    except Exception as exc:
        _log(f"hook_llm: get_provider({provider_name!r}, {model!r}) failed: {exc}")
        _HOOK_LLM_CACHE[cache_key] = "__unavailable__"
        return None

    try:
        ok, msg = provider.check_available()
    except Exception as exc:
        _log(f"hook_llm: check_available raised: {exc}")
        _HOOK_LLM_CACHE[cache_key] = "__unavailable__"
        return None

    if not ok:
        _log(f"hook_llm: provider unavailable: {msg}")
        _HOOK_LLM_CACHE[cache_key] = "__unavailable__"
        return None

    _HOOK_LLM_CACHE[cache_key] = provider
    return provider


def _llm_themes(messages: list[str], max_themes: int = 7) -> list[str]:
    """Ask gemma4 for 3-7 topical entities from recent messages.

    Returns ``[]`` on any failure (caller unions with keyword themes).
    Never raises — hooks must not crash when the LLM is misbehaving.

    Output contract: gemma is asked for JSON ``{"topics": [...]}`` so we
    can use Ollama's ``format=json`` mode and avoid prose-wrapper parsing
    headaches. Names are sanitized via :func:`sanitize_name` to drop any
    hallucinated path-traversal or null-byte payload before they reach
    drawer metadata.
    """
    provider = _get_hook_llm()
    if provider is None:
        return []
    if not messages:
        return []

    try:
        from .config import sanitize_name
    except Exception:
        return []

    # Trim aggressively — gemma at 4B has a small useful prompt window and
    # we want this under the per-call timeout budget. Last 30 messages,
    # 200 chars each ≈ 6KB, well under any reasonable context.
    sample = "\n".join(f"- {m[:200]}" for m in messages[-30:])
    system = (
        "You extract topical entities from chat messages. Return ONLY a JSON "
        'object of the shape {"topics": ["t1", "t2", ...]}. Each topic is a '
        "short bare noun or proper noun (1-3 words), no prose, no punctuation, "
        "no quotes inside the topic strings. 3 to 7 topics. No explanation."
    )
    user = f"Messages:\n{sample}\n\nReturn JSON only."

    try:
        resp = provider.classify(system=system, user=user, json_mode=True)
    except Exception as exc:
        _log(f"hook_llm: classify failed: {exc}")
        return []

    raw = (resp.text or "").strip()
    if not raw:
        return []

    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        # Some local models still wrap JSON in prose despite ``format=json``.
        # Cheap recovery: pull the first ``{...}`` block and try once.
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            return []
        try:
            parsed = json.loads(match.group(0))
        except (json.JSONDecodeError, TypeError):
            return []

    topics = parsed.get("topics") if isinstance(parsed, dict) else None
    if not isinstance(topics, list):
        return []

    out: list[str] = []
    seen: set = set()
    for t in topics:
        if not isinstance(t, str):
            continue
        candidate = t.strip().strip("\"'`")
        if not candidate:
            continue
        try:
            safe = sanitize_name(candidate, field_name="theme")
        except ValueError:
            continue
        key = safe.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(safe)
        if len(out) >= max_themes:
            break
    return out


def _extract_themes(messages: list[str], max_themes: int = 3) -> list[str]:
    """Pull 2-3 distinctive topic words from recent messages.

    When ``hook_llm_themes`` is on, gemma4-extracted topics are unioned
    with the keyword path. Keyword extraction stays as a fallback because
    code identifiers (function names, file basenames) often beat the LLM
    on technical sessions, while gemma catches natural-language topics
    keyword frequency misses.

    Note: stopword list is English-only; non-English corpora will produce noisy themes.
    """
    from collections import Counter

    words: Counter[str] = Counter()
    for msg in messages:
        for word in msg.lower().split():
            # Strip punctuation, keep words 4+ chars
            clean = word.strip(".,;:!?\"'`()[]{}#<>/\\-_=+@$%^&*~")
            if len(clean) >= 4 and clean not in _THEME_STOPWORDS and clean.isalpha():
                words[clean] += 1
    keyword_themes = [w for w, _ in words.most_common(max_themes)]

    # LLM upgrade — gated, fails silently to keyword path on any error.
    try:
        from .config import MempalaceConfig

        if not MempalaceConfig().hook_llm_themes:
            return keyword_themes
    except Exception:
        return keyword_themes

    llm_themes = _llm_themes(messages, max_themes=max_themes + 4)
    if not llm_themes:
        return keyword_themes

    # Union, LLM-first. We don't cap the union too aggressively because
    # diary readers can ignore extras, but truncating to 7 prevents an
    # over-long checkpoint topic line.
    merged: list[str] = []
    seen: set = set()
    for t in (*llm_themes, *keyword_themes):
        key = t.lower()
        if key in seen:
            continue
        seen.add(key)
        merged.append(t)
        if len(merged) >= 7:
            break
    return merged


# ── Piece 1: LLM-composed diary AAAK section ──────────────────────────────
# Produces a structured AAAK suffix appended to the mechanical CHECKPOINT
# header. Suffix shape (single line, pipe-delimited):
#
#   summary:<one-line arc>|decisions:<a;b>|blockers:<x;y>|rating:★N
#
# Why the pipe/semicolon split: the existing CHECKPOINT envelope already
# uses ``|`` as a field separator, and the AAAK dialect uses ``;`` for
# multi-value entries inside a field. Mirroring that here keeps any
# downstream parser (dialect.py / search) unchanged.

# Predicates allowed for the PreCompact KG extraction. Anything else the
# LLM emits gets dropped silently rather than poisoning the graph with
# hallucinated relationship types. Keep this set conservative — it's
# easier to add predicates than to hunt down bad triples later.
_HOOK_KG_ALLOWED_PREDICATES = frozenset(
    [
        "works_on",
        "assigned_to",
        "blocked_by",
        "decided",
        "decides",
        "mentions",
        "discusses",
        "depends_on",
        "related_to",
        "owns",
        "uses",
        "uses_tool",
        "reports_to",
        "asked_about",
        "completed",
    ]
)


def _llm_compose_diary_suffix(messages: list[str], themes: list[str]) -> str | None:
    """Ask gemma4 for a structured AAAK suffix to attach to the diary CHECKPOINT.

    Returns the suffix string (no leading pipe) or ``None`` on any failure.
    Caller appends it to the mechanical envelope; the mechanical
    ``recent:...`` line is preserved as a fallback when this returns None
    so we never lose checkpoint searchability on a flaky LLM.
    """
    provider = _get_hook_llm()
    if provider is None:
        return None
    if not messages:
        return None

    sample = "\n".join(f"- {m[:200]}" for m in messages[-30:])
    theme_hint = ", ".join(themes[:5]) if themes else "none yet"
    system = (
        "You compress chat sessions into one-line AAAK diary entries. "
        "Return ONLY a JSON object with this exact shape: "
        '{"summary": "...", "decisions": ["..."], "blockers": ["..."], '
        '"rating": 1..5}. '
        "summary: one short sentence describing the arc. "
        "decisions: 0-5 short bare phrases of choices made. "
        "blockers: 0-5 short bare phrases of unresolved obstacles. "
        "rating: integer 1-5 reflecting how memorable / important this "
        "session is for future recall. No prose outside the JSON."
    )
    user = f"Themes so far: {theme_hint}\n\nMessages:\n{sample}\n\nReturn JSON only."

    try:
        resp = provider.classify(system=system, user=user, json_mode=True)
    except Exception as exc:
        _log(f"hook_llm: diary classify failed: {exc}")
        return None

    raw = (resp.text or "").strip()
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            return None
        try:
            parsed = json.loads(match.group(0))
        except (json.JSONDecodeError, TypeError):
            return None

    if not isinstance(parsed, dict):
        return None

    # Helpers — strict shape checks. We never raise; bad fields just become
    # empty/missing in the suffix. The suffix itself remains well-formed
    # so downstream parsers don't see a half-broken pipe-line.
    summary = parsed.get("summary")
    if not isinstance(summary, str):
        summary = ""
    summary = summary.strip().replace("|", "/").replace("\n", " ")[:240]

    def _list_str(field: str, cap: int) -> list[str]:
        raw_list = parsed.get(field)
        if not isinstance(raw_list, list):
            return []
        items: list[str] = []
        for v in raw_list:
            if not isinstance(v, str):
                continue
            cleaned = v.strip().replace("|", "/").replace(";", ",").replace("\n", " ")
            if cleaned:
                items.append(cleaned[:120])
            if len(items) >= cap:
                break
        return items

    decisions = _list_str("decisions", 5)
    blockers = _list_str("blockers", 5)

    rating_raw = parsed.get("rating")
    try:
        rating = int(rating_raw)
    except (TypeError, ValueError):
        rating = 0
    rating = max(0, min(5, rating))
    stars = "★" * rating if rating else ""

    parts: list[str] = []
    if summary:
        parts.append(f"summary:{summary}")
    if decisions:
        parts.append("decisions:" + ";".join(decisions))
    if blockers:
        parts.append("blockers:" + ";".join(blockers))
    if stars:
        parts.append(f"rating:{stars}")
    if not parts:
        return None
    return "|".join(parts)


def _save_diary_direct(
    transcript_path: str,
    session_id: str,
    wing: str = "",
    toast: bool = False,
) -> dict:
    """Write a diary checkpoint by calling the tool function directly (no MCP roundtrip).

    If `wing` is set, the entry lands in that wing (typically the project wing
    derived from the transcript path). Otherwise falls back to `tool_diary_write`'s
    default of `wing_session-hook`.

    Returns {"count": N, "themes": [...]} on success, {"count": 0} on failure.
    """
    messages = _extract_recent_messages(transcript_path)
    if not messages:
        _log("No recent messages to save")
        return {"count": 0}

    themes = _extract_themes(messages)

    # Build a compressed diary entry from recent conversation
    now = datetime.now()
    topics = "|".join(m[:80] for m in messages[-10:])
    entry = (
        f"CHECKPOINT:{now.strftime('%Y-%m-%d')}|session:{session_id}"
        f"|msgs:{len(messages)}|recent:{topics}"
    )

    # Optional LLM-enhanced AAAK suffix. We *append* rather than replace
    # the mechanical "recent:..." segment — if a downstream tool only
    # knows the original 4-field shape it still parses cleanly, and any
    # LLM hallucination is confined to its own labelled fields.
    try:
        from .config import MempalaceConfig

        if MempalaceConfig().hook_llm_diary:
            suffix = _llm_compose_diary_suffix(messages, themes)
            if suffix:
                entry = f"{entry}|{suffix}"
    except Exception as exc:
        _log(f"hook_llm: diary suffix skipped: {exc}")

    try:
        from .mcp_server import tool_diary_write

        result = tool_diary_write(
            agent_name="session-hook",
            entry=entry,
            topic="checkpoint",
            wing=wing,
        )
        if result.get("success"):
            _log(f"Diary checkpoint saved: {result.get('entry_id', '?')}")
            # Write state for ack tool to read
            try:
                ack_file = STATE_DIR / "last_checkpoint"
                ack_file.write_text(
                    json.dumps({"msgs": len(messages), "ts": now.isoformat()}),
                    encoding="utf-8",
                )
            except OSError:
                pass
            if toast:
                _desktop_toast(f"Checkpoint saved \u2014 {len(messages)} messages archived")
            return {"count": len(messages), "themes": themes}
        else:
            _log(f"Diary checkpoint failed: {result.get('error', 'unknown')}")
    except Exception as e:
        _log(f"Diary checkpoint error: {e}")
    return {"count": 0}


def _ingest_transcript(transcript_path: str):
    """Mine a Claude Code session transcript into the palace as a conversation."""
    path = Path(transcript_path).expanduser()
    if not path.is_file() or path.stat().st_size < 100:
        return

    from .config import MempalaceConfig

    try:
        MempalaceConfig()  # validate config loads
    except Exception:
        return

    try:
        # Route through ``_spawn_mine`` so the per-target PID guard kicks
        # in here too — repeated Stop/PreCompact fires for the same
        # transcript should not stack up parallel ingest mines.
        _spawn_mine(
            [
                _mempalace_python(),
                "-m",
                "mempalace",
                "mine",
                str(path.parent),
                "--mode",
                "convos",
                "--wing",
                "sessions",
            ]
        )
        _log(f"Transcript ingest started: {path.name}")
    except OSError:
        pass


SUPPORTED_HARNESSES = {"claude-code", "codex"}


def _parse_harness_input(data: dict, harness: str) -> dict:
    """Parse stdin JSON according to the harness type."""
    if harness not in SUPPORTED_HARNESSES:
        print(f"Unknown harness: {harness}", file=sys.stderr)
        sys.exit(1)
    return {
        "session_id": _sanitize_session_id(str(data.get("session_id", "unknown"))),
        "stop_hook_active": data.get("stop_hook_active", False),
        "transcript_path": str(data.get("transcript_path", "")),
    }


def _wing_from_transcript_path(transcript_path: str) -> str:
    """Derive a project wing name from a Claude Code transcript path.

    Claude Code encodes the project's source directory by replacing path
    separators with dashes, producing folders like:
        ~/.claude/projects/-home-<user>-Projects-<project>/session.jsonl
        ~/.claude/projects/-home-<user>-dev-<parent>-<project>/session.jsonl
        ~/.claude/projects/-Users-<user>-<folder>-<project>/session.jsonl

    The project directory name is the final dash-separated token of the
    encoded folder. Returns ``wing_<project>`` (lowercased, spaces → ``_``).
    Falls back to ``wing_sessions`` if the path does not match a Claude Code
    project-folder layout.
    """
    # Normalize path separators for cross-platform (Windows backslashes)
    normalized = transcript_path.replace("\\", "/")
    # Primary: pull the encoded project folder out of ``.claude/projects/``
    # and take its last dash-separated token.
    match = re.search(r"/\.claude/projects/-([^/]+)", normalized)
    if match:
        encoded = match.group(1)
        project = encoded.rsplit("-", 1)[-1]
        if project:
            return f"wing_{project.lower().replace(' ', '_')}"
    # Legacy fallback: explicit ``-Projects-<name>`` segment, useful for
    # transcripts not under the standard Claude Code projects dir.
    match = re.search(r"-Projects-([^/]+?)(?:/|$)", normalized)
    if match:
        project = match.group(1).lower().replace(" ", "_")
        return f"wing_{project}"
    return "wing_sessions"


def hook_stop(data: dict, harness: str):
    """Stop hook: block every N messages for auto-save."""
    if not _palace_root_exists():
        _output({})
        return
    parsed = _parse_harness_input(data, harness)
    session_id = parsed["session_id"]
    stop_hook_active = parsed["stop_hook_active"]
    transcript_path = parsed["transcript_path"]

    # If already in a block-mode save cycle, let through (infinite-loop prevention).
    # Silent mode saves directly without returning {"decision":"block"}, so there's
    # no loop to prevent — and Claude Code's plugin dispatch sets this flag on every
    # fire after the first, which would otherwise suppress all subsequent auto-saves.
    if str(stop_hook_active).lower() in ("true", "1", "yes"):
        # Safe default: assume silent mode on any config-read failure so saves
        # proceed rather than being silently dropped. Silent mode is the default
        # (v3.3.0+), so if we can't read config, behave as if it's still on.
        silent_guard = True
        try:
            from .config import MempalaceConfig
        except ImportError as exc:
            _log(
                f"WARNING: could not import MempalaceConfig for stop guard: {exc}; defaulting to silent mode"
            )
        else:
            try:
                silent_guard = MempalaceConfig().hook_silent_save
            except AttributeError as exc:
                _log(f"WARNING: could not read hook_silent_save: {exc}; defaulting to silent mode")
        if not silent_guard:
            _output({})
            return

    # Count human messages
    exchange_count = _count_human_messages(transcript_path)

    # Track last save point
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    last_save_file = STATE_DIR / f"{session_id}_last_save"
    last_save = 0
    if last_save_file.is_file():
        try:
            last_save = int(last_save_file.read_text().strip())
        except (ValueError, OSError):
            last_save = 0

    since_last = exchange_count - last_save

    _log(f"Session {session_id}: {exchange_count} exchanges, {since_last} since last save")

    if since_last >= SAVE_INTERVAL and exchange_count > 0:
        _log(f"TRIGGERING SAVE at exchange {exchange_count}")

        # Read hook settings from config
        from .config import MempalaceConfig

        try:
            config = MempalaceConfig()
            silent = config.hook_silent_save
            toast = config.hook_desktop_toast
        except Exception:
            silent = True
            toast = False

        project_wing = _wing_from_transcript_path(transcript_path)

        if silent:
            # Save directly via Python API — systemMessage renders in terminal
            result = {"count": 0}
            if transcript_path:
                result = _save_diary_direct(
                    transcript_path, session_id, wing=project_wing, toast=toast
                )
                _ingest_transcript(transcript_path)
            _maybe_auto_ingest()
            # Only advance save marker after successful save
            count = result.get("count", 0)
            if count > 0:
                try:
                    last_save_file.write_text(str(exchange_count), encoding="utf-8")
                except OSError:
                    pass
                themes = result.get("themes", [])
                if themes:
                    tag = " \u2014 " + ", ".join(themes)
                else:
                    tag = ""
                _output(
                    {
                        "systemMessage": f"\u2726 {count} memories woven into the palace{tag}",
                    }
                )
            else:
                _output({})
        else:
            # Legacy: block and ask Claude to save via MCP tools.
            # Marker advances before confirmed save — best-effort; if Claude
            # fails to save, the checkpoint is lost but won't retry endlessly.
            try:
                last_save_file.write_text(str(exchange_count), encoding="utf-8")
            except OSError:
                pass
            if transcript_path:
                _ingest_transcript(transcript_path)
            _maybe_auto_ingest()
            reason = STOP_BLOCK_REASON + f" Write diary entry to wing={project_wing}."
            _output({"decision": "block", "reason": reason})
    else:
        _output({})


def hook_session_start(data: dict, harness: str):
    """Session start hook: initialize session tracking state."""
    if not _palace_root_exists():
        _output({})
        return
    parsed = _parse_harness_input(data, harness)
    session_id = parsed["session_id"]

    _log(f"SESSION START for session {session_id}")

    # Initialize session state directory
    STATE_DIR.mkdir(parents=True, exist_ok=True)

    # Pass through — no blocking on session start
    _output({})


# ── Piece 3: PreCompact KG extraction ─────────────────────────────────────
# Pulls (subject, predicate, object) triples out of the about-to-be-compacted
# transcript and lands them in the knowledge graph before they're lost.
# Mining picks up drawer content from disk later, but live-conversation
# entities + relationships die with the compaction window unless captured
# here.


def _kg_extract_from_transcript(transcript_path: str) -> list[dict]:
    """Ask gemma4 for KG triples from the recent transcript window.

    Returns a list of ``{"subject", "predicate", "object", "confidence"}``
    dicts that passed validation. Predicates are clamped to
    :data:`_HOOK_KG_ALLOWED_PREDICATES`; anything outside that set is
    dropped silently rather than allowed to invent novel relationship
    types.

    Never raises — PreCompact must not block compaction on an LLM error.
    """
    try:
        from .config import MempalaceConfig
        kg_timeout = MempalaceConfig().hook_llm_timeout_kg_s
    except Exception:
        kg_timeout = 60
    provider = _get_hook_llm(timeout=kg_timeout)
    if provider is None:
        return []
    if not transcript_path:
        return []

    messages = _extract_recent_messages(transcript_path, count=60)
    if not messages:
        return []

    sample = "\n".join(f"- {m[:200]}" for m in messages[-50:])
    allowed = ", ".join(sorted(_HOOK_KG_ALLOWED_PREDICATES))
    system = (
        "You extract knowledge-graph triples from chat. Return ONLY a JSON "
        'object of shape {"triples": [{"subject":"S","predicate":"P",'
        '"object":"O","confidence":0.0-1.0}, ...]}. '
        f"Predicate MUST be one of: {allowed}. Drop any fact that doesn't "
        "fit those predicates. Subject and object should be concrete "
        "entities (people, projects, files, tools) — never opinions or "
        "abstract claims. Confidence is your honest 0.0-1.0 estimate. "
        "Up to 12 triples. No prose outside the JSON."
    )
    user = f"Messages:\n{sample}\n\nReturn JSON only."

    try:
        resp = provider.classify(system=system, user=user, json_mode=True)
    except Exception as exc:
        _log(f"hook_llm: kg classify failed: {exc}")
        return []

    raw = (resp.text or "").strip()
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            return []
        try:
            parsed = json.loads(match.group(0))
        except (json.JSONDecodeError, TypeError):
            return []

    triples = parsed.get("triples") if isinstance(parsed, dict) else None
    if not isinstance(triples, list):
        return []

    try:
        from .config import sanitize_kg_value, sanitize_name
    except Exception:
        return []

    out: list[dict] = []
    for t in triples:
        if not isinstance(t, dict):
            continue
        subject = t.get("subject")
        predicate = t.get("predicate")
        obj = t.get("object")
        confidence = t.get("confidence", 0.0)
        if not (isinstance(subject, str) and isinstance(predicate, str) and isinstance(obj, str)):
            continue
        try:
            confidence_f = float(confidence)
        except (TypeError, ValueError):
            confidence_f = 0.0
        # Confidence floor — gemma is quite happy to invent low-confidence
        # connections. 0.7 keeps the graph high-signal at the cost of
        # missing weakly-stated facts (which we'd rather catch via mining
        # of the actual drawer content later anyway).
        if confidence_f < 0.7:
            continue
        pred_norm = predicate.strip().lower()
        if pred_norm not in _HOOK_KG_ALLOWED_PREDICATES:
            continue
        try:
            subject_v = sanitize_kg_value(subject, "subject")
            object_v = sanitize_kg_value(obj, "object")
            predicate_v = sanitize_name(pred_norm, "predicate")
        except ValueError:
            continue
        out.append(
            {
                "subject": subject_v,
                "predicate": predicate_v,
                "object": object_v,
                "confidence": confidence_f,
            }
        )
        if len(out) >= 12:
            break
    return out


def _kg_apply_triples(triples: list[dict]) -> int:
    """Persist accepted triples via the in-process KG writer.

    Returns the number of triples actually inserted. Failures inside
    individual ``tool_kg_add`` calls are logged but never raised — one
    bad triple should not kill the rest.
    """
    if not triples:
        return 0
    try:
        from .mcp_server import tool_kg_add
    except Exception as exc:
        _log(f"hook_llm: kg_add import failed: {exc}")
        return 0

    today = datetime.now().strftime("%Y-%m-%d")
    inserted = 0
    for t in triples:
        try:
            res = tool_kg_add(
                subject=t["subject"],
                predicate=t["predicate"],
                object=t["object"],
                valid_from=today,
                source_closet="precompact-hook",
            )
            if res.get("success"):
                inserted += 1
            else:
                _log(f"hook_llm: kg_add rejected {t}: {res.get('error')}")
        except Exception as exc:
            _log(f"hook_llm: kg_add raised on {t}: {exc}")
    return inserted


def hook_precompact(data: dict, harness: str):
    """Precompact hook: mine transcript synchronously, then allow compaction."""
    if not _palace_root_exists():
        _output({})
        return
    parsed = _parse_harness_input(data, harness)
    session_id = parsed["session_id"]
    transcript_path = parsed["transcript_path"]

    _log(f"PRE-COMPACT triggered for session {session_id}")

    # Capture tool output via our normalize path before compaction loses it
    if transcript_path:
        _ingest_transcript(transcript_path)

    # Optional KG extraction over the about-to-be-compacted window. Done
    # before _mine_sync because compaction can fire mid-mine and we want
    # the high-confidence triples committed first. Latency budget is
    # bounded by hook_llm_timeout_s (default 10s).
    try:
        from .config import MempalaceConfig

        if transcript_path and MempalaceConfig().hook_llm_precompact_kg:
            triples = _kg_extract_from_transcript(transcript_path)
            inserted = _kg_apply_triples(triples)
            if inserted:
                _log(f"hook_llm: precompact KG inserted {inserted} triple(s)")
    except Exception as exc:
        _log(f"hook_llm: precompact KG skipped: {exc}")

    # Mine MEMPAL_DIR synchronously so project data lands before
    # compaction proceeds. Transcript convos were already kicked off
    # above via _ingest_transcript.
    _mine_sync()

    _output({})


def run_hook(hook_name: str, harness: str):
    """Main entry point: read stdin JSON, dispatch to hook handler."""
    try:
        data = json.load(sys.stdin)
    except (json.JSONDecodeError, EOFError):
        _log("WARNING: Failed to parse stdin JSON, proceeding with empty data")
        data = {}

    hooks = {
        "session-start": hook_session_start,
        "stop": hook_stop,
        "precompact": hook_precompact,
    }

    handler = hooks.get(hook_name)
    if handler is None:
        print(f"Unknown hook: {hook_name}", file=sys.stderr)
        sys.exit(1)

    handler(data, harness)
