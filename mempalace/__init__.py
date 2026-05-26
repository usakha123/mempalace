"""MemPalace — Give your AI a memory. No API key required."""

import logging
import os
import re
from pathlib import Path

from .version import __version__  # noqa: E402


def _load_palace_env_file() -> None:
    """Load ``~/.mempalace/env`` and ``~/.mempalace/hook_env`` into ``os.environ``.

    Why: harnesses like Claude Code invoke ``mempalace hook run`` directly,
    skipping the shell wrappers that would otherwise source these files.
    Without this loader, hook subprocesses run with an empty environment
    and lose things like ``MEMPALACE_EMBEDDING_PROVIDER``,
    ``VOYAGE_API_KEY``, and ``ANTHROPIC_API_KEY``, which makes the palace
    fail to add drawers or run hook-LLM calls.

    Two files are read, in order:

    * ``~/.mempalace/env`` — non-secret config (provider names, model
      names, palace path). Also sourced by the parent shell via
      ``~/.zshenv``, so this loader is a defensive no-op there.
    * ``~/.mempalace/hook_env`` — hook-only secrets (API keys). This
      file is **never** sourced by the parent shell because exporting
      ``ANTHROPIC_API_KEY`` into Claude Code's own shell conflicts with
      the claude.ai login token. Hook subprocesses are the right scope
      for it, which is exactly what this loader provides.

    Existing ``os.environ`` values always win (``setdefault``) so users
    who deliberately override at the shell level keep their override.

    Errors are swallowed — this is a best-effort convenience, not a hard
    dependency. Malformed lines are skipped silently.
    """
    # Tolerant parser: ``export KEY="value"``, ``export KEY=value``,
    # or bare ``KEY=value``. Comments (#) and blank lines ignored.
    pattern = re.compile(
        r'^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*'
        r'(?:"([^"]*)"|\'([^\']*)\'|([^\s#]*))\s*(?:#.*)?$'
    )
    for relpath in ("~/.mempalace/env", "~/.mempalace/hook_env"):
        env_file = Path(os.path.expanduser(relpath))
        if not env_file.is_file():
            continue
        try:
            for line in env_file.read_text(encoding="utf-8").splitlines():
                if not line.strip() or line.lstrip().startswith("#"):
                    continue
                m = pattern.match(line)
                if not m:
                    continue
                key = m.group(1)
                val = m.group(2) or m.group(3) or m.group(4) or ""
                os.environ.setdefault(key, val)
        except Exception:
            # Best-effort. A broken env file should not break ``import mempalace``.
            continue


_load_palace_env_file()

# chromadb telemetry: posthog capture() was broken in 0.6.x causing noisy stderr
# warnings ("capture() takes 1 positional argument but 3 were given"). In 1.x the
# posthog client is a no-op stub, so this is now harmless — kept as a guard in
# case future chromadb versions re-introduce real telemetry calls.
logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)

# NOTE: the previous block set ``ORT_DISABLE_COREML=1`` on macOS arm64 as a
# supposed workaround for the #74 ARM64 segfault.  Two problems:
#
# 1. ONNX Runtime does not read that env var -- it has no global way to
#    disable a single execution provider, so the setdefault was a no-op.
# 2. #74 is a null-pointer crash in ``chromadb_rust_bindings.abi3.so``, not
#    an ONNX issue, so disabling CoreML would not have fixed it anyway.
#
# #521 has since traced the actual macOS arm64 crashes (both in mine and
# search paths) to the 0.x chromadb hnswlib binding.  Filtering
# CoreMLExecutionProvider at the ONNX layer leaves the hnswlib C++ crash
# intact, so the real fix is upgrading chromadb to 1.5.4+, which #581
# proposes.  See #397 for the history of this line.

__all__ = ["__version__"]
