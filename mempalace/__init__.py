"""MemPalace — Give your AI a memory. No API key required."""

import logging
import os
import re
from pathlib import Path

from .version import __version__  # noqa: E402


def _load_palace_env_file() -> None:
    """Load ``~/.mempalace/env`` into ``os.environ`` if it exists.

    Why: harnesses like Claude Code invoke ``mempalace hook run`` directly,
    skipping the shell wrappers that source ``~/.mempalace/env``. Without
    this loader, hook subprocesses run with an empty environment and lose
    things like ``MEMPALACE_EMBEDDING_PROVIDER`` and ``VOYAGE_API_KEY``,
    which makes a voyage-backed palace raise
    "Embedding function voyage not found" on every diary checkpoint.

    Existing ``os.environ`` values always win (``setdefault``) so users who
    deliberately override at the shell level keep their override.

    Errors are swallowed — this is a best-effort convenience, not a hard
    dependency. Malformed lines are skipped silently.
    """
    env_file = Path(os.path.expanduser("~/.mempalace/env"))
    if not env_file.is_file():
        return
    try:
        # Tolerant parser: ``export KEY="value"``, ``export KEY=value``,
        # or bare ``KEY=value``. Comments (#) and blank lines ignored.
        pattern = re.compile(
            r'^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*'
            r'(?:"([^"]*)"|\'([^\']*)\'|([^\s#]*))\s*(?:#.*)?$'
        )
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
        pass


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
