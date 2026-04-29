"""Embedding function factory with hardware acceleration and pluggable providers.

Routing (driven by ``MEMPALACE_EMBEDDING_PROVIDER``):

* ``voyage`` / ``voyageai``       — :class:`VoyageEmbeddingFunction`, API-backed
  with on-disk content-addressable cache. Requires ``VOYAGE_API_KEY``. BYOK,
  opt-in only.
* ``sentence-transformers``       — local ST model (default
  ``mixedbread-ai/mxbai-embed-large-v1``). Honours ``MEMPALACE_EMBEDDING_MODEL``
  and ``MEMPALACE_EMBEDDING_DEVICE`` (mps/cuda/cpu auto-detected).
* unset / anything else (default) — ChromaDB's bundled ``all-MiniLM-L6-v2``
  via ONNX Runtime, with hardware acceleration. The same 384-dim vectors
  ChromaDB ships by default are reused, so switching device does not
  invalidate existing palaces. **This is the project default** — voyage and
  ST paths are explicit opt-ins.

Supported devices for the default ONNX path (env ``MEMPALACE_EMBEDDING_DEVICE``
or ``embedding_device`` in ``~/.mempalace/config.json``):

* ``auto`` — prefer CUDA ▸ CoreML ▸ DirectML, fall back to CPU
* ``cpu`` — force CPU (the historical default)
* ``cuda`` — NVIDIA GPU via ``onnxruntime-gpu`` (``pip install mempalace[gpu]``)
* ``coreml`` — Apple Neural Engine (macOS)
* ``dml`` — DirectML (Windows / AMD / Intel GPUs)

Requesting an unavailable accelerator emits a warning and falls back to CPU
rather than hard-failing — mining must still work on a laptop without CUDA.
Voyage misconfiguration (e.g. missing API key) is a hard error and is **not**
silently fallen back, so users always know when their explicit provider
choice failed.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_VOYAGE_MODEL = "voyage-code-3"
DEFAULT_ST_MODEL = "mixedbread-ai/mxbai-embed-large-v1"

_PROVIDER_MAP = {
    "cpu": ["CPUExecutionProvider"],
    "cuda": ["CUDAExecutionProvider", "CPUExecutionProvider"],
    "coreml": ["CoreMLExecutionProvider", "CPUExecutionProvider"],
    "dml": ["DmlExecutionProvider", "CPUExecutionProvider"],
}

_DEVICE_EXTRA = {
    "cuda": "mempalace[gpu]",
    "coreml": "mempalace[coreml]",
    "dml": "mempalace[dml]",
}

_AUTO_ORDER = [
    ("CUDAExecutionProvider", "cuda"),
    ("CoreMLExecutionProvider", "coreml"),
    ("DmlExecutionProvider", "dml"),
]

_EF_CACHE: dict = {}
_WARNED: set = set()


def _resolve_providers(device: str) -> tuple[list, str]:
    """Return ``(provider_list, effective_device)`` for ``device``.

    Falls back to CPU (with a one-shot warning) when the requested
    accelerator is not compiled into the installed ``onnxruntime``.
    """
    device = (device or "auto").strip().lower()

    try:
        import onnxruntime as ort

        available = set(ort.get_available_providers())
    except ImportError:
        return (["CPUExecutionProvider"], "cpu")

    if device == "auto":
        for provider, name in _AUTO_ORDER:
            if provider in available:
                return ([provider, "CPUExecutionProvider"], name)
        return (["CPUExecutionProvider"], "cpu")

    requested = _PROVIDER_MAP.get(device)
    if requested is None:
        if device not in _WARNED:
            logger.warning("Unknown embedding_device %r — falling back to cpu", device)
            _WARNED.add(device)
        return (["CPUExecutionProvider"], "cpu")

    preferred = requested[0]
    if preferred == "CPUExecutionProvider":
        return (requested, "cpu")

    if preferred not in available:
        if device not in _WARNED:
            extra = _DEVICE_EXTRA.get(device, "the matching mempalace extra for your device")
            logger.warning(
                "embedding_device=%r requested but %s is not installed — "
                "falling back to CPU. Install %s.",
                device,
                preferred,
                extra,
            )
            _WARNED.add(device)
        return (["CPUExecutionProvider"], "cpu")

    return (requested, device)


def _build_ef_class():
    """Subclass ``ONNXMiniLM_L6_V2`` with name ``"default"``.

    Why the rename: ChromaDB 1.5 persists the EF identity on the collection
    and rejects reads that pass a differently-named EF (``onnx_mini_lm_l6_v2``
    vs ``default``). The vectors and model are identical — only the
    ``name()`` tag differs — so spoofing the name lets one EF class serve
    palaces created with ``DefaultEmbeddingFunction`` *and* palaces we
    create ourselves, with the same GPU-capable ``preferred_providers``.
    """
    from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2

    class _MempalaceONNX(ONNXMiniLM_L6_V2):
        @staticmethod
        def name() -> str:
            return "default"

    return _MempalaceONNX


def _detect_st_device() -> str:
    """Return 'mps' on Apple Silicon, 'cuda' if available, else 'cpu'.

    Used only by the sentence-transformers path; the default ONNX path has
    its own ``_resolve_providers`` device negotiation.
    """
    try:
        import torch

        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def _get_voyage_ef():
    """Build a Voyage AI embedding function. Hard-fails on misconfiguration."""
    from .voyage_ef import VoyageEmbeddingFunction

    model = os.environ.get("MEMPALACE_EMBEDDING_MODEL", DEFAULT_VOYAGE_MODEL)
    ef = VoyageEmbeddingFunction(model=model)
    logger.info("Using Voyage embedding model %s", model)
    return ef


def _get_st_ef(device_hint: Optional[str] = None):
    """Build a sentence-transformers EF. Returns None to fall back to ONNX default."""
    model_name = os.environ.get("MEMPALACE_EMBEDDING_MODEL", DEFAULT_ST_MODEL)

    if model_name.lower() in ("", "default", "chroma-default"):
        return None

    try:
        from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
    except Exception as exc:
        logger.warning(
            "SentenceTransformerEmbeddingFunction unavailable (%s); falling back to ONNX default",
            exc,
        )
        return None

    device = device_hint or os.environ.get("MEMPALACE_EMBEDDING_DEVICE") or _detect_st_device()
    try:
        ef = SentenceTransformerEmbeddingFunction(model_name=model_name, device=device)
        logger.info("Using sentence-transformers %s on device %s", model_name, device)
        return ef
    except Exception as exc:
        logger.warning(
            "Failed to load sentence-transformers %s on %s (%s); falling back to ONNX default",
            model_name,
            device,
            exc,
        )
        return None


def get_embedding_function(device: Optional[str] = None):
    """Return a cached embedding function based on provider + device config.

    Provider is selected by ``MEMPALACE_EMBEDDING_PROVIDER``:

    * ``voyage`` → :class:`VoyageEmbeddingFunction` (BYOK, hard-fails on
      misconfig)
    * ``sentence-transformers`` → local ST model; ``device`` is forwarded to
      ST and falls back to the ONNX default if ST loading fails
    * unset / other → ONNX-accelerated MiniLM (project default; preserves
      palace compatibility)

    ``device=None`` reads from :class:`MempalaceConfig.embedding_device`.
    The returned function is shared across calls with the same resolved
    provider list so we only pay model-load cost once per process.
    """
    provider = os.environ.get("MEMPALACE_EMBEDDING_PROVIDER", "").strip().lower()

    if provider in ("voyage", "voyageai"):
        cached = _EF_CACHE.get(("__voyage__",))
        if cached is not None:
            return cached
        # Voyage misconfig is a hard error — never silently fall back.
        ef = _get_voyage_ef()
        _EF_CACHE[("__voyage__",)] = ef
        return ef

    if provider in ("sentence-transformers", "st"):
        cache_key = ("__st__", os.environ.get("MEMPALACE_EMBEDDING_MODEL", DEFAULT_ST_MODEL))
        cached = _EF_CACHE.get(cache_key)
        if cached is not None:
            return cached
        st_ef = _get_st_ef(device_hint=device)
        if st_ef is not None:
            _EF_CACHE[cache_key] = st_ef
            return st_ef
        # Fall through to ONNX default if ST failed to load.

    # Default path: ONNX-accelerated MiniLM (preserves palace compatibility).
    if device is None:
        from .config import MempalaceConfig

        device = MempalaceConfig().embedding_device

    providers, effective = _resolve_providers(device)
    cache_key = tuple(providers)
    cached = _EF_CACHE.get(cache_key)
    if cached is not None:
        return cached

    ef_cls = _build_ef_class()
    ef = ef_cls(preferred_providers=providers)
    _EF_CACHE[cache_key] = ef
    logger.info("Embedding function initialized (device=%s providers=%s)", effective, providers)
    return ef


def describe_device(device: Optional[str] = None) -> str:
    """Return a short human-readable label for the resolved device.

    Used by the miner CLI header so users can see at a glance whether GPU
    acceleration actually engaged.
    """
    if device is None:
        from .config import MempalaceConfig

        device = MempalaceConfig().embedding_device
    _, effective = _resolve_providers(device)
    return effective
