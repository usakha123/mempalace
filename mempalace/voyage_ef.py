"""Voyage AI embedding function for ChromaDB + disk-backed content-addressable cache.

Design goals:
  * ChromaDB 1.x-compatible EmbeddingFunction (name, __call__, get_config, build_from_config)
  * Content-addressable cache (sha256(text) -> .npy) so re-mines don't re-spend tokens
  * Respect Voyage rate limits via SDK retries + conservative batching
  * Safe under concurrent subprocess access (atomic writes via os.replace)

Env vars consumed:
  VOYAGE_API_KEY              (required)
  MEMPALACE_EMBEDDING_MODEL   (default: voyage-code-3)
  MEMPALACE_VOYAGE_INPUT_TYPE (default: document)  -- "document" | "query" | None
  MEMPALACE_EMBEDDING_CACHE   (default: ~/.mempalace/embedding_cache)
  MEMPALACE_VOYAGE_BATCH_DOCS (default: 128)
  MEMPALACE_VOYAGE_BATCH_TOKENS (default: 100000)
"""

from __future__ import annotations

import hashlib
import logging
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "voyage-code-3"
DEFAULT_DIM = 1024  # voyage-code-3 native dim
DEFAULT_CACHE = Path.home() / ".mempalace" / "embedding_cache"


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _approx_tokens(text: str) -> int:
    # cheap heuristic, 4 chars/token; good enough for batching math
    return max(1, len(text) // 4)


class DiskEmbeddingCache:
    """Content-addressable cache for embeddings.

    Layout: <root>/<model>/<hash[:2]>/<hash[2:4]>/<hash>.npy
    Atomic writes via os.replace.
    """

    def __init__(self, root: Path | str, model: str):
        self.root = Path(root) / model
        self.root.mkdir(parents=True, exist_ok=True)
        self.model = model
        self._hits = 0
        self._misses = 0

    def _path(self, h: str) -> Path:
        return self.root / h[:2] / h[2:4] / f"{h}.npy"

    def get(self, text: str) -> np.ndarray | None:
        p = self._path(_hash_text(text))
        if not p.is_file():
            self._misses += 1
            return None
        try:
            arr = np.load(p)
            self._hits += 1
            return arr
        except Exception as e:
            logger.warning("cache corrupt at %s (%s), treating as miss", p, e)
            self._misses += 1
            return None

    def put(self, text: str, vec: np.ndarray) -> None:
        h = _hash_text(text)
        p = self._path(h)
        p.parent.mkdir(parents=True, exist_ok=True)
        # np.save auto-appends .npy when given a string path without that suffix;
        # use an open file handle to get exact-name behaviour, then atomic replace.
        tmp = p.with_name(p.name + f".tmp.{os.getpid()}")
        with open(tmp, "wb") as fh:
            np.save(fh, vec.astype(np.float32), allow_pickle=False)
        os.replace(tmp, p)

    def stats(self) -> dict:
        return {"hits": self._hits, "misses": self._misses, "model": self.model}


from chromadb.api.types import EmbeddingFunction as _ChromaEF  # type: ignore
from chromadb.utils.embedding_functions import (  # type: ignore
    register_embedding_function as _register_ef,
)


class VoyageEmbeddingFunction(_ChromaEF):
    """ChromaDB-compatible embedding function backed by Voyage AI + disk cache."""

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        input_type: str | None = None,
        cache_dir: str | Path | None = None,
        batch_docs: int | None = None,
        batch_tokens: int | None = None,
    ):
        import voyageai  # lazy import so mempalace works without voyage installed

        self.model = model or os.environ.get("MEMPALACE_EMBEDDING_MODEL", DEFAULT_MODEL)
        self.api_key = api_key or os.environ.get("VOYAGE_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "VOYAGE_API_KEY env var is required for VoyageEmbeddingFunction"
            )
        self.input_type = input_type or os.environ.get(
            "MEMPALACE_VOYAGE_INPUT_TYPE", "document"
        )
        if self.input_type in ("", "none", "None"):
            self.input_type = None
        self.batch_docs = int(
            batch_docs or os.environ.get("MEMPALACE_VOYAGE_BATCH_DOCS", 128)
        )
        self.batch_tokens = int(
            batch_tokens or os.environ.get("MEMPALACE_VOYAGE_BATCH_TOKENS", 100_000)
        )

        cache_root = (
            cache_dir
            or os.environ.get("MEMPALACE_EMBEDDING_CACHE")
            or DEFAULT_CACHE
        )
        self._cache = DiskEmbeddingCache(cache_root, self.model)
        # voyageai.Client picks up VOYAGE_API_KEY from env if api_key not passed
        self._client = voyageai.Client(api_key=self.api_key)

    # ------------------------------------------------------------------ chroma protocol
    @classmethod
    def name(cls) -> str:  # type: ignore[override]
        return "voyage"

    def get_config(self) -> dict[str, Any]:
        # Persisted with the collection. DO NOT include the api_key.
        return {
            "model": self.model,
            "input_type": self.input_type,
            "batch_docs": self.batch_docs,
            "batch_tokens": self.batch_tokens,
        }

    @classmethod
    def build_from_config(cls, config: dict[str, Any]) -> "VoyageEmbeddingFunction":
        return cls(
            model=config.get("model"),
            input_type=config.get("input_type"),
            batch_docs=config.get("batch_docs"),
            batch_tokens=config.get("batch_tokens"),
        )

    # ------------------------------------------------------------------ batching helpers
    def _form_batches(self, texts: list[str]) -> list[list[int]]:
        """Greedy pack indices into batches honoring batch_docs and batch_tokens caps."""
        batches: list[list[int]] = []
        cur: list[int] = []
        cur_tokens = 0
        for i, t in enumerate(texts):
            t_tok = _approx_tokens(t)
            if cur and (len(cur) >= self.batch_docs or cur_tokens + t_tok > self.batch_tokens):
                batches.append(cur)
                cur, cur_tokens = [], 0
            cur.append(i)
            cur_tokens += t_tok
        if cur:
            batches.append(cur)
        return batches

    def _embed_api(self, texts: list[str]) -> list[list[float]]:
        """One Voyage embed call with retry on transient errors."""
        last_err: Exception | None = None
        for attempt in range(6):
            try:
                kwargs: dict[str, Any] = {
                    "texts": texts,
                    "model": self.model,
                }
                if self.input_type:
                    kwargs["input_type"] = self.input_type
                resp = self._client.embed(**kwargs)
                return resp.embeddings
            except Exception as e:
                last_err = e
                msg = str(e).lower()
                is_rate = "rate" in msg or "429" in msg or "too many" in msg
                is_transient = is_rate or "timeout" in msg or "temporar" in msg or "503" in msg
                if not is_transient:
                    raise
                sleep = min(30, 2 ** attempt) + (0.1 * attempt)
                logger.warning(
                    "voyage transient error (attempt %d/6): %s -- sleeping %.1fs",
                    attempt + 1,
                    e,
                    sleep,
                )
                time.sleep(sleep)
        raise RuntimeError(f"voyage embed failed after retries: {last_err}")

    # ------------------------------------------------------------------ public API
    def __call__(self, input: list[str]) -> list[list[float]]:
        if not input:
            return []
        # 1) cache lookup
        out: list[list[float] | None] = [None] * len(input)
        todo_idx: list[int] = []
        for i, text in enumerate(input):
            hit = self._cache.get(text)
            if hit is not None:
                out[i] = hit.tolist()
            else:
                todo_idx.append(i)

        # 2) batch + fetch misses
        if todo_idx:
            miss_texts = [input[i] for i in todo_idx]
            batches = self._form_batches(miss_texts)
            for batch in batches:
                batch_texts = [miss_texts[j] for j in batch]
                vecs = self._embed_api(batch_texts)
                for local_j, vec in zip(batch, vecs):
                    global_i = todo_idx[local_j]
                    arr = np.asarray(vec, dtype=np.float32)
                    out[global_i] = arr.tolist()
                    self._cache.put(input[global_i], arr)

        # sanity: every slot filled
        assert all(v is not None for v in out), "voyage ef: missing embeddings"
        return out  # type: ignore[return-value]

    # ------------------------------------------------------------------ introspection
    def cache_stats(self) -> dict:
        return self._cache.stats()


# Register with chroma's EF registry so collection configs persist/restore cleanly.
try:
    _register_ef(VoyageEmbeddingFunction)
except Exception as _exc:  # already registered, or registry absent on older chroma
    logger.debug("voyage EF registration skipped: %s", _exc)
