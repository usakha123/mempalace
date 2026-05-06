#!/usr/bin/env python3
"""Re-embed every drawer in an existing palace with Voyage AI.

Use when a palace was bootstrapped with a different 1024-dim embedding
(e.g. mxbai-embed-large-v1) and you now want all vectors to come from
voyage-code-3 — a silent semantic mismatch otherwise hides search relevance
because identical content has divergent vectors depending on which writer
ingested it.

Idempotent: each migrated drawer gets a ``_voyage_migrated_at`` metadata
stamp; reruns skip stamped drawers, so crashes/SIGINT/rate-limit aborts are
resumable for free.

Parallel: uses a ThreadPoolExecutor to fan out Voyage embed calls. The
Voyage SDK is sync per call but IO-bound, so threads scale well. Default
16 workers; bump with ``--workers``.

Disk cache: VoyageEmbeddingFunction writes each (text → vector) result to
``~/.mempalace/embedding_cache/`` (sha256 keyed, atomic). Reruns of the
same texts cost zero tokens.

USAGE
=====

    # Dry run — count what would be migrated, no API calls:
    python scripts/migrate_to_voyage.py --palace ~/.mempalace/palace --dry-run

    # Real migration:
    python scripts/migrate_to_voyage.py --palace ~/.mempalace/palace --workers 16

    # Resume after a crash (just rerun — stamped rows are skipped):
    python scripts/migrate_to_voyage.py --palace ~/.mempalace/palace

ENV
===

    VOYAGE_API_KEY        required
    MEMPALACE_EMBEDDING_MODEL  optional, default voyage-code-3
"""

from __future__ import annotations

import argparse
import datetime as _dt
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import chromadb

# Make sure we can import the package even when run from repo root.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mempalace.voyage_ef import VoyageEmbeddingFunction  # noqa: E402

logger = logging.getLogger("migrate_to_voyage")

PAGE_SIZE = 1000  # how many drawers we read from chroma per get()
UPDATE_CHUNK = 500  # how many drawers we write back per update()
MIGRATED_KEY = "_voyage_migrated_at"


def _now_iso() -> str:
    return _dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _embed_slice(ef: VoyageEmbeddingFunction, texts: list[str]) -> list[list[float]]:
    """Worker: embed a slice of texts. Runs in a thread."""
    return ef(texts)


def migrate_collection(
    client: chromadb.PersistentClient,
    name: str,
    ef: VoyageEmbeddingFunction,
    workers: int,
    dry_run: bool,
) -> tuple[int, int]:
    """Re-embed every drawer in *name* that lacks a Voyage migration stamp.

    Returns (skipped, migrated).
    """
    col = client.get_collection(name)
    total = col.count()
    logger.info("[%s] total drawers: %d", name, total)

    skipped = 0
    migrated = 0
    started = time.time()

    offset = 0
    while offset < total:
        page = col.get(
            limit=PAGE_SIZE,
            offset=offset,
            include=["documents", "metadatas"],
        )
        ids = page.get("ids") or []
        docs = page.get("documents") or []
        metas = page.get("metadatas") or []
        if not ids:
            break

        # Filter to drawers needing migration.
        todo: list[tuple[str, str, dict]] = []
        for drawer_id, doc, meta in zip(ids, docs, metas):
            meta = dict(meta or {})
            if MIGRATED_KEY in meta:
                skipped += 1
                continue
            if not doc:
                # No text -> nothing to re-embed; stamp anyway so we don't reprocess.
                meta[MIGRATED_KEY] = _now_iso()
                if not dry_run:
                    col.update(ids=[drawer_id], metadatas=[meta])
                skipped += 1
                continue
            todo.append((drawer_id, doc, meta))

        if not todo:
            offset += PAGE_SIZE
            continue

        if dry_run:
            migrated += len(todo)
            offset += PAGE_SIZE
            elapsed = time.time() - started
            logger.info(
                "[%s] DRY-RUN page off=%d migrated_so_far=%d skipped_so_far=%d elapsed=%.1fs",
                name,
                offset,
                migrated,
                skipped,
                elapsed,
            )
            continue

        # Split into worker-sized slices. Each slice is one ef() call which itself
        # internally batches by token/doc caps — so workers parallelism is at the
        # request-fan-out level, not within a single batch.
        slice_size = max(1, len(todo) // workers + (1 if len(todo) % workers else 0))
        slices = [todo[i : i + slice_size] for i in range(0, len(todo), slice_size)]

        # Fan out.
        new_vectors: dict[str, list[float]] = {}
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(_embed_slice, ef, [t[1] for t in chunk]): chunk
                for chunk in slices
            }
            for fut in as_completed(futures):
                chunk = futures[fut]
                vecs = fut.result()  # raises on Voyage failure -> aborts page
                for (drawer_id, _doc, _meta), vec in zip(chunk, vecs):
                    new_vectors[drawer_id] = vec

        # Write back to chroma in UPDATE_CHUNK-sized batches. Single-threaded
        # (sqlite writer) but the heavy lifting was the embed calls.
        batch_ids: list[str] = []
        batch_embeddings: list[list[float]] = []
        batch_metas: list[dict] = []
        stamp = _now_iso()
        for drawer_id, _doc, meta in todo:
            meta[MIGRATED_KEY] = stamp
            batch_ids.append(drawer_id)
            batch_embeddings.append(new_vectors[drawer_id])
            batch_metas.append(meta)
            if len(batch_ids) >= UPDATE_CHUNK:
                col.update(
                    ids=batch_ids, embeddings=batch_embeddings, metadatas=batch_metas
                )
                batch_ids, batch_embeddings, batch_metas = [], [], []
        if batch_ids:
            col.update(
                ids=batch_ids, embeddings=batch_embeddings, metadatas=batch_metas
            )

        migrated += len(todo)
        offset += PAGE_SIZE
        elapsed = time.time() - started
        rate = migrated / elapsed if elapsed > 0 else 0
        eta_total = (total - offset) / rate if rate > 0 else 0
        cache = ef.cache_stats()
        logger.info(
            "[%s] page off=%d migrated=%d skipped=%d cache_hits=%d cache_misses=%d "
            "rate=%.1f drawers/s eta=%.0fs",
            name,
            offset,
            migrated,
            skipped,
            cache.get("hits", 0),
            cache.get("misses", 0),
            rate,
            eta_total,
        )

    return skipped, migrated


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--palace",
        default=os.environ.get("MEMPALACE_PALACE_PATH", "~/.mempalace/palace"),
        help="Path to palace (default: $MEMPALACE_PALACE_PATH or ~/.mempalace/palace)",
    )
    p.add_argument(
        "--collections",
        nargs="*",
        default=None,
        help="Collection names to migrate. Default: all collections in the palace.",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=16,
        help="Parallel Voyage embed workers per page (default 16).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Count work without calling Voyage or writing.",
    )
    p.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose logging (DEBUG).",
    )
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )

    if not os.environ.get("VOYAGE_API_KEY") and not args.dry_run:
        logger.error(
            "VOYAGE_API_KEY not set. Source ~/.mempalace/env or export it manually."
        )
        return 2

    palace_path = os.path.expanduser(args.palace)
    if not os.path.isdir(palace_path):
        logger.error("palace not found: %s", palace_path)
        return 2

    client = chromadb.PersistentClient(path=palace_path)

    if args.collections:
        names = args.collections
    else:
        names = [c.name for c in client.list_collections()]
    logger.info("palace=%s collections=%s workers=%d dry_run=%s",
                palace_path, names, args.workers, args.dry_run)

    # One EF instance shared across threads. Its disk cache uses atomic
    # os.replace so concurrent writes to the same hash are safe.
    ef = (
        VoyageEmbeddingFunction()
        if not args.dry_run
        else None  # type: ignore[assignment]
    )

    grand_skipped = 0
    grand_migrated = 0
    t0 = time.time()
    for name in names:
        s, m = migrate_collection(
            client, name, ef, workers=args.workers, dry_run=args.dry_run
        )
        grand_skipped += s
        grand_migrated += m

    elapsed = time.time() - t0
    logger.info(
        "DONE total_skipped=%d total_migrated=%d elapsed=%.1fs",
        grand_skipped,
        grand_migrated,
        elapsed,
    )
    if ef is not None:
        logger.info("voyage cache stats: %s", ef.cache_stats())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
