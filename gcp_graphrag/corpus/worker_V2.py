"""
Background embedding workers.

Each worker thread pulls file paths from a shared Queue,
calls ingest_file(), and updates a shared progress dict.
FAISS is rebuilt every `rebuild_every` files to amortise cost.
"""
import threading, logging
from pathlib import Path
from queue import Queue, Empty

from corpus.ingest import ingest_file
from corpus.store  import load_all, build_faiss
from retrieval.embedder import MultimodalEmbedder
from config.settings import PipelineConfig

logger = logging.getLogger(__name__)

_lock      = threading.Lock()
_queue_ref: Queue | None = None   # stored separately — never put in _progress

# Only JSON-serialisable values live here
_progress: dict = {
    "done":    0,
    "failed":  0,
    "current": "",
    "errors":  [],
}


def get_progress() -> dict:
    """
    Thread-safe snapshot returned to /ingest/progress and /ingest/stream.
    All values are JSON-serialisable (str / int / list[dict]).
    """
    with _lock:
        queued = _queue_ref.qsize() if _queue_ref is not None else 0
        return {
            "queued":  queued,
            "done":    _progress["done"],
            "failed":  _progress["failed"],
            "current": _progress["current"],
            "errors":  list(_progress["errors"]),   # shallow copy is enough
        }


def _upd(key: str, value=None, delta: int = 0) -> None:
    with _lock:
        if delta:
            _progress[key] += delta
        else:
            _progress[key] = value


def worker_loop(queue: Queue, embedder: MultimodalEmbedder,
                rebuild_every: int = 10) -> None:
    """Runs forever as a daemon thread."""
    since_rebuild = 0

    while True:
        try:
            filepath = queue.get(timeout=2)
        except Empty:
            continue

        _upd("current", Path(filepath).name)
        logger.info(f"[worker] Processing: {filepath}")

        try:
            n = ingest_file(filepath, embedder, rebuild_index=False)
            _upd("done", delta=1)
            since_rebuild += 1
            logger.info(f"[worker] Done: {Path(filepath).name} → {n} chunk(s)")
        except Exception as exc:
            _upd("failed", delta=1)
            with _lock:
                _progress["errors"].append({"file": str(filepath), "error": str(exc)})
            logger.error(f"[worker] Failed {filepath}: {exc}")
        finally:
            queue.task_done()
            _upd("current", "")

        # Rebuild FAISS every N files (amortise cost)
        if since_rebuild >= rebuild_every:
            try:
                logger.info("[worker] Rebuilding FAISS index…")
                _, embs = load_all()
                if len(embs) > 0:
                    build_faiss(embs)
                since_rebuild = 0
            except Exception as exc:
                logger.error(f"[worker] FAISS rebuild failed: {exc}")


def start_workers(queue: Queue, n_workers: int = 3,
                  rebuild_every: int = 10) -> list[threading.Thread]:
    """
    Spawn *n_workers* daemon threads sharing the same queue.
    Each thread gets its own MultimodalEmbedder instance.

    Recommended:
      n_workers=1   CPU-only / low RAM
      n_workers=3   multi-core or single GPU
      n_workers=6+  multi-GPU / cloud endpoint
    """
    global _queue_ref
    _queue_ref = queue          # store separately, never mixed into _progress

    cfg     = PipelineConfig()
    threads = []
    for i in range(n_workers):
        embedder = MultimodalEmbedder(cfg)
        t = threading.Thread(
            target=worker_loop,
            args=(queue, embedder, rebuild_every),
            daemon=True,
            name=f"ingest-worker-{i}",
        )
        t.start()
        threads.append(t)

    logger.info(f"[worker] Started {n_workers} worker thread(s)")
    return threads
