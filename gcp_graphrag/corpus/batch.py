"""
Batch enqueue — scans an existing folder and pushes all supported
file paths onto the job queue so workers pick them up on startup.
"""
import logging
from pathlib import Path
from queue import Queue

from corpus.ingest import ALL_EXTS

logger = logging.getLogger(__name__)


def enqueue_existing(folder: str, queue: Queue) -> int:
    """
    Walk *folder* recursively and push every supported file onto *queue*.
    Already-indexed files will be silently skipped by ``add_document``
    (SHA-256 deduplication) so this is safe to call on every startup.

    Returns the number of files enqueued.
    """
    root  = Path(folder)
    count = 0
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in ALL_EXTS:
            queue.put(str(path))
            count += 1
    logger.info(f"[batch] Enqueued {count} file(s) from {folder}")
    return count
