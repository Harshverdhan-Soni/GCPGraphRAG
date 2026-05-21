"""
Folder watcher — detects new files dropped into the watch directory
and pushes their paths onto a shared queue for worker threads.

Uses watchdog (OS-native inotify / FSEvents / ReadDirectoryChangesW).
"""
import logging
from pathlib import Path
from queue import Queue

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from corpus.ingest import ALL_EXTS

logger = logging.getLogger(__name__)


class _NewFileHandler(FileSystemEventHandler):
    def __init__(self, queue: Queue):
        self._queue = queue

    def on_created(self, event):
        if not event.is_directory:
            self._enqueue(event.src_path)

    def on_moved(self, event):
        """Handle files moved / renamed into the watched folder."""
        if not event.is_directory:
            self._enqueue(event.dest_path)

    def _enqueue(self, path: str):
        if Path(path).suffix.lower() in ALL_EXTS:
            logger.info(f"[watcher] Detected: {Path(path).name}")
            self._queue.put(path)


def start_watcher(watch_folder: str, queue: Queue) -> Observer:
    """
    Start watching *watch_folder* recursively.
    Returns the Observer — call ``observer.stop(); observer.join()`` to shut down.
    """
    Path(watch_folder).mkdir(parents=True, exist_ok=True)
    observer = Observer()
    observer.schedule(_NewFileHandler(queue), watch_folder, recursive=True)
    observer.start()
    logger.info(f"[watcher] Watching: {watch_folder}")
    return observer
