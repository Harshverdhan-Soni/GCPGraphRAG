"""
Persistent corpus store.
- SQLite: document content + metadata
- FAISS index (.index file): vector search
- NumPy (.npy file): raw embeddings matrix

All three files live under CORPUS_DIR (default: ./corpus/data/).
"""
import sqlite3, hashlib, logging
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

CORPUS_DIR = Path("corpus/data")
DB_PATH    = CORPUS_DIR / "corpus.db"
IDX_PATH   = CORPUS_DIR / "faiss.index"
EMB_PATH   = CORPUS_DIR / "embeddings.npy"


def _ensure_dirs():
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)


def _conn() -> sqlite3.Connection:
    _ensure_dirs()
    c = sqlite3.connect(DB_PATH)
    c.execute("""
        CREATE TABLE IF NOT EXISTS docs (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            sha256           TEXT    UNIQUE,
            content          TEXT,
            filepath         TEXT    DEFAULT '',
            modality         TEXT    DEFAULT 'text',
            source           TEXT    DEFAULT '',
            lang             TEXT    DEFAULT 'en',
            provenance_score REAL    DEFAULT 0.8,
            created_at       TEXT    DEFAULT (datetime('now'))
        )
    """)
    c.commit()
    return c


def add_document(doc: dict, embedding: np.ndarray) -> int:
    """
    Upsert one document into SQLite and append its embedding.
    Returns the row id (stable across restarts).
    Skips duplicates based on SHA-256 of content.
    """
    content = str(doc.get("content", ""))
    sha     = hashlib.sha256(content.encode()).hexdigest()

    with _conn() as c:
        cur = c.execute(
            """INSERT OR IGNORE INTO docs
               (sha256, content, filepath, modality, source, lang, provenance_score)
               VALUES (?,?,?,?,?,?,?)""",
            (sha, content, doc.get("filepath", ""),
             doc.get("modality", "text"), doc.get("source", ""),
             doc.get("lang", "en"), doc.get("provenance_score", 0.8)),
        )
        if cur.lastrowid:
            doc_id = cur.lastrowid
            _append_embedding(embedding)
            logger.debug(f"Added doc id={doc_id} source={doc.get('source')}")
        else:
            doc_id = c.execute(
                "SELECT id FROM docs WHERE sha256=?", (sha,)
            ).fetchone()[0]
            logger.debug(f"Skipped duplicate sha={sha[:8]}")
    return doc_id


def load_all() -> tuple[list[dict], np.ndarray]:
    """Load all metadata and embeddings. Returns aligned (meta_list, embs)."""
    with _conn() as c:
        rows = c.execute(
            "SELECT id,content,filepath,modality,source,lang,provenance_score FROM docs"
            " ORDER BY id"
        ).fetchall()
    if not rows:
        return [], np.empty((0, 768), dtype=np.float32)
    meta = [
        {"id": r[0], "content": r[1], "filepath": r[2], "modality": r[3],
         "source": r[4], "lang": r[5], "provenance_score": r[6]}
        for r in rows
    ]
    embs = np.load(EMB_PATH) if EMB_PATH.exists() else np.empty((0, 768), dtype=np.float32)
    # Guard against shape mismatch after manual edits
    if len(embs) != len(meta):
        logger.warning(f"Embedding count ({len(embs)}) != doc count ({len(meta)}). "
                       "Run rebuild_index() to fix.")
    return meta, embs


def build_faiss(embeddings: np.ndarray) -> None:
    """Build and persist a FAISS IndexFlatIP from the given embeddings."""
    if len(embeddings) == 0:
        return
    try:
        import faiss
        _ensure_dirs()
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings.astype(np.float32))
        faiss.write_index(index, str(IDX_PATH))
        logger.info(f"FAISS index saved: {len(embeddings)} vectors → {IDX_PATH}")
    except ImportError:
        logger.warning("faiss not installed – index not persisted.")


def load_faiss():
    """Load the persisted FAISS index, or None if it doesn't exist."""
    if not IDX_PATH.exists():
        return None
    try:
        import faiss
        return faiss.read_index(str(IDX_PATH))
    except ImportError:
        return None


def get_stats() -> dict:
    with _conn() as c:
        total = c.execute("SELECT COUNT(*) FROM docs").fetchone()[0]
        by_modality = dict(c.execute(
            "SELECT modality, COUNT(*) FROM docs GROUP BY modality"
        ).fetchall())
    embs = np.load(EMB_PATH) if EMB_PATH.exists() else np.empty((0,))
    return {
        "total_chunks":      total,
        "by_modality":       by_modality,
        "embedding_vectors": len(embs),
        "faiss_index_exists":IDX_PATH.exists(),
        "db_path":           str(DB_PATH),
    }


def delete_document(doc_id: int) -> bool:
    """Remove a document by id. Returns True if deleted."""
    with _conn() as c:
        cur = c.execute("DELETE FROM docs WHERE id=?", (doc_id,))
    return cur.rowcount > 0


def _append_embedding(emb: np.ndarray) -> None:
    _ensure_dirs()
    emb = emb.reshape(1, -1).astype(np.float32)
    if EMB_PATH.exists():
        existing = np.load(EMB_PATH)
        combined = np.vstack([existing, emb])
    else:
        combined = emb
    np.save(EMB_PATH, combined)
