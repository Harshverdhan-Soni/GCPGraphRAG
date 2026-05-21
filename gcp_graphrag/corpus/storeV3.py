"""
Persistent corpus store — v2.

Key design: SQLite stores an `emb_row` column (integer) pointing to the
exact row in embeddings.npy for that document.  On delete we just drop
those rows from the matrix — no re-inference at all.

Files:
  corpus/data/corpus.db      – SQLite: content + metadata + emb_row
  corpus/data/faiss.index    – FAISS ANN index
  corpus/data/embeddings.npy – float32 matrix  (N, dim)
"""
import sqlite3, hashlib, logging
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

CORPUS_DIR = Path("corpus/data")
DB_PATH    = CORPUS_DIR / "corpus.db"
IDX_PATH   = CORPUS_DIR / "faiss.index"
EMB_PATH   = CORPUS_DIR / "embeddings.npy"

# ── Internal helpers ──────────────────────────────────────────────────────

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
            emb_row          INTEGER DEFAULT -1,
            created_at       TEXT    DEFAULT (datetime('now'))
        )
    """)
    # Migration: add emb_row column if upgrading from v1
    try:
        c.execute("ALTER TABLE docs ADD COLUMN emb_row INTEGER DEFAULT -1")
        c.commit()
        logger.info("Migrated DB schema: added emb_row column.")
    except Exception:
        pass  # column already exists
    c.commit()
    return c


def _load_matrix() -> np.ndarray:
    """Load embeddings.npy or return empty (0, 768) array."""
    if EMB_PATH.exists():
        return np.load(EMB_PATH)
    return np.empty((0, 768), dtype=np.float32)


def _save_matrix(matrix: np.ndarray):
    _ensure_dirs()
    np.save(EMB_PATH, matrix.astype(np.float32))


# ── Public API ────────────────────────────────────────────────────────────

def add_document(doc: dict, embedding: np.ndarray) -> int:
    """
    Upsert one document + its embedding vector.
    Returns the SQLite row id.
    Skips exact duplicates (SHA-256 of content).
    """
    content = str(doc.get("content", ""))
    sha     = hashlib.sha256(content.encode()).hexdigest()

    # Append vector to matrix and get its row index
    matrix  = _load_matrix()
    emb_row = len(matrix)                        # next available row
    matrix  = np.vstack([matrix, embedding.reshape(1, -1).astype(np.float32)])
    _save_matrix(matrix)

    with _conn() as c:
        cur = c.execute(
            """INSERT OR IGNORE INTO docs
               (sha256, content, filepath, modality, source, lang, provenance_score, emb_row)
               VALUES (?,?,?,?,?,?,?,?)""",
            (sha, content, doc.get("filepath", ""),
             doc.get("modality", "text"), doc.get("source", ""),
             doc.get("lang", "en"), doc.get("provenance_score", 0.8),
             emb_row),
        )
        if cur.lastrowid and cur.lastrowid > 0:
            doc_id = cur.lastrowid
            logger.debug(f"Added doc id={doc_id} emb_row={emb_row} source={doc.get('source')}")
        else:
            # Duplicate — row was ignored; trim the vector we just appended
            matrix = matrix[:emb_row]
            _save_matrix(matrix)
            doc_id = c.execute(
                "SELECT id FROM docs WHERE sha256=?", (sha,)
            ).fetchone()[0]
            logger.debug(f"Skipped duplicate sha={sha[:8]}")
    return doc_id


def load_all() -> tuple[list[dict], np.ndarray]:
    """
    Returns (meta_list, embeddings) ordered by DB id.
    Embeddings are re-ordered to match meta_list using emb_row.
    """
    with _conn() as c:
        rows = c.execute(
            "SELECT id, content, filepath, modality, source, lang, "
            "provenance_score, emb_row FROM docs ORDER BY id"
        ).fetchall()

    if not rows:
        return [], np.empty((0, 768), dtype=np.float32)

    meta = [
        {"id": r[0], "content": r[1], "filepath": r[2], "modality": r[3],
         "source": r[4], "lang": r[5], "provenance_score": r[6], "_emb_row": r[7]}
        for r in rows
    ]

    matrix = _load_matrix()
    vecs   = []
    for m in meta:
        row = m.get("_emb_row", -1)
        if 0 <= row < len(matrix):
            vecs.append(matrix[row])
        else:
            # Fallback: stub vector (shouldn't happen normally)
            logger.warning(f"Missing embedding for doc id={m['id']} (emb_row={row})")
            rng = np.random.default_rng(m["id"])
            vecs.append(rng.standard_normal(768).astype(np.float32))

    embs = np.vstack(vecs).astype(np.float32) if vecs else np.empty((0, 768), dtype=np.float32)
    return meta, embs


def delete_by_source(source: str) -> int:
    """
    Delete all chunks for a source.
    Removes their rows from embeddings.npy by index — NO re-inference.
    Returns count deleted.
    """
    with _conn() as c:
        rows = c.execute(
            "SELECT emb_row FROM docs WHERE source=?", (source,)
        ).fetchall()
        emb_rows_to_delete = {r[0] for r in rows if r[0] >= 0}

        cur = c.execute("DELETE FROM docs WHERE source=?", (source,))
        deleted = cur.rowcount

    if deleted == 0:
        return 0

    # Slice those rows out of the matrix (fast — pure numpy, no GPU)
    matrix = _load_matrix()
    if len(matrix) > 0 and emb_rows_to_delete:
        keep_mask = np.ones(len(matrix), dtype=bool)
        for row_idx in emb_rows_to_delete:
            if row_idx < len(matrix):
                keep_mask[row_idx] = False
        new_matrix = matrix[keep_mask]
        _save_matrix(new_matrix)

        # Re-index emb_row values in DB (compact the gaps)
        _reindex_emb_rows(keep_mask)

    logger.info(f"Deleted {deleted} doc(s) with source='{source}'")
    return deleted


def delete_document(doc_id: int) -> bool:
    """Delete a single document by id. Returns True if deleted."""
    with _conn() as c:
        row = c.execute("SELECT emb_row FROM docs WHERE id=?", (doc_id,)).fetchone()
        if not row:
            return False
        emb_row = row[0]
        c.execute("DELETE FROM docs WHERE id=?", (doc_id,))

    if emb_row >= 0:
        matrix = _load_matrix()
        if emb_row < len(matrix):
            new_matrix = np.delete(matrix, emb_row, axis=0)
            _save_matrix(new_matrix)
            keep_mask = np.ones(len(matrix), dtype=bool)
            keep_mask[emb_row] = False
            _reindex_emb_rows(keep_mask)
    return True


def build_faiss(embeddings: np.ndarray) -> None:
    """Build and persist a FAISS IndexFlatIP."""
    if len(embeddings) == 0:
        if IDX_PATH.exists():
            IDX_PATH.unlink()
        return
    try:
        import faiss
        _ensure_dirs()
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings.astype(np.float32))
        faiss.write_index(index, str(IDX_PATH))
        logger.info(f"FAISS index saved: {len(embeddings)} vectors")
    except ImportError:
        logger.warning("faiss not installed – skipping index persist.")


def rebuild_index_from_matrix() -> int:
    """
    Rebuild FAISS from the current embeddings.npy — NO re-inference.
    This is now the fast path called after every delete.
    Returns number of vectors indexed.
    """
    matrix = _load_matrix()
    build_faiss(matrix)
    return len(matrix)


def rebuild_embeddings_from_db(embedder) -> int:
    """
    Full re-embed: runs the encoder on every document in SQLite.
    Only needed after a DB corruption or schema migration.
    Prefer rebuild_index_from_matrix() for normal use.
    """
    _ensure_dirs()
    with _conn() as c:
        rows = c.execute(
            "SELECT id, content, modality, lang FROM docs ORDER BY id"
        ).fetchall()

    if not rows:
        if EMB_PATH.exists(): EMB_PATH.unlink()
        if IDX_PATH.exists(): IDX_PATH.unlink()
        return 0

    vecs = []
    for row_id, content, modality, lang in rows:
        try:
            if modality == "image":
                p = Path(str(content))
                raw = p.read_bytes() if p.exists() else b""
                vec = embedder._encode_image(raw)
            elif modality == "audio":
                p = Path(str(content))
                raw = p.read_bytes() if p.exists() else b""
                vec = embedder._encode_audio(raw)
            else:
                vec = embedder._encode_text(str(content))
            vecs.append(embedder._normalise(vec))
        except Exception as e:
            logger.warning(f"Re-embed failed for id={row_id}: {e} – using stub.")
            rng = np.random.default_rng(row_id)
            vecs.append(embedder._normalise(
                rng.standard_normal(embedder.dim).astype("float32")
            ))

    matrix = np.vstack(vecs).astype(np.float32)
    _save_matrix(matrix)

    # Update emb_row values
    with _conn() as c:
        ids = [r[0] for r in rows]
        for new_row, doc_id in enumerate(ids):
            c.execute("UPDATE docs SET emb_row=? WHERE id=?", (new_row, doc_id))
        c.commit()

    build_faiss(matrix)
    logger.info(f"Full re-embed complete: {len(vecs)} vectors.")
    return len(vecs)


def get_stats() -> dict:
    with _conn() as c:
        total = c.execute("SELECT COUNT(*) FROM docs").fetchone()[0]
        by_modality = dict(c.execute(
            "SELECT modality, COUNT(*) FROM docs GROUP BY modality"
        ).fetchall())
    matrix = _load_matrix()
    return {
        "total_chunks":       total,
        "by_modality":        by_modality,
        "embedding_vectors":  len(matrix),
        "faiss_index_exists": IDX_PATH.exists(),
        "db_path":            str(DB_PATH),
    }


def load_faiss():
    if not IDX_PATH.exists(): return None
    try:
        import faiss
        return faiss.read_index(str(IDX_PATH))
    except ImportError:
        return None


# ── Internal ──────────────────────────────────────────────────────────────

def _reindex_emb_rows(keep_mask: np.ndarray):
    """
    After removing rows from the matrix, update emb_row in SQLite
    so each document still points to the right row in the compacted matrix.
    keep_mask[i] = True means old row i is kept; its new row = sum(keep_mask[:i]).
    """
    old_to_new = {}
    new_row = 0
    for old_row, kept in enumerate(keep_mask):
        if kept:
            old_to_new[old_row] = new_row
            new_row += 1

    with _conn() as c:
        docs = c.execute("SELECT id, emb_row FROM docs").fetchall()
        for doc_id, emb_row in docs:
            new = old_to_new.get(emb_row, -1)
            c.execute("UPDATE docs SET emb_row=? WHERE id=?", (new, doc_id))
        c.commit()
