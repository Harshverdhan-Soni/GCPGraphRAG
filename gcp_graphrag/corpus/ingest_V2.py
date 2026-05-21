"""
Ingest pipeline: read a file → chunk/transcribe → embed → store.

Supported formats
-----------------
Text  : .txt .md .pdf .docx .html .csv .json
Image : .jpg .jpeg .png .webp .gif .bmp
Audio : .mp3 .wav .m4a .ogg .flac .aac
"""
import logging
from pathlib import Path
import numpy as np

from corpus.store import add_document, build_faiss, load_all
from retrieval.embedder import MultimodalEmbedder
from config.settings import PipelineConfig

logger = logging.getLogger(__name__)

SUPPORTED: dict[str, list[str]] = {
    "text":  [".txt", ".md", ".pdf", ".docx", ".html", ".csv", ".json"],
    "image": [".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"],
    "audio": [".mp3", ".wav", ".m4a", ".ogg", ".flac", ".aac"],
}
ALL_EXTS = {ext for exts in SUPPORTED.values() for ext in exts}


# ── Public API ────────────────────────────────────────────────────────────

def ingest_file(
    filepath: str,
    embedder: MultimodalEmbedder,
    source: str = "",
    lang: str = "en",
    provenance_score: float = 0.85,
    chunk_size: int = 400,
    chunk_overlap: int = 50,
    rebuild_index: bool = True,
) -> int:
    """
    Embed and persist a single file.
    Returns the number of chunks added.
    """
    path     = Path(filepath)
    ext      = path.suffix.lower()
    modality = _modality(ext)

    if modality is None:
        logger.warning(f"Unsupported extension '{ext}' – skipping {path.name}")
        return 0

    src = source or path.stem

    if modality == "text":
        text   = _read_text(path)
        chunks = _chunk(text, chunk_size, chunk_overlap)
        docs   = [
            {"content": c, "filepath": str(path), "modality": "text",
             "source": src, "lang": lang, "provenance_score": provenance_score}
            for c in chunks
        ]
    else:
        raw  = path.read_bytes()
        docs = [{"content": raw, "filepath": str(path), "modality": modality,
                 "source": src, "lang": lang, "provenance_score": provenance_score}]

    for doc in docs:
        emb = _embed(embedder, doc)
        add_document(doc, emb)

    if rebuild_index:
        _, embs = load_all()
        if len(embs) > 0:
            build_faiss(embs)

    logger.info(f"Ingested {path.name}: {len(docs)} chunk(s)")
    return len(docs)


def ingest_folder(
    folder: str,
    embedder: MultimodalEmbedder,
    lang: str = "en",
    provenance_score: float = 0.85,
    chunk_size: int = 400,
    chunk_overlap: int = 50,
) -> dict:
    """
    Recursively ingest all supported files from a folder.
    Returns counts per modality.
    """
    root    = Path(folder)
    results = {"text": 0, "image": 0, "audio": 0, "skipped": 0, "errors": 0}

    files = [p for p in root.rglob("*") if p.is_file()]
    logger.info(f"Found {len(files)} files in {folder}")

    for path in files:
        ext = path.suffix.lower()
        if ext not in ALL_EXTS:
            results["skipped"] += 1
            continue
        modality = _modality(ext)
        try:
            n = ingest_file(
                str(path), embedder,
                lang=lang, provenance_score=provenance_score,
                chunk_size=chunk_size, chunk_overlap=chunk_overlap,
                rebuild_index=False,   # rebuild once at the end
            )
            results[modality] += n
        except Exception as e:
            logger.error(f"Failed {path.name}: {e}")
            results["errors"] += 1

    # Single rebuild after all files
    _, embs = load_all()
    if len(embs) > 0:
        build_faiss(embs)

    return results


# ── Helpers ───────────────────────────────────────────────────────────────

def _modality(ext: str) -> str | None:
    for m, exts in SUPPORTED.items():
        if ext in exts:
            return m
    return None


def _embed(embedder: MultimodalEmbedder, doc: dict) -> np.ndarray:
    m = doc["modality"]
    if m == "text":
        return embedder._normalise(embedder._encode_text(doc["content"]))
    elif m == "image":
        return embedder._normalise(embedder._encode_image(doc["content"]))
    else:
        return embedder._normalise(embedder._encode_audio(doc["content"]))


def _read_text(path: Path) -> str:
    ext = path.suffix.lower()
    try:
        if ext == ".pdf":
            from pypdf import PdfReader
            return "\n".join(p.extract_text() or "" for p in PdfReader(path).pages)
        if ext == ".docx":
            import docx
            return "\n".join(p.text for p in docx.Document(str(path)).paragraphs)
        if ext == ".html":
            from bs4 import BeautifulSoup
            return BeautifulSoup(path.read_bytes(), "html.parser").get_text(separator="\n")
        if ext == ".csv":
            import csv, io
            rows = list(csv.reader(io.StringIO(path.read_text(errors="ignore"))))
            return "\n".join(",".join(r) for r in rows)
        if ext == ".json":
            import json
            data = json.loads(path.read_text(errors="ignore"))
            return json.dumps(data, indent=2) if isinstance(data, (dict, list)) else str(data)
        # .txt .md fallthrough
        return path.read_text(errors="ignore")
    except Exception as e:
        logger.warning(f"Could not read {path.name}: {e} – treating as empty")
        return ""


def _chunk(text: str, size: int, overlap: int) -> list[str]:
    words  = text.split()
    if not words:
        return []
    chunks = []
    i      = 0
    while i < len(words):
        chunks.append(" ".join(words[i: i + size]))
        i += size - overlap
    return chunks
