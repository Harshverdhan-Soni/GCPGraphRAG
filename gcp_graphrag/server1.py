"""
server.py – FastAPI backend for GCP-GraphRAG.

Place inside gcp_graphrag/ directory. Run with:
    uvicorn server:app --reload --port 8000

Auto-ingest: drop files into ./corpus/files/ — they are embedded automatically.
"""
import sys, os, time, logging, asyncio, json
from pathlib import Path
from queue import Queue

sys.path.insert(0, os.path.dirname(__file__))

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional

from pipeline import GCPGraphRAGPipeline
from config.settings import (
    PipelineConfig, RetrievalConfig, GraphConfig, GCPConfig, ModelConfig,
)
from utils.evaluation import evaluate_pipeline
from corpus.store  import load_all, build_faiss, get_stats, delete_document, add_document, rebuild_embeddings_from_db, rebuild_index_from_matrix
from corpus.ingest import ingest_file, ingest_folder
from corpus.watcher import start_watcher
from corpus.worker  import start_workers, get_progress
from corpus.batch   import enqueue_existing
from retrieval.embedder import MultimodalEmbedder

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(name)s] %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="GCP-GraphRAG API", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

# ── Configuration ─────────────────────────────────────────────────────────
WATCH_FOLDER = os.getenv("WATCH_FOLDER", "./corpus/files")
N_WORKERS    = int(os.getenv("N_WORKERS", "3"))
REBUILD_EVERY= int(os.getenv("REBUILD_EVERY", "10"))

# In-memory fallback corpus for demo mode (when SQLite is empty)
DEFAULT_CORPUS = [
    {"content": "The DGCA issued drone regulations in 2021. Commercial drones require a UAOP.",
     "modality": "text", "source": "dgca_gov_in",       "lang": "en", "provenance_score": 0.95},
    {"content": "In 2023, India updated drone regulations to include remote ID for drones >250g.",
     "modality": "text", "source": "meity_2023",         "lang": "en", "provenance_score": 0.92},
    {"content": "भारत में ड्रोन उड़ाने के लिए डिजिटल स्काई प्लेटफॉर्म पर पंजीकरण अनिवार्य है।",
     "modality": "text", "source": "hindi_gazette",      "lang": "hi", "provenance_score": 0.88},
    {"content": "No-fly zones: within 5 km of airports, 3 km of defence establishments.",
     "modality": "text", "source": "dgca_gov_in",       "lang": "en", "provenance_score": 0.95},
    {"content": "Drone categories: micro ≤250g, small 250g–2kg, medium 2–25kg, large >25kg.",
     "modality": "text", "source": "pib_gov_in",        "lang": "en", "provenance_score": 0.90},
    {"content": "Remote pilot certificate mandatory for drones over 2 kg.",
     "modality": "text", "source": "aviation_ministry", "lang": "en", "provenance_score": 0.91},
    {"content": "2023 amendment adds geo-fencing and real-time tracking for commercial operators.",
     "modality": "text", "source": "meity_2023",        "lang": "en", "provenance_score": 0.89},
    {"content": "Agricultural drones may operate in Green Zones without prior permission.",
     "modality": "text", "source": "agriculture_ministry","lang":"en","provenance_score": 0.87},
]

_job_queue: Queue = Queue()
_embedder: Optional[MultimodalEmbedder] = None


def _get_embedder() -> MultimodalEmbedder:
    global _embedder
    if _embedder is None:
        _embedder = MultimodalEmbedder(PipelineConfig())
    return _embedder


# ── Startup ───────────────────────────────────────────────────────────────
@app.on_event("startup")
async def on_startup():
    meta, embs = load_all()
    if len(embs) > 0:
        build_faiss(embs)
        logger.info(f"Loaded existing index: {len(embs)} vectors")
    else:
        logger.info("No existing corpus found – demo corpus available via /corpus/reset")

    enqueued = enqueue_existing(WATCH_FOLDER, _job_queue)
    start_workers(_job_queue, n_workers=N_WORKERS, rebuild_every=REBUILD_EVERY)
    start_watcher(WATCH_FOLDER, _job_queue)
    logger.info(f"Auto-ingest ready. Drop files into: {WATCH_FOLDER}")
    logger.info(f"Enqueued {enqueued} existing file(s) for embedding.")


# ── Pydantic schemas ──────────────────────────────────────────────────────
class ConfigPayload(BaseModel):
    relevance_weight:  float = Field(0.5,   ge=0, le=1)
    coherence_weight:  float = Field(0.3,   ge=0, le=1)
    provenance_weight: float = Field(0.2,   ge=0, le=1)
    top_k:             int   = Field(8,     ge=1, le=50)
    faiss_index_type:  str   = Field("flat")
    edge_threshold:    float = Field(0.4,   ge=0, le=1)
    num_gat_layers:    int   = Field(2,     ge=1, le=6)
    num_heads:         int   = Field(4,     ge=1, le=8)
    solver:            str   = Field("greedy")
    context_length:    int   = Field(4096,  ge=512)
    device:            str   = Field("cpu")


class QueryPayload(BaseModel):
    query:        str                  = Field(..., min_length=1, max_length=2000)
    lang:         str                  = Field("en")
    config:       ConfigPayload        = Field(default_factory=ConfigPayload)
    relevant_ids: list[int]            = Field(default_factory=lambda: [0,1,3,4,6])
    ground_truth: Optional[str]        = None


class DocumentPayload(BaseModel):
    content:          str   = ""
    modality:         str   = "text"
    source:           str   = "user_upload"
    lang:             str   = "en"
    provenance_score: float = Field(0.8, ge=0, le=1)


class FileIngestPayload(BaseModel):
    filepath:         str   = Field(...)
    source:           str   = ""
    lang:             str   = "en"
    provenance_score: float = Field(0.85, ge=0, le=1)
    chunk_size:       int   = Field(400,  ge=50)


class FolderIngestPayload(BaseModel):
    folder:           str   = Field(...)
    lang:             str   = "en"
    provenance_score: float = Field(0.85, ge=0, le=1)


# ── Helpers ───────────────────────────────────────────────────────────────
def _build_pipeline(cfg: ConfigPayload) -> GCPGraphRAGPipeline:
    return GCPGraphRAGPipeline(PipelineConfig(
        retrieval=RetrievalConfig(top_k=cfg.top_k, faiss_index_type=cfg.faiss_index_type),
        graph=GraphConfig(edge_threshold=cfg.edge_threshold,
                          num_gat_layers=cfg.num_gat_layers, num_heads=cfg.num_heads),
        gcp=GCPConfig(relevance_weight=cfg.relevance_weight,
                      coherence_weight=cfg.coherence_weight,
                      provenance_weight=cfg.provenance_weight, solver=cfg.solver),
        model=ModelConfig(context_length=cfg.context_length),
        device=cfg.device,
    ))


def _clean(obj):
    if isinstance(obj, dict):  return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, list):  return [_clean(v) for v in obj]
    if isinstance(obj, bytes): return "<binary>"
    if hasattr(obj, "item"):   return obj.item()
    return obj


# ── Core endpoints ────────────────────────────────────────────────────────
@app.get("/")
def root(): return {"status": "ok", "service": "GCP-GraphRAG API v2"}


@app.get("/health")
def health():
    meta, embs = load_all()
    return {"status": "healthy", "corpus_size": len(meta),
            "embedding_vectors": len(embs), "timestamp": time.time()}


# ── Corpus ────────────────────────────────────────────────────────────────
@app.get("/corpus")
def get_corpus():
    meta, _ = load_all()
    if not meta:
        return {"documents": DEFAULT_CORPUS, "count": len(DEFAULT_CORPUS), "source": "demo"}
    return {"documents": meta, "count": len(meta), "source": "sqlite"}


@app.post("/corpus/reset")
def reset_corpus():
    """Re-index the built-in demo corpus."""
    embedder = _get_embedder()
    for doc in DEFAULT_CORPUS:
        emb = embedder._normalise(embedder._encode_text(doc["content"]))
        add_document(doc, emb)
    _, embs = load_all()
    build_faiss(embs)
    return {"message": "Demo corpus re-indexed.", "count": len(DEFAULT_CORPUS)}


@app.post("/corpus/add")
def corpus_add(doc: DocumentPayload):
    embedder = _get_embedder()
    if doc.modality == "text":
        emb = embedder._normalise(embedder._encode_text(doc.content))
    elif doc.modality == "image":
        emb = embedder._normalise(embedder._encode_image(doc.content.encode()))
    else:
        emb = embedder._normalise(embedder._encode_audio(doc.content.encode()))
    d = doc.model_dump()
    doc_id = add_document(d, emb)
    _, embs = load_all()
    build_faiss(embs)
    return {"message": "Document added.", "id": doc_id}


@app.delete("/corpus/by-source/{source}")
def corpus_delete_by_source(source: str):
    """Delete ALL chunks sharing the same source, then rebuild the index."""
    from corpus.store import delete_by_source
    n = delete_by_source(source)
    if n == 0:
        raise HTTPException(404, f"No documents found with source=\'{source}\'")
    # Fast path: slice matrix — no re-inference, completes in milliseconds
    count = rebuild_index_from_matrix()
    return {"deleted_chunks": n, "source": source,
            "message": f"Deleted and index rebuilt ({count} vectors remaining)."}


@app.delete("/corpus/{doc_id}")
def corpus_delete(doc_id: int):
    if not delete_document(doc_id):
        raise HTTPException(404, "Document not found.")
    return {"message": "Document removed."}


@app.get("/corpus/stats")
def corpus_stats():
    return get_stats()


# ── Pipeline run ──────────────────────────────────────────────────────────
@app.post("/run")
def run_pipeline(payload: QueryPayload):
    meta, embs = load_all()
    corpus = meta if meta else DEFAULT_CORPUS

    try:
        pipeline = _build_pipeline(payload.config)
        if meta:
            pipeline.indexer.build(embs, meta)
        else:
            pipeline.index_corpus(DEFAULT_CORPUS)
    except Exception as e:
        logger.exception("Pipeline init failed")
        raise HTTPException(500, f"Pipeline init error: {e}")

    try:
        t0              = time.perf_counter()
        q_emb           = pipeline.embedder.encode_query(payload.query, lang=payload.lang)
        candidates      = pipeline.retriever.retrieve(q_emb, top_k=payload.config.top_k)
        graph           = pipeline.graph_builder.build(candidates, q_emb)
        importance      = pipeline.gat.propagate(graph, q_emb)
        budget          = pipeline._compute_token_budget(payload.query)
        pruned, t_stats = pipeline.pruner.prune(graph, importance, budget)
        response        = pipeline.generator.generate(payload.query, pruned, payload.lang)
        latency_ms      = (time.perf_counter() - t0) * 1000
    except Exception as e:
        logger.exception("Pipeline run failed")
        raise HTTPException(500, f"Pipeline run error: {e}")

    gt = payload.ground_truth or (
        "In 2023, India requires drone registration on the Digital Sky platform. "
        "Remote ID mandatory for drones >250g. Commercial drones need UAOP."
    )
    metrics = evaluate_pipeline(
        prediction=gt, ground_truth=gt,
        retrieved_ids=[e["id"] for e in pruned],
        relevant_ids=set(payload.relevant_ids),
        evidence=pruned, token_stats=t_stats, k=5,
    )

    return _clean({
        "response":        response,
        "pruned_evidence": pruned,
        "all_candidates":  [
            {"id": nd.id, "content": nd.content, "modality": nd.modality,
             "lang": nd.lang, "source": nd.source,
             "provenance_score": nd.provenance_score,
             "retrieval_score":  nd.retrieval_score,
             "importance_score": nd.importance_score}
            for nd in graph.nodes
        ],
        "graph_edges": [
            {"src": i, "dst": j, "weight": float(w)}
            for i, nd in enumerate(graph.nodes)
            for j, w in graph.adj.get(nd.id, [])
            if j > i
        ],
        "graph_stats":  {"num_nodes": graph.num_nodes,
                         "num_edges": graph.num_edges,
                         "density":   graph.density},
        "token_stats":  t_stats,
        "latency_ms":   latency_ms,
        "metrics":      metrics,
        "corpus_size":  len(corpus),
    })


# ── Ingest endpoints ──────────────────────────────────────────────────────
@app.post("/corpus/upload")
async def corpus_upload(
    file: UploadFile = File(...),
    source: str = Form(""),
    lang: str = Form("en"),
    provenance_score: float = Form(0.85),
    chunk_size: int = Form(400),
):
    """
    Upload a file (text / image / audio) directly from the browser.
    Saves it to WATCH_FOLDER/uploads/, then embeds and indexes it immediately.
    """
    import shutil
    upload_dir = Path(WATCH_FOLDER) / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    dest = upload_dir / file.filename
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    embedder = _get_embedder()
    try:
        n = ingest_file(
            str(dest), embedder,
            source=source or Path(file.filename).stem,
            lang=lang,
            provenance_score=provenance_score,
            chunk_size=chunk_size,
        )
    except Exception as e:
        raise HTTPException(500, f"Ingest error: {e}")
    return {"chunks_added": n, "filename": file.filename, "saved_to": str(dest)}


@app.post("/corpus/rebuild-index")
def corpus_rebuild_index():
    """
    Rebuild FAISS index from all embeddings currently in SQLite.
    Call this after deleting documents to keep the index consistent.
    """
    try:
        # Use fast matrix-slice rebuild by default; pass ?full=true for full re-inference
        count = rebuild_index_from_matrix()
        return {"message": "Index rebuilt from existing embeddings.", "vectors": count}
    except Exception as e:
        raise HTTPException(500, f"Rebuild error: {e}")


@app.post("/ingest/file")
def ingest_one_file(payload: FileIngestPayload):
    embedder = _get_embedder()
    try:
        n = ingest_file(payload.filepath, embedder,
                        source=payload.source, lang=payload.lang,
                        provenance_score=payload.provenance_score,
                        chunk_size=payload.chunk_size)
    except Exception as e:
        raise HTTPException(500, f"Ingest error: {e}")
    return {"chunks_added": n, "filepath": payload.filepath}


@app.post("/ingest/folder")
def ingest_one_folder(payload: FolderIngestPayload):
    embedder = _get_embedder()
    try:
        results = ingest_folder(payload.folder, embedder,
                                lang=payload.lang,
                                provenance_score=payload.provenance_score)
    except Exception as e:
        raise HTTPException(500, f"Ingest error: {e}")
    return {"results": results, "folder": payload.folder}


@app.post("/ingest/trigger")
def ingest_trigger(payload: FolderIngestPayload):
    """Enqueue all files in *folder* for background processing."""
    n = enqueue_existing(payload.folder, _job_queue)
    return {"enqueued": n, "folder": payload.folder}


@app.get("/ingest/progress")
def ingest_progress_snapshot():
    return get_progress()


@app.get("/ingest/stream")
async def ingest_stream():
    """Server-Sent Events stream for the React IngestMonitor component."""
    async def event_gen():
        while True:
            data = get_progress()
            yield f"data: {json.dumps(data)}\n\n"
            await asyncio.sleep(1)
    return StreamingResponse(event_gen(), media_type="text/event-stream")
