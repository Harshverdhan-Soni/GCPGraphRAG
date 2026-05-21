# GCP-GraphRAG — Full Project

Graph Context Pruning for multimodal Retrieval-Augmented Generation.

---

## Project structure

```
project/
├── gcp_graphrag/                   ← Python backend
│   ├── server.py                   ← FastAPI server (start here)
│   ├── pipeline.py                 ← End-to-end pipeline orchestrator
│   ├── demo.py                     ← Quick sanity-check (no server needed)
│   ├── requirements.txt
│   │
│   ├── config/
│   │   └── settings.py             ← All hyperparameters
│   │
│   ├── corpus/                     ← Auto-ingest system (NEW)
│   │   ├── store.py                ← SQLite + FAISS persistence
│   │   ├── ingest.py               ← File reading, chunking, embedding
│   │   ├── watcher.py              ← Filesystem watcher (watchdog)
│   │   ├── worker.py               ← Background embedding threads
│   │   └── batch.py                ← Bulk enqueue on startup
│   │
│   ├── retrieval/
│   │   ├── embedder.py             ← mDeBERTa / CLIP / Whisper stubs
│   │   ├── indexer.py              ← FAISS wrapper
│   │   └── retriever.py            ← Top-k ANN retrieval
│   │
│   ├── graph/
│   │   ├── builder.py              ← Evidence graph G=(V,E,W)
│   │   └── gat.py                  ← Graph Attention Network
│   │
│   ├── core/
│   │   └── gcp.py                  ← Graph Context Pruning (knapsack)
│   │
│   ├── generation/
│   │   └── generator.py            ← Grounded multilingual response
│   │
│   └── utils/
│       ├── evaluation.py           ← Recall, F1, ROUGE-L, budget metrics
│       └── logger.py
│
├── gcp_graphrag_ui/                ← React frontend
│   ├── package.json
│   ├── public/index.html
│   └── src/
│       ├── App.jsx                 ← Main component
│       ├── App.css
│       ├── index.js
│       ├── api/client.js           ← All API calls
│       └── components/
│           ├── GraphCanvas.jsx     ← Evidence graph visualisation
│           ├── CorpusManager.jsx   ← Add / remove documents
│           └── IngestMonitor.jsx   ← Live ingest progress bar (SSE)
│
└── README.md
```

---

## Quick start

### 1. Install Python dependencies

```bash
cd gcp_graphrag/
pip install -r requirements.txt
```

### 2. (Optional) Verify with the demo script

```bash
python demo.py
```

### 3. Start the backend

```bash
uvicorn server:app --reload --port 8000
```

The server will:
- Load any existing FAISS index from `corpus/data/`
- Start 3 background embedding worker threads
- Start watching `corpus/files/` for new files

### 4. Start the React frontend

```bash
cd ../gcp_graphrag_ui/
npm install
npm start        # → http://localhost:3000
```

---

## Automated file ingestion

Drop **any** supported file into `corpus/files/` — it is embedded automatically
within seconds, no API call required.

```bash
# Text
cp report.pdf       gcp_graphrag/corpus/files/
cp notes.md         gcp_graphrag/corpus/files/

# Audio
cp interview.mp3    gcp_graphrag/corpus/files/audio/

# Images
cp diagram.png      gcp_graphrag/corpus/files/images/
```

### Supported formats

| Modality | Extensions |
|----------|-----------|
| Text | `.txt` `.md` `.pdf` `.docx` `.html` `.csv` `.json` |
| Image | `.jpg` `.jpeg` `.png` `.webp` `.gif` `.bmp` |
| Audio | `.mp3` `.wav` `.m4a` `.ogg` `.flac` `.aac` |

### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `WATCH_FOLDER` | `./corpus/files` | Folder to watch for new files |
| `N_WORKERS` | `3` | Number of parallel embedding threads |
| `REBUILD_EVERY` | `10` | Rebuild FAISS index every N files |

```bash
WATCH_FOLDER=./data N_WORKERS=6 uvicorn server:app --port 8000
```

---

## Enabling real embedding models

Open `retrieval/embedder.py` and uncomment the production blocks inside
`_encode_text`, `_encode_image`, and `_encode_audio`, then install:

```bash
pip install transformers torch sentencepiece open-clip-torch openai-whisper
```

No other files need to change — the pipeline, server, and UI work identically
with real or stub embeddings.

---

## API reference

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Server status |
| GET | `/corpus` | List all documents |
| POST | `/corpus/reset` | Re-index demo corpus |
| POST | `/corpus/add` | Add single document |
| DELETE | `/corpus/{id}` | Remove document |
| GET | `/corpus/stats` | Counts by modality |
| POST | `/run` | Run full GCP-GraphRAG pipeline |
| POST | `/ingest/file` | Embed a single file path |
| POST | `/ingest/folder` | Embed all files in a folder |
| POST | `/ingest/trigger` | Enqueue folder for background workers |
| GET | `/ingest/progress` | Snapshot of worker progress |
| GET | `/ingest/stream` | SSE stream for live progress (React) |

---

## React environment variable

```bash
# Override backend URL (e.g. for deployed backend)
REACT_APP_API_URL=https://my-api.example.com npm start
```

---

## Persistent storage layout

```
gcp_graphrag/
└── corpus/
    ├── files/          ← Drop files here for auto-ingestion
    └── data/
        ├── corpus.db   ← SQLite: content + metadata
        ├── faiss.index ← FAISS ANN index (survives restarts)
        └── embeddings.npy  ← Raw float32 embedding matrix
```
