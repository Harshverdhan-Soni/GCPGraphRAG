"""
FAISS vector index wrapper (Stage a – indexing side).
Falls back to NumPy brute-force if faiss is not installed.
"""
import logging
import numpy as np
from config.settings import PipelineConfig

logger = logging.getLogger(__name__)


class FAISSIndexer:
    def __init__(self, config: PipelineConfig):
        self.config   = config
        self.dim      = config.embedding.embedding_dim
        self.index    = None
        self.metadata: list[dict] = []

    def build(self, embeddings: np.ndarray, metadata: list[dict]) -> None:
        try:
            import faiss
            self._build_faiss(embeddings)
        except ImportError:
            logger.warning("faiss not installed – falling back to NumPy brute-force.")
            self._embeddings = embeddings
        self.metadata = metadata
        logger.info(f"Index built: {len(metadata)} documents, dim={self.dim}.")

    def search(self, query_vec: np.ndarray, top_k: int) -> list[dict]:
        query_vec = query_vec.reshape(1, -1).astype(np.float32)
        if hasattr(self, "_embeddings"):
            return self._numpy_search(query_vec, top_k)
        scores, indices = self.index.search(query_vec, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            entry = dict(self.metadata[idx])
            entry["score"] = float(score)
            entry["id"]    = int(idx)
            results.append(entry)
        return results

    def _build_faiss(self, embeddings: np.ndarray) -> None:
        import faiss
        idx_type = self.config.retrieval.faiss_index_type
        if idx_type == "flat":
            self.index = faiss.IndexFlatIP(self.dim)
        elif idx_type == "ivf":
            quantiser  = faiss.IndexFlatIP(self.dim)
            self.index = faiss.IndexIVFFlat(
                quantiser, self.dim,
                self.config.retrieval.ivf_nlist,
                faiss.METRIC_INNER_PRODUCT,
            )
            self.index.train(embeddings)
        elif idx_type == "hnsw":
            self.index = faiss.IndexHNSWFlat(self.dim, 32, faiss.METRIC_INNER_PRODUCT)
        else:
            raise ValueError(f"Unknown FAISS index type: {idx_type}")
        self.index.add(embeddings)

    def _numpy_search(self, query_vec: np.ndarray, top_k: int) -> list[dict]:
        scores      = (self._embeddings @ query_vec.T).squeeze()
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [
            {**dict(self.metadata[idx]), "score": float(scores[idx]), "id": int(idx)}
            for idx in top_indices
        ]
