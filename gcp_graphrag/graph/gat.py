"""Stage (d) – Query-Conditioned Graph Relevance Propagation (GAT)."""
import logging
import numpy as np
from graph.builder import EvidenceGraph
from config.settings import PipelineConfig

logger = logging.getLogger(__name__)


class GraphAttentionNetwork:
    def __init__(self, config: PipelineConfig):
        self.config      = config
        self.num_layers  = config.graph.num_gat_layers
        self.hidden_dim  = config.graph.hidden_dim
        self.num_heads   = config.graph.num_heads
        self.leaky_slope = config.graph.leaky_relu_slope

        rng    = np.random.default_rng(config.seed)
        self.W = rng.standard_normal(
            (config.embedding.embedding_dim, self.hidden_dim)
        ).astype(np.float32) * 0.02
        self.a = rng.standard_normal(2 * self.hidden_dim).astype(np.float32) * 0.02

    def propagate(self, graph: EvidenceGraph,
                  query_embedding: np.ndarray) -> dict[int, float]:
        n = graph.num_nodes
        if n == 0:
            return {}

        H    = np.vstack([nd.embedding for nd in graph.nodes]).astype(np.float32)
        H    = H @ self.W
        H    = np.tanh(H)

        for _ in range(self.num_layers):
            H = self._gat_layer(H, graph)

        importance = np.linalg.norm(H, axis=1)
        retrieval  = np.array([nd.retrieval_score for nd in graph.nodes])
        importance = 0.7 * importance + 0.3 * retrieval

        mn, mx = importance.min(), importance.max()
        if mx > mn:
            importance = (importance - mn) / (mx - mn)

        scores = {nd.id: float(importance[i]) for i, nd in enumerate(graph.nodes)}
        for nd in graph.nodes:
            nd.importance_score = scores[nd.id]
        return scores

    def _gat_layer(self, H: np.ndarray, graph: EvidenceGraph) -> np.ndarray:
        H_new = np.zeros_like(H)
        for i, node in enumerate(graph.nodes):
            nbrs = graph.neighbours(node.id)
            if not nbrs:
                H_new[i] = np.tanh(H[i])
                continue
            alphas = []
            for j_id, _ in nbrs:
                e_ij = self._leaky_relu(self.a @ np.concatenate([H[i], H[j_id]]))
                alphas.append((j_id, e_ij))
            raw  = np.array([e for _, e in alphas])
            raw -= raw.max()
            exp  = np.exp(raw)
            agg  = np.zeros(H.shape[1])
            for k, (j_id, _) in enumerate(alphas):
                agg += (exp[k] / exp.sum()) * H[j_id]
            H_new[i] = np.tanh(agg)
        return H_new

    def _leaky_relu(self, x: float) -> float:
        return x if x >= 0 else self.leaky_slope * x
