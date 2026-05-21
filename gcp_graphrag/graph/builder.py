"""Stage (c) – Multimodal Evidence Graph Construction."""
import logging
from dataclasses import dataclass, field
import numpy as np
from config.settings import PipelineConfig

logger = logging.getLogger(__name__)


@dataclass
class EvidenceNode:
    id:               int
    content:          str
    modality:         str
    lang:             str
    source:           str
    embedding:        np.ndarray
    retrieval_score:  float
    provenance_score: float = 1.0
    importance_score: float = 0.0


@dataclass
class EvidenceGraph:
    nodes: list[EvidenceNode]           = field(default_factory=list)
    adj:   dict[int, list[tuple[int, float]]] = field(default_factory=dict)

    @property
    def num_nodes(self) -> int: return len(self.nodes)

    @property
    def num_edges(self) -> int:
        return sum(len(v) for v in self.adj.values()) // 2

    @property
    def density(self) -> float:
        n = self.num_nodes
        return self.num_edges / (n * (n - 1) / 2) if n >= 2 else 0.0

    def neighbours(self, node_id: int) -> list[tuple[int, float]]:
        return self.adj.get(node_id, [])


class EvidenceGraphBuilder:
    def __init__(self, config: PipelineConfig):
        self.threshold = config.graph.edge_threshold

    def build(self, candidates: list[dict], query_embedding: np.ndarray) -> EvidenceGraph:
        graph = EvidenceGraph()
        for i, c in enumerate(candidates):
            emb  = self._get_or_stub_embedding(c, i)
            node = EvidenceNode(
                id=i, content=c.get("content", ""),
                modality=c.get("modality", "text"), lang=c.get("lang", "en"),
                source=c.get("source", f"doc_{i}"), embedding=emb,
                retrieval_score=c.get("score", 0.0),
                provenance_score=c.get("provenance_score", 1.0),
            )
            graph.nodes.append(node)
            graph.adj[i] = []

        n = graph.num_nodes
        for i in range(n):
            for j in range(i + 1, n):
                weight = self._edge_weight(graph.nodes[i], graph.nodes[j])
                if weight > 0:
                    graph.adj[i].append((j, weight))
                    graph.adj[j].append((i, weight))

        logger.debug(f"Graph: {graph.num_nodes}N {graph.num_edges}E density={graph.density:.3f}")
        return graph

    def _edge_weight(self, ni: EvidenceNode, nj: EvidenceNode) -> float:
        cos_sim = float(np.dot(ni.embedding, nj.embedding))
        if ni.source == nj.source:
            return max(cos_sim, 0.5)
        if ni.modality != nj.modality:
            return cos_sim if cos_sim > self.threshold * 0.9 else 0.0
        return cos_sim if cos_sim > self.threshold else 0.0

    @staticmethod
    def _get_or_stub_embedding(candidate: dict, idx: int) -> np.ndarray:
        if "embedding" in candidate and candidate["embedding"] is not None:
            return np.asarray(candidate["embedding"], dtype=np.float32)
        rng = np.random.default_rng(idx)
        vec = rng.standard_normal(768).astype(np.float32)
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 1e-9 else vec
