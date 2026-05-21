"""Stage (e) – Graph Context Pruning (GCP). Greedy 0/1 knapsack."""
import logging, re
from graph.builder import EvidenceGraph, EvidenceNode
from config.settings import PipelineConfig

logger = logging.getLogger(__name__)


class GraphContextPruner:
    def __init__(self, config: PipelineConfig):
        self.cfg    = config.gcp
        self.w_rel  = config.gcp.relevance_weight
        self.w_coh  = config.gcp.coherence_weight
        self.w_prov = config.gcp.provenance_weight

    def prune(self, graph: EvidenceGraph, importance_scores: dict[int, float],
              token_budget: int) -> tuple[list[dict], dict]:
        nodes = graph.nodes
        if not nodes:
            return [], {"tokens_used": 0, "token_budget": token_budget}

        utilities    = self._compute_utilities(nodes, graph, importance_scores)
        sorted_nodes = sorted(utilities.items(), key=lambda x: x[1], reverse=True)

        selected: list[EvidenceNode] = []
        tokens_used = 0
        total_raw   = sum(self._token_count(n) for n in nodes)

        for node_id, _ in sorted_nodes:
            node = nodes[node_id]
            cost = self._token_count(node)
            if tokens_used + cost <= token_budget:
                selected.append(node)
                tokens_used += cost

        evidence   = [self._node_to_dict(n) for n in selected]
        token_stats = {
            "token_budget":     token_budget,
            "tokens_used":      tokens_used,
            "total_tokens_raw": total_raw,
            "reduction_ratio":  1 - tokens_used / max(total_raw, 1),
            "num_selected":     len(selected),
            "num_candidates":   len(nodes),
            "redundancy_ratio": 1 - len(selected) / max(len(nodes), 1),
        }
        return evidence, token_stats

    def _compute_utilities(self, nodes, graph, importance_scores):
        utilities = {}
        for node in nodes:
            rel  = importance_scores.get(node.id, node.retrieval_score)
            coh  = self._coherence(node, graph)
            prov = node.provenance_score
            utilities[node.id] = self.w_rel * rel + self.w_coh * coh + self.w_prov * prov
        return utilities

    def _coherence(self, node: EvidenceNode, graph: EvidenceGraph) -> float:
        nbrs = graph.neighbours(node.id)
        return sum(w for _, w in nbrs) / len(nbrs) if nbrs else 0.0

    def _token_count(self, node: EvidenceNode) -> int:
        if node.modality == "text":
            return max(1, len(re.split(r"\s+", str(node.content).strip())))
        return 256 if node.modality == "image" else 128

    @staticmethod
    def _node_to_dict(node: EvidenceNode) -> dict:
        return {
            "id":               node.id,
            "content":          node.content,
            "modality":         node.modality,
            "lang":             node.lang,
            "source":           node.source,
            "provenance_score": node.provenance_score,
            "retrieval_score":  node.retrieval_score,
            "importance_score": node.importance_score,
        }
