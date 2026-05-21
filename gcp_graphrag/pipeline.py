"""GCP-GraphRAG: End-to-end pipeline orchestrator."""
import time, logging
from typing import Optional
from config.settings import PipelineConfig
from retrieval.embedder import MultimodalEmbedder
from retrieval.indexer import FAISSIndexer
from retrieval.retriever import CandidateRetriever
from graph.builder import EvidenceGraphBuilder
from graph.gat import GraphAttentionNetwork
from core.gcp import GraphContextPruner
from generation.generator import MultilingualGenerator
from utils.logger import setup_logger

logger = setup_logger(__name__)


class GCPGraphRAGPipeline:
    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config       = config or PipelineConfig()
        self.embedder     = MultimodalEmbedder(self.config)
        self.indexer      = FAISSIndexer(self.config)
        self.retriever    = CandidateRetriever(self.indexer, self.config)
        self.graph_builder= EvidenceGraphBuilder(self.config)
        self.gat          = GraphAttentionNetwork(self.config)
        self.pruner       = GraphContextPruner(self.config)
        self.generator    = MultilingualGenerator(self.config)
        logger.info("Pipeline ready.")

    def index_corpus(self, documents: list[dict]) -> None:
        logger.info(f"Indexing {len(documents)} documents…")
        embeddings, metadata = self.embedder.encode_corpus(documents)
        self.indexer.build(embeddings, metadata)
        logger.info("Corpus indexed.")

    def run(self, query: str, lang: str = "en", top_k: int = None) -> dict:
        t0    = time.perf_counter()
        top_k = top_k or self.config.retrieval.top_k

        query_embedding = self.embedder.encode_query(query, lang=lang)
        candidates      = self.retriever.retrieve(query_embedding, top_k=top_k)
        graph           = self.graph_builder.build(candidates, query_embedding)
        importance      = self.gat.propagate(graph, query_embedding)
        token_budget    = self._compute_token_budget(query)
        pruned, token_stats = self.pruner.prune(graph, importance, token_budget)
        response        = self.generator.generate(query=query, evidence=pruned, lang=lang)

        return {
            "response":        response,
            "pruned_evidence": pruned,
            "graph_stats": {
                "num_nodes": graph.num_nodes,
                "num_edges": graph.num_edges,
                "density":   graph.density,
            },
            "token_stats": token_stats,
            "latency_ms":  (time.perf_counter() - t0) * 1000,
        }

    def _compute_token_budget(self, query: str) -> int:
        tq = len(query.split())
        return self.config.model.context_length - tq - self.config.model.system_prompt_tokens
