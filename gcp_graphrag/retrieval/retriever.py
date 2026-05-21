"""Stage (b) – Initial Candidate Retrieval."""
import logging
import numpy as np
from retrieval.indexer import FAISSIndexer
from config.settings import PipelineConfig

logger = logging.getLogger(__name__)


class CandidateRetriever:
    def __init__(self, indexer: FAISSIndexer, config: PipelineConfig):
        self.indexer = indexer
        self.config  = config

    def retrieve(self, query_embedding: np.ndarray, top_k: int = None) -> list[dict]:
        k          = top_k or self.config.retrieval.top_k
        candidates = self.indexer.search(query_embedding, top_k=k)
        logger.debug(f"Retrieved {len(candidates)} candidates.")
        return candidates
