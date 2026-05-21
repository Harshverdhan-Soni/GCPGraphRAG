"""
GCP-GraphRAG configuration.
All hyperparameters live here — edit this file to change defaults.

Current stack:
  Embedding  : google/muril-base-cased (17 Indian languages, 768-dim)
  Generation : tinyllama via Ollama (637MB, fits in 4-8GB RAM)
  Index      : FAISS Flat (exact, best for <10k docs)
"""
from dataclasses import dataclass, field


@dataclass
class EmbeddingConfig:
    text_model:     str = "google/muril-base-cased"
    vision_model:   str = "openai/clip-vit-base-patch32"
    audio_model:    str = "openai/whisper-base"
    embedding_dim:  int = 768
    max_seq_length: int = 512
    batch_size:     int = 32


@dataclass
class RetrievalConfig:
    top_k:            int = 8       # 8 candidates — good for 120-doc corpus
    faiss_index_type: str = "flat"  # flat = exact, best for <10k docs
    ivf_nlist:        int = 100


@dataclass
class GraphConfig:
    edge_threshold:    float = 0.40  # cosine sim threshold for graph edges
    num_gat_layers:    int   = 2     # 2 = research sweet spot, no over-smoothing
    num_heads:         int   = 4     # 4 attention heads, best tradeoff
    hidden_dim:        int   = 256
    leaky_relu_slope:  float = 0.2
    attention_dropout: float = 0.1


@dataclass
class GCPConfig:
    relevance_weight:    float = 0.55  # semantic similarity to query
    coherence_weight:    float = 0.30  # graph connectivity score
    provenance_weight:   float = 0.15  # source trustworthiness
    max_tokens_per_unit: int   = 256
    solver:              str   = "greedy"  # greedy = fast, ~98% optimal


@dataclass
class ModelConfig:
    decoder:              str   = "tinyllama"
    # tinyllama max context = 2048 tokens total
    # ~1020 usable after system prompt + query overhead
    context_length:       int   = 1020
    system_prompt_tokens: int   = 100
    max_new_tokens:       int   = 256
    temperature:          float = 0.1   # low = factual, less hallucination
    top_p:                float = 0.9


@dataclass
class PipelineConfig:
    embedding:  EmbeddingConfig  = field(default_factory=EmbeddingConfig)
    retrieval:  RetrievalConfig  = field(default_factory=RetrievalConfig)
    graph:      GraphConfig      = field(default_factory=GraphConfig)
    gcp:        GCPConfig        = field(default_factory=GCPConfig)
    model:      ModelConfig      = field(default_factory=ModelConfig)
    device:     str              = "cpu"
    seed:       int              = 42
