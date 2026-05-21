"""
GCP-GraphRAG configuration.
All hyperparameters and model identifiers live here.
"""
from dataclasses import dataclass, field


@dataclass
class EmbeddingConfig:
    # sentence-transformers model — reliable on Windows, no SentencePiece issues
    text_model:    str = "paraphrase-multilingual-mpnet-base-v2"
    vision_model:  str = "openai/clip-vit-base-patch32"
    audio_model:   str = "openai/whisper-base"
    embedding_dim: int = 768
    max_seq_length:int = 512
    batch_size:    int = 32


@dataclass
class RetrievalConfig:
    top_k:            int = 8       # reduced from 20 — return most relevant only
    faiss_index_type: str = "flat"
    ivf_nlist:        int = 100


@dataclass
class GraphConfig:
    edge_threshold:    float = 0.4  # lowered so more edges form
    num_gat_layers:    int   = 2
    num_heads:         int   = 4
    hidden_dim:        int   = 256
    leaky_relu_slope:  float = 0.2
    attention_dropout: float = 0.1


@dataclass
class GCPConfig:
    provenance_weight:   float = 0.2
    coherence_weight:    float = 0.3
    relevance_weight:    float = 0.5
    max_tokens_per_unit: int   = 256
    solver:              str   = "greedy"


@dataclass
class ModelConfig:
    decoder:              str   = "facebook/mbart-large-cc25"
    # mBART max usable context = 1020 tokens
    context_length:       int   = 1020
    system_prompt_tokens: int   = 100
    max_new_tokens:       int   = 256
    temperature:          float = 0.7
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
