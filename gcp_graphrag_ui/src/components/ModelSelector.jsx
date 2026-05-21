// src/components/ModelSelector.jsx
import { useState, useEffect } from "react";
import { api } from "../api/client";

const CONTEXT_MAP = {
  "tinyllama": 1020, "phi3:mini": 3072, "phi3": 4096,
  "mistral": 8192,   "llama3": 8192,    "stub": 4096,
};

const RAM_MAP = {
  "tinyllama": 0.6, "phi3:mini": 5.0, "phi3": 5.5,
  "mistral": 5.0,   "llama3": 6.0,    "stub": 0,
};

// Three recommended presets
const PRESETS = [
  {
    id:        "sbert_phi3",
    label:     "⚡ Balanced",
    sublabel:  "Best quality for 8GB RAM",
    embedding: "sentence-transformers",
    generator: "phi3:mini",
    ram:       "~6 GB",
    rec:       true,
    color:     "cyan",
  },
  {
    id:        "muril_tiny",
    label:     "🇮🇳 Indian",
    sublabel:  "Best for Indian languages",
    embedding: "muril",
    generator: "tinyllama",
    ram:       "~4 GB",
    rec:       false,
    color:     "violet",
  },
  {
    id:        "sbert_tiny",
    label:     "🪶 Lightweight",
    sublabel:  "Fits in 4GB RAM",
    embedding: "sentence-transformers",
    generator: "tinyllama",
    ram:       "~3 GB",
    rec:       false,
    color:     "green",
  },
];

export default function ModelSelector({ config, onConfigChange }) {
  const [models,  setModels]  = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    api.getModelConfig()
      .then(d => { setModels(d); setLoading(false); })
      .catch(() => setLoading(false));
  }, []);

  function applyPreset(preset) {
    onConfigChange("embedding_model", preset.embedding);
    onConfigChange("generator_model", preset.generator);
    onConfigChange("context_length",  CONTEXT_MAP[preset.generator] || 1020);
  }

  function setEmbedding(id) {
    onConfigChange("embedding_model", id);
  }

  function setGenerator(id) {
    onConfigChange("generator_model", id);
    onConfigChange("context_length",  CONTEXT_MAP[id] || 1020);
  }

  const currentPreset = PRESETS.find(
    p => p.embedding === config.embedding_model &&
         p.generator === config.generator_model
  );

  const embeddingModels = models?.embedding_models || [
    { id: "sentence-transformers", name: "Sentence-Transformers", desc: "50+ languages, 400MB RAM", recommended: true,  paper: false },
    { id: "muril",                 name: "MuRIL (Google)",        desc: "17 Indian languages, 900MB RAM", recommended: false, paper: false },
    { id: "mdeberta",              name: "mDeBERTa-v3-base",      desc: "Paper proposed, 900MB RAM",      recommended: false, paper: true  },
    { id: "indic-bert",            name: "IndicBERT",             desc: "12 Indian scripts, 500MB RAM",   recommended: false, paper: false },
  ];

  const generatorModels = models?.generation_models || [
    { id: "phi3:mini", name: "Phi3 Mini 3.8B",  ram_gb: 5.0, context: 3072, desc: "Good quality, needs 5GB free RAM", recommended: true  },
    { id: "tinyllama", name: "TinyLlama 1.1B",  ram_gb: 0.6, context: 1020, desc: "Basic quality, fits 4GB RAM",      recommended: false },
    { id: "mistral",   name: "Mistral 7B",       ram_gb: 5.0, context: 8192, desc: "High quality, needs 8GB RAM",     recommended: false },
    { id: "llama3",    name: "Llama3 8B",        ram_gb: 6.0, context: 8192, desc: "Best quality, needs 16GB RAM",    recommended: false },
    { id: "stub",      name: "Stub (No LLM)",    ram_gb: 0.0, context: 4096, desc: "Chunks only, no generation",      recommended: false },
  ];

  return (
    <>
      {/* ── Preset combinations ── */}
      <div className="sidebar-section">
        <div className="sidebar-label">Quick Presets</div>
        <div className="preset-grid">
          {PRESETS.map(p => (
            <button
              key={p.id}
              className={`preset-btn preset-${p.color} ${currentPreset?.id === p.id ? "preset-active" : ""}`}
              onClick={() => applyPreset(p)}
              title={`${p.embedding} + ${p.generator}`}
            >
              <span className="preset-label">{p.label}</span>
              <span className="preset-sublabel">{p.sublabel}</span>
              <span className="preset-ram">{p.ram}</span>
            </button>
          ))}
        </div>
        {currentPreset && (
          <div className="preset-active-note">
            ✓ {currentPreset.label} preset active
          </div>
        )}
      </div>

      <hr className="divider"/>

      {/* ── Individual embedding model picker ── */}
      <div className="sidebar-section">
        <div className="sidebar-label">Embedding Model</div>
        {embeddingModels.map(m => (
          <div
            key={m.id}
            className={`model-option ${config.embedding_model === m.id ? "model-selected" : ""}`}
            onClick={() => setEmbedding(m.id)}
          >
            <div className="model-option-top">
              <span className="model-name">{m.name}</span>
              {m.recommended && <span className="model-badge-rec">✓ rec</span>}
              {m.paper        && <span className="model-badge-paper">paper</span>}
            </div>
            <div className="model-desc">{m.desc}</div>
          </div>
        ))}
      </div>

      <hr className="divider"/>

      {/* ── Individual generator model picker ── */}
      <div className="sidebar-section">
        <div className="sidebar-label">Generation Model</div>
        {generatorModels.map(m => (
          <div
            key={m.id}
            className={`model-option ${config.generator_model === m.id ? "model-selected" : ""}`}
            onClick={() => setGenerator(m.id)}
          >
            <div className="model-option-top">
              <span className="model-name">{m.name}</span>
              {m.recommended && <span className="model-badge-rec">✓ rec</span>}
              <span className="model-ram">{m.ram_gb > 0 ? `${m.ram_gb}GB` : "free"}</span>
            </div>
            <div className="model-desc">{m.desc}</div>
            <div className="model-ctx">ctx: {(m.context||1020).toLocaleString()} tokens</div>
          </div>
        ))}
        <div className="model-ctx-display">
          Context window set to <span>{(config.context_length||1020).toLocaleString()}</span> tokens
        </div>
      </div>
    </>
  );
}
