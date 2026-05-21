// src/App.jsx
import { useState, useCallback, useRef, useEffect } from "react";
import { api, runStream } from "./api/client";
import ModelSelector from "./components/ModelSelector";
import GraphCanvas from "./components/GraphCanvas";
import CorpusManager from "./components/CorpusManager";
import IngestMonitor from "./components/IngestMonitor";
import GenerationProgress from "./components/GenerationProgress";
import ReportModal from "./components/ReportModal";
import "./App.css";

// ── Pipeline stages ──────────────────────────────────────────────────────────
const STAGES = [
  { key: "a", label: "(a) Embed" },
  { key: "b", label: "(b) Retrieve" },
  { key: "c", label: "(c) Graph" },
  { key: "d", label: "(d) GAT" },
  { key: "e", label: "(e) GCP Prune" },
  { key: "gen", label: "Generate" },
];

const STAGE_DELAYS_MS = [0, 300, 600, 900, 1200, 1500];

const DEFAULT_CONFIG = {
  relevance_weight: 0.5,
  coherence_weight: 0.3,
  provenance_weight: 0.2,
  top_k: 8,
  faiss_index_type: "flat",
  edge_threshold: 0.4,
  num_gat_layers: 2,
  num_heads: 4,
  solver: "greedy",
  context_length: 4096,
  device: "cpu",
};

// ── Helpers ──────────────────────────────────────────────────────────────────
function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }
function fmt(n, d = 3) { return typeof n === "number" ? (n * 100).toFixed(1) + "%" : "—"; }

// ── Main component ────────────────────────────────────────────────────────────
export default function App() {
  const [query, setQuery] = useState("What are the drone regulations in India in 2023?");
  const [lang, setLang] = useState("en");
  const [config, setConfig] = useState(DEFAULT_CONFIG);
  const [running, setRunning] = useState(false);
  const [activeStage, setActiveStage] = useState(-1);
  const [doneStages, setDoneStages] = useState(new Set());
  const [result, setResult] = useState(null);
  const [genProgress, setGenProgress] = useState(null); // {stage, pct, message}
  const [reportLoading, setReportLoading] = useState(false);
  const [error, setError] = useState(null);
  const [serverOk, setServerOk] = useState(null);
  const [modelOptions, setModelOptions] = useState({
    embedding_models: [
      { id: "sentence-transformers", label: "Sentence-Transformers — 50+ langs ✓", context: 1020 },
      { id: "muril",    label: "MuRIL (Google) — 17 Indian langs", context: 1020 },
      { id: "mdeberta", label: "mDeBERTa-v3-base (paper) — 96 langs", context: 1020 },
      { id: "clip",     label: "CLIP ViT-B/32 — Image+Text joint",    context: 1020 },
    ],
    generator_models: [
      { id: "phi3:mini", label: "Phi3 Mini 3.8B  — 5GB RAM ✓",   context: 3072 },
      { id: "tinyllama", label: "TinyLlama 1.1B  — 0.6GB RAM", context: 1020 },
      { id: "mistral",   label: "Mistral 7B      — 5GB RAM",   context: 8192 },
      { id: "llama3",    label: "Llama3 8B       — 6GB RAM",   context: 8192 },
      { id: "mbart",     label: "mBART (paper)   — CPU slow",  context: 1020 },
      { id: "stub",      label: "Stub (No LLM)   — chunks only",context: 4096 },
    ],
  });
  const [embeddingModel, setEmbeddingModel] = useState("sentence-transformers");
  const [generatorModel, setGeneratorModel] = useState("phi3:mini");
  const [showReport, setShowReport] = useState(false);
  const [genStage, setGenStage] = useState(-1);
  const [showCorpus, setShowCorpus] = useState(false);
  const [corpusSize, setCorpusSize] = useState(8);
  const [darkTheme, setDarkTheme]   = useState(true);
  const animRef = useRef(null);

  // Health check on mount
  useEffect(() => {
    api.health()
      .then(d => { setServerOk(true); setCorpusSize(d.corpus_size || 8); })
      .catch(() => setServerOk(false));
    api.getModelConfig()
      .then(d => {
        // Merge server data safely — never allow null/undefined arrays
        if (d && typeof d === 'object') {
          setModelOptions(prev => ({
            embedding_models: Array.isArray(d.embedding_models) && d.embedding_models.length > 0
              ? d.embedding_models : prev.embedding_models,
            generator_models: Array.isArray(d.generator_models) && d.generator_models.length > 0
              ? d.generator_models : prev.generator_models,
          }));
        }
      })
      .catch(() => {}); // keep defaults on failure
  }, []);

  const updateConfig = useCallback((key, value) => {
    setConfig(c => ({ ...c, [key]: value }));
  }, []);

  const resetConfig = useCallback(() => {
    setConfig(DEFAULT_CONFIG);
  }, []);

  // ── Run pipeline ─────────────────────────────────────────────────────────
  async function runPipeline() {
    if (!query.trim() || running) return;
    setRunning(true);
    setError(null);
    setResult(null);
    setGenProgress({ stage: "init", pct: 0, message: "Starting pipeline…" });
    setActiveStage(-1);
    setDoneStages(new Set());

    const STAGE_MAP = {
      embed: 1, retrieve: 2, graph: 3, gat: 4, prune: 5, generate: 5, done: 6
    };

    const payload = {
      query, lang,
      config: {
        ...config,
        embedding_model: config.embedding_model || "sentence-transformers",
        generator_model: config.generator_model || "phi3:mini",
      },
      relevant_ids: [0,1,3,4,6],
    };

    const cancel = runStream(
      payload,
      // onStage
      (msg) => {
        setGenProgress(msg);
        const stageIdx = STAGE_MAP[msg.stage] ?? 0;
        setActiveStage(stageIdx);
      },
      // onDone
      (data) => {
        setResult(data);
        setGenProgress({ stage: "done", pct: 100, message: "Complete" });
        const done = new Set([0,1,2,3,4,5,6]);
        setDoneStages(done);
        setActiveStage(-1);
        setRunning(false);
      },
      // onError
      (err) => {
        setError(err.message);
        setGenProgress(null);
        setRunning(false);
      }
    );
    return cancel;
  }

  // ── Derived graph data for canvas ─────────────────────────────────────────
  // allCandidates: all nodes (selected + pruned). Backend returns selected only,
  // so we reconstruct the full set from result.all_candidates when server sends it,
  // otherwise fall back to pruned_evidence so the graph still shows edges.
  const allCandidates = result?.all_candidates ?? result?.pruned_evidence ?? [];
  const prunedEvidence = result?.pruned_evidence ?? [];
  // Use real server edges when available; GraphCanvas will compute them client-side as fallback
  const serverEdges = result?.graph_edges ?? [];


  async function generateReport() {
    if (!result) return;
    setReportLoading(true);
    try {
      const payload = {
        query,
        result,
        metrics: result.metrics || {},
        config,
      };
      const data = await api.generateReport(payload);
      // Download as .md file
      const blob = new Blob([data.markdown], { type: "text/markdown" });
      const url  = URL.createObjectURL(blob);
      const a    = document.createElement("a");
      a.href     = url;
      a.download = `gcp_graphrag_report_${Date.now()}.md`;
      a.click();
      URL.revokeObjectURL(url);
    } catch (e) {
      setError(`Report failed: ${e.message}`);
    } finally {
      setReportLoading(false);
    }
  }

  // ── Render ────────────────────────────────────────────────────────────────
  return (
    <div className={`shell ${darkTheme ? "theme-dark" : "theme-light"}`}>
      {/* ── Header ── */}
      <header className="header">
        <div className="logo">
          <svg width="32" height="32" viewBox="0 0 32 32" fill="none">
            <circle cx="16" cy="16" r="6" stroke="#00e5ff" strokeWidth="1.5" fill="none"/>
            <circle cx="6" cy="8" r="3" stroke="#7c3aed" strokeWidth="1.5" fill="none"/>
            <circle cx="26" cy="8" r="3" stroke="#7c3aed" strokeWidth="1.5" fill="none"/>
            <circle cx="6" cy="24" r="3" stroke="#10b981" strokeWidth="1.5" fill="none"/>
            <circle cx="26" cy="24" r="3" stroke="#10b981" strokeWidth="1.5" fill="none"/>
            <line x1="10" y1="16" x2="7" y2="10" stroke="#00e5ff" strokeWidth="0.8" strokeOpacity="0.6"/>
            <line x1="10" y1="16" x2="7" y2="22" stroke="#00e5ff" strokeWidth="0.8" strokeOpacity="0.6"/>
            <line x1="22" y1="16" x2="25" y2="10" stroke="#00e5ff" strokeWidth="0.8" strokeOpacity="0.6"/>
            <line x1="22" y1="16" x2="25" y2="22" stroke="#00e5ff" strokeWidth="0.8" strokeOpacity="0.6"/>
          </svg>
          <div>
            <div className="logo-text">GCP-GraphRAG</div>
            <div className="logo-sub">Graph Context Pruning</div>
          </div>
        </div>

        <div className="header-right">
          <IngestMonitor />
          <button
            className="theme-toggle-btn"
            onClick={() => setDarkTheme(d => !d)}
            title={darkTheme ? "Switch to light theme" : "Switch to dark theme"}
          >
            {darkTheme ? "☀" : "🌙"}
          </button>
          <button className="corpus-btn" onClick={() => setShowCorpus(true)}>
            ⊞ Corpus <span className="corpus-count">{corpusSize}</span>
          </button>
          <div className={`server-dot ${serverOk === true ? "ok" : serverOk === false ? "err" : "checking"}`}
               title={serverOk === true ? "API connected" : serverOk === false ? "API unreachable – start server.py" : "Checking…"} />
          <span className="header-chip chip-cyan">v1.0</span>
          <span className="header-chip chip-violet">Research</span>
        </div>
      </header>

      {/* ── Sidebar ── */}
      <aside className="sidebar">
        <SidebarSection label="GCP Weights">
          <RangeRow label="Relevance" id="relevance_weight" value={config.relevance_weight}
            onChange={v => updateConfig("relevance_weight", v)} min={0} max={1} step={0.05} />
          <RangeRow label="Coherence" id="coherence_weight" value={config.coherence_weight}
            onChange={v => updateConfig("coherence_weight", v)} min={0} max={1} step={0.05} />
          <RangeRow label="Provenance" id="provenance_weight" value={config.provenance_weight}
            onChange={v => updateConfig("provenance_weight", v)} min={0} max={1} step={0.05} />
        </SidebarSection>

        <hr className="divider"/>

        <SidebarSection label="Retrieval">
          <RangeRow label="Top-K" id="top_k" value={config.top_k}
            onChange={v => updateConfig("top_k", v)} min={2} max={20} step={1} integer />
          <SelectRow label="FAISS Index" value={config.faiss_index_type}
            onChange={v => updateConfig("faiss_index_type", v)}
            options={[["flat","Flat (Exact)"],["ivf","IVF (Approx)"],["hnsw","HNSW (Fast)"]]} />
        </SidebarSection>

        <hr className="divider"/>

        <SidebarSection label="Graph · GAT">
          <RangeRow label="Edge θ" id="edge_threshold" value={config.edge_threshold}
            onChange={v => updateConfig("edge_threshold", v)} min={0.1} max={0.9} step={0.05} />
          <RangeRow label="GAT Layers" id="num_gat_layers" value={config.num_gat_layers}
            onChange={v => updateConfig("num_gat_layers", v)} min={1} max={6} step={1} integer />
          <RangeRow label="Attn Heads" id="num_heads" value={config.num_heads}
            onChange={v => updateConfig("num_heads", v)} min={1} max={8} step={1} integer />
        </SidebarSection>

        <hr className="divider"/>

        <SidebarSection label="Pruning">
          <SelectRow label="Solver" value={config.solver}
            onChange={v => updateConfig("solver", v)}
            options={[["greedy","Greedy O(n log n)"],["ilp","ILP (Exact)"]]} />
          <SelectRow label="Ctx Window" value={config.context_length}
            onChange={v => updateConfig("context_length", parseInt(v))}
            options={[[2048,"2048 tokens"],[4096,"4096 tokens"],[8192,"8192 tokens"],[16384,"16384 tokens"]]} />
        </SidebarSection>

        <hr className="divider"/>

        <ModelSelector config={config} onConfigChange={updateConfig} />

        <hr className="divider"/>

        <SidebarSection label="Device">
          <SelectRow label="Device" value={config.device}
            onChange={v => updateConfig("device", v)}
            options={[["cpu","CPU"],["cuda","CUDA (GPU)"],["mps","MPS (Apple)"]]} />
        </SidebarSection>

        <hr className="divider"/>

        <SidebarSection label="Embedding Model">
          <div className="select-row">
            <select className="select-input" value={embeddingModel} onChange={e => setEmbeddingModel(e.target.value)}>
              {(modelOptions.embedding_models ?? []).map(m => (
                <option key={m.id} value={m.id}>{m.label}</option>
              ))}
              {(modelOptions.embedding_models?.length ?? 0) === 0 && (
                <option value="muril">MuRIL (17 Indian languages)</option>
              )}
            </select>
          </div>
        </SidebarSection>

        <hr className="divider"/>

        <SidebarSection label="Generator Model">
          <div className="select-row">
            <select className="select-input" value={generatorModel} onChange={e => {
              setGeneratorModel(e.target.value);
              const m = modelOptions.generator_models.find(x => x.id === e.target.value);
              if (m) updateConfig("context_length", m.context);
            }}>
              {(modelOptions.generator_models ?? []).map(m => (
                <option key={m.id} value={m.id}>{m.label}</option>
              ))}
              {(modelOptions.generator_models?.length ?? 0) === 0 && (
                <option value="tinyllama">TinyLlama 1.1B (637MB)</option>
              )}
            </select>
          </div>
          <div className="select-row" style={{marginTop:"6px"}}>
            <label className="select-label">Ctx window</label>
            <select className="select-input" value={config.context_length}
              onChange={e => updateConfig("context_length", parseInt(e.target.value))}>
              <option value={1020}>1020 tokens (tinyllama)</option>
              <option value={2048}>2048 tokens</option>
              <option value={3072}>3072 tokens (phi3:mini)</option>
              <option value={4096}>4096 tokens</option>
              <option value={7168}>7168 tokens (llama3/mistral)</option>
            </select>
          </div>
        </SidebarSection>

        <hr className="divider"/>

        <div className="sidebar-section">
          <button className="reset-config-btn" onClick={resetConfig} title="Restore all settings to defaults">
            ↺ Reset All Settings
          </button>
        </div>

        {result && (
          <>
            <hr className="divider"/>
            <SidebarSection label="Last Run">
              <div className="latency-display">
                {result.latency_ms?.toFixed(1)}<span className="latency-unit"> ms</span>
              </div>
              <div className="sidebar-stat">
                <span>Tokens used</span>
                <span className="stat-accent">{result.token_stats?.tokens_used} / {result.token_stats?.token_budget}</span>
              </div>
              <div className="sidebar-stat">
                <span>Reduction</span>
                <span className="stat-accent">{fmt(result.token_stats?.reduction_ratio)}</span>
              </div>
              <div className="sidebar-stat">
                <span>Selected</span>
                <span className="stat-accent">{result.token_stats?.num_selected} / {result.token_stats?.num_candidates}</span>
              </div>
            </SidebarSection>
          </>
        )}
      </aside>

      {/* ── Main ── */}
      <main className="main">

        {/* Query bar */}
        <div className="query-area">
          <div className="query-row">
            <textarea
              className="query-input"
              value={query}
              onChange={e => {
                const val = e.target.value;
                setQuery(val);
                // Auto-detect Indian scripts and switch language
                const indicRe = /[ऀ-ॿঀ-৿஀-௿ఀ-೿ഀ-ൿ]/;
                const bengaliRe = /[ঀ-৿]/;
                const tamilRe = /[஀-௿]/;
                const teluguRe = /[ఀ-౿]/;
                if (indicRe.test(val)) {
                  if (bengaliRe.test(val)) setLang("bn");
                  else if (tamilRe.test(val)) setLang("ta");
                  else if (teluguRe.test(val)) setLang("te");
                  else setLang("hi");
                }
              }}
              placeholder="Enter your query…"
              rows={2}
              onKeyDown={e => { if (e.key === "Enter" && e.ctrlKey) runPipeline(); }}
            />
            <select className="lang-select" value={lang} onChange={e => setLang(e.target.value)}>
              {["en","hi","ta","fr","de","es","zh","ar"].map(l => <option key={l}>{l}</option>)}
            </select>
            {result && (
              <button
                className="report-btn"
                onClick={generateReport}
                disabled={reportLoading}
                title="Download research report as Markdown"
              >
                {reportLoading ? <span className="btn-spinner"/> : "⬇ Report"}
              </button>
            )}
            <button className="run-btn" onClick={runPipeline} disabled={running}>
              {running ? <span className="btn-spinner"/> : "▶"} Run
            </button>
          </div>

          {error && <div className="error-bar">⚠ {error}</div>}

          {/* Generation progress bar */}
          {genProgress && running && (
            <div className="gen-progress-wrap">
              <div className="gen-progress-labels">
                <span className="gen-progress-stage">{genProgress.message || genProgress.stage}</span>
                <span className="gen-progress-pct">{genProgress.pct}%</span>
              </div>
              <div className="gen-progress-track">
                <div className="gen-progress-fill" style={{width: `${genProgress.pct}%`}}/>
              </div>
            </div>
          )}

          {/* Pipeline stage bar */}
          <div className="stage-bar">
            {STAGES.map((s, i) => (
              <div key={s.key} className="stage-item">
                <span className={`stage-dot
                  ${doneStages.has(i) ? "done" : ""}
                  ${activeStage === i ? "active" : ""}`} />
                <span className={`stage-label
                  ${doneStages.has(i) ? "done" : ""}
                  ${activeStage === i ? "active" : ""}`}>
                  {s.label}
                </span>
                {i < STAGES.length - 1 && <span className="stage-arrow">→</span>}
              </div>
            ))}
          </div>
        </div>

        {/* 2×2 output grid */}
        <div className="output-grid">

          {/* Evidence panel */}
          <Panel
            icon="📄"
            title="Pruned Evidence"
            badge={result ? `${result.pruned_evidence?.length ?? 0} units` : null}
            badgeClass="badge-cyan"
          >
            {running && !result ? (
              <Empty spinner text="Retrieving & pruning evidence…" />
            ) : result?.pruned_evidence?.length > 0 ? (
              result.pruned_evidence.map((ev, i) => (
                <EvidenceCard key={i} ev={ev} delay={i * 50} />
              ))
            ) : (
              <Empty text="Run the pipeline to see pruned evidence" />
            )}
          </Panel>

          {/* Graph panel */}
          <Panel
            icon="⬡"
            title="Evidence Graph"
            badge={result ? `${result.graph_stats?.num_nodes ?? 0}N · ${result.graph_stats?.num_edges ?? 0}E` : null}
            badgeClass="badge-violet"
          >
            {running && !result ? (
              <Empty spinner text="Constructing evidence graph…" />
            ) : result ? (
              <div style={{ width: "100%", height: "100%", display: "flex", flexDirection: "column" }}>
                <div style={{ flex: 1, minHeight: 0 }}>
                  <GraphCanvas
                    allNodes={allCandidates}
                    prunedEvidence={prunedEvidence}
                    edgeThreshold={config.edge_threshold}
                    serverEdges={serverEdges}
                  />
                </div>
                <div className="graph-stats-row">
                  <StatPill label="Nodes" val={result.graph_stats?.num_nodes} color="cyan" />
                  <StatPill label="Edges" val={result.graph_stats?.num_edges} color="violet" />
                  <StatPill label="Density" val={result.graph_stats?.density?.toFixed(3)} color="green" />
                </div>
              </div>
            ) : (
              <Empty text="Graph renders after pipeline run" />
            )}
          </Panel>

          {/* Metrics panel */}
          <Panel
            icon="▦"
            title="Evaluation Metrics"
            badge={result?.metrics ? `F1 ${(result.metrics.f1 * 100).toFixed(1)}%` : null}
            badgeClass="badge-green"
          >
            {running && !result ? (
              <Empty spinner text="Evaluating metrics…" />
            ) : result?.metrics ? (
              <MetricsGrid metrics={result.metrics} />
            ) : (
              <Empty text="Metrics shown after evaluation" />
            )}
          </Panel>

          {/* Response panel */}
          <Panel
            icon="✦"
            title="Generated Response"
            badge={result?.token_stats ? `${result.token_stats.tokens_used} tokens` : null}
            badgeClass="badge-amber"
            extra={result ? (
              <button className="report-btn" onClick={() => setShowReport(true)}
                title="Generate research report">
                📄 Report
              </button>
            ) : null}
          >
            {running ? (
              <div style={{padding:"12px"}}>
                <GenerationProgress running={running} stageIndex={genStage} />
              </div>
            ) : result ? (
              <ResponsePanel result={result} />
            ) : (
              <Empty text="Response will appear here" />
            )}
          </Panel>

        </div>
      </main>

      {showCorpus && (
        <CorpusManager
          onClose={() => setShowCorpus(false)}
          onCorpusChange={() => api.health().then(d => setCorpusSize(d.corpus_size || 0))}
        />
      )}

      {showReport && result && (
        <ReportModal
          result={result}
          query={query}
          config={{ ...config, embedding_model: embeddingModel, generator_model: generatorModel }}
          onClose={() => setShowReport(false)}
        />
      )}
    </div>
  );
}

// ── Sub-components ────────────────────────────────────────────────────────────

function SidebarSection({ label, children }) {
  return (
    <div className="sidebar-section">
      <div className="sidebar-label">{label}</div>
      {children}
    </div>
  );
}

function RangeRow({ label, id, value, onChange, min, max, step, integer }) {
  const display = integer ? String(Math.round(value)) : Number(value).toFixed(2);
  return (
    <div className="range-row">
      <label className="range-label">{label}</label>
      <input
        type="range" id={id} min={min} max={max} step={step} value={value}
        onChange={e => onChange(integer ? parseInt(e.target.value) : parseFloat(e.target.value))}
        className="range-input"
      />
      <span className="range-val">{display}</span>
    </div>
  );
}

function SelectRow({ label, value, onChange, options }) {
  return (
    <div className="select-row">
      <label className="select-label">{label}</label>
      <select className="select-input" value={value} onChange={e => onChange(e.target.value)}>
        {options.map(([v, l]) => <option key={v} value={v}>{l}</option>)}
      </select>
    </div>
  );
}

function Panel({ icon, title, badge, badgeClass, extra, children }) {
  return (
    <div className="panel">
      <div className="panel-header">
        <span className="panel-icon">{icon}</span>
        <span className="panel-title">{title}</span>
        {badge && <span className={`panel-badge ${badgeClass}`}>{badge}</span>}
        {extra && <span style={{marginLeft:"auto"}}>{extra}</span>}
      </div>
      <div className="panel-body">{children}</div>
    </div>
  );
}

function Empty({ text, spinner }) {
  return (
    <div className="empty-state">
      {spinner && <span className="spinner" />}
      <span>{text}</span>
    </div>
  );
}

const BASE_URL = process.env.REACT_APP_API_URL || "http://localhost:8000";

function EvidenceCard({ ev, delay }) {
  const imp = Math.min(ev.importance_score || 0, 1);
  const tagClass = ev.modality === "image" ? "tag-image" : ev.modality === "audio" ? "tag-audio" : "tag-text";

  // Detect if content is a filepath (new format) vs raw bytes/text
  const content = String(ev.content || "");
  const isFilepath = (
    ev.modality !== "text" &&
    (content.match(/^[A-Za-z]:\\/) ||   // Windows path
     content.startsWith("/") ||            // Unix path
     content.startsWith("."))              // relative path
  );
  const isBinaryHex = content.startsWith("b'\\x") || content.startsWith("b\"\\x");

  // For images: show a thumbnail if we have the file path
  function renderContent() {
    if (ev.modality === "image") {
      if (isFilepath) {
        // Serve image via backend proxy endpoint
        const fname = content.split(/[/\\]/).pop();
        return (
          <div className="ev-image-wrap">
            <img
              src={`${BASE_URL}/corpus/image?path=${encodeURIComponent(content)}`}
              alt={fname}
              className="ev-image-thumb"
              onError={e => { e.target.style.display="none"; e.target.nextSibling.style.display="block"; }}
            />
            <div className="ev-image-fallback" style={{display:"none"}}>
              🖼️ {fname}
            </div>
          </div>
        );
      }
      if (isBinaryHex) {
        return <div className="ev-binary-note">🖼️ Image (binary data — re-ingest to enable preview)</div>;
      }
      return <div className="ev-binary-note">🖼️ {content.slice(0, 80)}</div>;
    }

    if (ev.modality === "audio") {
      if (isFilepath) {
        const fname = content.split(/[/\\]/).pop();
        return (
          <div className="ev-audio-wrap">
            🎵 <span className="ev-filepath">{fname}</span>
          </div>
        );
      }
      return <div className="ev-binary-note">🎵 Audio file</div>;
    }

    // Text — show content, clean up [IMAGE: ...] prefix
    return <div className="ev-content">{content.slice(0, 300)}{content.length > 300 ? "…" : ""}</div>;
  }

  return (
    <div className="ev-card" style={{ animationDelay: `${delay}ms` }}>
      <div className="ev-header">
        <span className={`ev-tag ${tagClass}`}>{ev.modality}</span>
        <span className="ev-tag tag-lang">{ev.lang?.toUpperCase()}</span>
        <span className="ev-source">{ev.source}</span>
      </div>
      {renderContent()}
      <div className="ev-scores">
        <ScorePill label="imp" value={imp} />
        <ScorePill label="prov" value={ev.provenance_score} noBar />
        <ScorePill label="ret" value={ev.retrieval_score} noBar />
      </div>
    </div>
  );
}

function ScorePill({ label, value, noBar }) {
  const v = typeof value === "number" ? value : 0;
  return (
    <span className="score-pill">
      {label} <span className="score-val">{v.toFixed(3)}</span>
      {!noBar && <span className="score-bar"><span className="score-fill" style={{ width: `${v * 100}%` }} /></span>}
    </span>
  );
}

const METRIC_DEFS = [
  { key: "f1", label: "F1 Score", color: "#10b981" },
  { key: "rouge_l", label: "ROUGE-L", color: "#00e5ff" },
  { key: "recall@5", label: "Recall@5", color: "#7c3aed" },
  { key: "evidence_density", label: "Evid. Density", color: "#f59e0b" },
  { key: "budget_adherence", label: "Budget Adherence", color: "#00e5ff" },
  { key: "context_reduction_ratio", label: "Context Reduction", color: "#7c3aed" },
  { key: "redundancy_ratio", label: "Redundancy Ratio", color: "#f59e0b" },
  { key: "relevance_per_token", label: "Rel / Token", color: "#10b981", raw: true },
];

function MetricsGrid({ metrics }) {
  return (
    <div className="metrics-grid">
      {METRIC_DEFS.map((m, i) => {
        const raw = metrics[m.key] ?? 0;
        const pct = m.raw ? Math.min(raw * 100, 100) : raw * 100;
        return (
          <div className="metric-card" key={m.key} style={{ animationDelay: `${i * 60}ms` }}>
            <div className="metric-name">{m.label}</div>
            <div className="metric-value" style={{ color: m.color }}>
              {m.raw ? raw.toFixed(4) : pct.toFixed(1)}<span className="metric-unit">{m.raw ? "" : "%"}</span>
            </div>
            <div className="metric-bar">
              <div className="metric-fill" style={{ width: `${Math.min(pct, 100)}%`, background: m.color }} />
            </div>
          </div>
        );
      })}
    </div>
  );
}

function ResponsePanel({ result }) {
  const ts = result.token_stats || {};
  return (
    <>
      <div className="response-box">
        {result.response?.split("\n").map((line, i) => (
          <span key={i}>{line}<br/></span>
        ))}
      </div>
      <div className="token-stat-row">
        {[
          ["budget", ts.token_budget],
          ["used", ts.tokens_used],
          ["reduction", `${((ts.reduction_ratio || 0) * 100).toFixed(1)}%`],
          ["selected", `${ts.num_selected}/${ts.num_candidates}`],
        ].map(([k, v]) => (
          <div className="stat-chip" key={k}>
            <span className="chip-key">{k}</span>
            <span className="chip-val">{v}</span>
          </div>
        ))}
      </div>
    </>
  );
}

function StatPill({ label, val, color }) {
  const cls = `stat-pill-${color}`;
  return (
    <div className={`stat-pill ${cls}`}>
      <span className="sp-label">{label}</span>
      <span className="sp-val">{val}</span>
    </div>
  );
}
