// src/components/GenerationProgress.jsx
import { useEffect, useState } from "react";

const BASE = process.env.REACT_APP_API_URL || "http://localhost:8000";

const STAGES = [
  { key: "embed",    label: "Embedding query",       pct: 10 },
  { key: "retrieve", label: "Retrieving candidates", pct: 25 },
  { key: "graph",    label: "Building graph",        pct: 45 },
  { key: "gat",      label: "GAT propagation",       pct: 60 },
  { key: "prune",    label: "GCP pruning",           pct: 75 },
  { key: "generate", label: "Generating response",   pct: 90 },
  { key: "done",     label: "Complete",              pct: 100 },
];

export default function GenerationProgress({ running, stageIndex }) {
  const [animPct, setAnimPct] = useState(0);

  const targetPct = running
    ? STAGES[Math.min(stageIndex, STAGES.length - 1)]?.pct || 0
    : 0;

  useEffect(() => {
    if (!running) { setAnimPct(0); return; }
    const timer = setInterval(() => {
      setAnimPct(prev => {
        if (prev < targetPct) return Math.min(prev + 2, targetPct);
        return prev;
      });
    }, 40);
    return () => clearInterval(timer);
  }, [running, targetPct]);

  if (!running) return null;

  const stage = STAGES[Math.min(stageIndex, STAGES.length - 1)];

  return (
    <div className="gen-progress-wrap">
      <div className="gen-progress-header">
        <span className="gen-progress-label">
          <span className="gen-spinner" />
          {stage?.label || "Processing…"}
        </span>
        <span className="gen-progress-pct">{Math.round(animPct)}%</span>
      </div>
      <div className="gen-progress-track">
        <div
          className="gen-progress-fill"
          style={{ width: `${animPct}%` }}
        />
      </div>
      <div className="gen-stage-dots">
        {STAGES.map((s, i) => (
          <div
            key={s.key}
            className={`gen-stage-dot ${
              i < stageIndex ? "done" : i === stageIndex ? "active" : ""
            }`}
            title={s.label}
          />
        ))}
      </div>
    </div>
  );
}
