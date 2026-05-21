// src/components/IngestMonitor.jsx
import { useEffect, useState } from "react";

const BASE = process.env.REACT_APP_API_URL || "http://localhost:8000";

export default function IngestMonitor() {
  const [prog, setProg] = useState(null);
  const [open, setOpen] = useState(false);

  useEffect(() => {
    const es = new EventSource(`${BASE}/ingest/stream`);
    es.onmessage = e => setProg(JSON.parse(e.data));
    es.onerror = () => {};
    return () => es.close();
  }, []);

  if (!prog) return null;

  const total = prog.done + prog.failed + (prog.queued || 0);
  const pct = total > 0 ? Math.round((prog.done / total) * 100) : 0;
  const busy = !!prog.current || (prog.queued > 0);

  return (
    <div className="ingest-monitor">
      <span className="ingest-label">Ingest</span>
      <div className="ingest-bar" title={`${pct}% complete`}>
        <div className="ingest-fill" style={{ width: `${pct}%` }} />
      </div>
      <span className="ingest-counts">
        {prog.done}<span className="ingest-sep">/</span>{total}
        {prog.failed > 0 && <span className="ingest-failed"> · {prog.failed} err</span>}
      </span>
      {busy && <span className="ingest-spinner" />}
      {prog.current && (
        <span className="ingest-current" title={prog.current}>
          {prog.current.length > 18 ? "…" + prog.current.slice(-15) : prog.current}
        </span>
      )}
      {prog.errors?.length > 0 && (
        <button className="ingest-err-btn" onClick={() => setOpen(o => !o)}>
          {open ? "hide errors" : "show errors"}
        </button>
      )}
      {open && prog.errors?.length > 0 && (
        <div className="ingest-err-list">
          {prog.errors.map((e, i) => (
            <div key={i} className="ingest-err-row">
              <span className="ingest-err-file">{e.file}</span>
              <span className="ingest-err-msg">{e.error}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
