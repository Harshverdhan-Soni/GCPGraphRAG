// src/components/CorpusManager.jsx
import { useState, useEffect, useRef, useCallback } from "react";
import { api } from "../api/client";

const LANG_OPTIONS = ["en","hi","ta","fr","de","es","zh","ar"];
const MOD_ICON     = { text:"📄", image:"🖼️", audio:"🎵" };
const ACCEPT_FILES = ".txt,.md,.pdf,.docx,.html,.csv,.json,.jpg,.jpeg,.png,.webp,.gif,.bmp,.mp3,.wav,.m4a,.ogg,.flac,.aac";

// Delete has 3 backend stages: remove chunks → rebuild embeddings → rebuild FAISS
const DELETE_STAGES = ["Removing chunks from DB…", "Slicing embedding matrix…", "Rebuilding FAISS index…", "Done"];
// Simulated timing weights (ms) for each stage — real op is single API call
const DELETE_STAGE_DURATIONS = [300, 400, 500];

function isBinaryContent(doc) {
  return doc.modality === "image" || doc.modality === "audio";
}

function previewText(doc) {
  if (isBinaryContent(doc)) return null;
  const c = String(doc.content ?? "");
  if (c.startsWith("b'\\x") || c.startsWith("b\"\\x")) return null;
  return c.length > 160 ? c.slice(0, 160) + "…" : c;
}

function groupBySource(docs) {
  const map = new Map();
  docs.forEach(doc => {
    const key = String(doc.source || "unknown");
    if (!map.has(key)) map.set(key, {
      source: key, modality: doc.modality || "text",
      lang: doc.lang || "en", provenance_score: doc.provenance_score ?? 0.85,
      filepath: doc.filepath || "", chunks: [],
    });
    map.get(key).chunks.push(doc);
  });
  return [...map.values()];
}

// Animated progress bar that fills smoothly over `duration` ms
function ProgressBar({ pct, label, sublabel, color = "var(--accent)" }) {
  return (
    <div className="cm-progress-wrap">
      <div className="cm-progress-labels">
        <span className="cm-progress-label">{label}</span>
        <span className="cm-progress-pct">{Math.round(pct)}%</span>
      </div>
      {sublabel && <div className="cm-progress-sublabel">{sublabel}</div>}
      <div className="cm-progress-track">
        <div
          className="cm-progress-fill"
          style={{ width: `${pct}%`, background: color,
                   transition: "width 0.35s cubic-bezier(0.4,0,0.2,1)" }}
        />
      </div>
    </div>
  );
}

// Hook that animates a value from 0 → target over duration ms
function useAnimatedProgress(active, target, duration) {
  const [val, setVal] = useState(0);
  const rafRef = useRef();
  const startRef = useRef();

  useEffect(() => {
    if (!active) { setVal(0); return; }
    startRef.current = performance.now();
    function step(now) {
      const elapsed = now - startRef.current;
      const progress = Math.min(elapsed / duration, 1);
      // ease-out
      setVal(target * (1 - Math.pow(1 - progress, 3)));
      if (progress < 1) rafRef.current = requestAnimationFrame(step);
    }
    rafRef.current = requestAnimationFrame(step);
    return () => cancelAnimationFrame(rafRef.current);
  }, [active, target, duration]);

  return val;
}

export default function CorpusManager({ onClose, onCorpusChange }) {
  const [groups,    setGroups]    = useState([]);
  const [totalDocs, setTotalDocs] = useState(0);
  const [loading,   setLoading]   = useState(true);
  const [error,     setError]     = useState(null);
  const [success,   setSuccess]   = useState(null);
  const [tab,       setTab]       = useState("browse");
  const [rebuilding,setRebuilding]= useState(false);
  const [rebuildPct,setRebuildPct]= useState(0);

  // Per-source delete progress: { [source]: { stage, pct, done, error } }
  const [deleteProgress, setDeleteProgress] = useState({});

  // Text-add form
  const [textForm, setTextForm] = useState({ content:"", source:"", lang:"en", provenance_score:0.85 });
  const [adding,   setAdding]   = useState(false);
  const [addPct,   setAddPct]   = useState(0);

  // Upload queue
  const [queue,     setQueue]    = useState([]);
  const [uploading, setUploading]= useState(false);
  const [uploadStats, setUploadStats] = useState({ done:0, total:0 });
  const fileRef = useRef();

  // ── Helpers ────────────────────────────────────────────────────────
  const flash = useCallback((msg, isErr=false) => {
    isErr ? setError(msg) : setSuccess(msg);
    setTimeout(() => isErr ? setError(null) : setSuccess(null), 5000);
  }, []);

  const fetchCorpus = useCallback(async () => {
    setLoading(true);
    try {
      const data = await api.getCorpus();
      const docs = data.documents || [];
      setTotalDocs(docs.length);
      setGroups(groupBySource(docs));
    } catch (e) { flash(e.message, true); }
    finally { setLoading(false); }
  }, [flash]);

  useEffect(() => { fetchCorpus(); }, [fetchCorpus]);

  function modTagCls(m) {
    if (m === "image") return "ev-tag tag-image";
    if (m === "audio") return "ev-tag tag-audio";
    return "ev-tag tag-text";
  }

  const byMod = groups.reduce((acc, g) => {
    acc[g.modality] = (acc[g.modality]||0) + g.chunks.length; return acc;
  }, {});

  // ── Delete with animated stage progress ───────────────────────────
  async function handleDeleteSource(source) {
    // Initialise progress for this source
    const setDP = (patch) => setDeleteProgress(prev => ({
      ...prev,
      [source]: { ...(prev[source] || {}), ...patch }
    }));

    setDP({ stage: 0, pct: 0, done: false, error: null });

    // Animate stages 0 and 1 while the real API call runs in parallel
    const totalAnimTime = DELETE_STAGE_DURATIONS.reduce((a,b)=>a+b,0);
    let elapsed = 0;
    const startTime = performance.now();

    // Animate progress smoothly while API runs
    const animInterval = setInterval(() => {
      const now = performance.now() - startTime;
      // which stage are we in?
      let stageStart = 0;
      for (let s = 0; s < DELETE_STAGE_DURATIONS.length; s++) {
        const stageEnd = stageStart + DELETE_STAGE_DURATIONS[s];
        if (now < stageEnd) {
          const stagePct = ((now - stageStart) / DELETE_STAGE_DURATIONS[s]) * 100;
          // Overall pct = stages done + current stage fraction
          const overallPct = ((s + stagePct / 100) / DELETE_STAGES.length) * 95; // cap at 95 until API returns
          setDP({ stage: s, pct: overallPct });
          break;
        }
        stageStart = stageEnd;
      }
    }, 60);

    try {
      const res = await api.removeBySource(source);
      clearInterval(animInterval);
      // Jump to 100%
      setDP({ stage: DELETE_STAGES.length - 1, pct: 100, done: true });
      flash(`Deleted "${source}" — ${res.deleted_chunks} chunk(s) removed, index rebuilt.`);
      await fetchCorpus();
      onCorpusChange();
    } catch (e) {
      clearInterval(animInterval);
      setDP({ pct: 0, done: false, error: e.message });
      flash(`Delete failed: ${e.message}`, true);
    } finally {
      // Clear after 2 s so user can read "Done"
      setTimeout(() => setDeleteProgress(prev => {
        const next = { ...prev };
        delete next[source];
        return next;
      }), 2000);
    }
  }

  // ── Rebuild index with progress ────────────────────────────────────
  async function handleRebuild() {
    setRebuilding(true);
    setRebuildPct(0);
    // Animate up to 90% while waiting
    const interval = setInterval(() => {
      setRebuildPct(prev => prev < 88 ? prev + 4 : prev);
    }, 200);
    try {
      const r = await api.rebuildIndex();
      clearInterval(interval);
      setRebuildPct(100);
      setTimeout(() => { setRebuilding(false); setRebuildPct(0); }, 1200);
      flash(`Index rebuilt — ${r.vectors} vectors.`);
    } catch (e) {
      clearInterval(interval);
      setRebuilding(false); setRebuildPct(0);
      flash(e.message, true);
    }
  }

  async function handleReset() {
    try {
      await api.resetCorpus();
      await fetchCorpus();
      onCorpusChange();
      flash("Demo corpus loaded and indexed.");
    } catch (e) { flash(e.message, true); }
  }

  // ── Add text with progress ─────────────────────────────────────────
  async function handleAddText(e) {
    e.preventDefault();
    if (!textForm.content.trim()) return;
    setAdding(true); setAddPct(0);
    const interval = setInterval(() => {
      setAddPct(prev => prev < 85 ? prev + 7 : prev);
    }, 180);
    try {
      await api.addDocument({ ...textForm, modality: "text" });
      clearInterval(interval); setAddPct(100);
      await fetchCorpus(); onCorpusChange();
      setTextForm({ content:"", source:"", lang:"en", provenance_score:0.85 });
      flash("Document added and indexed.");
      setTimeout(() => { setAdding(false); setAddPct(0); setTab("browse"); }, 800);
    } catch (e) {
      clearInterval(interval); setAdding(false); setAddPct(0);
      flash(e.message, true);
    }
  }

  // ── File upload with per-file + overall progress ───────────────────
  function pickFiles(files) {
    setQueue(prev => [...prev, ...[...files].map(f => ({
      file: f,
      source: f.name.replace(/\.[^/.]+$/, ""),
      lang: "en", prov: 0.85,
      status: "pending", error: "", chunks: null,
      pct: 0,
    }))]);
  }

  async function uploadAll() {
    if (uploading) return;
    const pending = queue.filter(u => u.status === "pending");
    if (pending.length === 0) return;
    setUploading(true);
    setUploadStats({ done: 0, total: pending.length });
    let doneCount = 0;
    let anyDone = false;

    for (let i = 0; i < queue.length; i++) {
      if (queue[i].status !== "pending") continue;

      // Mark as uploading with animated progress
      setQueue(prev => prev.map((u,j) => j===i ? {...u, status:"uploading", pct:0} : u));

      // Animate individual file progress bar
      const fileInterval = setInterval(() => {
        setQueue(prev => prev.map((u,j) =>
          j===i && u.status==="uploading"
            ? { ...u, pct: Math.min((u.pct||0) + 3, 82) }
            : u));
      }, 200);

      try {
        const fd = new FormData();
        fd.append("file",             queue[i].file);
        fd.append("source",           queue[i].source);
        fd.append("lang",             queue[i].lang);
        fd.append("provenance_score", String(queue[i].prov));
        const res = await api.uploadFile(fd);
        clearInterval(fileInterval);
        setQueue(prev => prev.map((u,j) =>
          j===i ? { ...u, status:"done", pct:100, chunks: res.chunks_added } : u));
        doneCount++;
        setUploadStats({ done: doneCount, total: pending.length });
        anyDone = true;
      } catch (err) {
        clearInterval(fileInterval);
        // Show exact error message so user knows what went wrong
        const msg = err.message || "Upload failed";
        setQueue(prev => prev.map((u,j) =>
          j===i ? { ...u, status:"error", pct:0,
                    error: msg.length > 80 ? msg.slice(0,77)+"…" : msg } : u));
        flash(`Failed to upload "${queue[i].file.name}": ${msg}`, true);
      }
    }

    setUploading(false);
    if (anyDone) {
      await fetchCorpus(); onCorpusChange();
      flash(`Upload complete — ${doneCount} file(s) embedded and indexed.`);
    }
  }

  // Overall upload pct
  const overallUploadPct = uploading && uploadStats.total > 0
    ? Math.round(
        (queue.reduce((sum, u) =>
          sum + (u.status === "done" ? 100 : u.pct || 0), 0)
        ) / queue.length
      )
    : 0;

  // ── Render ─────────────────────────────────────────────────────────
  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal cm-modal" onClick={e => e.stopPropagation()}>

        {/* Header */}
        <div className="modal-header">
          <span className="modal-title">Corpus Manager</span>
          <span className="modal-badge">{groups.length} sources · {totalDocs} chunks</span>
          <div className="cm-stat-pills">
            {Object.entries(byMod).map(([m,n]) => (
              <span key={m} className={`cm-stat-pill ${modTagCls(m)}`}>
                {MOD_ICON[m]} {n}
              </span>
            ))}
          </div>
          <button className="modal-close" onClick={onClose}>✕</button>
        </div>

        {/* Flash bars */}
        {error   && <div className="modal-error">⚠ {error}</div>}
        {success && <div className="modal-success">✓ {success}</div>}

        {/* Rebuild progress (shown across all tabs) */}
        {rebuilding && (
          <div className="cm-global-progress">
            <ProgressBar
              pct={rebuildPct}
              label="Rebuilding FAISS index…"
              sublabel="Re-embedding all corpus documents"
              color="var(--accent)"
            />
          </div>
        )}

        {/* Tabs */}
        <div className="cm-tabs">
          {[["browse","Browse Corpus"],["add-text","Add Text"],["upload","Upload Files"]].map(
            ([key, label]) => (
              <button key={key}
                className={`cm-tab${tab===key?" cm-tab-active":""}`}
                onClick={() => setTab(key)}>
                {label}
              </button>
            )
          )}
        </div>

        {/* ═══════════════ TAB: BROWSE ═══════════════ */}
        {tab === "browse" && (
          <div className="cm-body">
            {loading ? (
              <div className="modal-empty"><span className="spinner"/> Loading corpus…</div>
            ) : groups.length === 0 ? (
              <div className="modal-empty">
                No documents yet.<br/>
                <span style={{color:"var(--text-muted)"}}>
                  Use "Upload Files" or "Add Text" to get started.
                </span>
              </div>
            ) : (
              groups.map(g => {
                const dp = deleteProgress[g.source];
                const isDeleting = !!dp && !dp.done && !dp.error;
                const preview = g.chunks[0] ? previewText(g.chunks[0]) : null;

                return (
                  <div className="cm-group" key={g.source}>
                    <div className="cm-group-header">
                      <span className={modTagCls(g.modality)}>
                        {MOD_ICON[g.modality]||"📄"} {g.modality}
                      </span>
                      <span className="ev-tag tag-lang">{g.lang}</span>
                      <span className="cm-source-name" title={g.source}>{g.source}</span>
                      <span className="cm-chunk-count">
                        {g.chunks.length} chunk{g.chunks.length!==1?"s":""}
                      </span>
                      {g.filepath && (
                        <span className="cm-filepath" title={g.filepath}>
                          {g.filepath.split(/[/\\]/).pop()}
                        </span>
                      )}
                      <span className="cm-prov">prov {g.provenance_score?.toFixed(2)}</span>
                      <button
                        className="cm-delete-btn"
                        onClick={() => !isDeleting && handleDeleteSource(g.source)}
                        disabled={isDeleting}
                        title={`Delete all ${g.chunks.length} chunk(s) from "${g.source}"`}>
                        {isDeleting
                          ? <><span className="spinner-sm"/> Deleting…</>
                          : "🗑 Delete"}
                      </button>
                    </div>

                    {/* Delete progress bar */}
                    {dp && !dp.done && !dp.error && (
                      <div className="cm-delete-progress">
                        <ProgressBar
                          pct={dp.pct || 0}
                          label={DELETE_STAGES[dp.stage || 0]}
                          sublabel={`Stage ${(dp.stage||0)+1} of ${DELETE_STAGES.length - 1}`}
                          color="#ef4444"
                        />
                      </div>
                    )}
                    {dp && dp.done && (
                      <div className="cm-delete-progress cm-delete-done">
                        ✓ Deleted and index rebuilt
                      </div>
                    )}

                    {/* Content preview */}
                    {!dp && preview && (
                      <div className="cm-preview">
                        {preview}
                        {g.chunks.length > 1 && (
                          <span className="cm-more"> +{g.chunks.length-1} more chunks</span>
                        )}
                      </div>
                    )}
                    {!dp && isBinaryContent(g.chunks[0]) && (
                      <div className="cm-preview cm-preview-binary">
                        {MOD_ICON[g.modality]} Binary file
                        {g.filepath ? ` — ${g.filepath.split(/[/\\]/).pop()}` : ""}
                      </div>
                    )}
                  </div>
                );
              })
            )}
          </div>
        )}

        {/* ═══════════════ TAB: ADD TEXT ═══════════════ */}
        {tab === "add-text" && (
          <div className="cm-body">
            <form className="cm-form" onSubmit={handleAddText}>
              <label className="cm-field-label">Content</label>
              <textarea className="form-textarea cm-textarea" rows={7}
                placeholder="Paste or type document text here…"
                value={textForm.content}
                onChange={e => setTextForm(f=>({...f, content:e.target.value}))}
              />
              <div className="cm-field-row">
                <div className="cm-field">
                  <label className="cm-field-label">Source name</label>
                  <input className="form-input" placeholder="e.g. annual_report_2024"
                    value={textForm.source}
                    onChange={e => setTextForm(f=>({...f, source:e.target.value}))} />
                </div>
                <div className="cm-field cm-field-sm">
                  <label className="cm-field-label">Language</label>
                  <select className="form-select" value={textForm.lang}
                    onChange={e => setTextForm(f=>({...f, lang:e.target.value}))}>
                    {LANG_OPTIONS.map(l=><option key={l}>{l}</option>)}
                  </select>
                </div>
                <div className="cm-field cm-field-sm">
                  <label className="cm-field-label">Provenance</label>
                  <input className="form-input" type="number" step="0.05" min="0" max="1"
                    value={textForm.provenance_score}
                    onChange={e => setTextForm(f=>({...f, provenance_score:parseFloat(e.target.value)}))} />
                </div>
              </div>

              {/* Add progress */}
              {adding && (
                <ProgressBar
                  pct={addPct}
                  label={addPct < 100 ? "Embedding document…" : "Done — switching to browse"}
                  sublabel="Running mDeBERTa encoder"
                  color="var(--accent2)"
                />
              )}

              <button className="cm-submit-btn" type="submit"
                disabled={adding || !textForm.content.trim()}>
                {adding
                  ? <><span className="spinner-sm"/> Embedding…</>
                  : "+ Add & Index"}
              </button>
            </form>
          </div>
        )}

        {/* ═══════════════ TAB: UPLOAD FILES ═══════════════ */}
        {tab === "upload" && (
          <div className="cm-body">
            {/* Drop zone */}
            <div className="cm-dropzone"
              onClick={() => fileRef.current?.click()}
              onDragOver={e=>{e.preventDefault();e.currentTarget.classList.add("cm-dz-over");}}
              onDragLeave={e=>e.currentTarget.classList.remove("cm-dz-over")}
              onDrop={e=>{
                e.preventDefault();
                e.currentTarget.classList.remove("cm-dz-over");
                pickFiles(e.dataTransfer.files);
              }}>
              <input ref={fileRef} type="file" multiple accept={ACCEPT_FILES}
                style={{display:"none"}} onChange={e=>pickFiles(e.target.files)}/>
              <div className="cm-dz-icon">⬆</div>
              <div className="cm-dz-text">Click or drag &amp; drop files here</div>
              <div className="cm-dz-hint">
                📄 PDF, DOCX, TXT, MD, HTML, CSV, JSON &nbsp;|&nbsp;
                🖼️ JPG, PNG, WEBP, GIF &nbsp;|&nbsp;
                🎵 MP3, WAV, M4A, OGG, FLAC
              </div>
            </div>

            {/* Overall upload progress */}
            {uploading && (
              <div className="cm-overall-progress">
                <ProgressBar
                  pct={overallUploadPct}
                  label={`Uploading & embedding files — ${uploadStats.done} / ${uploadStats.total} done`}
                  sublabel="Each file is chunked, embedded, and added to the FAISS index"
                  color="var(--accent3)"
                />
              </div>
            )}

            {/* Queue */}
            {queue.length > 0 && (
              <>
                <div className="cm-queue-hdr">
                  <span className="cm-field-label">{queue.length} file(s) queued</span>
                  <button className="cm-clear-btn" onClick={()=>setQueue([])} disabled={uploading}>
                    Clear all
                  </button>
                </div>

                <div className="cm-queue">
                  {queue.map((item, i) => (
                    <div key={i} className={`cm-qrow cm-qrow-${item.status}`}>
                      <div className="cm-qrow-top">
                        <span className="cm-qicon">
                          {item.status==="done"      ? "✓" :
                           item.status==="error"     ? "✕" :
                           item.status==="uploading" ? <span className="spinner-sm"/> : "○"}
                        </span>
                        <span className="cm-qname" title={item.file.name}>
                          {item.file.name.length>28 ? item.file.name.slice(0,25)+"…" : item.file.name}
                        </span>
                        <span className="cm-qsize">
                          {item.file.size < 1024*1024
                            ? `${(item.file.size/1024).toFixed(0)} KB`
                            : `${(item.file.size/1024/1024).toFixed(1)} MB`}
                        </span>
                        <input className="form-input cm-qsource" placeholder="source name"
                          value={item.source} disabled={item.status!=="pending"}
                          onChange={e=>setQueue(prev=>prev.map((u,j)=>j===i?{...u,source:e.target.value}:u))}/>
                        <select className="form-select cm-qlang"
                          value={item.lang} disabled={item.status!=="pending"}
                          onChange={e=>setQueue(prev=>prev.map((u,j)=>j===i?{...u,lang:e.target.value}:u))}>
                          {LANG_OPTIONS.map(l=><option key={l}>{l}</option>)}
                        </select>
                        {item.status==="done" && item.chunks!=null && (
                          <span className="cm-qdone">{item.chunks} chunks</span>
                        )}
                        {item.status==="error" && (
                          <span className="cm-qerr" title={item.error}>⚠ {item.error.slice(0,40)}</span>
                        )}
                        {item.status==="pending" && (
                          <button className="cm-qrm"
                            onClick={()=>setQueue(prev=>prev.filter((_,j)=>j!==i))}>✕</button>
                        )}
                      </div>

                      {/* Per-file progress bar */}
                      {item.status === "uploading" && (
                        <div className="cm-qprogress">
                          <div className="cm-qprogress-track">
                            <div className="cm-qprogress-fill"
                              style={{ width: `${item.pct||0}%` }}/>
                          </div>
                          <span className="cm-qprogress-pct">{Math.round(item.pct||0)}%</span>
                        </div>
                      )}
                      {item.status === "done" && (
                        <div className="cm-qprogress">
                          <div className="cm-qprogress-track">
                            <div className="cm-qprogress-fill cm-qprogress-done"
                              style={{ width:"100%" }}/>
                          </div>
                          <span className="cm-qprogress-pct" style={{color:"var(--accent3)"}}>100%</span>
                        </div>
                      )}
                    </div>
                  ))}
                </div>

                <button className="cm-submit-btn" onClick={uploadAll}
                  disabled={uploading || queue.every(u=>u.status==="done")}>
                  {uploading
                    ? <><span className="spinner-sm"/> Uploading {uploadStats.done}/{uploadStats.total}…</>
                    : `⬆ Upload ${queue.filter(u=>u.status==="pending").length} file(s) & Index`}
                </button>
              </>
            )}
          </div>
        )}

        {/* Footer */}
        <div className="modal-footer cm-footer">
          <button className="btn-reset" onClick={handleReset}>↺ Load Demo Corpus</button>
          <button className="btn-reset cm-rebuild-btn" onClick={handleRebuild} disabled={rebuilding}>
            {rebuilding
              ? <><span className="spinner-sm"/> Rebuilding…</>
              : "⟳ Rebuild Index"}
          </button>
        </div>
      </div>
    </div>
  );
}
