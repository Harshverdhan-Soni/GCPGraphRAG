// src/api/client.js
const BASE = process.env.REACT_APP_API_URL || "http://localhost:8000";

async function request(method, path, body) {
  const res = await fetch(`${BASE}${path}`, {
    method,
    headers: { "Content-Type": "application/json" },
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || "Request failed");
  }
  return res.json();
}

async function upload(path, formData) {
  const res = await fetch(`${BASE}${path}`, { method: "POST", body: formData });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || "Upload failed");
  }
  return res.json();
}

export const api = {
  // Health
  health:          ()         => request("GET",    "/health"),
  // Corpus
  getCorpus:       ()         => request("GET",    "/corpus"),
  resetCorpus:     ()         => request("POST",   "/corpus/reset"),
  addDocument:     (doc)      => request("POST",   "/corpus/add", doc),
  removeDocument:  (id)       => request("DELETE", `/corpus/${id}`),
  removeBySource:  (source)   => request("DELETE", `/corpus/by-source/${encodeURIComponent(source)}`),
  rebuildIndex:    ()         => request("POST",   "/corpus/rebuild-index"),
  corpusStats:     ()         => request("GET",    "/corpus/stats"),
  uploadFile:      (fd)       => upload("/corpus/upload", fd),
  // Pipeline
  run:             (payload)  => request("POST",   "/run", payload),
  // Models
  getModelConfig:  ()         => request("GET",    "/models/config"),
  // Report
  generateReport:  (payload)  => request("POST",   "/report/generate", payload),
  // Ingest
  ingestFile:      (payload)  => request("POST",   "/ingest/file", payload),
  ingestFolder:    (payload)  => request("POST",   "/ingest/folder", payload),
  ingestTrigger:   (payload)  => request("POST",   "/ingest/trigger", payload),
  ingestProgress:  ()         => request("GET",    "/ingest/progress"),
  // SSE streams (use EventSource directly)
  // new EventSource(`${BASE}/ingest/stream`)
  // new EventSource(`${BASE}/run/stream`) with POST body — use fetchEventSource instead
};

// Streaming pipeline run using fetch + SSE
export function runStream(payload, onStage, onDone, onError) {
  const ctrl = new AbortController();

  // Notify UI immediately so it shows "Starting…" right away
  onStage({ stage: "init", pct: 5, message: "Starting pipeline…" });

  fetch(`${BASE}/run/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
    signal: ctrl.signal,
  }).then(async res => {
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      onError(new Error(err.detail || `HTTP ${res.status}`));
      return;
    }
    const reader  = res.body.getReader();
    const decoder = new TextDecoder();
    let buf = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      const parts = buf.split("\n\n");
      buf = parts.pop();
      for (const part of parts) {
        const line = part.replace(/^data:\s*/, "").trim();
        if (!line) continue;
        try {
          const msg = JSON.parse(line);
          if (msg.stage === "done") {
            try {
              const result = typeof msg.message === "string"
                ? JSON.parse(msg.message)
                : msg.message;
              onDone(result);
            } catch (parseErr) {
              onError(new Error("Failed to parse result: " + parseErr.message));
            }
          } else if (msg.stage === "error") {
            onError(new Error(msg.message));
          } else {
            onStage(msg);
          }
        } catch (_) {}
      }
    }
  }).catch(e => {
    if (e.name !== "AbortError") onError(e);
  });

  return () => ctrl.abort();
}
