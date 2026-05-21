// src/components/GraphCanvas.jsx
import { useEffect, useRef } from "react";

// Modality colours + icons
const MOD_COLOR = { text: "#00e5ff", image: "#7c3aed", audio: "#f59e0b", default: "#10b981" };
const MOD_ICON  = { text: "T", image: "I", audio: "A" };

function seededRand(seed) {
  const x = Math.sin(Math.abs(seed) + 1) * 10000;
  return x - Math.floor(x);
}
function strHash(s) {
  let h = 0;
  for (let i = 0; i < s.length; i++) h = (Math.imul(31, h) + s.charCodeAt(i)) | 0;
  return Math.abs(h);
}
function computeEdges(allNodes, edgeThreshold) {
  const edges = [];
  for (let i = 0; i < allNodes.length; i++) {
    for (let j = i + 1; j < allNodes.length; j++) {
      const ni = allNodes[i], nj = allNodes[j];
      let weight;
      if (ni.source === nj.source) {
        weight = Math.max(0.5, seededRand(strHash(ni.source) % 9999) * 0.5 + 0.5);
      } else {
        const seed = strHash((ni.content || "") + (nj.content || "")) % 77777;
        const sim  = seededRand(seed);
        const thr  = ni.modality !== nj.modality ? edgeThreshold * 0.9 : edgeThreshold;
        weight = sim > thr ? sim : 0;
      }
      if (weight > 0) edges.push({ src: i, dst: j, weight });
    }
  }
  return edges;
}

// Detect light theme from DOM
function isLightTheme() {
  return document.querySelector(".theme-light") !== null;
}

export default function GraphCanvas({
  allNodes = [],
  prunedEvidence = [],
  edgeThreshold = 0.4,
  serverEdges = [],
}) {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    const nodes  = allNodes.length > 0 ? allNodes : prunedEvidence;
    if (!canvas || nodes.length === 0) return;

    const parent = canvas.parentElement;
    const W = parent.offsetWidth  || 400;
    const H = parent.offsetHeight || 280;
    canvas.width  = W;
    canvas.height = H;
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, W, H);

    const light = isLightTheme();

    // ── Colours ──────────────────────────────────────────────────────
    const C = {
      nodePruned:   light ? "#e5f0ff" : "#1a2535",
      nodePrunedBd: light ? "#a0b8d8" : "#2e3f52",
      labelPruned:  light ? "#6b7280" : "#3a4a5c",
      labelSelected:light ? "#111827" : "#c8daea",
      edgeFaint:    light ? "rgba(150,170,200,0.35)" : "rgba(30,42,56,0.7)",
      legendTxt:    light ? "#374151" : "#6b7c93",
      edgeWeightTxt:light ? "#374151" : "#9ca3af",
      impArc:       "rgba(16,185,129,0.75)",
    };

    const n   = nodes.length;
    const cx  = W / 2;
    const cy  = (H - 34) / 2 + 6;
    const r   = Math.min(W * 0.40, (H - 64) * 0.40);

    const positions = nodes.map((_, i) => ({
      x: cx + r * Math.cos((2 * Math.PI * i / n) - Math.PI / 2),
      y: cy + r * Math.sin((2 * Math.PI * i / n) - Math.PI / 2),
    }));

    const selectedIds = new Set(prunedEvidence.map(e => e.id));
    const edges = serverEdges.length > 0 ? serverEdges : computeEdges(nodes, edgeThreshold);

    // ── Draw edges ────────────────────────────────────────────────────
    for (const edge of edges) {
      const s = positions[edge.src];
      const d = positions[edge.dst];
      if (!s || !d) continue;

      const ni = nodes[edge.src];
      const nj = nodes[edge.dst];
      const srcSel = selectedIds.has(ni?.id);
      const dstSel = selectedIds.has(nj?.id);
      const bothSel   = srcSel && dstSel;
      const eitherSel = srcSel || dstSel;

      // Cross-modality edge = dashed
      const crossModal = ni?.modality !== nj?.modality;
      if (crossModal) {
        ctx.setLineDash([4, 3]);
      } else {
        ctx.setLineDash([]);
      }

      ctx.beginPath();
      ctx.moveTo(s.x, s.y);
      ctx.lineTo(d.x, d.y);

      if (bothSel) {
        // Colour based on average modality
        const baseColor = ni?.modality === nj?.modality
          ? (MOD_COLOR[ni?.modality] || MOD_COLOR.default)
          : "#a78bfa";
        const alpha = 0.15 + edge.weight * 0.5;
        ctx.strokeStyle = hexToRgba(baseColor, alpha);
        ctx.lineWidth   = 1 + edge.weight * 2;
      } else if (eitherSel) {
        ctx.strokeStyle = light ? "rgba(0,119,204,0.12)" : "rgba(0,229,255,0.08)";
        ctx.lineWidth   = 0.8;
      } else {
        ctx.strokeStyle = C.edgeFaint;
        ctx.lineWidth   = 0.5;
      }
      ctx.stroke();
      ctx.setLineDash([]);

      // ── Edge weight label (only on selected-selected edges) ───────
      if (bothSel && edge.weight > 0.3) {
        const mx = (s.x + d.x) / 2;
        const my = (s.y + d.y) / 2;
        const wLabel = edge.weight.toFixed(2);
        ctx.font      = '7px "Space Mono", monospace';
        ctx.textAlign = "center";
        ctx.fillStyle = C.edgeWeightTxt;
        // Small background pill
        const tw = ctx.measureText(wLabel).width + 4;
        ctx.fillStyle = light ? "rgba(255,255,255,0.75)" : "rgba(6,8,16,0.65)";
        ctx.fillRect(mx - tw/2, my - 5, tw, 10);
        ctx.fillStyle = C.edgeWeightTxt;
        ctx.fillText(wLabel, mx, my + 3);
      }
    }

    // ── Draw nodes ────────────────────────────────────────────────────
    nodes.forEach((doc, i) => {
      const pos      = positions[i];
      const selected = selectedIds.has(doc.id);
      const mod      = doc.modality || "text";
      const modColor = MOD_COLOR[mod] || MOD_COLOR.default;
      const nodeR    = selected ? 9 : 5;

      // Glow halo
      if (selected) {
        const grd = ctx.createRadialGradient(pos.x, pos.y, 0, pos.x, pos.y, 22);
        grd.addColorStop(0, hexToRgba(modColor, 0.25));
        grd.addColorStop(1, hexToRgba(modColor, 0));
        ctx.fillStyle = grd;
        ctx.beginPath();
        ctx.arc(pos.x, pos.y, 22, 0, Math.PI * 2);
        ctx.fill();
      }

      // Node fill
      ctx.beginPath();
      ctx.arc(pos.x, pos.y, nodeR, 0, Math.PI * 2);
      ctx.fillStyle   = selected ? modColor : C.nodePruned;
      ctx.strokeStyle = selected ? modColor : C.nodePrunedBd;
      ctx.lineWidth   = 1.5;
      ctx.fill();
      ctx.stroke();

      // Modality icon inside node
      ctx.font      = `bold ${selected ? 7 : 5}px "Space Mono", monospace`;
      ctx.textAlign = "center";
      ctx.fillStyle = selected
        ? (light ? "#fff" : "#0d1117")
        : C.labelPruned;
      ctx.fillText(MOD_ICON[mod] || "?", pos.x, pos.y + (selected ? 2.5 : 1.8));

      // Importance arc ring
      if (selected) {
        const imp = Math.min(doc.importance_score ?? 0.5, 1);
        ctx.beginPath();
        ctx.arc(pos.x, pos.y, nodeR + 4,
          -Math.PI / 2,
          -Math.PI / 2 + imp * Math.PI * 2);
        ctx.strokeStyle = C.impArc;
        ctx.lineWidth   = 2;
        ctx.stroke();

        // Importance score label
        ctx.font      = '7px "Space Mono", monospace';
        ctx.fillStyle = C.labelSelected;
        ctx.textAlign = "center";
        const impLabel = imp.toFixed(2);
        ctx.fillText(impLabel, pos.x, pos.y + nodeR + 14);
      }

      // Source label above node
      ctx.font      = `bold 7px "Space Mono", monospace`;
      ctx.fillStyle = selected ? C.labelSelected : C.labelPruned;
      ctx.textAlign = "center";
      const label = (doc.source || `node_${i}`).split("_")[0].toUpperCase().slice(0, 5);
      ctx.fillText(label, pos.x, pos.y - (nodeR + 7));
    });

    // ── Legend ────────────────────────────────────────────────────────
    const ly = H - 18;
    ctx.font      = '8px "Space Mono", monospace';
    ctx.textAlign = "left";

    // Modality legend
    let lx = 10;
    for (const [mod, color] of Object.entries(MOD_COLOR)) {
      if (mod === "default") continue;
      ctx.fillStyle = color;
      ctx.beginPath(); ctx.arc(lx + 4, ly, 4, 0, Math.PI * 2); ctx.fill();
      ctx.fillStyle = C.legendTxt;
      ctx.fillText(mod, lx + 11, ly + 3);
      lx += 45;
    }

    // Selected indicator
    ctx.fillStyle = light ? "#059669" : "rgba(16,185,129,0.8)";
    ctx.beginPath(); ctx.arc(lx + 4, ly, 4, 0, Math.PI * 2); ctx.fill();
    ctx.beginPath();
    ctx.arc(lx + 4, ly, 7, -Math.PI/2, Math.PI/2);
    ctx.strokeStyle = C.impArc; ctx.lineWidth = 1.5; ctx.stroke();
    ctx.fillStyle = C.legendTxt;
    ctx.fillText("imp arc", lx + 11, ly + 3);
    lx += 58;

    // Cross-modal edge
    ctx.setLineDash([3, 2]);
    ctx.beginPath(); ctx.moveTo(lx, ly); ctx.lineTo(lx + 20, ly);
    ctx.strokeStyle = C.legendTxt; ctx.lineWidth = 1; ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = C.legendTxt;
    ctx.fillText("cross-modal", lx + 24, ly + 3);

  }, [allNodes, prunedEvidence, edgeThreshold, serverEdges]);

  return <canvas ref={canvasRef} style={{ width: "100%", height: "100%", display: "block" }} />;
}

function hexToRgba(hex, alpha) {
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  return `rgba(${r},${g},${b},${alpha})`;
}
