"""
Grounded multilingual response generation.

DEFAULT : Stub mode — structured summary of top evidence. Fast, always works.

OLLAMA  : Set USE_REAL_GENERATOR=1 and ensure Ollama is running with phi3:mini.
          Start with: .\start_server.ps1  (sets all env vars automatically)
"""
import os
import logging
from config.settings import PipelineConfig

logger = logging.getLogger(__name__)

USE_REAL_GENERATOR = os.getenv("USE_REAL_GENERATOR", "0") == "1"
_OLLAMA_MODEL      = os.getenv("OLLAMA_MODEL", "phi3:mini")
_OLLAMA_URL        = os.getenv("OLLAMA_URL", "http://localhost:11434")


class MultilingualGenerator:
    def __init__(self, config: PipelineConfig):
        self.config         = config
        self._ollama_ready  = False  # True after first warmup call

    # ── Public API ────────────────────────────────────────────────────

    def generate(self, query: str, evidence: list[dict], lang: str = "en") -> str:
        if USE_REAL_GENERATOR:
            try:
                return self._generate_ollama(query, evidence, lang)
            except Exception as e:
                logger.error(f"Ollama generation failed: {e}")
                logger.info("Falling back to structured stub response.")

        return self._structured_stub(query, evidence, lang)

    # ── Ollama (phi3:mini) ────────────────────────────────────────────

    def _generate_ollama(self, query: str, evidence: list[dict], lang: str) -> str:
        import requests

        model   = _OLLAMA_MODEL
        context = self._linearise(evidence, max_chars_per_chunk=500)
        prompt  = self._build_prompt(query, context, lang)

        # Warm up on first call so model weights are in RAM
        if not self._ollama_ready:
            logger.info(f"Warming up {model} — first call may take 30–60 s…")
            try:
                requests.post(
                    f"{_OLLAMA_URL}/api/generate",
                    json={"model": model, "prompt": " ", "stream": False,
                          "options": {"num_predict": 1}},
                    timeout=120,
                )
                self._ollama_ready = True
                logger.info(f"{model} warmed up ✓")
            except Exception as e:
                logger.warning(f"Warmup failed ({e}) — continuing anyway.")

        # phi3 uses /api/chat; other models use /api/generate
        is_phi3 = "phi3" in model.lower()

        # Build a full prompt with system instruction for /api/generate
        # Works for ALL models including phi3:mini — more reliable than /api/chat
        full_prompt = (
            f"<|system|>\n"
            f"You are a precise research assistant. "
            f"Answer questions using ONLY the provided evidence. "
            f"Be concise, factual, and write a complete paragraph answer. "
            f"Do NOT repeat the evidence verbatim. Summarise and synthesise. "
            f"Respond in language: {lang}.\n"
            f"<|end|>\n"
            f"<|user|>\n{prompt}<|end|>\n"
            f"<|assistant|>\n"
        ) if is_phi3 else (
            f"[INST] You are a precise research assistant. "
            f"Answer using ONLY the provided evidence. "
            f"Respond in language: {lang}.\n\n{prompt} [/INST]"
        )

        payload = {
            "model": model,
            "prompt": full_prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,
                "top_p": 0.9,
                "num_predict": 512,
                "stop": ["<|end|>", "<|user|>", "</s>"],
            },
        }
        res = requests.post(
            f"{_OLLAMA_URL}/api/generate",
            json=payload,
            timeout=300,
        )

        if not res.ok:
            # Log the actual Ollama error for debugging
            try:
                err = res.json().get("error", res.text)
            except Exception:
                err = res.text
            raise RuntimeError(f"Ollama {res.status_code}: {err}")

        response_text = res.json().get("response", "").strip()
        if not response_text:
            raise RuntimeError("Ollama returned empty response")
        return response_text

    # ── Structured stub (default) ─────────────────────────────────────

    def _structured_stub(self, query: str, evidence: list[dict], lang: str) -> str:
        top     = sorted(evidence, key=lambda e: e.get("importance_score", 0), reverse=True)
        sources = list(dict.fromkeys(e["source"] for e in evidence))

        lines = [f"Query: {query}\n", "─" * 60]

        text_ev  = [e for e in top if e.get("modality") == "text"]
        image_ev = [e for e in top if e.get("modality") == "image"]
        audio_ev = [e for e in top if e.get("modality") == "audio"]

        if text_ev:
            lines.append(f"Relevant content from {len(text_ev)} text chunk(s):\n")
            for i, ev in enumerate(text_ev[:5], 1):
                content = str(ev.get("content", "")).strip()
                if len(content) > 500:
                    content = content[:500] + "…"
                lines.append(f"[{i}] {ev['source']}  (relevance: {ev.get('importance_score', 0):.3f})")
                lines.append(content)
                lines.append("")

        if image_ev:
            lines.append(f"Images retrieved: {', '.join(e['source'] for e in image_ev)}")
        if audio_ev:
            lines.append(f"Audio retrieved:  {', '.join(e['source'] for e in audio_ev)}")

        lines.append("─" * 60)
        lines.append(f"Sources ({len(sources)}): {', '.join(sources[:10])}")
        lines.append("")
        lines.append(
            "Tip: start server with .\\start_server.ps1 to enable phi3:mini responses."
        )
        return "\n".join(lines)

    # ── Helpers ───────────────────────────────────────────────────────

    def _linearise(self, evidence: list[dict], max_chars_per_chunk: int = 500) -> str:
        sorted_ev = sorted(evidence, key=lambda e: e.get("importance_score", 0), reverse=True)
        parts = []
        for e in sorted_ev:
            content = str(e.get("content", "")).strip()
            if len(content) > max_chars_per_chunk:
                content = content[:max_chars_per_chunk] + "…"
            parts.append(f"[{e['source']} | {e['modality']} | {e['lang']}]\n{content}")
        return "\n\n".join(parts)

    def _build_prompt(self, query: str, context: str, lang: str) -> str:
        return (
            f"Using ONLY the evidence below, answer the question accurately.\n"
            f"If the evidence does not contain the answer, say so clearly.\n"
            f"Response language: {lang}.\n\n"
            f"--- EVIDENCE ---\n{context}\n--- END EVIDENCE ---\n\n"
            f"Question: {query}\n\nAnswer:"
        )
