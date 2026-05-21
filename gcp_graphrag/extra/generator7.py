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

        if is_phi3:
            payload = {
                "model": model,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are a precise research assistant. "
                            "Answer questions using ONLY the provided evidence. "
                            "Be concise and factual. "
                            f"Respond in language: {lang}."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "top_p": 0.9,
                    "num_predict": 512,
                },
            }
            res = requests.post(
                f"{_OLLAMA_URL}/api/chat",
                json=payload,
                timeout=300,
            )
            res.raise_for_status()
            return res.json()["message"]["content"].strip()

        else:
            # llama3, mistral, gemma2, etc.
            payload = {
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.2, "num_predict": 512},
            }
            res = requests.post(
                f"{_OLLAMA_URL}/api/generate",
                json=payload,
                timeout=300,
            )
            res.raise_for_status()
            return res.json().get("response", "").strip()

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
