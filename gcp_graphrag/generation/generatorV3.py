"""
Grounded multilingual response generation.

DEFAULT : Stub mode — structured summary of top evidence. Fast, always works.
OLLAMA  : Set USE_REAL_GENERATOR=1. Start with: .\\start_server.ps1

Language handling:
  - Auto-detects Hindi/Indian scripts from query text
  - For Hindi queries: translates evidence summary to Hindi via the LLM
  - tinyllama is English-only; for Hindi use phi3:mini or mistral
"""
import os
import re
import logging
from config.settings import PipelineConfig

logger = logging.getLogger(__name__)

USE_REAL_GENERATOR = os.getenv("USE_REAL_GENERATOR", "0") == "1"
_OLLAMA_MODEL      = os.getenv("OLLAMA_MODEL", "tinyllama")
_OLLAMA_URL        = os.getenv("OLLAMA_URL", "http://localhost:11434")

# Devanagari and other Indian script Unicode ranges
_INDIAN_SCRIPT_RE = re.compile(
    r"[\u0900-\u097F\u0980-\u09FF\u0A00-\u0A7F\u0A80-\u0AFF"
    r"\u0B00-\u0B7F\u0B80-\u0BFF\u0C00-\u0C7F\u0C80-\u0CFF"
    r"\u0D00-\u0D7F\u0600-\u06FF]"
)

def _detect_lang(text: str, declared_lang: str) -> str:
    """Auto-detect Indian script languages from text."""
    if _INDIAN_SCRIPT_RE.search(text):
        # Count script characters to determine language
        devanagari = len(re.findall(r"[\u0900-\u097F]", text))
        bengali    = len(re.findall(r"[\u0980-\u09FF]", text))
        tamil      = len(re.findall(r"[\u0B80-\u0BFF]", text))
        telugu     = len(re.findall(r"[\u0C00-\u0C7F]", text))
        if devanagari >= max(bengali, tamil, telugu):
            return "hi"
        if bengali > 0:  return "bn"
        if tamil > 0:    return "ta"
        if telugu > 0:   return "te"
    return declared_lang


class MultilingualGenerator:
    def __init__(self, config: PipelineConfig):
        self.config        = config
        self._ollama_ready = False

    # ── Public API ────────────────────────────────────────────────────

    def generate(self, query: str, evidence: list[dict], lang: str = "en") -> str:
        # Auto-detect language from query script
        detected_lang = _detect_lang(query, lang)
        if detected_lang != lang:
            logger.info(f"Auto-detected language: {detected_lang} (declared: {lang})")
        lang = detected_lang

        if USE_REAL_GENERATOR:
            try:
                return self._generate_ollama(query, evidence, lang)
            except Exception as e:
                logger.error(f"Ollama generation failed: {e}")
                logger.info("Falling back to structured stub.")

        return self._structured_stub(query, evidence, lang)

    # ── Ollama generation ─────────────────────────────────────────────

    def _generate_ollama(self, query: str, evidence: list[dict], lang: str) -> str:
        import requests

        model = _OLLAMA_MODEL

        # Warm up on first call
        if not self._ollama_ready:
            logger.info(f"Warming up {model}…")
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
                logger.warning(f"Warmup failed ({e})")

        is_indian = lang in ("hi", "bn", "ta", "te", "mr", "gu", "pa", "ur")
        is_tinyllama = "tinyllama" in model.lower()
        is_phi3      = "phi3" in model.lower()

        # For Indian languages with tinyllama:
        # Build English prompt but ask it to respond in the detected language
        # tinyllama can produce SOME Hindi but quality is low
        # Better: extract key info in English, then state it was in Hindi source
        context = self._linearise(evidence, max_chars_per_chunk=400)

        if is_indian and is_tinyllama:
            # Strategy: ask tinyllama to summarise in English first,
            # then we wrap with Hindi context note
            prompt = self._build_prompt_hindi_tinyllama(query, context, lang)
        else:
            prompt = self._build_prompt(query, context, lang)

        # Build model-specific full prompt
        if is_phi3:
            full_prompt = (
                f"<|system|>\n"
                f"You are a precise multilingual research assistant. "
                f"Answer ONLY from the provided evidence. "
                f"Be concise and factual. "
                f"{'Respond in Hindi (Devanagari script).' if lang == 'hi' else f'Respond in language: {lang}.'}\n"
                f"<|end|>\n"
                f"<|user|>\n{prompt}<|end|>\n"
                f"<|assistant|>\n"
            )
        elif is_tinyllama:
            full_prompt = (
                f"<s>[INST] <<SYS>>\n"
                f"You are a helpful assistant. Answer using ONLY the evidence given. "
                f"Be factual. Do NOT add commentary, postscripts, or blog-style text. "
                f"Stop after answering the question.\n"
                f"<</SYS>>\n\n"
                f"{prompt} [/INST]"
            )
        else:
            # mistral, llama3, gemma2 etc.
            full_prompt = (
                f"[INST] You are a precise research assistant. "
                f"Answer using ONLY the provided evidence. "
                f"{'Respond in Hindi.' if lang == 'hi' else f'Respond in language: {lang}.'}\n\n"
                f"{prompt} [/INST]"
            )

        payload = {
            "model": model,
            "prompt": full_prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,
                "top_p":       0.9,
                "num_predict": 400,
                "stop": ["</s>", "[/INST]", "<|end|>", "<|user|>",
                         "<|system|>", "Postscript", "Note:", "I hope"],
            },
        }
        res = requests.post(f"{_OLLAMA_URL}/api/generate", json=payload, timeout=300)

        if not res.ok:
            err = res.json().get("error", res.text) if res.content else res.text
            raise RuntimeError(f"Ollama {res.status_code}: {err}")

        response_text = res.json().get("response", "").strip()

        # Clean up common tinyllama artifacts
        response_text = self._clean_response(response_text, lang)

        if not response_text:
            raise RuntimeError("Ollama returned empty response")

        # For Indian language queries: add a note if model responded in English
        if is_indian and not _INDIAN_SCRIPT_RE.search(response_text):
            lang_names = {
                "hi": "हिंदी", "bn": "বাংলা", "ta": "தமிழ்", "te": "తెలుగు"
            }
            lang_name = lang_names.get(lang, lang.upper())
            response_text = (
                f"[Note: Your query was in {lang_name}. "
                f"The model ({model}) responded in English as it has limited "
                f"multilingual generation capability. "
                f"Use phi3:mini or mistral for Hindi responses.]\n\n"
                + response_text
            )

        return response_text

    # ── Structured stub ───────────────────────────────────────────────

    def _structured_stub(self, query: str, evidence: list[dict], lang: str) -> str:
        top     = sorted(evidence, key=lambda e: e.get("importance_score", 0), reverse=True)
        sources = list(dict.fromkeys(e["source"] for e in evidence))

        is_indian = lang in ("hi", "bn", "ta", "te", "mr", "gu", "pa", "ur")
        header = "प्रश्न" if lang == "hi" else "Query"
        content_hdr = "प्रासंगिक सामग्री" if lang == "hi" else "Relevant content"

        lines = [f"{header}: {query}\n", "─" * 60]

        text_ev  = [e for e in top if e.get("modality") == "text"]
        image_ev = [e for e in top if e.get("modality") == "image"]
        audio_ev = [e for e in top if e.get("modality") == "audio"]

        if text_ev:
            lines.append(f"{content_hdr} from {len(text_ev)} chunk(s):\n")
            for i, ev in enumerate(text_ev[:5], 1):
                c = str(ev.get("content", "")).strip()
                if len(c) > 500: c = c[:500] + "…"
                lines.append(f"[{i}] {ev['source']} (relevance: {ev.get('importance_score',0):.3f})")
                lines.append(c)
                lines.append("")

        if image_ev:
            lines.append(f"🖼️ Images: {', '.join(e['source'] for e in image_ev)}")
        if audio_ev:
            lines.append(f"🎵 Audio: {', '.join(e['source'] for e in audio_ev)}")

        lines.append("─" * 60)
        lines.append(f"Sources: {', '.join(sources[:8])}")

        if is_indian:
            lines.append(
                f"\n[Hindi/Indian language queries: Use phi3:mini or mistral "
                f"for better multilingual responses. "
                f"Set OLLAMA_MODEL=phi3:mini in start_server.ps1]"
            )

        return "\n".join(lines)

    # ── Helpers ───────────────────────────────────────────────────────

    def _build_prompt(self, query: str, context: str, lang: str) -> str:
        return (
            f"Using ONLY the evidence below, answer the question accurately.\n"
            f"If the evidence does not contain the answer, say so clearly.\n"
            f"Do NOT add postscripts, blog comments, or unrelated text.\n"
            f"Response language: {lang}.\n\n"
            f"--- EVIDENCE ---\n{context}\n--- END EVIDENCE ---\n\n"
            f"Question: {query}\n\nAnswer:"
        )

    def _build_prompt_hindi_tinyllama(self, query: str, context: str, lang: str) -> str:
        """Simplified prompt for tinyllama handling Hindi queries."""
        return (
            f"The following evidence is in Hindi. "
            f"Summarise the key information from the evidence to answer the question.\n"
            f"Give a factual answer. Stop after the answer.\n\n"
            f"EVIDENCE:\n{context}\n\n"
            f"QUESTION: {query}\n\n"
            f"ANSWER (summarise the evidence):"
        )

    def _linearise(self, evidence: list[dict], max_chars_per_chunk: int = 400) -> str:
        sorted_ev = sorted(
            evidence, key=lambda e: e.get("importance_score", 0), reverse=True
        )
        parts = []
        for e in sorted_ev:
            c = str(e.get("content", "")).strip()
            if len(c) > max_chars_per_chunk:
                c = c[:max_chars_per_chunk] + "…"
            parts.append(f"[{e['source']} | {e['modality']} | {e['lang']}]\n{c}")
        return "\n\n".join(parts)

    def _clean_response(self, text: str, lang: str) -> str:
        """Remove common LLM artifacts from the response."""
        # Remove postscript / blog artifacts
        cutoffs = [
            "Postscript:", "P.S.", "I hope you found",
            "If you have any further", "Feel free to ask",
            "Note: This response", "Disclaimer:",
        ]
        for cutoff in cutoffs:
            idx = text.find(cutoff)
            if idx > 80:  # only cut if there's actual content before it
                text = text[:idx].strip()

        # Remove repeated content (tinyllama sometimes loops)
        lines  = text.splitlines()
        seen   = set()
        unique = []
        for line in lines:
            stripped = line.strip()
            if stripped and stripped not in seen:
                seen.add(stripped)
                unique.append(line)
            elif not stripped:
                unique.append(line)
        text = "\n".join(unique)

        return text.strip()
