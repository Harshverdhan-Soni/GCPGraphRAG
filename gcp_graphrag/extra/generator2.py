"""
Grounded multilingual response generation.

Decoder: facebook/mbart-large-cc25 (max position = 1024 tokens).
The prompt is hard-truncated to MBART_MAX_TOKENS before passing to the model
to prevent IndexError in embed_positions.

To enable real generation:
  1. pip install transformers torch sentencepiece
  2. Set USE_REAL_GENERATOR=1 as an environment variable
  3. Restart the server — the model downloads on first /run call (~2.4 GB)
"""
import os
import logging
from config.settings import PipelineConfig

logger = logging.getLogger(__name__)

# mBART-large-cc25 hard position limit (2 reserved = 1022 usable)
MBART_MAX_TOKENS = 1020
USE_REAL_GENERATOR = os.getenv("USE_REAL_GENERATOR", "0") == "1"


class MultilingualGenerator:
    def __init__(self, config: PipelineConfig):
        self.config     = config
        self._model     = None
        self._tokenizer = None

    # ── Public API ────────────────────────────────────────────────────

    def generate(self, query: str, evidence: list[dict], lang: str = "en") -> str:
        context = self._linearise(evidence)
        prompt  = self._build_prompt(query, context, lang)
        logger.debug(f"Prompt length (chars): {len(prompt)}")

        if USE_REAL_GENERATOR:
            try:
                return self._generate_real(prompt, lang)
            except Exception as e:
                logger.error(f"Real generation failed ({e}) – falling back to stub.")

        return self._stub_response(query, evidence)

    # ── Real generation (mBART) ───────────────────────────────────────

    def _generate_real(self, prompt: str, lang: str) -> str:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        import torch

        if self._model is None:
            name = self.config.model.decoder
            logger.info(f"Loading generator: {name}")
            self._tokenizer = AutoTokenizer.from_pretrained(name)
            self._model = AutoModelForSeq2SeqLM.from_pretrained(
                name, torch_dtype=torch.float32
            )
            self._model.eval()
            logger.info("Generator loaded.")

        # ── CRITICAL: truncate to mBART's position limit ──────────────
        # mBART-large-cc25 has a position embedding table of size 1026
        # (indices 0-1025). Sequences longer than 1020 usable tokens
        # cause IndexError: index out of range in embed_positions.
        inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=MBART_MAX_TOKENS,   # hard cap — never exceed this
            padding=False,
        )

        # Extra safety: clamp position IDs in case tokenizer miscounts
        seq_len = inputs["input_ids"].shape[1]
        if seq_len > MBART_MAX_TOKENS:
            inputs["input_ids"]      = inputs["input_ids"][:, :MBART_MAX_TOKENS]
            inputs["attention_mask"] = inputs["attention_mask"][:, :MBART_MAX_TOKENS]

        logger.debug(f"Generator input length: {inputs['input_ids'].shape[1]} tokens")

        # Set forced BOS token for target language
        lang_code_map = {
            "en": "en_XX", "hi": "hi_IN", "ta": "ta_IN",
            "fr": "fr_XX", "de": "de_DE", "es": "es_XX",
            "zh": "zh_CN", "ar": "ar_AR",
        }
        lang_code = lang_code_map.get(lang, "en_XX")
        try:
            forced_bos = self._tokenizer.lang_code_to_id[lang_code]
        except (KeyError, AttributeError):
            forced_bos = self._tokenizer.bos_token_id

        import torch
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                forced_bos_token_id=forced_bos,
                max_new_tokens=min(self.config.model.max_new_tokens, 256),
                num_beams=4,
                early_stopping=True,
                # Avoid do_sample with mBART — it causes instability
            )

        return self._tokenizer.decode(outputs[0], skip_special_tokens=True)

    # ── Stub response ─────────────────────────────────────────────────

    def _stub_response(self, query: str, evidence: list[dict]) -> str:
        sources = list({e["source"] for e in evidence})
        chunks  = len(evidence)
        # Build a readable summary from the top evidence chunks
        top = sorted(evidence, key=lambda e: e.get("importance_score", 0), reverse=True)[:3]
        excerpts = []
        for ev in top:
            if ev.get("modality") == "text":
                text = str(ev.get("content", ""))[:200]
                excerpts.append(f'• [{ev["source"]}] {text}{"…" if len(text)==200 else ""}')
        excerpt_block = "\n".join(excerpts) if excerpts else "(no text evidence)"

        return (
            f"Query: {query}\n\n"
            f"Top evidence ({chunks} chunks from {len(sources)} source(s)):\n"
            f"{excerpt_block}\n\n"
            f"Sources: {', '.join(sources[:8])}\n\n"
            f"[To enable real answer generation, set USE_REAL_GENERATOR=1 "
            f"and restart the server. Requires ~2.4 GB download of mBART-large-cc25.]"
        )

    # ── Helpers ───────────────────────────────────────────────────────

    def _linearise(self, evidence: list[dict]) -> str:
        sorted_ev = sorted(
            evidence, key=lambda e: e.get("importance_score", 0), reverse=True
        )
        parts = []
        for e in sorted_ev:
            header = (f"[SOURCE: {e['source']} | "
                      f"MODALITY: {e['modality']} | LANG: {e['lang']}]")
            content = str(e.get("content", ""))
            # Truncate individual evidence chunks to avoid prompt explosion
            if len(content) > 800:
                content = content[:800] + "…"
            parts.append(f"{header}\n{content}")
        return "\n\n".join(parts)

    def _build_prompt(self, query: str, context: str, lang: str) -> str:
        return (
            f"You are a helpful multilingual assistant. "
            f"Answer the question using ONLY the evidence below. "
            f"Language: {lang}.\n\n"
            f"EVIDENCE:\n{context}\n\n"
            f"QUESTION: {query}\n\nANSWER:"
        )
