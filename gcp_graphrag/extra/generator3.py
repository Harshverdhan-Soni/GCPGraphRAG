"""
Grounded multilingual response generation.

DEFAULT: Stub mode — returns a structured summary of top evidence.
         Fast, reliable, and shows retrieved content clearly.

TO ENABLE real LLM generation, set USE_REAL_GENERATOR=1 and choose
a backend (see options below). mBART is NOT recommended on CPU —
use Ollama (local) or OpenAI API instead.
"""
import os
import logging
from config.settings import PipelineConfig

logger = logging.getLogger(__name__)

USE_REAL_GENERATOR = os.getenv("USE_REAL_GENERATOR", "0") == "1"
GENERATOR_BACKEND  = os.getenv("GENERATOR_BACKEND", "ollama")   # "ollama" | "openai" | "mbart"

# mBART hard position limit
MBART_MAX_TOKENS = 1020


class MultilingualGenerator:
    def __init__(self, config: PipelineConfig):
        self.config      = config
        self._model      = None
        self._tokenizer  = None

    # ── Public API ────────────────────────────────────────────────────

    def generate(self, query: str, evidence: list[dict], lang: str = "en") -> str:
        if USE_REAL_GENERATOR:
            try:
                if GENERATOR_BACKEND == "ollama":
                    return self._generate_ollama(query, evidence, lang)
                elif GENERATOR_BACKEND == "openai":
                    return self._generate_openai(query, evidence, lang)
                elif GENERATOR_BACKEND == "mbart":
                    context = self._linearise(evidence)
                    prompt  = self._build_prompt(query, context, lang)
                    return self._generate_mbart(prompt, lang)
            except Exception as e:
                logger.error(f"[{GENERATOR_BACKEND}] generation failed: {e}")

        return self._structured_stub(query, evidence, lang)

    # ── Option 1: Ollama (recommended for local use, free, fast) ─────
    # Setup: install Ollama from https://ollama.com, then run:
    #   ollama pull llama3          (4GB, best quality)
    #   ollama pull mistral         (4GB, fast)
    #   ollama pull phi3            (2GB, lightweight)
    # Then: set USE_REAL_GENERATOR=1 GENERATOR_BACKEND=ollama

    def _generate_ollama(self, query: str, evidence: list[dict], lang: str) -> str:
        import requests
        model   = os.getenv("OLLAMA_MODEL", "llama3")
        context = self._linearise(evidence, max_chars_per_chunk=600)
        prompt  = self._build_prompt(query, context, lang)

        res = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=120,
        )
        res.raise_for_status()
        return res.json().get("response", "").strip()

    # ── Option 2: OpenAI API ──────────────────────────────────────────
    # Setup: pip install openai
    # Then: set USE_REAL_GENERATOR=1 GENERATOR_BACKEND=openai
    #       set OPENAI_API_KEY=sk-...

    # def _generate_openai(self, query: str, evidence: list[dict], lang: str) -> str:
    #     from openai import OpenAI
    #     client  = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    #     context = self._linearise(evidence, max_chars_per_chunk=800)
    #     prompt  = self._build_prompt(query, context, lang)

    #     resp = client.chat.completions.create(
    #         model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    #         messages=[
    #             {"role": "system",
    #              "content": "You are a precise multilingual assistant. "
    #                          "Answer using ONLY the provided evidence. "
    #                          f"Respond in language: {lang}."},
    #             {"role": "user", "content": prompt},
    #         ],
    #         max_tokens=512,
    #         temperature=0.3,
    #     )
    #     return resp.choices[0].message.content.strip()

    # # ── Option 3: mBART (NOT recommended on CPU — ~2min per query) ───
    # # Setup: pip install transformers torch sentencepiece protobuf
    # # Then: set USE_REAL_GENERATOR=1 GENERATOR_BACKEND=mbart

    # def _generate_mbart(self, prompt: str, lang: str) -> str:
    #     from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    #     import torch

    #     if self._model is None:
    #         name = self.config.model.decoder
    #         logger.info(f"Loading mBART: {name} (~2.4 GB, slow on CPU)")
    #         self._tokenizer = AutoTokenizer.from_pretrained(name)
    #         self._model = AutoModelForSeq2SeqLM.from_pretrained(
    #             name, torch_dtype=torch.float32)
    #         self._model.eval()

    #     inputs = self._tokenizer(
    #         prompt, return_tensors="pt",
    #         truncation=True, max_length=MBART_MAX_TOKENS,
    #     )
    #     seq_len = inputs["input_ids"].shape[1]
    #     if seq_len > MBART_MAX_TOKENS:
    #         inputs["input_ids"]      = inputs["input_ids"][:, :MBART_MAX_TOKENS]
    #         inputs["attention_mask"] = inputs["attention_mask"][:, :MBART_MAX_TOKENS]

    #     lang_map = {"en":"en_XX","hi":"hi_IN","ta":"ta_IN",
    #                 "fr":"fr_XX","de":"de_DE","es":"es_XX"}
    #     try:
    #         forced_bos = self._tokenizer.lang_code_to_id[lang_map.get(lang,"en_XX")]
    #     except Exception:
    #         forced_bos = self._tokenizer.bos_token_id

    #     import torch
    #     with torch.no_grad():
    #         out = self._model.generate(
    #             **inputs, forced_bos_token_id=forced_bos,
    #             max_new_tokens=256, num_beams=4, early_stopping=True,
    #         )
    #     return self._tokenizer.decode(out[0], skip_special_tokens=True)

    # ── Structured stub (default — always works) ──────────────────────

    def _structured_stub(self, query: str, evidence: list[dict], lang: str) -> str:
        """
        Returns a clean structured answer built directly from the top evidence
        chunks — readable, fast, and honest about what the corpus contains.
        """
        top = sorted(evidence,
                     key=lambda e: e.get("importance_score", 0),
                     reverse=True)

        sources = list(dict.fromkeys(e["source"] for e in evidence))  # preserve order

        lines = [f"Query: {query}\n"]

        text_chunks = [e for e in top if e.get("modality") == "text"]
        image_chunks = [e for e in top if e.get("modality") == "image"]
        audio_chunks = [e for e in top if e.get("modality") == "audio"]

        if text_chunks:
            lines.append("─" * 60)
            lines.append(f"Relevant content from {len(text_chunks)} text chunk(s):\n")
            for i, ev in enumerate(text_chunks[:5], 1):
                content = str(ev.get("content", "")).strip()
                if len(content) > 500:
                    content = content[:500] + "…"
                imp = ev.get("importance_score", 0)
                lines.append(f"[{i}] {ev['source']}  (relevance: {imp:.3f})")
                lines.append(content)
                lines.append("")

        if image_chunks:
            lines.append(f"📷 {len(image_chunks)} image(s) retrieved: "
                         f"{', '.join(e['source'] for e in image_chunks)}")

        if audio_chunks:
            lines.append(f"🎵 {len(audio_chunks)} audio clip(s) retrieved: "
                         f"{', '.join(e['source'] for e in audio_chunks)}")

        lines.append("─" * 60)
        lines.append(f"Sources ({len(sources)}): {', '.join(sources[:10])}")
        lines.append("")
        lines.append(
            "💡 To get a natural-language answer, enable a real LLM:\n"
            "   • Ollama (free, local): install from ollama.com, then\n"
            "     set USE_REAL_GENERATOR=1 GENERATOR_BACKEND=ollama\n"
            "   • OpenAI: set USE_REAL_GENERATOR=1 GENERATOR_BACKEND=openai\n"
            "     OPENAI_API_KEY=sk-..."
        )

        return "\n".join(lines)

    # ── Helpers ───────────────────────────────────────────────────────

    def _linearise(self, evidence: list[dict], max_chars_per_chunk: int = 600) -> str:
        sorted_ev = sorted(evidence,
                           key=lambda e: e.get("importance_score", 0),
                           reverse=True)
        parts = []
        for e in sorted_ev:
            content = str(e.get("content", "")).strip()
            if len(content) > max_chars_per_chunk:
                content = content[:max_chars_per_chunk] + "…"
            header = (f"[{e['source']} | {e['modality']} | {e['lang']}]")
            parts.append(f"{header}\n{content}")
        return "\n\n".join(parts)

    def _build_prompt(self, query: str, context: str, lang: str) -> str:
        return (
            f"Answer the question using ONLY the evidence below.\n"
            f"Respond in language: {lang}.\n\n"
            f"EVIDENCE:\n{context}\n\n"
            f"QUESTION: {query}\n\nANSWER:"
        )
