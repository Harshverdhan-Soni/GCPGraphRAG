"""Grounded multilingual response generation."""
import logging
from config.settings import PipelineConfig

logger = logging.getLogger(__name__)


class MultilingualGenerator:
    def __init__(self, config: PipelineConfig):
        self.config     = config
        self._model     = None
        self._tokenizer = None

    def generate(self, query: str, evidence: list[dict], lang: str = "en") -> str:
        context = self._linearise(evidence)
        prompt  = self._build_prompt(query, context, lang)
        logger.debug(f"Prompt length (chars): {len(prompt)}")

        # PRODUCTION — uncomment to enable real generation:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        if self._model is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.config.model.decoder)
            self._model     = AutoModelForSeq2SeqLM.from_pretrained(self.config.model.decoder)
            self._model.eval()
        inputs = self._tokenizer(prompt, return_tensors="pt", truncation=True,
                                 max_length=self.config.model.context_length)
        import torch
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs, max_new_tokens=self.config.model.max_new_tokens,
                temperature=self.config.model.temperature, top_p=self.config.model.top_p,
                do_sample=True)
        return self._tokenizer.decode(outputs[0], skip_special_tokens=True)

        sources = list({e["source"] for e in evidence})
        return (
            f"[Stub response – plug in a real decoder to enable generation]\n\n"
            f"Query: {query}\n"
            f"Evidence used: {len(evidence)} units from {len(sources)} sources.\n"
            f"Sources: {', '.join(sources[:5])}"
        )

    def _linearise(self, evidence: list[dict]) -> str:
        sorted_ev = sorted(evidence, key=lambda e: e.get("importance_score", 0), reverse=True)
        parts = []
        for e in sorted_ev:
            header = (f"[SOURCE: {e['source']} | "
                      f"MODALITY: {e['modality']} | LANG: {e['lang']}]")
            parts.append(f"{header}\n{e['content']}")
        return "\n\n".join(parts)

    def _build_prompt(self, query: str, context: str, lang: str) -> str:
        return (
            f"You are a helpful multilingual assistant. "
            f"Answer the question below using ONLY the provided evidence. "
            f"Respond in language: {lang}.\n\n"
            f"EVIDENCE:\n{context}\n\n"
            f"QUESTION: {query}\n\nANSWER:"
        )
