"""
Stage (a) – Multimodal Embedding.

Encodes text, image, and audio evidence units into a shared
embedding space using:
  - mDeBERTa-v3  for multilingual text
  - CLIP          for images
  - Whisper       for speech → text → embedding

Stub mode (default): deterministic random vectors for testing.
Production mode: uncomment the transformer blocks below.
"""
import logging
from typing import Any
import numpy as np
from config.settings import PipelineConfig

logger = logging.getLogger(__name__)

class MultimodalEmbedder:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.dim = config.embedding.embedding_dim
        self._text_model   = None
        self._vision_model = None
        self._audio_model  = None

    # ── Public API ────────────────────────────────────────────────────

    def encode_query(self, query: str, lang: str = "en") -> np.ndarray:
        vec = self._encode_text(query)
        return self._normalise(vec)

    def encode_corpus(self, documents: list[dict]) -> tuple[np.ndarray, list[dict]]:
        vectors, metadata = [], []
        for doc in documents:
            vec = self._encode_document(doc)
            if vec is not None:
                vectors.append(self._normalise(vec))
                metadata.append({
                    "source":           doc.get("source", "unknown"),
                    "modality":         doc.get("modality", "text"),
                    "lang":             doc.get("lang", "en"),
                    "content":          doc.get("content", ""),
                    "provenance_score": doc.get("provenance_score", 1.0),
                })
        embeddings = np.vstack(vectors).astype(np.float32)
        return embeddings, metadata

    # ── Internal helpers ──────────────────────────────────────────────

    def _encode_document(self, doc: dict) -> np.ndarray | None:
        modality = doc.get("modality", "text")
        if modality == "text":
            return self._encode_text(doc["content"])
        elif modality == "image":
            return self._encode_image(doc["content"])
        elif modality == "audio":
            return self._encode_audio(doc["content"])
        logger.warning(f"Unknown modality '{modality}' – skipping.")
        return None

    def _encode_text(self, text: str) -> np.ndarray:
        """
        PRODUCTION — uncomment to use real embeddings:
        """

        from transformers import AutoTokenizer, AutoModel
        import torch
        if self._text_model is None:
            tok = AutoTokenizer.from_pretrained(self.config.embedding.text_model)
            mdl = AutoModel.from_pretrained(self.config.embedding.text_model)
            mdl.eval()
            self._text_model = (tok, mdl)
        tok, mdl = self._text_model
        inputs = tok(str(text), return_tensors="pt", truncation=True,
                     max_length=self.config.embedding.max_seq_length, padding=True)
        with torch.no_grad():
            output = mdl(**inputs)
        return output.last_hidden_state.mean(dim=1).squeeze().numpy()
        
        rng = np.random.default_rng(abs(hash(str(text))) % (2**32))
        return rng.standard_normal(self.dim).astype(np.float32)

    def _encode_image(self, image_bytes: Any) -> np.ndarray:
        """
        PRODUCTION — uncomment to use CLIP:
        """
        from transformers import CLIPProcessor, CLIPModel
        from PIL import Image
        import io, torch
        if self._vision_model is None:
            proc = CLIPProcessor.from_pretrained(self.config.embedding.vision_model)
            mdl  = CLIPModel.from_pretrained(self.config.embedding.vision_model)
            mdl.eval()
            self._vision_model = (proc, mdl)
        proc, mdl = self._vision_model
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        inputs = proc.image_processor(images=img, return_tensors="pt")
        with torch.no_grad():
            features = mdl.get_image_features(**inputs)
        return features.cpu().numpy().flatten().astype(np.float32)
        
        rng = np.random.default_rng(42)
        return rng.standard_normal(self.dim).astype(np.float32)


    # def _encode_image(self, image_bytes: Any) -> np.ndarray:
    #     from transformers import CLIPProcessor, CLIPModel
    #     from PIL import Image
    #     import io, torch

    #     if self._vision_model is None:
    #         proc = CLIPProcessor.from_pretrained(self.config.embedding.vision_model)
    #         mdl  = CLIPModel.from_pretrained(self.config.embedding.vision_model)
    #         mdl.eval()
    #         self._vision_model = (proc, mdl)

    #     proc, mdl = self._vision_model
    #     img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    #     inputs = proc.image_processor(images=img, return_tensors="pt")

    #     with torch.no_grad():
    #         features = mdl.get_image_features(**inputs)  # returns tensor directly

    #     return features.cpu().numpy().flatten().astype(np.float32)  # ✅ no .pooler_output

    def _encode_audio(self, audio_bytes: Any) -> np.ndarray:
        """
        PRODUCTION — uncomment to use Whisper:
        """
        import whisper, tempfile, os
        if self._audio_model is None:
            self._audio_model = whisper.load_model("base")
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(audio_bytes); tmp = f.name
        result = self._audio_model.transcribe(tmp)
        os.unlink(tmp)
        return self._encode_text(result["text"])
        
        rng = np.random.default_rng(7)
        return rng.standard_normal(self.dim).astype(np.float32)

    @staticmethod
    def _normalise(vec: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 1e-9 else vec
