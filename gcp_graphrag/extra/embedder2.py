"""
Stage (a) – Multimodal Embedding.

Text  : mDeBERTa-v3-base (768-dim, multilingual)
Image : CLIP ViT-B/32    (512-dim → projected to 768)
Audio : Whisper-base → transcript → mDeBERTa

Set env var USE_STUB_EMBEDDINGS=1 to skip real models (for testing).
"""
import os
import logging
from typing import Any
import numpy as np
from config.settings import PipelineConfig

logger = logging.getLogger(__name__)

_USE_STUB = os.getenv("USE_STUB_EMBEDDINGS", "0") == "1"
_CLIP_DIM = 512


class MultimodalEmbedder:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.dim    = config.embedding.embedding_dim   # 768
        self._text_model   = None
        self._vision_model = None
        self._audio_model  = None

    # ── Public API ────────────────────────────────────────────────────

    def encode_query(self, query: str, lang: str = "en") -> np.ndarray:
        return self._normalise(self._encode_text(str(query)))

    def encode_corpus(self, documents):
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
        if not vectors:
            return np.empty((0, self.dim), dtype=np.float32), metadata
        return np.vstack(vectors).astype(np.float32), metadata

    # ── Dispatch ──────────────────────────────────────────────────────

    def _encode_document(self, doc):
        m = doc.get("modality", "text")
        if m == "text":   return self._encode_text(doc.get("content", ""))
        if m == "image":  return self._encode_image(doc.get("content", b""))
        if m == "audio":  return self._encode_audio(doc.get("content", b""))
        logger.warning(f"Unknown modality '{m}' – skipping.")
        return None

    # ── Text: mDeBERTa-v3-base ────────────────────────────────────────

    def _encode_text(self, text: str) -> np.ndarray:
        if _USE_STUB:
            return self._stub(text, 0)
        try:
            from transformers import AutoTokenizer, AutoModel
            import torch

            if self._text_model is None:
                name = self.config.embedding.text_model
                logger.info(f"Loading text model: {name}")
                tok = AutoTokenizer.from_pretrained(name)
                # Force float32 — avoids Float/Half dtype mismatch on Windows
                mdl = AutoModel.from_pretrained(
                    name,
                    ignore_mismatched_sizes=True,
                    torch_dtype=torch.float32,
                )
                mdl.eval()
                self._text_model = (tok, mdl)

            tok, mdl = self._text_model

            inputs = tok(
                str(text),
                return_tensors="pt",
                truncation=True,
                max_length=self.config.embedding.max_seq_length,
                padding=True,
            )
            # Cast input tensors to match the model dtype (float32)
            inputs = {k: v.to(torch.float32) if v.dtype.is_floating_point else v
                      for k, v in inputs.items()}

            with torch.no_grad():
                out = mdl(**inputs)

            # Mean-pool: (1, seq_len, 768) → (768,)
            vec = out.last_hidden_state.mean(dim=1).squeeze(0)
            return vec.cpu().to(torch.float32).numpy().astype(np.float32)

        except Exception as e:
            logger.warning(f"Text encoding failed ({e}), using stub.")
            return self._stub(text, 0)

    # ── Image: CLIP ViT-B/32 ──────────────────────────────────────────

    def _encode_image(self, image_bytes: Any) -> np.ndarray:
        if _USE_STUB:
            return self._stub("image", 1)
        try:
            from transformers import CLIPProcessor, CLIPModel
            from PIL import Image
            import io, torch

            if self._vision_model is None:
                name = self.config.embedding.vision_model
                logger.info(f"Loading vision model: {name}")
                proc = CLIPProcessor.from_pretrained(name)
                mdl  = CLIPModel.from_pretrained(name, torch_dtype=torch.float32)
                mdl.eval()
                proj = torch.nn.Linear(_CLIP_DIM, self.dim, bias=False)
                torch.nn.init.xavier_uniform_(proj.weight)
                proj.eval()
                self._vision_model = (proc, mdl, proj)

            proc, mdl, proj = self._vision_model

            if isinstance(image_bytes, (bytes, bytearray)):
                img = Image.open(io.BytesIO(bytes(image_bytes))).convert("RGB")
            else:
                img = Image.open(str(image_bytes)).convert("RGB")

            inputs = proc.image_processor(images=img, return_tensors="pt")
            # Ensure pixel_values are float32
            inputs = {k: v.to(torch.float32) if v.dtype.is_floating_point else v
                      for k, v in inputs.items()}

            with torch.no_grad():
                feat = mdl.get_image_features(**inputs)          # (1, 512)
                vec  = proj(feat.to(torch.float32)).squeeze(0)   # (768,)

            return vec.cpu().numpy().astype(np.float32)

        except Exception as e:
            logger.warning(f"Image encoding failed ({e}), using stub.")
            return self._stub("image", 1)

    # ── Audio: Whisper → transcript → mDeBERTa ────────────────────────

    def _encode_audio(self, audio_bytes: Any) -> np.ndarray:
        if _USE_STUB:
            return self._stub("audio", 2)
        try:
            import whisper, tempfile, os as _os

            if self._audio_model is None:
                logger.info("Loading Whisper model: base")
                self._audio_model = whisper.load_model("base")

            if isinstance(audio_bytes, (bytes, bytearray)):
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    f.write(bytes(audio_bytes))
                    tmp = f.name
                try:
                    transcript = self._audio_model.transcribe(tmp).get("text", "")
                finally:
                    _os.unlink(tmp)
            else:
                transcript = self._audio_model.transcribe(
                    str(audio_bytes)).get("text", "")

            if not transcript.strip():
                logger.warning("Whisper returned empty transcript – using stub.")
                return self._stub("audio", 2)

            return self._encode_text(transcript)

        except Exception as e:
            logger.warning(f"Audio encoding failed ({e}), using stub.")
            return self._stub("audio", 2)

    # ── Helpers ───────────────────────────────────────────────────────

    def _stub(self, text: str, seed_offset: int) -> np.ndarray:
        seed = (abs(hash(str(text))) + seed_offset) % (2 ** 32)
        return np.random.default_rng(seed).standard_normal(self.dim).astype(np.float32)

    @staticmethod
    def _normalise(vec: np.ndarray) -> np.ndarray:
        v = vec.astype(np.float32)
        norm = np.linalg.norm(v)
        return (v / norm) if norm > 1e-9 else v
