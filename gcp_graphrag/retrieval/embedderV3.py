"""
Stage (a) – Multimodal Embedding.

Text encoder hierarchy (Indian multilingual use case):
  PRIMARY  : google/muril-base-cased
             — 17 Indian languages, cross-script, trained on IndicCorp + Dakshina
  FALLBACK : sentence-transformers/paraphrase-multilingual-mpnet-base-v2
             — 50+ languages, reliable on Windows

Image : CLIP ViT-B/32 (512-dim → projected to 768)
Audio : Whisper-base → transcript → text encoder

Set USE_STUB_EMBEDDINGS=1 to skip models (fast testing).
"""
import os
import logging
from typing import Any
import numpy as np
from config.settings import PipelineConfig

logger = logging.getLogger(__name__)
_USE_STUB = os.getenv("USE_STUB_EMBEDDINGS", "0") == "1"
_CLIP_DIM = 512

# Indian language script ranges for detection
_INDIAN_SCRIPT_RANGES = [
    (0x0900, 0x097F),   # Devanagari (Hindi, Marathi, Sanskrit)
    (0x0980, 0x09FF),   # Bengali
    (0x0A00, 0x0A7F),   # Gurmukhi (Punjabi)
    (0x0A80, 0x0AFF),   # Gujarati
    (0x0B00, 0x0B7F),   # Odia
    (0x0B80, 0x0BFF),   # Tamil
    (0x0C00, 0x0C7F),   # Telugu
    (0x0C80, 0x0CFF),   # Kannada
    (0x0D00, 0x0D7F),   # Malayalam
    (0x0600, 0x06FF),   # Arabic/Urdu
]

def _has_indian_script(text: str) -> bool:
    """Detect if text contains Indian script characters."""
    for ch in str(text)[:200]:
        cp = ord(ch)
        if any(lo <= cp <= hi for lo, hi in _INDIAN_SCRIPT_RANGES):
            return True
    return False


class MultimodalEmbedder:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.dim    = config.embedding.embedding_dim   # 768
        self._muril_model       = None   # (tokenizer, model) — MuRIL primary
        self._sbert_model       = None   # SentenceTransformer — fallback
        self._vision_model      = None   # (CLIPProcessor, CLIPModel, projection)
        self._audio_model       = None   # Whisper

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
        if m == "text":  return self._encode_text(doc.get("content", ""))
        if m == "image": return self._encode_image(doc.get("content", b""))
        if m == "audio": return self._encode_audio(doc.get("content", b""))
        logger.warning(f"Unknown modality '{m}' – skipping.")
        return None

    # ── Text: MuRIL primary, sentence-transformers fallback ───────────

    def _encode_text(self, text: str) -> np.ndarray:
        if _USE_STUB:
            return self._stub(text, 0)

        # Try MuRIL first (best for Indian scripts)
        vec = self._encode_muril(text)
        if vec is not None:
            return vec

        # Fallback to sentence-transformers (reliable on Windows)
        vec = self._encode_sbert(text)
        if vec is not None:
            return vec

        logger.warning("Both encoders failed — using stub.")
        return self._stub(text, 0)

    def _encode_muril(self, text: str) -> np.ndarray | None:
        """MuRIL: google/muril-base-cased — best for Indian multilingual text."""
        try:
            from transformers import AutoTokenizer, AutoModel
            import torch

            if self._muril_model is None:
                name = self.config.embedding.text_model  # google/muril-base-cased
                logger.info(f"Loading MuRIL: {name}")
                tok = AutoTokenizer.from_pretrained(name)
                mdl = AutoModel.from_pretrained(
                    name,
                    ignore_mismatched_sizes=True,
                    torch_dtype=torch.float32,
                )
                mdl.eval()
                self._muril_model = (tok, mdl)
                logger.info("MuRIL loaded ✓")

            tok, mdl = self._muril_model
            text_str = str(text)[:3000]

            inputs = tok(
                text_str,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True,
            )

            # Clamp input_ids to valid vocab range (safety net)
            vocab_size = mdl.config.vocab_size
            inputs["input_ids"] = inputs["input_ids"].clamp(0, vocab_size - 1)

            # Ensure float32
            inputs = {k: v.to(torch.float32) if v.dtype.is_floating_point else v
                      for k, v in inputs.items()}

            with torch.no_grad():
                out = mdl(**inputs)

            # Mean-pool over token dimension → (768,)
            vec = out.last_hidden_state.mean(dim=1).squeeze(0)
            return vec.cpu().to(torch.float32).numpy().astype(np.float32)

        except ImportError:
            logger.warning("transformers not installed — skipping MuRIL.")
            return None
        except Exception as e:
            logger.warning(f"MuRIL encoding failed ({e}) — trying fallback.")
            return None

    def _encode_sbert(self, text: str) -> np.ndarray | None:
        """sentence-transformers fallback — multilingual, Windows-friendly."""
        try:
            from sentence_transformers import SentenceTransformer

            if self._sbert_model is None:
                name = "paraphrase-multilingual-mpnet-base-v2"
                logger.info(f"Loading sentence-transformer fallback: {name}")
                self._sbert_model = SentenceTransformer(name)
                logger.info("Sentence-transformer fallback loaded ✓")

            text_str = str(text)[:3000]
            vec = self._sbert_model.encode(
                text_str,
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            return vec.astype(np.float32)

        except ImportError:
            logger.warning("sentence-transformers not installed.")
            return None
        except Exception as e:
            logger.warning(f"SBERT fallback failed ({e}).")
            return None

    # ── Image: CLIP ViT-B/32 ──────────────────────────────────────────

    def _encode_image(self, image_source: Any) -> np.ndarray:
        """Accepts filepath string (new) or raw bytes (legacy)."""
        if _USE_STUB:
            return self._stub(str(image_source), 1)
        try:
            from transformers import CLIPProcessor, CLIPModel
            from PIL import Image
            import io, torch

            if self._vision_model is None:
                name = self.config.embedding.vision_model
                logger.info(f"Loading CLIP: {name}")
                proc = CLIPProcessor.from_pretrained(name)
                mdl  = CLIPModel.from_pretrained(name, torch_dtype=torch.float32)
                mdl.eval()
                proj = torch.nn.Linear(_CLIP_DIM, self.dim, bias=False)
                torch.nn.init.xavier_uniform_(proj.weight)
                proj.eval()
                self._vision_model = (proc, mdl, proj)
                logger.info("CLIP loaded ✓")

            proc, mdl, proj = self._vision_model

            if isinstance(image_source, (bytes, bytearray)):
                img = Image.open(io.BytesIO(bytes(image_source))).convert("RGB")
            else:
                img = Image.open(str(image_source)).convert("RGB")

            inputs = proc(images=img, return_tensors="pt")
            inputs = {k: v.to(torch.float32) if v.dtype.is_floating_point else v
                      for k, v in inputs.items()}

            with torch.no_grad():
                feat = mdl.get_image_features(**inputs)
                vec  = proj(feat.to(torch.float32)).squeeze(0)

            return vec.cpu().numpy().astype(np.float32)

        except ImportError:
            logger.warning("CLIP needs: pip install transformers torch Pillow")
            return self._stub(str(image_source), 1)
        except FileNotFoundError:
            logger.warning(f"Image file not found: {image_source}")
            return self._stub(str(image_source), 1)
        except Exception as e:
            logger.warning(f"Image encoding failed ({e}), using stub.")
            return self._stub(str(image_source), 1)

    # ── Audio: Whisper → transcript → text encoder ────────────────────

    def _encode_audio(self, audio_bytes: Any) -> np.ndarray:
        if _USE_STUB:
            return self._stub("audio", 2)
        try:
            import whisper, tempfile, os as _os

            if self._audio_model is None:
                logger.info("Loading Whisper: base")
                self._audio_model = whisper.load_model("base")
                logger.info("Whisper loaded ✓")

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
                return self._stub("audio", 2)

            logger.debug(f"Whisper transcript: {transcript[:100]}")
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
        v    = vec.astype(np.float32)
        norm = np.linalg.norm(v)
        return (v / norm) if norm > 1e-9 else v
