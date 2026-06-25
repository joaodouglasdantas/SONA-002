"""Dublês de teste — substituem a IA pesada (InsightFace/CLIP/Replicate)."""
from __future__ import annotations

import io

import numpy as np
from PIL import Image

from app.face.detector import DetectedFace
from app.face.realness import RealnessResult
from app.generation.base import (GenerationProvider, GenerationRequest,
                                 GenerationResult)


def png_bytes(color=(128, 128, 128), size=(256, 256)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


def make_face(bbox, score=0.9, emb=None, dim=512) -> DetectedFace:
    if emb is None:
        emb = np.ones(dim, dtype=np.float32)
    emb = np.asarray(emb, dtype=np.float32)
    emb = emb / (np.linalg.norm(emb) or 1.0)
    x1, y1, x2, y2 = bbox
    kps = np.zeros((5, 2), dtype=np.float32)
    return DetectedFace(bbox=tuple(map(float, bbox)), det_score=score,
                        embedding=emb, kps=kps)


class FakeDetector:
    """Retorna listas de rostos pré-configuradas, em sequência por chamada."""

    def __init__(self, sequence):
        # sequence: lista de listas de DetectedFace (uma por chamada de detect)
        self.sequence = list(sequence)
        self.calls = 0

    def detect(self, image_bgr):
        idx = min(self.calls, len(self.sequence) - 1)
        self.calls += 1
        return self.sequence[idx]


class FakeRealness:
    def __init__(self, is_real=True, score=0.9):
        self._is_real = is_real
        self._score = score

    def score(self, image, threshold=0.65):
        return RealnessResult(is_real=self._is_real, score=self._score,
                              threshold=threshold)


class FakeProvider(GenerationProvider):
    name = "fake"

    def __init__(self, out_png=None):
        self.out_png = out_png or png_bytes((200, 180, 160))
        self.requests = []

    def generate(self, req: GenerationRequest) -> GenerationResult:
        self.requests.append(req)
        return GenerationResult(image_png=self.out_png, provider=self.name,
                                seed=req.seed, meta={"fake": True})
