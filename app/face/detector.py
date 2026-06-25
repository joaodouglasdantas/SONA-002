"""Detecção facial + embedding de identidade via InsightFace (SCRFD + ArcFace).

Import preguiçoso de insightface/onnxruntime: o módulo importa sem GPU/modelos
instalados (importante para subir o servidor e rodar os testes). O modelo só é
carregado na primeira detecção real.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class DetectedFace:
    bbox: tuple          # (x1, y1, x2, y2)
    det_score: float
    embedding: np.ndarray  # ArcFace 512-d, normalizado
    kps: np.ndarray        # 5 keypoints (olhos, nariz, cantos da boca)

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.bbox
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    def center(self) -> tuple:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


class FaceDetector:
    """Wrapper fino sobre insightface.app.FaceAnalysis."""

    def __init__(self, device: str = "cpu", det_size: int = 640):
        self.device = device
        self.det_size = det_size
        self._app = None  # carregado preguiçosamente

    def _ensure_loaded(self):
        if self._app is not None:
            return
        from insightface.app import FaceAnalysis  # import pesado, adiado

        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if self.device == "cuda"
            else ["CPUExecutionProvider"]
        )
        app = FaceAnalysis(name="buffalo_l", providers=providers)
        ctx_id = 0 if self.device == "cuda" else -1
        app.prepare(ctx_id=ctx_id, det_size=(self.det_size, self.det_size))
        self._app = app

    def detect(self, image_bgr: np.ndarray) -> List[DetectedFace]:
        """Retorna todos os rostos detectados (pode ser lista vazia)."""
        self._ensure_loaded()
        faces = self._app.get(image_bgr)
        out: List[DetectedFace] = []
        for f in faces:
            emb = f.normed_embedding  # já normalizado (L2)
            out.append(
                DetectedFace(
                    bbox=tuple(float(v) for v in f.bbox),
                    det_score=float(f.det_score),
                    embedding=np.asarray(emb, dtype=np.float32),
                    kps=np.asarray(f.kps, dtype=np.float32),
                )
            )
        return out
