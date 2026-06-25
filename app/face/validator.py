"""Porteiro de validação: decide se a foto é aceita e qual rosto usar.

Combina B1 (detecção), B2 (rosto real), B3 (múltiplos rostos) e B4 (embedding).
Detector e classificador são injetados → fácil de mockar nos testes.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

from ..config import Settings
from ..utils.image_io import LoadedImage
from .detector import DetectedFace, FaceDetector
from .realness import RealnessClassifier


class RejectReason(str, Enum):
    NO_FACE = "no_face"                  # nenhum rosto detectado
    LOW_CONFIDENCE = "low_confidence"    # rosto com score baixo
    FACE_TOO_SMALL = "face_too_small"    # rosto pequeno demais
    NOT_REAL_HUMAN = "not_real_human"    # desenho/3D/animal/objeto/etc.
    AMBIGUOUS_MULTIFACE = "ambiguous_multiface"  # múltiplos rostos equivalentes


@dataclass
class ValidationResult:
    accepted: bool
    reason: Optional[RejectReason] = None
    message: str = ""
    primary_face: Optional[DetectedFace] = None
    num_faces: int = 0
    realness_score: Optional[float] = None


def _primary_score(face: DetectedFace, img_w: int, img_h: int) -> float:
    """Pontuação para eleger o rosto principal: área × score × centralidade."""
    cx, cy = face.center()
    dx = (cx - img_w / 2.0) / (img_w / 2.0)
    dy = (cy - img_h / 2.0) / (img_h / 2.0)
    centrality = max(0.0, 1.0 - (dx * dx + dy * dy) ** 0.5)
    return face.area * face.det_score * (0.5 + 0.5 * centrality)


class FaceValidator:
    def __init__(self, settings: Settings,
                 detector: Optional[FaceDetector] = None,
                 realness: Optional[RealnessClassifier] = None):
        self.s = settings
        self.detector = detector or FaceDetector(device=settings.device)
        self.realness = realness or RealnessClassifier(device=settings.device)

    def validate(self, image: LoadedImage) -> ValidationResult:
        faces = self.detector.detect(image.to_numpy_bgr())

        # B1 — existe rosto?
        if not faces:
            return ValidationResult(False, RejectReason.NO_FACE,
                                    "Nenhum rosto humano foi detectado na imagem.")

        # filtra por confiança
        confident = [f for f in faces if f.det_score >= self.s.det_score_min]
        if not confident:
            return ValidationResult(
                False, RejectReason.LOW_CONFIDENCE,
                "Não foi possível confirmar um rosto com confiança suficiente.",
                num_faces=len(faces))

        # B3 — escolher rosto principal / detectar ambiguidade
        ranked = sorted(
            confident,
            key=lambda f: _primary_score(f, image.width, image.height),
            reverse=True,
        )
        primary = ranked[0]

        if len(ranked) >= 2:
            second = ranked[1]
            if second.area >= self.s.multiface_ambiguous_ratio * primary.area:
                return ValidationResult(
                    False, RejectReason.AMBIGUOUS_MULTIFACE,
                    "Há mais de um rosto de destaque na imagem. Envie uma foto "
                    "com apenas um rosto principal.",
                    num_faces=len(confident))

        # B1 (cont.) — tamanho mínimo do rosto principal
        x1, y1, x2, y2 = primary.bbox
        face_side = min(x2 - x1, y2 - y1)
        if face_side < self.s.face_min_ratio * image.min_side:
            return ValidationResult(
                False, RejectReason.FACE_TOO_SMALL,
                "O rosto está pequeno demais. Aproxime mais a câmera do rosto.",
                num_faces=len(confident))

        # B2 — é foto de humano REAL? (recorte do rosto, com margem)
        realness_score = None
        if self.s.realness_enabled:
            crop = _crop_with_margin(image, primary.bbox, margin=0.35)
            r = self.realness.score(crop, threshold=self.s.realness_min)
            realness_score = r.score
            if not r.is_real:
                return ValidationResult(
                    False, RejectReason.NOT_REAL_HUMAN,
                    "A imagem não parece ser a foto real de um rosto humano "
                    "(desenhos, personagens, renders 3D e animais não são aceitos).",
                    num_faces=len(confident), realness_score=realness_score)

        # aprovado
        return ValidationResult(
            True, None, "Rosto válido.", primary_face=primary,
            num_faces=len(confident), realness_score=realness_score)


def _crop_with_margin(image: LoadedImage, bbox, margin: float = 0.35):
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    mx, my = w * margin, h * margin
    left = max(0, int(x1 - mx))
    top = max(0, int(y1 - my))
    right = min(image.width, int(x2 + mx))
    bottom = min(image.height, int(y2 + my))
    return image.pil.crop((left, top, right, bottom))
