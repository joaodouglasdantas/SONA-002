"""Orquestrador do fluxo A -> B -> C -> D.

A: pre-processamento   B: validacao de rosto (porteiro)
C: geracao             D: verificacao de fidelidade + re-tentativa

Dependencias (validator, provider, detector) sao injetaveis para teste.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional

from .config import Settings, get_settings
from .face.detector import FaceDetector
from .face.validator import FaceValidator, RejectReason, ValidationResult
from .generation.base import (GenerationProvider, GenerationRequest,
                              get_provider)
from .generation.prompts import build_flux_prompt, build_negative, build_prompt
from .utils.image_io import load_image
from .verify.similarity import cosine_similarity


@dataclass
class PipelineResult:
    accepted: bool
    reason: Optional[RejectReason] = None
    message: str = ""
    num_faces: int = 0
    realness_score: Optional[float] = None
    image_png: Optional[bytes] = None
    similarity: float = 0.0
    fidelity_ok: bool = False
    attempts: int = 0
    provider: str = ""
    meta: dict = field(default_factory=dict)

    @classmethod
    def from_rejection(cls, v: ValidationResult) -> "PipelineResult":
        return cls(accepted=False, reason=v.reason, message=v.message,
                   num_faces=v.num_faces, realness_score=v.realness_score)


class Pipeline:
    def __init__(self, settings: Optional[Settings] = None,
                 validator: Optional[FaceValidator] = None,
                 provider: Optional[GenerationProvider] = None,
                 detector: Optional[FaceDetector] = None):
        self.s = settings or get_settings()
        self.detector = detector or FaceDetector(device=self.s.device)
        self.validator = validator or FaceValidator(self.s, detector=self.detector)
        self._provider = provider

    @property
    def provider(self) -> GenerationProvider:
        if self._provider is None:
            self._provider = get_provider(self.s)
        return self._provider

    def run(self, data: bytes, extra_style: str = "",
            seed: Optional[int] = None) -> PipelineResult:
        # A - pre-processamento
        image = load_image(data, max_side=self.s.max_side_px)

        # B - validacao (porteiro)
        v = self.validator.validate(image)
        if not v.accepted:
            return PipelineResult.from_rejection(v)

        original_emb = v.primary_face.embedding
        image_png = image.to_png_bytes()

        # prompt depende do caminho de geracao
        if self.s.provider in ("flux_pulid", "comfyui_replicate"):
            use_lora_trigger = self.s.provider == "comfyui_replicate"
            prompt = build_flux_prompt(extra_style, use_lora_trigger=use_lora_trigger)
        else:
            prompt = build_prompt(extra_style)
        negative = build_negative()

        # C + D - gerar e verificar fidelidade, re-tentando se preciso
        best = None
        attempts = 0
        for attempt in range(self.s.max_retries + 1):
            attempts = attempt + 1
            use_seed = seed if seed is not None else random.randint(0, 2**31 - 1)
            id_strength = min(0.95, 0.85 + 0.05 * attempt)
            req = GenerationRequest(
                image_png=image_png, prompt=prompt, negative_prompt=negative,
                identitynet_strength=id_strength, seed=use_seed,
                style_lora=self.s.style_lora,
            )
            result = self.provider.generate(req)

            similarity = self._measure_similarity(original_emb, result.image_png)
            fidelity_ok = similarity >= self.s.similarity_min

            if best is None or similarity > best[0]:
                best = (similarity, result, fidelity_ok)

            if fidelity_ok:
                break

        sim, result, fidelity_ok = best
        return PipelineResult(
            accepted=True, image_png=result.image_png, similarity=sim,
            fidelity_ok=fidelity_ok, attempts=attempts, provider=result.provider,
            num_faces=v.num_faces, realness_score=v.realness_score,
            meta=result.meta)

    def _measure_similarity(self, original_emb, generated_png: bytes) -> float:
        """Detecta rosto na saida e compara identidade com a entrada."""
        gen_img = load_image(generated_png, max_side=self.s.max_side_px)
        faces = self.detector.detect(gen_img.to_numpy_bgr())
        if not faces:
            return 0.0
        gen_face = max(faces, key=lambda f: f.area)
        return cosine_similarity(original_emb, gen_face.embedding)
