"""Modelos Pydantic de resposta da API."""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class RejectionResponse(BaseModel):
    accepted: bool = False
    reason: str
    message: str
    num_faces: int = 0
    realness_score: Optional[float] = None


class TransformResponse(BaseModel):
    accepted: bool = True
    image_base64: str          # PNG da versão The Sims 3 (base64)
    similarity: float          # cosseno ArcFace entrada×saída [0..1]
    fidelity_ok: bool          # similarity >= limiar
    attempts: int
    provider: str
    num_faces: int
    realness_score: Optional[float] = None


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str
    provider: str
