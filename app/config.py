"""Configuração central via variáveis de ambiente.

Todos os limiares são ajustáveis sem mexer no código. Carregado uma vez e
reutilizado (singleton via lru_cache).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache

# Carrega o .env automaticamente, se python-dotenv estiver instalado.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def _f(name: str, default: float) -> float:
    return float(os.getenv(name, default))


def _i(name: str, default: int) -> int:
    return int(os.getenv(name, default))


@dataclass(frozen=True)
class Settings:
    # ---- Pré-processamento ----
    max_upload_mb: float = field(default_factory=lambda: _f("SONA_MAX_UPLOAD_MB", 15))
    max_side_px: int = field(default_factory=lambda: _i("SONA_MAX_SIDE_PX", 1280))
    allowed_mime: tuple = ("image/jpeg", "image/png", "image/webp")

    # ---- Detecção de rosto (B1) ----
    det_score_min: float = field(default_factory=lambda: _f("SONA_DET_SCORE_MIN", 0.55))
    # fração mínima da menor dimensão da imagem que o rosto deve ocupar
    face_min_ratio: float = field(default_factory=lambda: _f("SONA_FACE_MIN_RATIO", 0.10))

    # ---- "Rosto humano real?" (B2, CLIP) ----
    realness_min: float = field(default_factory=lambda: _f("SONA_REALNESS_MIN", 0.65))
    realness_enabled: bool = field(
        default_factory=lambda: os.getenv("SONA_REALNESS_ENABLED", "1") == "1"
    )

    # ---- Múltiplos rostos (B3) ----
    # se o 2º maior rosto tiver área >= este % do maior, considera ambíguo e rejeita
    multiface_ambiguous_ratio: float = field(
        default_factory=lambda: _f("SONA_MULTIFACE_AMBIGUOUS_RATIO", 0.80)
    )

    # ---- Verificação de fidelidade (D) ----
    similarity_min: float = field(default_factory=lambda: _f("SONA_SIMILARITY_MIN", 0.45))
    max_retries: int = field(default_factory=lambda: _i("SONA_MAX_RETRIES", 2))

    # ---- Geração ----
    provider: str = field(default_factory=lambda: os.getenv("SONA_PROVIDER", "replicate"))
    replicate_model: str = field(
        default_factory=lambda: os.getenv(
            "SONA_REPLICATE_MODEL",
            "zsxkib/instant-id:2e4785a4d80dadf580077b2244c8d7c05d8e3faac04a04c02d8e099dd2876789",
        )
    )
    replicate_api_token: str = field(
        default_factory=lambda: os.getenv("REPLICATE_API_TOKEN", "")
    )
    comfyui_url: str = field(default_factory=lambda: os.getenv("SONA_COMFYUI_URL", ""))
    style_lora: str = field(default_factory=lambda: os.getenv("SONA_STYLE_LORA", ""))

    # device para os modelos de CV (cpu / cuda)
    device: str = field(default_factory=lambda: os.getenv("SONA_DEVICE", "cpu"))


@lru_cache
def get_settings() -> Settings:
    return Settings()
