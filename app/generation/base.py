"""Interface abstrata do provider de geração.

Trocar de Replicate para ComfyUI self-hosted (ou outro) = nova implementação
desta interface, sem tocar no pipeline. Isso evita lock-in e controla custo
conforme a escala.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass, field


@dataclass
class GenerationRequest:
    image_png: bytes              # foto original validada (PNG)
    prompt: str
    negative_prompt: str
    # força do condicionamento de identidade (InstantID IdentityNet)
    identitynet_strength: float = 0.85
    # influência da imagem de referência (adapter)
    adapter_strength: float = 0.80
    guidance_scale: float = 5.0
    num_steps: int = 30
    seed: int | None = None
    style_lora: str = ""          # opcional: LoRA de estilo The Sims 3
    extra: dict = field(default_factory=dict)


@dataclass
class GenerationResult:
    image_png: bytes
    provider: str
    seed: int | None = None
    meta: dict = field(default_factory=dict)


class GenerationError(RuntimeError):
    pass


class GenerationProvider(abc.ABC):
    name: str = "base"

    @abc.abstractmethod
    def generate(self, req: GenerationRequest) -> GenerationResult:
        """Gera a versão estilizada preservando identidade/pose."""
        raise NotImplementedError


def get_provider(settings) -> GenerationProvider:
    """Fábrica: escolhe o provider conforme a config."""
    name = settings.provider.lower()
    if name == "replicate":
        from .replicate_provider import ReplicateProvider
        return ReplicateProvider(settings)
    if name == "comfyui":
        from .comfyui_provider import ComfyUIProvider
        return ComfyUIProvider(settings)
    raise GenerationError(f"provider desconhecido: {settings.provider!r}")
