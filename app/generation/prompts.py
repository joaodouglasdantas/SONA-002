"""Engenharia de prompt para reproduzir o estilo visual de The Sims.

A identidade e travada pelo modelo (InstantID/PuLID); o prompt cuida APENAS do
estilo artistico. Por isso o prompt e deliberadamente neutro quanto a genero,
idade e tracos - para nao empurrar a geracao para longe da pessoa real.
"""
from __future__ import annotations

# --- Caminho SDXL / InstantID ---
SIMS3_STYLE = (
    "official character portrait in the art style of The Sims 3 video game, "
    "stylized 3D rendered character, smooth CG skin shading, soft studio lighting, "
    "clean simple background, Maxis art style, high quality 3D render, "
    "head and shoulders portrait"
)

NEGATIVE = (
    "photorealistic, real photo, photograph, anime, 2d, cartoon sketch, "
    "deformed, distorted, disfigured, extra limbs, extra fingers, "
    "lowres, blurry, bad anatomy, watermark, text, signature, "
    "different person, multiple faces, oversaturated"
)


def build_prompt(extra_style: str = "") -> str:
    """Monta o prompt final (SDXL/InstantID)."""
    parts = [SIMS3_STYLE]
    if extra_style:
        parts.append(extra_style.strip())
    return ", ".join(parts)


def build_negative() -> str:
    return NEGATIVE


# --- Caminho Flux (PuLID e/ou LoRA Sims) ---

# Gatilho da LoRA dvyio/flux-lora-the-sims (Hugging Face).
SIMS_LORA_TRIGGER = "video game screenshot in the style of THSMS"

SIMS_FLUX_STYLE = (
    "portrait of a person as a The Sims video game character, "
    "stylized 3D game render, smooth CG skin, soft lighting, "
    "clean simple background, head and shoulders"
)


def build_flux_prompt(extra_style: str = "", use_lora_trigger: bool = True) -> str:
    """Prompt para o caminho Flux.

    Com a LoRA Sims ativa, incluir o gatilho THSMS melhora muito o estilo.
    Sem LoRA (apenas PuLID), o gatilho e inofensivo e o estilo vem do texto.
    """
    parts = []
    if use_lora_trigger:
        parts.append(SIMS_LORA_TRIGGER)
    parts.append(SIMS_FLUX_STYLE)
    if extra_style:
        parts.append(extra_style.strip())
    return ", ".join(parts)
