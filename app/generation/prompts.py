"""Engenharia de prompt para reproduzir o estilo visual de The Sims 3.

A identidade é travada pelo InstantID (embedding + keypoints); o prompt cuida
APENAS do estilo artístico. Por isso o prompt é deliberadamente neutro quanto a
gênero, idade e traços — para não empurrar a geração para longe da pessoa real.
"""
from __future__ import annotations

# Núcleo do estilo The Sims 3: render 3D estilizado, pele levemente CG,
# sombreamento suave, estética Maxis, retrato oficial de criação de personagem.
SIMS3_STYLE = (
    "official character portrait in the art style of The Sims 3 video game, "
    "stylized 3D rendered character, smooth CG skin shading, soft studio lighting, "
    "clean simple background, Maxis art style, high quality 3D render, "
    "head and shoulders portrait"
)

# Negative prompt: evita fotorrealismo, deformações e mudança de identidade.
NEGATIVE = (
    "photorealistic, real photo, photograph, anime, 2d, cartoon sketch, "
    "deformed, distorted, disfigured, extra limbs, extra fingers, "
    "lowres, blurry, bad anatomy, watermark, text, signature, "
    "different person, multiple faces, oversaturated"
)


def build_prompt(extra_style: str = "") -> str:
    """Monta o prompt final. `extra_style` permite ajustes finos opcionais."""
    parts = [SIMS3_STYLE]
    if extra_style:
        parts.append(extra_style.strip())
    return ", ".join(parts)


def build_negative() -> str:
    return NEGATIVE
