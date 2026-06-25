"""Decodificação e normalização de imagens de entrada.

Corrige orientação EXIF, converte para RGB, redimensiona para um limite máximo e
expõe conversões entre PIL / numpy / bytes. Mantemos PIL/numpy como deps leves;
nada de torch aqui.
"""
from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageOps


class ImageDecodeError(ValueError):
    """Bytes não formam uma imagem válida/legível."""


@dataclass
class LoadedImage:
    pil: Image.Image          # RGB
    width: int
    height: int

    @property
    def min_side(self) -> int:
        return min(self.width, self.height)

    def to_numpy_rgb(self) -> np.ndarray:
        return np.asarray(self.pil)

    def to_numpy_bgr(self) -> np.ndarray:
        """InsightFace/OpenCV trabalham em BGR."""
        return np.asarray(self.pil)[:, :, ::-1].copy()

    def to_png_bytes(self) -> bytes:
        buf = io.BytesIO()
        self.pil.save(buf, format="PNG")
        return buf.getvalue()


def load_image(data: bytes, max_side: int = 1280) -> LoadedImage:
    """Decodifica bytes → RGB com EXIF corrigido e lado máximo limitado."""
    if not data:
        raise ImageDecodeError("arquivo vazio")
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except Exception as exc:  # noqa: BLE001
        raise ImageDecodeError(f"não foi possível decodificar a imagem: {exc}") from exc

    img = ImageOps.exif_transpose(img)  # respeita orientação da câmera
    img = img.convert("RGB")

    w, h = img.size
    scale = max_side / max(w, h)
    if scale < 1.0:
        img = img.resize((round(w * scale), round(h * scale)), Image.LANCZOS)

    return LoadedImage(pil=img, width=img.size[0], height=img.size[1])
