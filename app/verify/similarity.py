"""Verificação de fidelidade: similaridade de cosseno entre embeddings ArcFace.

Compara a identidade da foto original com a da imagem gerada. Embeddings ArcFace
já vêm L2-normalizados, então o produto interno é o cosseno em [-1, 1].
"""
from __future__ import annotations

import numpy as np


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))
