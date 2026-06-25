"""Classificador 'rosto humano REAL?' via CLIP zero-shot.

Detector facial dispara também em cartoons, renders 3D e personagens de jogo,
então a detecção sozinha não cumpre o requisito. Aqui comparamos a imagem (ou o
recorte do rosto) contra dois grupos de rótulos textuais e devolvemos a
probabilidade de ser FOTO REAL de humano.

Import preguiçoso de torch/open_clip. Plugável: pode ser trocado por um modelo
anti-spoofing/liveness dedicado mantendo a mesma interface `RealnessClassifier`.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

# Rótulos positivos (foto real de humano) e negativos (tudo que deve ser rejeitado).
REAL_PROMPTS = [
    "a real photograph of a human face",
    "a candid photo of a real person",
    "a selfie of a real human being",
]
FAKE_PROMPTS = [
    "a drawing or illustration of a face",
    "a 3d rendered character",
    "a video game character",
    "an anime or cartoon character",
    "a painting of a person",
    "a photo of an animal",
    "a statue or doll face",
    "an object, a landscape, or text",
]


@dataclass
class RealnessResult:
    is_real: bool
    score: float          # P(foto real) em [0, 1]
    threshold: float


class RealnessClassifier:
    def __init__(self, device: str = "cpu", model_name: str = "ViT-B-32",
                 pretrained: str = "laion2b_s34b_b79k"):
        self.device = device
        self.model_name = model_name
        self.pretrained = pretrained
        self._model = None
        self._preprocess = None
        self._text_features = None

    def _ensure_loaded(self):
        if self._model is not None:
            return
        import torch
        import open_clip

        model, _, preprocess = open_clip.create_model_and_transforms(
            self.model_name, pretrained=self.pretrained, device=self.device
        )
        model.eval()
        tokenizer = open_clip.get_tokenizer(self.model_name)

        prompts = REAL_PROMPTS + FAKE_PROMPTS
        with torch.no_grad():
            tokens = tokenizer(prompts).to(self.device)
            tf = model.encode_text(tokens)
            tf /= tf.norm(dim=-1, keepdim=True)

        self._model = model
        self._preprocess = preprocess
        self._text_features = tf
        self._n_real = len(REAL_PROMPTS)

    def score(self, image: Image.Image, threshold: float = 0.65) -> RealnessResult:
        self._ensure_loaded()
        import torch

        with torch.no_grad():
            x = self._preprocess(image).unsqueeze(0).to(self.device)
            feat = self._model.encode_image(x)
            feat /= feat.norm(dim=-1, keepdim=True)
            logits = (100.0 * feat @ self._text_features.T).softmax(dim=-1)[0]
            real_p = float(logits[: self._n_real].sum().item())

        return RealnessResult(is_real=real_p >= threshold, score=real_p,
                              threshold=threshold)
