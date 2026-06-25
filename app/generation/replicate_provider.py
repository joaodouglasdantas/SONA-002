"""Provider de geração usando Replicate (InstantID sobre SDXL).

InstantID preserva identidade a partir de UMA foto (sem treino) e mantém
pose/enquadramento pelos keypoints faciais. Aqui só passamos a imagem + prompts;
o modelo extrai o embedding e os keypoints internamente.

Import preguiçoso de `replicate`. Requer REPLICATE_API_TOKEN no ambiente.
"""
from __future__ import annotations

import base64
import io

from ..config import Settings
from .base import (GenerationError, GenerationProvider, GenerationRequest,
                   GenerationResult)


class ReplicateProvider(GenerationProvider):
    name = "replicate"

    def __init__(self, settings: Settings):
        self.s = settings
        if not settings.replicate_api_token:
            raise GenerationError(
                "REPLICATE_API_TOKEN não configurado (.env). Necessário para gerar.")

    def generate(self, req: GenerationRequest) -> GenerationResult:
        import replicate  # adiado

        client = replicate.Client(api_token=self.s.replicate_api_token)

        data_uri = "data:image/png;base64," + base64.b64encode(req.image_png).decode()
        model_input = {
            "image": data_uri,
            "prompt": req.prompt,
            "negative_prompt": req.negative_prompt,
            "ip_adapter_scale": req.adapter_strength,
            "controlnet_conditioning_scale": req.identitynet_strength,
            "guidance_scale": req.guidance_scale,
            "num_inference_steps": req.num_steps,
        }
        if req.seed is not None:
            model_input["seed"] = req.seed
        if req.style_lora:
            model_input["lora"] = req.style_lora

        try:
            output = client.run(self.s.replicate_model, input=model_input)
        except Exception as exc:  # noqa: BLE001
            raise GenerationError(f"falha na chamada ao Replicate: {exc}") from exc

        png = _read_output_png(output)
        if png is None:
            raise GenerationError("Replicate não retornou imagem utilizável.")

        return GenerationResult(image_png=png, provider=self.name, seed=req.seed,
                                meta={"model": self.s.replicate_model})


def _read_output_png(output) -> bytes | None:
    """Normaliza as várias formas de saída do Replicate (lista, url, file-like)."""
    item = output[0] if isinstance(output, (list, tuple)) and output else output
    if item is None:
        return None
    # objeto file-like (replicate>=0.25 retorna FileOutput com .read())
    if hasattr(item, "read"):
        return item.read()
    # URL (str)
    if isinstance(item, str) and item.startswith("http"):
        import urllib.request
        with urllib.request.urlopen(item) as resp:  # noqa: S310
            return resp.read()
    if isinstance(item, (bytes, bytearray)):
        return bytes(item)
    if isinstance(item, io.IOBase):
        return item.read()
    return None
