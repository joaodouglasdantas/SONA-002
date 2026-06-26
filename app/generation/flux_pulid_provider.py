"""Provider Flux + PuLID via Replicate (bytedance/flux-pulid).

Preserva identidade a partir de UMA foto (PuLID injeta o ID no FLUX.1-dev). Nao
aplica LoRA - o estilo The Sims vem do prompt (gatilho THSMS). Funciona ja com o
token do Replicate, sem GPU propria.

Para combinar com a LoRA Sims, use o provider `comfyui_replicate`.
"""
from __future__ import annotations

import base64

from ..config import Settings
from .base import (GenerationError, GenerationProvider, GenerationRequest,
                   GenerationResult)
from .replicate_provider import _read_output_png, resolve_ref


class FluxPulidProvider(GenerationProvider):
    name = "flux_pulid"

    def __init__(self, settings: Settings):
        self.s = settings
        if not settings.replicate_api_token:
            raise GenerationError(
                "REPLICATE_API_TOKEN nao configurado (.env). Necessario para gerar.")

    def generate(self, req: GenerationRequest) -> GenerationResult:
        import replicate

        client = replicate.Client(api_token=self.s.replicate_api_token)
        data_uri = "data:image/png;base64," + base64.b64encode(req.image_png).decode()

        model_input = {
            "main_face_image": data_uri,
            "prompt": req.prompt,
            "negative_prompt": req.negative_prompt,
            "num_steps": req.num_steps,
            "guidance_scale": req.guidance_scale,
            "id_weight": min(3.0, max(0.0, req.identitynet_strength * 1.15)),
            "start_step": int(req.extra.get("start_step", 0)),
            "true_cfg": float(req.extra.get("true_cfg", 1.0)),
            "output_format": "png",
        }
        if req.seed is not None:
            model_input["seed"] = req.seed

        try:
            ref = resolve_ref(client, self.s.flux_pulid_model)
            output = client.run(ref, input=model_input)
        except Exception as exc:  # noqa: BLE001
            raise GenerationError(f"falha na chamada ao flux-pulid: {exc}") from exc

        png = _read_output_png(output)
        if png is None:
            raise GenerationError("flux-pulid nao retornou imagem utilizavel.")

        return GenerationResult(image_png=png, provider=self.name, seed=req.seed,
                                meta={"model": self.s.flux_pulid_model})
