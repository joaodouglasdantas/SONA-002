"""Provider self-hosted via ComfyUI (caminho de escala / custo zero por imagem).

Esboço funcional documentado: envia um workflow InstantID para uma instância
ComfyUI rodando numa GPU própria. Mantém a MESMA interface do ReplicateProvider,
então o pipeline não muda — só a config `SONA_PROVIDER=comfyui`.

Para ativar: subir o ComfyUI com os nós InstantID + SDXL e preencher um
workflow JSON em `workflows/instantid.json` (parametrizado por prompt/imagem).
"""
from __future__ import annotations

from ..config import Settings
from .base import (GenerationError, GenerationProvider, GenerationRequest,
                   GenerationResult)


class ComfyUIProvider(GenerationProvider):
    name = "comfyui"

    def __init__(self, settings: Settings):
        self.s = settings
        if not settings.comfyui_url:
            raise GenerationError("SONA_COMFYUI_URL não configurado (.env).")

    def generate(self, req: GenerationRequest) -> GenerationResult:
        # Implementação de produção:
        #   1) carregar workflow InstantID (JSON) e injetar prompt/imagem/strengths
        #   2) POST {comfyui_url}/prompt  → obter prompt_id
        #   3) poll {comfyui_url}/history/{prompt_id} até concluir
        #   4) baixar a imagem via {comfyui_url}/view
        raise GenerationError(
            "ComfyUIProvider é um esboço. Implemente o envio do workflow InstantID "
            "para a sua instância ComfyUI antes de usar SONA_PROVIDER=comfyui.")
