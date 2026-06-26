"""Provider Flux + PuLID + LoRA Sims via ComfyUI rodando no Replicate.

Usa `fofr/any-comfyui-workflow`, que executa um workflow ComfyUI arbitrário na
nuvem (sem GPU própria). Assim conseguimos combinar, numa só chamada:
  - FLUX.1-dev
  - PuLID-Flux (preservação de identidade pela foto)
  - a LoRA de estilo The Sims

Como funciona aqui:
  1. carregamos um TEMPLATE de workflow (API-format do ComfyUI) com placeholders;
  2. injetamos prompt, negative, nome do arquivo de imagem, LoRA e seed;
  3. enviamos a foto como `input_file` e o JSON como `workflow_json`.

IMPORTANTE: o template em `workflows/flux_pulid_lora.json` é um PONTO DE PARTIDA.
Valide-o no seu ComfyUI (Save → API Format) garantindo que os nós PuLID-Flux e o
LoraLoader existam na instância do runner. O provider é agnóstico ao workflow:
basta o JSON conter os placeholders abaixo.
"""
from __future__ import annotations

import io
import json
from pathlib import Path

from ..config import Settings
from .base import (GenerationError, GenerationProvider, GenerationRequest,
                   GenerationResult)
from .replicate_provider import _read_output_png, resolve_ref

# placeholders reconhecidos dentro do template de workflow
PH_PROMPT = "{{PROMPT}}"
PH_NEGATIVE = "{{NEGATIVE}}"
PH_IMAGE = "{{IMAGE}}"
PH_LORA = "{{LORA}}"
PH_SEED = "{{SEED}}"

INPUT_FILENAME = "sona_input.png"  # nome com que a foto entra no ComfyUI


def _inject(obj, mapping: dict):
    """Substitui placeholders recursivamente em strings do workflow."""
    if isinstance(obj, dict):
        return {k: _inject(v, mapping) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_inject(v, mapping) for v in obj]
    if isinstance(obj, str):
        out = obj
        for ph, val in mapping.items():
            if ph in out:
                # placeholder de seed vira número inteiro se for o valor todo
                if ph == PH_SEED and out == ph:
                    return int(val)
                out = out.replace(ph, str(val))
        return out
    return obj


class ComfyUIReplicateProvider(GenerationProvider):
    name = "comfyui_replicate"

    def __init__(self, settings: Settings):
        self.s = settings
        if not settings.replicate_api_token:
            raise GenerationError(
                "REPLICATE_API_TOKEN não configurado (.env). Necessário para gerar.")
        self.workflow_path = (
            settings.comfyui_workflow_path
            or str(Path(__file__).resolve().parents[2] / "workflows" / "flux_pulid_lora.json")
        )

    def _build_workflow(self, req: GenerationRequest) -> str:
        path = Path(self.workflow_path)
        if not path.exists():
            raise GenerationError(
                f"workflow ComfyUI não encontrado: {path}. "
                "Exporte um workflow Flux+PuLID+LoRA do ComfyUI (API Format).")
        template = json.loads(path.read_text(encoding="utf-8"))
        mapping = {
            PH_PROMPT: req.prompt,
            PH_NEGATIVE: req.negative_prompt,
            PH_IMAGE: INPUT_FILENAME,
            PH_LORA: req.style_lora or self.s.style_lora,
            PH_SEED: req.seed if req.seed is not None else 0,
        }
        return json.dumps(_inject(template, mapping))

    def generate(self, req: GenerationRequest) -> GenerationResult:
        import replicate

        client = replicate.Client(api_token=self.s.replicate_api_token)
        workflow_json = self._build_workflow(req)

        # a foto precisa chegar com o nome esperado pelo nó LoadImage do workflow
        face_file = io.BytesIO(req.image_png)
        face_file.name = INPUT_FILENAME

        model_input = {
            "workflow_json": workflow_json,
            "input_file": face_file,
            "randomise_seeds": req.seed is None,
            "return_temp_files": False,
        }

        try:
            ref = resolve_ref(client, self.s.comfyui_replicate_model)
            output = client.run(ref, input=model_input)
        except Exception as exc:  # noqa: BLE001
            raise GenerationError(f"falha na chamada ao any-comfyui-workflow: {exc}") from exc

        png = _read_output_png(output)
        if png is None:
            raise GenerationError("any-comfyui-workflow não retornou imagem utilizável.")

        return GenerationResult(image_png=png, provider=self.name, seed=req.seed,
                                meta={"model": self.s.comfyui_replicate_model,
                                      "workflow": str(self.workflow_path)})
