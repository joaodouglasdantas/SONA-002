"""Testes dos providers Flux (com o SDK do Replicate mockado)."""
import json
import sys
import types

from app.config import Settings
from app.generation.base import GenerationRequest, get_provider
from app.generation.comfyui_replicate_provider import (PH_IMAGE, PH_LORA,
                                                       PH_PROMPT, PH_SEED)

from .fakes import png_bytes


def _fake_replicate(capture: dict, output):
    """Injeta um modulo 'replicate' falso que captura o input e devolve `output`."""
    mod = types.ModuleType("replicate")

    class _Ver:
        id = "fakehash123"

    class _Model:
        latest_version = _Ver()

    class _Models:
        def get(self, ref):
            return _Model()

    class Client:
        def __init__(self, api_token=None):
            capture["token"] = api_token
            self.models = _Models()

        def run(self, model, input=None):
            capture["model"] = model
            capture["input"] = input
            return output

    mod.Client = Client
    sys.modules["replicate"] = mod
    return mod


def _req():
    return GenerationRequest(
        image_png=png_bytes(), prompt="video game screenshot in the style of THSMS, sims",
        negative_prompt="blurry", identitynet_strength=0.85, seed=123,
        style_lora="dvyio/flux-lora-the-sims",
    )


def test_factory_selects_flux_pulid():
    s = Settings(provider="flux_pulid", replicate_api_token="x")
    assert get_provider(s).name == "flux_pulid"


def test_flux_pulid_builds_input_and_reads_output():
    cap = {}
    _fake_replicate(cap, [b"\x89PNGfake"])
    s = Settings(provider="flux_pulid", replicate_api_token="tok")
    prov = get_provider(s)
    res = prov.generate(_req())

    assert res.image_png == b"\x89PNGfake"
    inp = cap["input"]
    assert inp["main_face_image"].startswith("data:image/png;base64,")
    assert inp["seed"] == 123
    assert abs(inp["id_weight"] - 0.85 * 1.15) < 1e-6
    assert cap["token"] == "tok"
    # versao resolvida automaticamente (owner/model:hash)
    assert cap["model"].endswith(":fakehash123")


def test_comfyui_workflow_injection_no_placeholders_left():
    s = Settings(provider="comfyui_replicate", replicate_api_token="tok")
    prov = get_provider(s)
    workflow = prov._build_workflow(_req())

    for ph in (PH_PROMPT, PH_LORA, PH_IMAGE, "{{NEGATIVE}}"):
        assert ph not in workflow
    data = json.loads(workflow)
    assert data["20"]["inputs"]["lora_name"] == "dvyio/flux-lora-the-sims"
    assert data["33"]["inputs"]["image"] == "sona_input.png"
    assert data["40"]["inputs"]["text"].startswith("video game screenshot")
    assert data["60"]["inputs"]["seed"] == 123


def test_comfyui_generate_sends_workflow_and_file():
    cap = {}
    _fake_replicate(cap, b"\x89PNGfake2")
    s = Settings(provider="comfyui_replicate", replicate_api_token="tok")
    prov = get_provider(s)
    res = prov.generate(_req())

    assert res.image_png == b"\x89PNGfake2"
    assert cap["model"].startswith("fofr/any-comfyui-workflow")
    inp = cap["input"]
    assert "workflow_json" in inp and isinstance(inp["workflow_json"], str)
    assert getattr(inp["input_file"], "name", None) == "sona_input.png"
