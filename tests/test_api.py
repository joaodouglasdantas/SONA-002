"""Teste de fiação da API (sem IA): injeta um Pipeline dublê."""
import io

import pytest
from PIL import Image

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

import app.main as main  # noqa: E402
from app.pipeline import PipelineResult  # noqa: E402
from app.face.validator import RejectReason  # noqa: E402


def _png():
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), (120, 120, 120)).save(buf, format="PNG")
    return buf.getvalue()


class StubPipeline:
    def __init__(self, result):
        self._result = result

    def run(self, data, extra_style="", seed=None):
        return self._result


def _client(result):
    main._pipeline = StubPipeline(result)
    return TestClient(main.app)


def test_health():
    c = TestClient(main.app)
    r = c.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_transform_success():
    res = PipelineResult(accepted=True, image_png=b"\x89PNG_fake",
                         similarity=0.82, fidelity_ok=True, attempts=1,
                         provider="fake", num_faces=1, realness_score=0.9)
    c = _client(res)
    r = c.post("/transform", files={"file": ("a.png", _png(), "image/png")})
    assert r.status_code == 200
    body = r.json()
    assert body["accepted"] and body["fidelity_ok"] and body["similarity"] == 0.82
    assert "image_base64" in body


def test_transform_rejected_returns_422():
    res = PipelineResult(accepted=False, reason=RejectReason.NOT_REAL_HUMAN,
                         message="não é humano real", num_faces=0)
    c = _client(res)
    r = c.post("/transform", files={"file": ("a.png", _png(), "image/png")})
    assert r.status_code == 422
    assert r.json()["reason"] == "not_real_human"


def test_transform_unsupported_type():
    c = TestClient(main.app)
    r = c.post("/transform", files={"file": ("a.gif", b"GIF89a", "image/gif")})
    assert r.status_code == 415
