"""Testes do orquestrador: rejeição, sucesso, loop de fidelidade."""
import numpy as np

from app.config import Settings
from app.face.validator import FaceValidator
from app.pipeline import Pipeline

from .fakes import FakeDetector, FakeProvider, FakeRealness, make_face, png_bytes

IMG = png_bytes(size=(400, 400))
ONES = np.ones(512)
ORTH = np.array([1, -1] + [0] * 510, dtype=np.float32)  # cosseno 0 com ONES


def _pipeline(detector_seq, provider, realness=None, **overrides):
    s = Settings(**overrides)
    det = FakeDetector(detector_seq)
    validator = FaceValidator(s, detector=det, realness=realness or FakeRealness())
    return Pipeline(settings=s, validator=validator, provider=provider, detector=det)


def test_rejection_passthrough():
    p = _pipeline([[]], FakeProvider())
    r = p.run(IMG)
    assert not r.accepted
    assert r.image_png is None


def test_success_high_fidelity_single_attempt():
    face = make_face((100, 100, 300, 300), emb=ONES)
    # call0: validação | call1: pós-geração (mesma identidade → sim=1.0)
    p = _pipeline([[face], [face]], FakeProvider())
    r = p.run(IMG, seed=42)
    assert r.accepted and r.fidelity_ok
    assert r.attempts == 1
    assert r.similarity > 0.99
    assert r.image_png is not None


def test_low_fidelity_triggers_retries():
    val_face = make_face((100, 100, 300, 300), emb=ONES)
    gen_face = make_face((100, 100, 300, 300), emb=ORTH)  # identidade diferente
    # validação + repetições de pós-geração (FakeDetector repete o último)
    p = _pipeline([[val_face], [gen_face]], FakeProvider(), max_retries=2)
    r = p.run(IMG, seed=1)
    assert r.accepted               # foto válida, mas...
    assert not r.fidelity_ok        # ...identidade não bate
    assert r.attempts == 3          # max_retries + 1


def test_provider_receives_sims3_prompt_and_escalating_identity():
    val_face = make_face((100, 100, 300, 300), emb=ONES)
    gen_face = make_face((100, 100, 300, 300), emb=ORTH)
    prov = FakeProvider()
    p = _pipeline([[val_face], [gen_face]], prov, max_retries=2)
    p.run(IMG, seed=7)
    assert all("Sims 3" in req.prompt for req in prov.requests)
    strengths = [req.identitynet_strength for req in prov.requests]
    assert strengths == sorted(strengths)        # reforça identidade a cada retry
    assert strengths[0] < strengths[-1]
