"""Testes do porteiro de validação (B1–B4)."""
import numpy as np

from app.config import Settings
from app.face.validator import FaceValidator, RejectReason
from app.utils.image_io import load_image

from .fakes import FakeDetector, FakeRealness, make_face, png_bytes


def _img():
    return load_image(png_bytes(size=(400, 400)), max_side=1280)


def _validator(sequence, realness=None, **overrides):
    s = Settings(**overrides)
    return FaceValidator(s, detector=FakeDetector(sequence),
                         realness=realness or FakeRealness())


def test_no_face_rejected():
    v = _validator([[]])
    r = v.validate(_img())
    assert not r.accepted and r.reason == RejectReason.NO_FACE


def test_low_confidence_rejected():
    face = make_face((100, 100, 300, 300), score=0.2)
    r = _validator([[face]]).validate(_img())
    assert not r.accepted and r.reason == RejectReason.LOW_CONFIDENCE


def test_face_too_small_rejected():
    face = make_face((10, 10, 40, 40), score=0.9)  # ~30px num lado de 400
    r = _validator([[face]]).validate(_img())
    assert not r.accepted and r.reason == RejectReason.FACE_TOO_SMALL


def test_not_real_human_rejected():
    face = make_face((100, 100, 300, 300), score=0.9)
    v = _validator([[face]], realness=FakeRealness(is_real=False, score=0.1))
    r = v.validate(_img())
    assert not r.accepted and r.reason == RejectReason.NOT_REAL_HUMAN


def test_ambiguous_multiface_rejected():
    a = make_face((20, 100, 180, 300), score=0.9)    # área semelhante
    b = make_face((220, 100, 380, 300), score=0.9)
    r = _validator([[a, b]]).validate(_img())
    assert not r.accepted and r.reason == RejectReason.AMBIGUOUS_MULTIFACE


def test_primary_face_selected_when_one_dominates():
    big = make_face((50, 50, 350, 350), score=0.95, emb=np.ones(512))
    small = make_face((360, 360, 395, 395), score=0.9, emb=np.zeros(512) + 0.1)
    r = _validator([[big, small]]).validate(_img())
    assert r.accepted
    assert r.primary_face.bbox == big.bbox  # o maior/central vence


def test_valid_face_accepted():
    face = make_face((100, 100, 300, 300), score=0.9)
    r = _validator([[face]]).validate(_img())
    assert r.accepted and r.reason is None and r.realness_score == 0.9
