import pytest
import torch

from gidi import device as dev


def _fake(monkeypatch, *, mps: bool, cuda: bool):
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)


@pytest.mark.parametrize(
    ("mps", "cuda", "expected"),
    [(True, True, "mps"), (False, True, "cuda"), (False, False, "cpu")],
)
def test_auto_prefers_mps_then_cuda_then_cpu(monkeypatch, mps, cuda, expected):
    _fake(monkeypatch, mps=mps, cuda=cuda)
    assert dev.resolve_device().type == expected


def test_explicit_unavailable_device_raises(monkeypatch):
    _fake(monkeypatch, mps=False, cuda=False)
    with pytest.raises(RuntimeError, match="mps"):
        dev.resolve_device("mps")


def test_explicit_cpu_always_works(monkeypatch):
    _fake(monkeypatch, mps=True, cuda=True)
    assert dev.resolve_device("CPU").type == "cpu"


def test_unknown_preference_rejected():
    with pytest.raises(ValueError):
        dev.resolve_device("tpu")
