"""Held-out leakage invariant of the annotation-v3 encoder retrain set (training-v1)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "build_annotation_v3_training", REPO_ROOT / "scripts" / "build_annotation_v3_training.py"
)
bt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bt)


def test_committed_set_is_reproducible_and_leak_free():
    # --check rebuilds in memory (the leakage gate runs inside) and compares the files on disk.
    assert bt.main(["--root", str(REPO_ROOT), "--check"]) == 0


def test_gate_rejects_held_out_id_and_nfc_lowercase_text():
    sets = {"human-value-01": [{"id": "hv-1", "text": "Còn nợ Hùng 300k"}]}
    clean = {"id": "a", "text": "mua cơm 30k"}
    bt.assert_no_leakage([clean], sets, [])
    with pytest.raises(ValueError, match="id in human-value-01"):
        bt.assert_no_leakage([{"id": "hv-1", "text": "khác"}], sets, [])
    decomposed = "Co\u0300n nợ Hùng 300K "  # NFD, other case, trailing space
    with pytest.raises(ValueError, match="text in human-value-01"):
        bt.assert_no_leakage([{"id": "b", "text": decomposed}], sets, [])
    with pytest.raises(ValueError, match="excluded id in train"):
        bt.assert_no_leakage([{"id": "hv-1", "text": "khác"}], {"x": []} | sets, ["hv-1"])
