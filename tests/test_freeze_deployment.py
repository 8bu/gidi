"""``freeze_deployment.py``: the v1 rebuild stays byte-identical; a value-head protocol works."""

from __future__ import annotations

import importlib.util
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file

from gidi.inference import GidiPredictor, verify_bundle
from gidi.inference.bundle import MODEL_FILE, load_config
from gidi.modeling.checkpoint import META_FILE, WEIGHTS_FILE

ROOT = Path(__file__).resolve().parents[1]
V1_PROTOCOL = ROOT / "experiments" / "deployment-v1" / "protocol.json"
V1_BUNDLE = ROOT / "models" / "gidi-finance-v1"
# float32 values that are not short decimals, so a lossy JSON round trip would show
TEST_CRF = {
    "start_transitions": np.array([0.1, -0.0047, 1 / 3], dtype=np.float32),
    "end_transitions": np.array([-0.0532, 2 / 3, -0.0791], dtype=np.float32),
    "transitions": (np.arange(9, dtype=np.float32).reshape(3, 3) - 4) / 7,
}

_SCRIPT = ROOT / "scripts" / "freeze_deployment.py"
_SPEC = importlib.util.spec_from_file_location("freeze_deployment", _SCRIPT)
freeze = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(freeze)

needs_sources = pytest.mark.skipif(
    not (V1_BUNDLE.is_dir() and V1_PROTOCOL.is_file()), reason="v1 bundle or protocol missing"
)


def value_protocol(tmp_path: Path, onnx_int8: Path, value_tags: list[str] | None) -> Path:
    """A value-head protocol over the v1 sources, with ``onnx_int8`` as its INT8 model.

    The checkpoint is the v1 one with ``value_tags`` added to its metadata (when given) and a
    weights file holding only the three CRF tensors of a value head; the FP32 ONNX is a copy of
    ``onnx_int8`` (freeze only hashes it and reads its interface).
    """
    protocol = json.loads(V1_PROTOCOL.read_text(encoding="utf-8"))
    onnx_dir = tmp_path / "onnx"
    onnx_dir.mkdir()
    shutil.copyfile(onnx_int8, onnx_dir / MODEL_FILE)
    shutil.copyfile(onnx_int8, onnx_dir / "model.onnx")
    source = Path(protocol["source_checkpoint"]).resolve()
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    for item in source.iterdir():
        if item.name not in (META_FILE, WEIGHTS_FILE):
            (checkpoint / item.name).symlink_to(item)
    save_file(
        {f"value_head.crf.{name}": tensor for name, tensor in TEST_CRF.items()},
        str(checkpoint / WEIGHTS_FILE),
    )
    meta = json.loads((source / META_FILE).read_text(encoding="utf-8"))
    if value_tags is not None:
        meta["value_tags"] = value_tags
    (checkpoint / META_FILE).write_text(json.dumps(meta), encoding="utf-8")
    protocol.update(
        version="gidi-finance-v2-test",
        source_onnx=str(onnx_dir),
        source_checkpoint=str(checkpoint),
        value_head=True,
        value_decoding="crf_viterbi",
    )
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(protocol), encoding="utf-8")
    return path


@needs_sources
def test_default_protocol_rebuild_is_byte_identical_to_the_frozen_v1_bundle(monkeypatch, tmp_path):
    monkeypatch.chdir(ROOT)
    existing = json.loads((V1_BUNDLE / "manifest.json").read_text(encoding="utf-8"))
    built = tmp_path / "gidi-finance-v1"
    freeze.build_bundle(built, existing["created_at"])
    try:
        assert freeze.differences(built, V1_BUNDLE) == []
    finally:
        freeze._remove(built)


@needs_sources
def test_value_head_protocol_builds_a_loadable_v2_bundle(monkeypatch, tmp_path, v2_bundle):
    monkeypatch.chdir(ROOT)
    protocol = value_protocol(tmp_path, v2_bundle / MODEL_FILE, ["O", "B-VALUE", "I-VALUE"])
    built = tmp_path / "gidi-finance-v2-test"
    freeze.build_bundle(built, "2026-01-01T00:00:00Z", protocol)
    try:
        config = load_config(built)
        assert config.value_labels == ("O", "B-VALUE", "I-VALUE")
        assert config.output_names == ("type_logits", "tag_logits", "value_logits")
        manifest = json.loads((built / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["value_labels"] == ["O", "B-VALUE", "I-VALUE"]
        assert manifest["model_version"] == "gidi-finance-v2-test"
        raw = json.loads((built / "config.json").read_text(encoding="utf-8"))
        decoding = raw["value_decoding"]
        assert decoding["method"] == "crf_viterbi"
        for name, tensor in TEST_CRF.items():  # float32 -> JSON -> float64 loses nothing
            assert np.array_equal(np.array(decoding[name], dtype=np.float64), tensor)
        assert raw["architecture"]["kind"] == "dual_encoder"
        assert raw["architecture"]["path_a"]["role"] == "type+target"
        assert manifest["parameters"]["path_b"] == sum(t.size for t in TEST_CRF.values())
        verify_bundle(built)
        assert "value_text" in GidiPredictor.from_bundle(built).predict("cơm tấm 100").to_dict()
    finally:
        freeze._remove(built)


@needs_sources
def test_value_head_protocol_rejects_an_onnx_without_the_value_output(monkeypatch, tmp_path):
    monkeypatch.chdir(ROOT)
    protocol = value_protocol(tmp_path, V1_BUNDLE / MODEL_FILE, ["O", "B-VALUE", "I-VALUE"])
    with pytest.raises(SystemExit, match="unexpected ONNX interface"):
        freeze.build_bundle(tmp_path / "out", "2026-01-01T00:00:00Z", protocol)


@needs_sources
def test_value_head_protocol_rejects_a_checkpoint_without_value_tags(
    monkeypatch, tmp_path, v2_bundle
):
    monkeypatch.chdir(ROOT)
    protocol = value_protocol(tmp_path, v2_bundle / MODEL_FILE, None)
    with pytest.raises(SystemExit, match="value_tags"):
        freeze.build_bundle(tmp_path / "out", "2026-01-01T00:00:00Z", protocol)


@needs_sources
def test_base_checkpoint_must_share_the_tokenizer_with_the_value_checkpoint(
    monkeypatch, tmp_path, v2_bundle
):
    monkeypatch.chdir(ROOT)
    protocol_path = value_protocol(tmp_path, v2_bundle / MODEL_FILE, ["O", "B-VALUE", "I-VALUE"])
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    base = Path(protocol["source_checkpoint"]).parent / "base"
    base.mkdir()
    original = Path(json.loads(V1_PROTOCOL.read_text(encoding="utf-8"))["source_checkpoint"])
    for item in original.iterdir():
        (base / item.name).symlink_to(item.resolve())
    checkpoint = Path(protocol["source_checkpoint"])
    meta_path = checkpoint / META_FILE
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["frozen_base"] = {"model_safetensors_sha256": freeze.sha256_file(base / WEIGHTS_FILE)}
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    protocol["base_checkpoint"] = str(base)
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
    (checkpoint / "tokenizer.json").unlink()
    (checkpoint / "tokenizer.json").write_text(
        (original / "tokenizer.json").read_text(encoding="utf-8") + " ", encoding="utf-8"
    )
    with pytest.raises(SystemExit, match="differs between the base checkpoint"):
        freeze.build_bundle(tmp_path / "out", "2026-01-01T00:00:00Z", protocol_path)


def test_rule_parser_protocol_pins_the_parser_version_and_the_word_snap():
    from gidi import value_parser

    rules = freeze._runtime_rules({"value_source": "rule-parser", "target_snap": "words"}, False)
    assert rules == {
        "value_source": "rule-parser",
        "value_parser": {"name": "gidi.value_parser", "version": value_parser.VERSION},
        "target_snap": "words",
    }
    assert freeze._runtime_rules({}, False) == {}  # v1/v2 bundles stay byte-identical


@pytest.mark.parametrize(
    ("protocol", "value_head"),
    [
        ({"value_source": "rule-parser"}, False),  # snap is required with the rule parser
        ({"value_source": "rule-parser", "target_snap": "chars"}, False),
        ({"value_source": "crf", "target_snap": "words"}, False),
        ({"value_source": "rule-parser", "target_snap": "words"}, True),  # excludes a value head
        ({"target_snap": "words"}, False),  # snap without a declared value source
    ],
)
def test_inconsistent_rule_parser_protocols_are_rejected(protocol, value_head):
    with pytest.raises(SystemExit):
        freeze._runtime_rules(protocol, value_head)
