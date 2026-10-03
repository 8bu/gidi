"""``verify_deployment.py``: prediction-dict validation and per-protocol configuration.

The full verification needs the trained checkpoints and takes minutes, so it is run by hand
(``uv run python scripts/verify_deployment.py``); these tests cover the parts that decide what
counts as a valid v1 / value-head prediction.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_script():
    spec = importlib.util.spec_from_file_location(
        "verify_deployment_under_test", ROOT / "scripts" / "verify_deployment.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


V1 = {
    "type": "expense",
    "type_confidence": 0.9,
    "target": "Mai",
    "target_span": [3, 6],
    "target_confidence": 0.8,
    "truncated": False,
    "model_version": "gidi-finance-v1",
}
TEXT = "ăn Mai 100k"
VALUE = {"value_text": "100k", "value_span": [7, 11], "value_confidence": 0.7}


@pytest.fixture
def verify():
    return load_script()


def test_default_module_state_is_the_v1_setup(verify):
    assert (verify.MODEL_VERSION, verify.HAS_VALUE, verify.FREEZE_PROTOCOL_ARGS) == (
        "gidi-finance-v1",
        False,
        [],
    )
    assert verify.BUNDLE == ROOT / "models" / "gidi-finance-v1"
    assert verify.validate_prediction_dict(TEXT, V1) == []
    assert verify.validate_prediction_dict(TEXT, {**V1, **VALUE})  # value keys are not v1 keys


def test_configure_switches_to_another_version_and_its_value_head(verify, tmp_path):
    protocol = tmp_path / "experiments" / "deployment-vx" / "protocol.json"
    protocol.parent.mkdir(parents=True)
    protocol.write_text(
        json.dumps(
            {
                "version": "gidi-finance-vx",
                "source_checkpoint": "models/some/checkpoint",
                "source_onnx": "models/some-onnx",
                "value_head": True,
                "int8_size_mb": 29.5,
            }
        ),
        encoding="utf-8",
    )
    verify.configure(protocol)
    assert verify.MODEL_VERSION == "gidi-finance-vx"
    assert verify.EXPERIMENT == "deployment-vx"
    assert protocol.parent == verify.DEPLOY_DIR
    assert verify.BUNDLE == ROOT / "models" / "gidi-finance-vx"
    assert verify.CHECKPOINT == ROOT / "models/some/checkpoint"
    assert verify.FP32_ONNX == ROOT / "models/some-onnx/model.onnx"
    assert verify.INT8_SIZE_MB == 29.5
    assert ["--protocol", str(protocol)] == verify.FREEZE_PROTOCOL_ARGS
    assert verify.prediction_keys() == verify.PREDICTION_KEYS | verify.VALUE_KEYS


def test_value_head_predictions_are_validated_against_the_callers_string(verify):
    verify.HAS_VALUE = True
    verify.MODEL_VERSION = "gidi-finance-v1"
    good = {**V1, **VALUE}
    assert verify.validate_prediction_dict(TEXT, good) == []
    assert verify.validate_prediction_dict(TEXT, V1)  # missing value keys
    assert verify.validate_prediction_dict(TEXT, {**good, "value_text": None})  # null mismatch
    assert verify.validate_prediction_dict(TEXT, {**good, "value_text": "100"})  # not the slice
    assert verify.validate_prediction_dict(TEXT, {**good, "value_span": [7, 99]})  # out of range
    assert verify.validate_prediction_dict(TEXT, {**good, "value_confidence": 1.5})
    padded = {**good, "value_text": " 100k", "value_span": [6, 11]}
    assert verify.validate_prediction_dict(TEXT, padded)  # whitespace around the span
    none = {**good, "value_text": None, "value_span": None}
    assert verify.validate_prediction_dict(TEXT, none) == []


def test_prediction_key_compares_the_value_span_only_when_there_is_one(verify):
    assert verify.pred_key(V1) == ("expense", (3, 6), "Mai")
    a, b = {**V1, **VALUE}, {**V1, **VALUE, "value_span": [7, 10], "value_text": "100"}
    assert verify.pred_key(a) != verify.pred_key(b)


def test_reports_with_lone_surrogate_inputs_stay_utf8_encodable_and_round_trip(verify):
    text = verify.dumps_json({"text": "ăn \ud83d 35k"}, indent=2)
    text.encode("utf-8")  # a raw surrogate would raise here
    assert json.loads(text) == {"text": "ăn \ud83d 35k"}
