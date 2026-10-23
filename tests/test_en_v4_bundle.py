"""The gidi-en-v4 bundle contract: argmax value head, value snap, WordPiece tokenizer type."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from gidi.inference import BundleError, GidiPredictor
from gidi.inference.bundle import load_config
from gidi.inference.decode import decode_value_argmax
from gidi.modeling.value_decode import decode_value
from gidi.release.build import PREDICTION_KEYS, runtime_from_config
from gidi.release.manifest import runtime_keys

ROOT = Path(__file__).resolve().parents[1]
V4_BUNDLE = ROOT / "models" / "gidi-en-v4-bundle"

CONFIG = {
    "model_version": "gidi-en-v4",
    "types": ["expense", "income", "borrow", "lend", "repayment_in", "repayment_out", "transfer"]
    + ["refund"],
    "tags": ["O", "B-TARGET", "I-TARGET"],
    "target_snap": "words",
    "max_length": 32,
    "onnx": {
        "inputs": ["input_ids", "attention_mask"],
        "outputs": ["type_logits", "tag_logits", "value_logits"],
        "opset": 17,
    },
    "value_labels": ["O", "B-VALUE", "I-VALUE"],
    "value_decoding": {"method": "argmax", "span_selection": "highest_confidence"},
    "value_snap": "words",
    "tokenizer_type": "wordpiece",
}


def write(tmp_path: Path, **changes: object) -> Path:
    config = {**CONFIG, **changes}
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    return tmp_path


def test_config_declares_argmax_value_snap_and_wordpiece(tmp_path):
    config = load_config(write(tmp_path))
    assert config.value_labels == ("O", "B-VALUE", "I-VALUE")
    assert config.value_decoding is not None and config.value_decoding.method == "argmax"
    assert config.value_decoding.transitions is None
    assert config.value_snap == "words" and config.tokenizer_type == "wordpiece"


def test_a_bundle_without_the_new_keys_keeps_the_bpe_defaults(tmp_path):
    config = {k: v for k, v in CONFIG.items() if k != "tokenizer_type"}
    for key in ("value_labels", "value_decoding", "value_snap"):
        del config[key]
    config["onnx"] = {**CONFIG["onnx"], "outputs": ["type_logits", "tag_logits"]}
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    loaded = load_config(tmp_path)
    assert loaded.tokenizer_type == "bpe" and loaded.value_snap is None
    assert loaded.value_labels is None and loaded.value_decoding is None


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"tokenizer_type": "unigram"}, "tokenizer_type"),
        ({"value_snap": "chars"}, "value_snap"),
        (
            {
                "value_decoding": {
                    "method": "argmax",
                    "span_selection": "highest_confidence",
                    "start_transitions": [0, 0, 0],
                }
            },
            "argmax takes no CRF",
        ),
        ({"value_decoding": {"method": "beam", "span_selection": "highest_confidence"}}, "method"),
        ({"value_decoding": {"method": "argmax", "span_selection": "first"}}, "span_selection"),
    ],
)
def test_bad_contracts_are_refused(tmp_path, changes, message):
    with pytest.raises(BundleError, match=message):
        load_config(write(tmp_path, **changes))


def test_value_snap_needs_a_value_head(tmp_path):
    config = {k: v for k, v in CONFIG.items() if k not in ("value_labels", "value_decoding")}
    config["onnx"] = {**CONFIG["onnx"], "outputs": ["type_logits", "tag_logits"]}
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(BundleError, match="value_snap needs a value head"):
        load_config(tmp_path)


def test_runtime_block_records_both_snaps_and_the_tokenizer():
    runtime = runtime_from_config(CONFIG)
    assert runtime["value_labels"] == ["O", "B-VALUE", "I-VALUE"]
    assert runtime["value_decoding"] == "argmax"
    assert runtime["target_snap"] == "words" and runtime["value_snap"] == "words"
    assert runtime["tokenizer_type"] == "wordpiece"
    assert runtime["prediction_keys"] == list(PREDICTION_KEYS)
    assert runtime_keys(runtime)[-3:] == ("target_snap", "value_snap", "tokenizer_type")


def test_runtime_argmax_decode_equals_the_training_side_decode():
    rng = np.random.default_rng(7)
    text = "lent bro $20 for 2 beers"
    offsets = [(0, 0)]
    for word in text.split(" "):
        start = text.index(word, offsets[-1][1])
        offsets.append((start, start + len(word)))
    offsets.append((0, 0))
    real = [False] + [True] * (len(offsets) - 2) + [False]
    for _ in range(200):
        logits = rng.normal(size=(len(offsets), 3)) * 3
        mine = decode_value_argmax(logits, offsets, text, real)
        theirs = decode_value(logits, offsets, text, real)
        assert mine == theirs


@pytest.mark.skipif(not V4_BUNDLE.is_dir(), reason="gidi-en-v4 bundle not built")
def test_v4_predictor_reads_the_amount_and_snaps_it_to_whole_words():
    predictor = GidiPredictor.from_bundle(V4_BUNDLE)
    assert predictor.has_value_head
    for text, amount in [
        ("venmo'd mom 50", "50"),
        ("coffee 5k", "5k"),
        ("2 grand for the car", "2 grand"),
        ("twenty bucks to sam", "twenty bucks"),
    ]:
        out = predictor.predict(text).to_dict()
        start, end = out["value_span"]
        assert text[start:end] == out["value_text"] == amount
        assert 0.0 < out["value_confidence"] <= 1.0
    assert predictor.predict("paid rent").to_dict()["value_text"] is None
    # the snap never cuts a word: the span edges sit on whitespace or the note's ends
    for text in ["lent bro $20, thanks", "got paid 2,400!", "uber 12.50/ride"]:
        out = predictor.predict(text).to_dict()
        if out["value_span"] is not None:
            start, end = out["value_span"]
            assert start == 0 or text[start - 1].isspace() or not text[start - 1].isalnum()
            assert end == len(text) or text[end].isspace() or not text[end].isalnum()
