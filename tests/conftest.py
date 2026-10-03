"""Shared fixtures.

``v2_bundle`` is a throwaway *value-head* bundle built at test time from the frozen v1 bundle, so
the v2 runtime path can be tested without a trained value head: the v1 INT8 graph gets one extra
output, ``value_logits``, defined as ``tag_logits`` shifted one token to the left (padded with
zero logits, i.e. ``O``). Its value span is therefore the token just before the target span,
which is deterministic, differs from the target span, and needs no training.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
V1_BUNDLE = ROOT / "models" / "gidi-finance-v1"

V2_TEST_VERSION = "gidi-finance-v2-test"


def add_shifted_value_output(source: Path, dest: Path) -> None:
    """Write ``source`` with an extra ``value_logits`` output (tag_logits shifted left by one)."""
    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    model = onnx.load(str(source))
    graph = model.graph
    starts = numpy_helper.from_array(np.array([1], dtype=np.int64), "value_shift_starts")
    int_max = np.array([np.iinfo(np.int64).max], dtype=np.int64)
    ends = numpy_helper.from_array(int_max, "value_shift_ends")
    axes = numpy_helper.from_array(np.array([1], dtype=np.int64), "value_shift_axes")
    pad_values = np.array([0, 0, 0, 0, 1, 0], dtype=np.int64)  # one zero row after the last token
    pads = numpy_helper.from_array(pad_values, "value_shift_pads")
    graph.initializer.extend([starts, ends, axes, pads])
    graph.node.extend(
        [
            helper.make_node(
                "Slice",
                ["tag_logits", "value_shift_starts", "value_shift_ends", "value_shift_axes"],
                ["value_shifted"],
                name="value_shift_slice",
            ),
            helper.make_node(
                "Pad",
                ["value_shifted", "value_shift_pads"],
                ["value_logits"],
                name="value_shift_pad",
            ),
        ]
    )
    graph.output.append(
        helper.make_tensor_value_info("value_logits", TensorProto.FLOAT, ["batch", "seq", 3])
    )
    onnx.save(model, str(dest))


def build_v2_bundle(dest: Path, source_bundle: Path = V1_BUNDLE) -> Path:
    """Copy ``source_bundle`` to ``dest`` as a value-head bundle (config, manifest, ONNX)."""
    from gidi.inference.bundle import (
        BUNDLE_FILES,
        CONFIG_FILE,
        MANIFEST_FILE,
        MODEL_FILE,
        sha256_file,
    )

    dest.mkdir(parents=True)
    for name in BUNDLE_FILES:
        shutil.copyfile(source_bundle / name, dest / name)
    add_shifted_value_output(source_bundle / MODEL_FILE, dest / MODEL_FILE)

    config = json.loads((source_bundle / CONFIG_FILE).read_text(encoding="utf-8"))
    value_labels = ["O", "B-VALUE", "I-VALUE"]
    config["model_version"] = V2_TEST_VERSION
    config["annotation_version"] = "annotation-v2"
    config["value_labels"] = value_labels
    config["value_decoding"] = {
        "method": "crf_viterbi",
        "start_transitions": [0.0, 0.0, 0.0],
        "end_transitions": [0.0, 0.0, 0.0],
        "transitions": [[0.0] * 3 for _ in range(3)],  # zero scores: Viterbi == argmax
        "span_selection": "highest_confidence",
        "confidence": "geometric mean of softmax(value_logits) P(Viterbi tag) over span tokens; "
        "no span: min P(O) over real tokens",
    }
    config["onnx"]["outputs"] = [*config["onnx"]["outputs"], "value_logits"]
    (dest / CONFIG_FILE).write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n")

    manifest = json.loads((source_bundle / MANIFEST_FILE).read_text(encoding="utf-8"))
    manifest["model_version"] = V2_TEST_VERSION
    manifest["annotation_version"] = "annotation-v2"
    manifest["value_labels"] = value_labels
    manifest["files"] = {
        name: {"sha256": sha256_file(dest / name), "bytes": (dest / name).stat().st_size}
        for name in BUNDLE_FILES
    }
    (dest / MANIFEST_FILE).write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return dest


@pytest.fixture(scope="session")
def v2_bundle(tmp_path_factory) -> Path:
    if not (V1_BUNDLE / "model.int8.onnx").is_file():
        pytest.skip("v1 deployment bundle not built")
    return build_v2_bundle(tmp_path_factory.mktemp("v2") / "gidi-finance-v2-test")
