"""Read a deployment bundle directory (``models/gidi-finance-v1/``) and validate it.

The runtime reads nothing but the bundle: ``config.json`` (labels, limits, ONNX interface),
``tokenizer.json`` and ``model.int8.onnx``. ``manifest.json`` and ``vocab_map.json`` are
provenance; ``verify_bundle`` checks them against the files.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

CONFIG_FILE = "config.json"
MANIFEST_FILE = "manifest.json"
MODEL_FILE = "model.int8.onnx"
TOKENIZER_FILE = "tokenizer.json"
TOKENIZER_CONFIG_FILE = "tokenizer_config.json"
VOCAB_MAP_FILE = "vocab_map.json"

# The optional value-span head of a v2 bundle: label order of ``value_logits``.
VALUE_LABELS = ("O", "B-VALUE", "I-VALUE")
VALUE_OUTPUT = "value_logits"
VALUE_DECODING_METHOD = "crf_viterbi"
VALUE_SPAN_SELECTION = "highest_confidence"

# Files whose sha256 the manifest lists under ``files`` (it cannot list itself).
BUNDLE_FILES = (
    MODEL_FILE,
    TOKENIZER_FILE,
    TOKENIZER_CONFIG_FILE,
    VOCAB_MAP_FILE,
    CONFIG_FILE,
)


class BundleError(ValueError):
    """The bundle directory is missing a file or its metadata is inconsistent."""


@dataclass(frozen=True)
class ValueDecoding:
    """CRF scores of the value head (``config.json`` ``value_decoding``).

    ``transitions[i][j]`` scores tag ``i`` followed by tag ``j``; tags are ``VALUE_LABELS``.
    """

    start: tuple[float, ...]
    end: tuple[float, ...]
    transitions: tuple[tuple[float, ...], ...]


@dataclass(frozen=True)
class BundleConfig:
    model_version: str
    types: tuple[str, ...]
    tags: tuple[str, ...]
    max_length: int
    input_names: tuple[str, ...]
    output_names: tuple[str, ...]
    # ``None`` for a v1 bundle (no value head); ``("O", "B-VALUE", "I-VALUE")`` for v2.
    value_labels: tuple[str, ...] | None = None
    # Set exactly when ``value_labels`` is.
    value_decoding: ValueDecoding | None = None


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise BundleError(f"bundle file missing: {path}") from None


def _finite_numbers(values: Any, count: int, name: str) -> tuple[float, ...]:
    if not isinstance(values, list) or len(values) != count:
        raise BundleError(f"value_decoding.{name} must be a list of {count} numbers")
    for value in values:
        ok = isinstance(value, (int, float)) and not isinstance(value, bool)
        if not ok or not math.isfinite(value):
            raise BundleError(f"value_decoding.{name} must hold finite numbers, got {value!r}")
    return tuple(float(value) for value in values)


def _parse_value_decoding(block: Any) -> ValueDecoding:
    """Validate the ``value_decoding`` block of a bundle that has a value head."""
    if not isinstance(block, dict):
        raise BundleError("a bundle with value_labels needs a value_decoding object")
    if block.get("method") != VALUE_DECODING_METHOD:
        raise BundleError(f"value_decoding.method must be {VALUE_DECODING_METHOD!r}")
    if block.get("span_selection") != VALUE_SPAN_SELECTION:
        raise BundleError(f"value_decoding.span_selection must be {VALUE_SPAN_SELECTION!r}")
    size = len(VALUE_LABELS)
    rows = block.get("transitions")
    if not isinstance(rows, list) or len(rows) != size:
        raise BundleError(f"value_decoding.transitions must be a {size}x{size} matrix")
    return ValueDecoding(
        start=_finite_numbers(block.get("start_transitions"), size, "start_transitions"),
        end=_finite_numbers(block.get("end_transitions"), size, "end_transitions"),
        transitions=tuple(_finite_numbers(row, size, "transitions") for row in rows),
    )


def load_config(bundle_dir: str | Path) -> BundleConfig:
    config = _read_json(Path(bundle_dir) / CONFIG_FILE)
    try:
        onnx = config["onnx"]
        loaded = BundleConfig(
            model_version=str(config["model_version"]),
            types=tuple(config["types"]),
            tags=tuple(config["tags"]),
            max_length=int(config["max_length"]),
            input_names=tuple(onnx["inputs"]),
            output_names=tuple(onnx["outputs"]),
            value_labels=None if "value_labels" not in config else tuple(config["value_labels"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise BundleError(f"invalid {CONFIG_FILE} in {bundle_dir}: {error!r}") from error
    if loaded.tags != ("O", "B-TARGET", "I-TARGET"):
        raise BundleError(f"unsupported tag set {loaded.tags!r}")
    expected_outputs = 2 if loaded.value_labels is None else 3
    if loaded.value_labels not in (None, VALUE_LABELS):
        raise BundleError(f"unsupported value label set {loaded.value_labels!r}")
    if len(loaded.output_names) != expected_outputs or loaded.max_length < 3:
        raise BundleError(f"unsupported ONNX interface or max_length in {bundle_dir}")
    if loaded.value_labels is not None and loaded.output_names[2] != VALUE_OUTPUT:
        raise BundleError(f"value head must be the third ONNX output {VALUE_OUTPUT!r}")
    if loaded.value_labels is None:
        return loaded
    return replace(loaded, value_decoding=_parse_value_decoding(config.get("value_decoding")))


def verify_bundle(bundle_dir: str | Path) -> dict[str, str]:
    """Check every bundle file against ``manifest.json``; return ``{name: sha256}``.

    Raises ``BundleError`` on a missing file, a hash or size mismatch.
    """
    root = Path(bundle_dir)
    manifest = _read_json(root / MANIFEST_FILE)
    config_labels = _read_json(root / CONFIG_FILE).get("value_labels")
    if manifest.get("value_labels") != config_labels:
        raise BundleError(f"value_labels differ between {CONFIG_FILE} and {MANIFEST_FILE}")
    listed = manifest.get("files", {})
    actual: dict[str, str] = {}
    for name in BUNDLE_FILES:
        path = root / name
        if not path.is_file():
            raise BundleError(f"bundle file missing: {path}")
        entry = listed.get(name)
        if entry is None:
            raise BundleError(f"{name} is not listed in {MANIFEST_FILE}")
        actual[name] = sha256_file(path)
        if actual[name] != entry["sha256"] or path.stat().st_size != entry["bytes"]:
            raise BundleError(f"{name} does not match {MANIFEST_FILE}")
    return actual
