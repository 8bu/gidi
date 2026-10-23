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
# The learned head of gidi-en v4 is decoded without a CRF: per-token argmax over the labels.
VALUE_DECODING_ARGMAX = "argmax"
VALUE_DECODING_METHODS = (VALUE_DECODING_METHOD, VALUE_DECODING_ARGMAX)
VALUE_SPAN_SELECTION = "highest_confidence"

# A v3 bundle has no value head: the value span comes from the rule parser, and the decoded
# target span is snapped to whole words (config.json ``value_source`` / ``target_snap``).
VALUE_SOURCE_RULE_PARSER = "rule-parser"
VALUE_PARSER_NAME = "gidi.value_parser"
TARGET_SNAP_WORDS = "words"
# Optional: the decoded value span is extended to whole words too (``value_snap`` of config.json).
VALUE_SNAP_WORDS = "words"

# ``tokenizer.json`` family: byte-level BPE (roberta-based students, the default when the config
# does not say) or uncased WordPiece (MiniLM-based gidi-en-v4, ``tokenizer_type`` in config.json).
TOKENIZER_TYPE_BPE = "bpe"
TOKENIZER_TYPE_WORDPIECE = "wordpiece"
TOKENIZER_TYPES = (TOKENIZER_TYPE_BPE, TOKENIZER_TYPE_WORDPIECE)

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
    """How the value head is decoded (``config.json`` ``value_decoding``).

    ``crf_viterbi``: CRF scores; ``transitions[i][j]`` scores tag ``i`` followed by tag ``j``;
    tags are ``VALUE_LABELS``. ``argmax``: no CRF, the three score fields are ``None``.
    """

    method: str
    start: tuple[float, ...] | None
    end: tuple[float, ...] | None
    transitions: tuple[tuple[float, ...], ...] | None


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
    # v3 bundles: ``"rule-parser"`` (value span from ``gidi.value_parser``, no value head) and
    # the parser version the bundle was frozen with; ``"words"`` for whole-word target snap.
    # ``None`` keeps the v1/v2 behaviour.
    value_source: str | None = None
    value_parser_version: str | None = None
    target_snap: str | None = None
    # ``"words"``: the decoded value span is extended to whole words (gidi-en v4); ``None``: not.
    value_snap: str | None = None
    tokenizer_type: str = TOKENIZER_TYPE_BPE


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
    method = block.get("method")
    if method not in VALUE_DECODING_METHODS:
        raise BundleError(f"value_decoding.method must be one of {VALUE_DECODING_METHODS!r}")
    if block.get("span_selection") != VALUE_SPAN_SELECTION:
        raise BundleError(f"value_decoding.span_selection must be {VALUE_SPAN_SELECTION!r}")
    if method == VALUE_DECODING_ARGMAX:
        if any(key in block for key in ("start_transitions", "end_transitions", "transitions")):
            raise BundleError("value_decoding argmax takes no CRF transitions")
        return ValueDecoding(method, None, None, None)
    size = len(VALUE_LABELS)
    rows = block.get("transitions")
    if not isinstance(rows, list) or len(rows) != size:
        raise BundleError(f"value_decoding.transitions must be a {size}x{size} matrix")
    return ValueDecoding(
        method=method,
        start=_finite_numbers(block.get("start_transitions"), size, "start_transitions"),
        end=_finite_numbers(block.get("end_transitions"), size, "end_transitions"),
        transitions=tuple(_finite_numbers(row, size, "transitions") for row in rows),
    )


def _parse_runtime_rules(config: dict[str, Any]) -> dict[str, str | None]:
    """Validate ``value_source``/``value_parser``/``target_snap``/``value_snap``/``tokenizer_type``.

    Absent keys keep the v1/v2 behaviour (value from the CRF head if there is one, no snap, BPE).
    """
    source = config.get("value_source")
    parser = config.get("value_parser")
    snap = config.get("target_snap")
    value_snap = config.get("value_snap")
    tokenizer_type = config.get("tokenizer_type", TOKENIZER_TYPE_BPE)
    if tokenizer_type not in TOKENIZER_TYPES:
        raise BundleError(f"unsupported tokenizer_type {tokenizer_type!r}")
    if value_snap not in (None, VALUE_SNAP_WORDS):
        raise BundleError(f"unsupported value_snap {value_snap!r}")
    if value_snap is not None and "value_labels" not in config:
        raise BundleError("value_snap needs a value head (value_labels)")
    common = {"value_snap": value_snap, "tokenizer_type": tokenizer_type}
    if source not in (None, VALUE_SOURCE_RULE_PARSER):
        raise BundleError(f"unsupported value_source {source!r}")
    if snap not in (None, TARGET_SNAP_WORDS):
        raise BundleError(f"unsupported target_snap {snap!r}")
    if source is None:
        if parser is not None:
            raise BundleError("value_parser needs value_source")
        return {
            "value_source": None,
            "value_parser_version": None,
            "target_snap": snap,
            **common,
        }
    if "value_labels" in config:
        raise BundleError("value_source rule-parser excludes a value head (value_labels)")
    ok = isinstance(parser, dict) and parser.get("name") == VALUE_PARSER_NAME
    if not ok or not isinstance(parser.get("version"), str) or not parser["version"]:
        raise BundleError(f"value_parser must be {{name: {VALUE_PARSER_NAME!r}, version: str}}")
    return {
        "value_source": source,
        "value_parser_version": parser["version"],
        "target_snap": snap,
        **common,
    }


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
    loaded = replace(loaded, **_parse_runtime_rules(config))
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
