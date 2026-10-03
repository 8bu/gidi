"""Verify a deployment bundle (default gidi-finance-v1) end to end (CPU, 1 thread, batch 1).

Writes ``<protocol dir>/verification.json`` (v1: ``experiments/deployment-v1/``). Sections:

- ``golden``: PyTorch reference (seed-1 checkpoint, HF tokenizer via
  ``gidi.modeling.preprocessing``) vs the runtime on the FP32 ONNX vs the runtime on the bundle
  INT8, over the golden suite.
- ``tokenizer``: bundle tokenizer vs the training HF tokenizer (ids, offsets, truncation) on the
  golden, robustness and every train/validation text (the test split is never read).
- ``offsets``: spans map back to the caller's string; NFC/NFD pairs; determinism.
- ``robustness``: every hand-written input returns a result or the expected ``EmptyInputError``.
- ``performance``: cold start (fresh subprocesses), warm latency, memory, sizes.
- ``isolation``: a subprocess on a *copy* of the bundle with an audit hook on file access.
- ``reproducibility``: ``freeze_deployment.py --check`` and a re-quantization of the source FP32.
- ``acceptance``: one boolean per gate in ``protocol.json``.

Usage: ``uv run python scripts/verify_deployment.py``  (exit code 1 when a gate fails)
Another version: ``... verify_deployment.py --protocol experiments/deployment-v2/protocol.json``;
a protocol with ``"value_head": true`` also verifies the value head: the golden value spans are
decoded with the CRF, and ``verify_value_deployment.py`` adds the sections ``parity``,
``int8_vs_fp32``, ``path_a_invariance``, ``value_quality``, ``hardening`` (cases in
``<protocol dir>/hardening-cases.jsonl``, built by ``build_golden_suite.py --with-value``),
``performance`` (v1 vs v2, components, single tokenization) and ``summary``, replaces
``acceptance`` by the literal gate of that protocol (the v1 booleans stay under
``legacy_acceptance_v1_thresholds``) and writes ``mismatches.jsonl`` next to the report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import unicodedata
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

_LONE_SURROGATE = re.compile("[\ud800-\udfff]")

ROOT = Path(__file__).resolve().parents[1]
DEPLOY_DIR = ROOT / "experiments" / "deployment-v1"
BUNDLE = ROOT / "models" / "gidi-finance-v1"
CHECKPOINT = (
    ROOT
    / "models/compression-v3/supervised"
    / "student-4x768-pretrained-pos32-vocab-B-rank-8000-ffn-combined-K2048/seed1"
)
FP32_ONNX = ROOT / "models/compression-v3-onnx/pos32-vocabB8000-ffn2048-4x768-seed1/model.onnx"
SPLITS = ROOT / "datasets" / "annotation-v1" / "splits"
MODEL_VERSION = "gidi-finance-v1"
EXPERIMENT = "deployment-v1"
HAS_VALUE = False  # True when the bundle under test has a value head (see configure)
FREEZE_PROTOCOL_ARGS: list[str] = []  # extra freeze_deployment.py args for a non-default protocol
PREDICTION_KEYS = {
    "type",
    "type_confidence",
    "target",
    "target_span",
    "target_confidence",
    "truncated",
    "model_version",
}
VALUE_KEYS = {"value_text", "value_span", "value_confidence"}

# Pass thresholds (declared here, applied below).
FP32_LOGIT_TOL = 1e-3  # torch vs ONNX FP32 max |logit diff|
INT8_MIN_EXACT_AGREEMENT = 0.95  # INT8 vs FP32 runtime: ">a few %" disagreements fail
INT8_SIZE_MB = 28.66
INT8_SIZE_TOL_MB = 0.05
COLD_RUNS = 5
WARM_TIMED_CALLS = 2000
WARM_UP_CALLS = 300


def prediction_keys() -> set[str]:
    """Keys of ``Prediction.to_dict()`` for the bundle under test."""
    return PREDICTION_KEYS | VALUE_KEYS if HAS_VALUE else PREDICTION_KEYS


def configure(protocol_path: Path) -> None:
    """Verify another version: bundle, sources and value head come from its protocol JSON.

    Without this call the module verifies ``gidi-finance-v1`` exactly as it always has. A
    protocol may name ``bundle`` (default ``models/<version>``), ``splits`` (the dataset splits
    whose texts feed the tokenizer check), ``experiment`` (the report's name, default the
    protocol's directory name), ``int8_size_mb`` and ``value_head`` (verify the value head too).
    """
    global DEPLOY_DIR, BUNDLE, CHECKPOINT, FP32_ONNX, SPLITS, MODEL_VERSION, EXPERIMENT
    global HAS_VALUE, INT8_SIZE_MB, FREEZE_PROTOCOL_ARGS
    path = ROOT / protocol_path
    protocol = json.loads(path.read_text(encoding="utf-8"))
    DEPLOY_DIR = path.parent
    MODEL_VERSION = protocol["version"]
    EXPERIMENT = protocol.get("experiment", path.parent.name)
    BUNDLE = ROOT / protocol.get("bundle", f"models/{MODEL_VERSION}")
    CHECKPOINT = ROOT / protocol["source_checkpoint"]
    FP32_ONNX = ROOT / protocol["source_onnx"] / "model.onnx"
    SPLITS = ROOT / protocol.get("splits", "datasets/annotation-v1/splits")
    HAS_VALUE = bool(protocol.get("value_head", False))
    INT8_SIZE_MB = protocol.get("int8_size_mb")  # a protocol without one has no size gate
    shown = path.relative_to(ROOT) if path.is_relative_to(ROOT) else path
    FREEZE_PROTOCOL_ARGS = ["--protocol", str(shown)]


# ---------------------------------------------------------------------------------------------
# helpers


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def pct(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round(q * (len(ordered) - 1))))]


def dumps_json(obj: Any, **kwargs: Any) -> str:
    """``json.dumps(ensure_ascii=False)`` that stays encodable as UTF-8: a lone surrogate (a
    robustness input) is written as a ``\\uXXXX`` escape, which JSON parsers read back."""
    return _LONE_SURROGATE.sub(
        lambda m: f"\\u{ord(m.group()):04x}", json.dumps(obj, ensure_ascii=False, **kwargs)
    )


def softmax(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def pred_key(pred: dict) -> tuple:
    span = pred["target_span"]
    key = pred["type"], None if span is None else tuple(span), pred["target"]
    if "value_span" in pred:
        value_span = pred["value_span"]
        key += (None if value_span is None else tuple(value_span), pred["value_text"])
    return key


# ---------------------------------------------------------------------------------------------
# PyTorch reference (the training-side path, never imported by the runtime)


class TorchReference:
    """The training-side model. ``checkpoint`` / ``has_value`` default to the version under test;
    a value-head version's value spans are decoded with its CRF (the ``decode_value(..., crf=)``
    path that produced every v7 metric)."""

    def __init__(self, checkpoint: Path | None = None, has_value: bool | None = None) -> None:
        import torch

        from gidi.modeling.checkpoint import load_checkpoint
        from gidi.modeling.preprocessing import TAGS, TYPES

        torch.set_num_threads(1)
        self.torch = torch
        self.checkpoint = CHECKPOINT if checkpoint is None else checkpoint
        self.has_value = HAS_VALUE if has_value is None else has_value
        self.crf = None
        if self.has_value:
            from gidi.modeling.value import load_value_checkpoint, value_crf

            self.model, self.tokenizer, self.meta = load_value_checkpoint(self.checkpoint)
            self.crf = value_crf(self.model)
        else:
            self.model, self.tokenizer, self.meta = load_checkpoint(self.checkpoint)
        self.model.eval()
        assert tuple(self.meta["types"]) == TYPES and tuple(self.meta["tags"]) == TAGS
        self.types = TYPES

    def raw(self, text: str) -> dict[str, Any]:
        """Logits + encoding for the NFC form of ``text`` through the training encode path."""
        from gidi.modeling.preprocessing import encode

        nfc = unicodedata.normalize("NFC", text)
        enc = encode(self.tokenizer, [nfc], [None])
        with self.torch.inference_mode():
            type_logits, tag_logits, *value_logits = self.model(enc.input_ids, enc.attention_mask)
        ids = enc.input_ids[0].tolist()
        special = self.tokenizer.get_special_tokens_mask(ids, already_has_special_tokens=True)
        raw = {
            "text": nfc,
            "ids": ids,
            "offsets": [tuple(o) for o in enc.offsets[0]],
            "special": [bool(s) for s in special],
            "hf_special_tokens_mask": [bool(s) for s in special],
            "truncated": bool(enc.truncated[0]),
            "type_logits": type_logits[0].numpy(),
            "tag_logits": tag_logits[0].numpy(),
        }
        if self.has_value:
            # The v7 metrics (gidi.evaluation.value_predict.decode_predictions) treat a token as
            # real when its offsets are non-empty, so `<unk>` tokens are real; the HF special
            # mask would flag them. The runtime's mask (`tokenizers`) agrees with the metrics.
            raw["special"] = [e <= s for s, e in raw["offsets"]]
            from gidi.inference.crf import viterbi_masked

            raw["value_logits"] = value_logits[0][0].numpy()
            real = [not s for s in raw["special"]]
            raw["value_tags"] = viterbi_masked(raw["value_logits"], real, self.crf, fill=0)
        return raw

    def predict(self, text: str) -> tuple[dict, dict]:
        """Reference prediction dict (offsets in the NFC string) and the raw outputs."""
        from gidi.modeling.preprocessing import spans_from_tags, trim_span

        raw = self.raw(text)
        nfc = raw["text"]
        type_probs = softmax(raw["type_logits"])
        type_idx = int(np.argmax(raw["type_logits"]))
        tags = np.argmax(raw["tag_logits"], axis=-1)
        span = spans_from_tags(raw["offsets"], tags.tolist(), nfc)
        tag_probs = softmax(raw["tag_logits"])
        if span is None:
            real = [i for i, s in enumerate(raw["special"]) if not s]
            conf = float(min(tag_probs[i, 0] for i in real)) if real else 1.0
        else:
            members = []
            started = False
            for i, ((s, e), tag) in enumerate(zip(raw["offsets"], tags.tolist(), strict=False)):
                s, e = trim_span(nfc, s, e)
                if s >= e:
                    continue
                if tag == 1:
                    if started:
                        break
                    started = True
                    members.append(i)
                elif tag == 2:
                    started = True
                    members.append(i)
                elif started:
                    break
            conf = float(
                math.exp(np.mean([math.log(float(tag_probs[i, tags[i]])) for i in members]))
            )
        value_fields: dict[str, Any] = {}
        if self.has_value:
            from gidi.modeling.value_decode import decode_value

            real = [not s for s in raw["special"]]
            value_span, value_conf = decode_value(
                raw["value_logits"], raw["offsets"], nfc, real, crf=self.crf
            )
            value_fields = {
                "value_text": None
                if value_span is None
                else nfc[value_span.start : value_span.end],
                "value_span": None if value_span is None else [value_span.start, value_span.end],
                "value_confidence": value_conf,
            }
        pred = {
            "type": self.types[type_idx],
            "type_confidence": float(type_probs[type_idx]),
            "target": None if span is None else span["text"],
            "target_span": None if span is None else [span["start"], span["end"]],
            "target_confidence": conf,
            **value_fields,
            "truncated": raw["truncated"],
            "model_version": MODEL_VERSION,
        }
        return pred, raw


# ---------------------------------------------------------------------------------------------
# checks


def golden_section(
    cases: list[dict], torch_ref: TorchReference, fp32: Any, int8: Any
) -> tuple[dict, dict[str, dict]]:
    rows = []
    max_t_f_type = max_t_f_tag = max_f_i_type = max_f_i_tag = 0.0
    max_t_f_value = max_f_i_value = 0.0
    value_conf = {"value": 0.0} if HAS_VALUE else {}
    conf_diffs_tf = {"type": 0.0, "target": 0.0, **value_conf}
    conf_diffs_fi = {"type": 0.0, "target": 0.0, **value_conf}
    torch_cache: dict[str, dict] = {}
    for case in cases:
        text = case["text"]
        t_pred, t_raw = torch_ref.predict(text)
        f_raw, i_raw = fp32.run(text), int8.run(text)
        f_pred, i_pred = fp32.predict(text).to_dict(), int8.predict(text).to_dict()
        torch_cache[case["id"]] = {"pred": t_pred, "raw": t_raw}

        ids_equal = list(f_raw.input_ids) == t_raw["ids"] and list(i_raw.input_ids) == t_raw["ids"]
        tf_type = float(np.abs(t_raw["type_logits"] - f_raw.type_logits).max())
        tf_tag = float(np.abs(t_raw["tag_logits"] - f_raw.tag_logits).max())
        fi_type = float(np.abs(f_raw.type_logits - i_raw.type_logits).max())
        fi_tag = float(np.abs(f_raw.tag_logits - i_raw.tag_logits).max())
        tf_value = fi_value = 0.0
        if HAS_VALUE:
            tf_value = float(np.abs(t_raw["value_logits"] - f_raw.value_logits).max())
            fi_value = float(np.abs(f_raw.value_logits - i_raw.value_logits).max())
        max_t_f_type, max_t_f_tag = max(max_t_f_type, tf_type), max(max_t_f_tag, tf_tag)
        max_f_i_type, max_f_i_tag = max(max_f_i_type, fi_type), max(max_f_i_tag, fi_tag)
        max_t_f_value, max_f_i_value = max(max_t_f_value, tf_value), max(max_f_i_value, fi_value)
        confidences = [("type_confidence", "type"), ("target_confidence", "target")]
        if HAS_VALUE:
            confidences.append(("value_confidence", "value"))
        for key, name in confidences:
            conf_diffs_tf[name] = max(conf_diffs_tf[name], abs(t_pred[key] - f_pred[key]))
            conf_diffs_fi[name] = max(conf_diffs_fi[name], abs(f_pred[key] - i_pred[key]))

        gold = case["gold"]
        gold_target = gold["target"]
        row = {
            "id": case["id"],
            "text": text,
            "gold_type": gold["type"],
            "gold_target": None if gold_target is None else gold_target["text"],
            "gold_span": None
            if gold_target is None
            else [gold_target["start"], gold_target["end"]],
            "categories": case["categories"],
            "torch": t_pred,
            "onnx_fp32": f_pred,
            "onnx_int8": i_pred,
            "ids_equal": ids_equal,
            "torch_eq_fp32": pred_key(t_pred) == pred_key(f_pred),
            "torch_eq_fp32_type": t_pred["type"] == f_pred["type"],
            "torch_eq_fp32_span": t_pred["target_span"] == f_pred["target_span"],
            "fp32_eq_int8": pred_key(f_pred) == pred_key(i_pred),
            "fp32_eq_int8_type": f_pred["type"] == i_pred["type"],
            "fp32_eq_int8_span": f_pred["target_span"] == i_pred["target_span"],
            "max_abs_logit_diff_torch_fp32": max(tf_type, tf_tag, tf_value),
            "max_abs_logit_diff_fp32_int8": max(fi_type, fi_tag, fi_value),
        }
        if HAS_VALUE:
            gold_value = gold.get("value")
            row.update(
                gold_value=None if gold_value is None else gold_value["text"],
                gold_value_span=None
                if gold_value is None
                else [gold_value["start"], gold_value["end"]],
                gold_value_status=gold.get("value_status"),
                torch_eq_fp32_value_span=t_pred["value_span"] == f_pred["value_span"],
                fp32_eq_int8_value_span=f_pred["value_span"] == i_pred["value_span"],
            )
        rows.append(row)

    n = len(rows)

    def rate(key: str) -> float:
        return sum(r[key] for r in rows) / n

    def accuracy(model: str) -> dict:
        scored = [r for r in rows if r["gold_type"] is not None]
        type_ok = sum(r[model]["type"] == r["gold_type"] for r in scored)
        span_ok = sum(
            (r[model]["target_span"] == r["gold_span"]) and (r[model]["target"] == r["gold_target"])
            for r in scored
        )
        exact = sum(
            r[model]["type"] == r["gold_type"]
            and r[model]["target_span"] == r["gold_span"]
            and r[model]["target"] == r["gold_target"]
            for r in scored
        )
        out = {
            "scored_cases": len(scored),
            "type_accuracy": type_ok / len(scored),
            "span_accuracy": span_ok / len(scored),
            "joint_accuracy": exact / len(scored),
            "note": "information only: golden cases come from the training/validation splits",
        }
        if HAS_VALUE:
            value_scored = [r for r in rows if r["gold_value_status"] == "complete"]
            value_ok = sum(
                r[model]["value_span"] == r["gold_value_span"]
                and r[model]["value_text"] == r["gold_value"]
                for r in value_scored
            )
            out["value_scored_cases"] = len(value_scored)
            out["value_span_accuracy"] = value_ok / max(1, len(value_scored))
        return out

    skipped = [
        {
            "id": r["id"],
            "text": r["text"],
            "torch": [r["torch"]["type"], r["torch"]["type_confidence"]],
            "onnx_fp32": [r["onnx_fp32"]["type"], r["onnx_fp32"]["type_confidence"]],
            "onnx_int8": [r["onnx_int8"]["type"], r["onnx_int8"]["type_confidence"]],
        }
        for r in rows
        if r["gold_type"] is None
    ]
    disagree_fi = [
        {k: r[k] for k in ("id", "text", "torch", "onnx_fp32", "onnx_int8")}
        for r in rows
        if not r["fp32_eq_int8"]
    ]
    disagree_tf = [
        {k: r[k] for k in ("id", "text", "torch", "onnx_fp32", "onnx_int8")}
        for r in rows
        if not r["torch_eq_fp32"]
    ]
    section = {
        "n_cases": n,
        "torch_vs_fp32": {
            "type_agreement": rate("torch_eq_fp32_type"),
            "span_agreement": rate("torch_eq_fp32_span"),
            "exact_agreement": rate("torch_eq_fp32"),
            "input_ids_identical": all(r["ids_equal"] for r in rows),
            "max_abs_logit_diff_type": max_t_f_type,
            "max_abs_logit_diff_tag": max_t_f_tag,
            "max_abs_confidence_diff": conf_diffs_tf,
            "disagreements": disagree_tf,
        },
        "fp32_vs_int8": {
            "type_agreement": rate("fp32_eq_int8_type"),
            "span_agreement": rate("fp32_eq_int8_span"),
            "exact_agreement": rate("fp32_eq_int8"),
            "max_abs_logit_diff_type": max_f_i_type,
            "max_abs_logit_diff_tag": max_f_i_tag,
            "max_abs_confidence_diff": conf_diffs_fi,
            "n_disagreements": len(disagree_fi),
            "disagreements": disagree_fi,
        },
        "gold_accuracy": {m: accuracy(m) for m in ("torch", "onnx_fp32", "onnx_int8")},
        "skipped_cases_predictions": skipped,
        "cases": rows,
    }
    if HAS_VALUE:
        section["torch_vs_fp32"].update(
            value_span_agreement=rate("torch_eq_fp32_value_span"),
            max_abs_logit_diff_value=max_t_f_value,
        )
        section["fp32_vs_int8"].update(
            value_span_agreement=rate("fp32_eq_int8_value_span"),
            max_abs_logit_diff_value=max_f_i_value,
        )
    return section, torch_cache


def tokenizer_section(torch_ref: TorchReference, int8: Any, texts: list[str]) -> dict:
    """Bundle tokenizer vs the HF training tokenizer: ids, offsets, truncation flag."""
    hf = torch_ref.tokenizer
    mismatches = []
    n = 0
    for text in texts:
        if not text.strip():
            continue
        nfc = unicodedata.normalize("NFC", text)
        try:
            ref = hf(nfc, truncation=True, max_length=32, return_offsets_mapping=True)
            full = len(hf(nfc, truncation=False)["input_ids"])
        except Exception as exc:  # the HF path rejecting an input is itself a finding
            mismatches.append({"text": text, "error": f"HF tokenizer: {type(exc).__name__}: {exc}"})
            continue
        raw = int8.run(text)
        n += 1
        same_ids = list(raw.input_ids) == list(ref["input_ids"])
        same_off = [tuple(o) for o in raw.normalized_offsets] == [
            tuple(o) for o in ref["offset_mapping"]
        ]
        same_trunc = raw.truncated == (full > 32)
        same_text = raw.normalized_text == nfc
        if not (same_ids and same_off and same_trunc and same_text):
            mismatches.append(
                {
                    "text": text,
                    "ids": same_ids,
                    "offsets": same_off,
                    "truncated": same_trunc,
                    "normalized_text": same_text,
                }
            )
    return {
        "texts_compared": n,
        "mismatches": mismatches,
        "identical": not mismatches,
        "bundle_tokenizer_json_sha256": sha256_file(BUNDLE / "tokenizer.json"),
        "checkpoint_tokenizer_json_sha256": sha256_file(CHECKPOINT / "tokenizer.json"),
    }


def _span_problems(text: str, label: str, span: Any, sliced: Any) -> list[str]:
    """Consistency problems of one ``<label>_span`` / sliced-text pair of a prediction dict."""
    if (span is None) != (sliced is None):
        return [f"{label}/{label}_span null mismatch"]
    if span is None:
        return []
    s, e = span
    if not (isinstance(s, int) and isinstance(e, int) and 0 <= s < e <= len(text)):
        return [f"bad span {span} for len {len(text)}"]
    if text[s:e] != sliced:
        return [f"text[{s}:{e}]={text[s:e]!r} != {label} {sliced!r}"]
    if sliced != sliced.strip():
        return [f"{label} has leading/trailing whitespace"]
    return []


def validate_prediction_dict(text: str, d: dict) -> list[str]:
    problems = []
    if set(d) != prediction_keys():
        problems.append(f"keys {sorted(d)}")
        return problems
    if d["model_version"] != MODEL_VERSION:
        problems.append("model_version")
    confidences = ("type_confidence", "target_confidence")
    for key in (*confidences, *(("value_confidence",) if HAS_VALUE else ())):
        v = d[key]
        if not isinstance(v, float) or not math.isfinite(v) or not 0.0 <= v <= 1.0 + 1e-9:
            problems.append(f"{key}={v!r}")
    if not isinstance(d["truncated"], bool):
        problems.append("truncated type")
    problems += _span_problems(text, "target", d["target_span"], d["target"])
    if HAS_VALUE:
        problems += _span_problems(text, "value", d["value_span"], d["value_text"])
    return problems


def offsets_and_robustness(
    robust: list[dict], cases: list[dict], int8: Any, fp32: Any, torch_ref: TorchReference
) -> tuple[dict, dict]:
    from gidi.inference import EmptyInputError

    # ---- robustness
    results = []
    failures = []
    span_problems: list[str] = []
    extras = [
        {
            "id": "robust-lone-surrogate",
            "text": "ăn sáng \ud83d 35k",
            "group": "control",
            "expect_empty_error": False,
            "nfc_nfd_pair": None,
        },
        {
            "id": "robust-lone-surrogate-low",
            "text": "\udc00 mượn anh Nam 2 triệu",
            "group": "control",
            "expect_empty_error": False,
            "nfc_nfd_pair": None,
        },
    ]
    all_inputs = robust + extras
    for item in all_inputs:
        text = item["text"]
        row: dict[str, Any] = {
            "id": item["id"],
            "group": item["group"],
            "expect_empty_error": item["expect_empty_error"],
            "text_repr": repr(text) if len(text) <= 120 else repr(text[:117]) + "...",
            "n_code_points": len(text),
        }
        for label, predictor in (("int8", int8), ("fp32", fp32)):
            try:
                pred = predictor.predict(text).to_dict()
            except EmptyInputError:
                row[label] = {"raised": "EmptyInputError"}
                if not item["expect_empty_error"]:
                    failures.append(f"{item['id']} [{label}]: unexpected EmptyInputError")
                continue
            except Exception as exc:
                row[label] = {"raised": f"{type(exc).__name__}: {exc}"}
                failures.append(f"{item['id']} [{label}]: {type(exc).__name__}: {exc}")
                continue
            if item["expect_empty_error"]:
                failures.append(f"{item['id']} [{label}]: expected EmptyInputError, got a result")
            problems = validate_prediction_dict(text, pred)
            if problems:
                failures.append(f"{item['id']} [{label}]: {problems}")
                span_problems.extend(f"{item['id']} [{label}]: {p}" for p in problems)
            row[label] = pred
        # length probes
        probe = item.get("length_probe")
        if probe and "type" in row.get("int8", {}):
            raw = int8.run(text)
            row["n_tokens_kept"] = len(raw.input_ids)
            expect = {
                "exactly-32-tokens": (32, False),
                "just-over-33-tokens": (32, True),
                "very_long": (32, True),
            }[probe]
            if (len(raw.input_ids), row["int8"]["truncated"]) != expect:
                failures.append(
                    f"{item['id']}: tokens/truncated {len(raw.input_ids)}/"
                    f"{row['int8']['truncated']} != {expect}"
                )
        # torch reference parity of logits on inputs the HF path accepts
        if not item["expect_empty_error"] and "type" in row.get("int8", {}):
            try:
                t_raw = torch_ref.raw(text)
                f_raw = fp32.run(text)
                logit_diffs = [
                    np.abs(t_raw["type_logits"] - f_raw.type_logits).max(),
                    np.abs(t_raw["tag_logits"] - f_raw.tag_logits).max(),
                ]
                if HAS_VALUE:
                    logit_diffs.append(np.abs(t_raw["value_logits"] - f_raw.value_logits).max())
                row["torch_vs_fp32"] = {
                    "ids_equal": list(f_raw.input_ids) == t_raw["ids"],
                    "max_abs_logit_diff": float(max(logit_diffs)),
                    "type_equal": torch_ref.types[int(np.argmax(t_raw["type_logits"]))]
                    == row["fp32"]["type"],
                }
            except Exception as exc:
                row["torch_vs_fp32"] = {"error": f"{type(exc).__name__}: {exc}"}
        results.append(row)

    type_errors = []
    for bad in (None, 123, b"bytes", ["x"]):
        for label, predictor in (("int8", int8), ("fp32", fp32)):
            try:
                predictor.predict(bad)  # type: ignore[arg-type]
                type_errors.append(f"{label}: {bad!r} did not raise")
            except TypeError:
                pass
            except Exception as exc:
                type_errors.append(f"{label}: {bad!r} raised {type(exc).__name__}")
    failures.extend(type_errors)
    ok_rows = [r for r in results if "type" in r.get("int8", {}) and "type" in r.get("fp32", {})]
    torch_parity = [
        r["torch_vs_fp32"] for r in results if "ids_equal" in r.get("torch_vs_fp32", {})
    ]
    robustness = {
        "n_inputs": len(all_inputs),
        "n_expect_empty_error": sum(i["expect_empty_error"] for i in all_inputs),
        "failures": failures,
        "non_str_raises_typeerror": not type_errors,
        "torch_vs_fp32_ids_equal_all": all(p["ids_equal"] for p in torch_parity),
        "torch_vs_fp32_type_equal_all": all(p["type_equal"] for p in torch_parity),
        "torch_vs_fp32_max_abs_logit_diff": max(
            (p["max_abs_logit_diff"] for p in torch_parity), default=0.0
        ),
        "torch_reference_errors": [
            r["id"] for r in results if "error" in r.get("torch_vs_fp32", {})
        ],
        "int8_vs_fp32_info": {
            "inputs": len(ok_rows),
            "type_agreement": sum(r["int8"]["type"] == r["fp32"]["type"] for r in ok_rows)
            / max(1, len(ok_rows)),
            "span_agreement": sum(
                r["int8"]["target_span"] == r["fp32"]["target_span"] for r in ok_rows
            )
            / max(1, len(ok_rows)),
            "disagreeing_ids": [
                r["id"] for r in ok_rows if pred_key(r["int8"]) != pred_key(r["fp32"])
            ],
        },
        "passed": not failures,
        "results": results,
    }

    # ---- offset stability
    problems: list[str] = list(span_problems)
    checked = sum(2 for r in results if "type" in r.get("int8", {}))
    for case in cases:
        for label, predictor in (("int8", int8), ("fp32", fp32)):
            d = predictor.predict(case["text"]).to_dict()
            checked += 1
            for p in validate_prediction_dict(case["text"], d):
                problems.append(f"{case['id']} [{label}]: {p}")

    pairs: dict[str, list[dict]] = {}
    for item in all_inputs:
        if item.get("nfc_nfd_pair"):
            pairs.setdefault(item["nfc_nfd_pair"], []).append(item)
    pair_rows = []
    for name, members in pairs.items():
        for label, predictor in (("int8", int8), ("fp32", fp32)):
            preds = [(m, predictor.predict(m["text"]).to_dict()) for m in members]
            types = {p["type"] for _, p in preds}
            targets_nfc = {
                None if p["target"] is None else unicodedata.normalize("NFC", p["target"])
                for _, p in preds
            }
            same = len(types) == 1 and len(targets_nfc) == 1
            pair_rows.append(
                {
                    "pair": name,
                    "predictor": label,
                    "same_type": len(types) == 1,
                    "same_target_after_nfc": len(targets_nfc) == 1,
                    "targets": [p["target"] for _, p in preds],
                    "spans": [p["target_span"] for _, p in preds],
                    "texts_equal": len({m["text"] for m in members}) == 1,
                }
            )
            if not same:
                problems.append(f"NFC/NFD pair {name} [{label}] disagrees")
    # the pair members must actually differ as strings (otherwise the check is vacuous)
    for row in pair_rows:
        if row["texts_equal"]:
            problems.append(f"pair {row['pair']} members are identical strings")

    determinism_bad = []
    fresh = type(int8).from_bundle(BUNDLE)
    for case in cases:
        text = case["text"]
        runs = [int8.predict(text).to_dict() for _ in range(3)] + [fresh.predict(text).to_dict()]
        raws = [int8.run(text), int8.run(text), fresh.run(text)]
        same_logits = all(
            raws[0].type_logits.tobytes() == r.type_logits.tobytes()
            and raws[0].tag_logits.tobytes() == r.tag_logits.tobytes()
            and (not HAS_VALUE or raws[0].value_logits.tobytes() == r.value_logits.tobytes())
            for r in raws[1:]
        )
        if any(r != runs[0] for r in runs[1:]) or not same_logits:
            determinism_bad.append(case["id"])
    for item in all_inputs:
        if item["expect_empty_error"]:
            continue
        runs = [int8.predict(item["text"]).to_dict() for _ in range(3)]
        if any(r != runs[0] for r in runs[1:]):
            determinism_bad.append(item["id"])

    offsets = {
        "predictions_checked": checked,
        "span_problems": problems,
        "nfc_nfd_pairs": pair_rows,
        "determinism": {
            "runs_per_input": 3,
            "plus_fresh_predictor_instance": True,
            "non_deterministic_inputs": determinism_bad,
        },
        "stable": not problems and not determinism_bad,
    }
    return offsets, robustness


# ---------------------------------------------------------------------------------------------
# performance


_COLD_CHILD = r"""
import json, os, resource, subprocess, sys, time
bundle, texts = sys.argv[1], json.loads(sys.argv[2])
def rss_mb():
    return int(subprocess.check_output(["ps", "-o", "rss=", "-p", str(os.getpid())])) / 1024
def peak_mb():
    v = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return v / 1e6 if sys.platform == "darwin" else v / 1e3  # macOS bytes, Linux KiB
out = {"peak_mb_start": peak_mb()}
t0 = time.perf_counter()
from gidi.inference import GidiPredictor
t1 = time.perf_counter()
out["peak_mb_after_import"] = peak_mb()
p = GidiPredictor.from_bundle(bundle)
t2 = time.perf_counter()
out["peak_mb_after_load"] = peak_mb()
p.predict(texts[0])
t3 = time.perf_counter()
out["peak_mb_after_first_predict"] = peak_mb()
for i in range(200):
    p.predict(texts[i % len(texts)])
out["peak_mb_after_200_predicts"] = peak_mb()
out["rss_mb_after_200_predicts"] = rss_mb()
out.update(import_s=t1 - t0, load_s=t2 - t1, first_predict_s=t3 - t2,
           ready_s=t3 - t0)
print(json.dumps(out))
"""

_BARE_CHILD = r"""
import json, resource, sys
def peak_mb():
    v = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return v / 1e6 if sys.platform == "darwin" else v / 1e3
out = {"peak_mb_python": peak_mb()}
import numpy, onnxruntime, tokenizers
out["peak_mb_python_plus_imports"] = peak_mb()
print(json.dumps(out))
"""


def run_child(code: str, *args: str, cwd: Path | None = None) -> tuple[dict, float]:
    t0 = time.perf_counter()
    proc = subprocess.run(
        [sys.executable, "-c", code, *args],
        capture_output=True,
        text=True,
        cwd=cwd or ROOT,
        check=False,
        timeout=300,
    )
    wall = time.perf_counter() - t0
    if proc.returncode != 0:
        raise RuntimeError(f"child failed: {proc.stderr[-2000:]}")
    return json.loads(proc.stdout.strip().splitlines()[-1]), wall


def warm_latency(predictor: Any, texts: list[str]) -> dict:
    """Warm ``predict()`` latency: ``WARM_UP_CALLS`` untimed calls, then ``WARM_TIMED_CALLS``."""
    for i in range(WARM_UP_CALLS):
        predictor.predict(texts[i % len(texts)])
    times = []
    for i in range(WARM_TIMED_CALLS):
        text = texts[i % len(texts)]
        t0 = time.perf_counter_ns()
        predictor.predict(text)
        times.append((time.perf_counter_ns() - t0) / 1e6)
    return {
        "timed_calls": WARM_TIMED_CALLS,
        "warmup_calls": WARM_UP_CALLS,
        "texts": len(texts),
        "p50_ms": statistics.median(times),
        "p95_ms": pct(times, 0.95),
        "p99_ms": pct(times, 0.99),
        "mean_ms": statistics.fmean(times),
        "max_ms": max(times),
        "scope": "predict(): NFC + tokenize + ONNX run + decode, batch 1, 1 thread",
    }


def performance_section(
    texts: list[str], int8: Any, fp32: Any | None, bundle: Path | None = None
) -> dict:
    """Cold start, warm latency, memory and sizes of ``bundle`` (default: the one under test)."""
    bundle = BUNDLE if bundle is None else bundle
    cold_runs = []
    for _ in range(COLD_RUNS):
        out, wall = run_child(_COLD_CHILD, str(bundle), json.dumps(texts))
        out["process_wall_s"] = wall
        cold_runs.append(out)
    bare_runs = [run_child(_BARE_CHILD)[0] for _ in range(3)]

    def med(key: str, runs: list[dict]) -> float:
        return statistics.median(r[key] for r in runs)

    cold = {
        "runs": COLD_RUNS,
        "note": "fresh subprocess per run; OS file cache is warm after the first run",
        **{
            f"median_{k}": med(k, cold_runs)
            for k in ("import_s", "load_s", "first_predict_s", "ready_s", "process_wall_s")
        },
        "all_runs": cold_runs,
    }
    bare = {
        "python_peak_mb": med("peak_mb_python", bare_runs),
        "python_plus_numpy_onnxruntime_tokenizers_peak_mb": med(
            "peak_mb_python_plus_imports", bare_runs
        ),
    }
    loaded_peak = med("peak_mb_after_200_predicts", cold_runs)
    memory = {
        "units_note": "ru_maxrss is peak RSS; reported in MB (macOS bytes / 1e6, Linux KiB / 1e3)",
        "bare": bare,
        "peak_mb_after_load_median": med("peak_mb_after_load", cold_runs),
        "peak_mb_after_first_predict_median": med("peak_mb_after_first_predict", cold_runs),
        "peak_mb_after_200_predicts_median": loaded_peak,
        "current_rss_mb_after_200_predicts_median": med("rss_mb_after_200_predicts", cold_runs),
        "delta_peak_mb_vs_python_plus_imports": loaded_peak
        - bare["python_plus_numpy_onnxruntime_tokenizers_peak_mb"],
        "delta_peak_mb_vs_bare_python": loaded_peak - bare["python_peak_mb"],
    }

    files = {p.name: p.stat().st_size for p in sorted(bundle.iterdir()) if p.is_file()}
    tok_names = ("tokenizer.json", "tokenizer_config.json", "vocab_map.json")
    sizes = {
        "files_bytes": files,
        "model_int8_bytes": files["model.int8.onnx"],
        "model_int8_mb": files["model.int8.onnx"] / 1e6,
        "tokenizer_artifacts_bytes": sum(files[n] for n in tok_names if n in files),
        "tokenizer_artifacts_mb": sum(files[n] for n in tok_names if n in files) / 1e6,
        "bundle_total_bytes": sum(files.values()),
        "bundle_total_mb": sum(files.values()) / 1e6,
        "mb_definition": "1 MB = 1e6 bytes",
    }
    out = {
        "cpu": platform.processor() or platform.machine(),
        "threads": 1,
        "cold_start": cold,
        "warm_inference_int8": warm_latency(int8, texts),
        "memory": memory,
        "sizes": sizes,
    }
    if fp32 is not None:
        out["warm_inference_fp32_for_reference"] = warm_latency(fp32, texts)
    return out


# ---------------------------------------------------------------------------------------------
# isolation


_ISOLATION_CHILD = r"""
import json, os, site, sys
events = []
def hook(event, args):
    if event in ("open", "os.listdir", "os.scandir") and args:
        a = args[0]
        if isinstance(a, int) or a is None:
            return
        try:
            events.append((event, os.fsdecode(a)))
        except Exception:
            pass
sys.addaudithook(hook)
bundle, texts = sys.argv[1], json.loads(sys.argv[2])
from gidi.inference import GidiPredictor
p = GidiPredictor.from_bundle(bundle)
preds = [p.predict(t).to_dict() for t in texts]
src_root = os.path.dirname(os.path.dirname(sys.modules["gidi"].__file__))
forbidden_mods = sorted(
    m for m in sys.modules
    if m.split(".")[0] in ("torch", "transformers", "safetensors", "onnx")
    or m.startswith(("gidi.modeling", "gidi.export", "gidi.training", "gidi.corpus",
                     "gidi.annotation", "gidi.compression", "gidi.distillation"))
)
roots = {os.path.realpath(x) for x in
         [sys.prefix, sys.base_prefix, sys.exec_prefix, src_root, *site.getsitepackages()]}
cwd = os.getcwd()
paths = []
for ev, path in events:
    full = os.path.realpath(path if os.path.isabs(path) else os.path.join(cwd, path))
    paths.append((ev, full))
print(json.dumps({"paths": paths, "roots": sorted(roots), "forbidden_modules": forbidden_mods,
                  "loaded_gidi_modules": sorted(m for m in sys.modules if m.startswith("gidi")),
                  "predictions": preds}))
"""


def isolation_section(texts: list[str], expected: list[dict]) -> dict:
    forbidden_dirs = [os.path.realpath(ROOT / d) for d in ("datasets", "corpus", "experiments")]
    with tempfile.TemporaryDirectory() as tmp:
        copy = Path(tmp) / "bundle-copy"
        shutil.copytree(BUNDLE, copy)
        work = Path(tmp) / "cwd"
        work.mkdir()
        out, _ = run_child(_ISOLATION_CHILD, str(copy), json.dumps(texts), cwd=work)
        copy_real = os.path.realpath(copy)
    paths = [(ev, p) for ev, p in out["paths"]]
    roots = out["roots"]

    def under(path: str, base: str) -> bool:
        return path == base or path.startswith(base + os.sep)

    touched_forbidden = sorted({p for _, p in paths if any(under(p, d) for d in forbidden_dirs)})
    in_bundle = sorted({p for _, p in paths if under(p, copy_real)})
    other = sorted(
        {p for _, p in paths if not under(p, copy_real) and not any(under(p, r) for r in roots)}
    )
    repo_non_src = sorted(
        {
            p
            for _, p in paths
            if under(p, os.path.realpath(ROOT))
            and not under(p, os.path.realpath(ROOT / "src"))
            and not any(under(p, r) for r in roots)
        }
    )
    predictions_match = out["predictions"] == expected
    return {
        "method": "sys.addaudithook on 'open', 'os.listdir', 'os.scandir' in a subprocess "
        "(cwd = empty temp dir) that loads a COPY of the bundle from a temp dir; native ONNX "
        "runtime file reads bypass Python audit events, so the model path is by construction "
        "the bundle copy",
        "audited_events": len(paths),
        "bundle_files_opened": [os.path.relpath(p, copy_real) for p in in_bundle],
        "forbidden_dirs_touched": touched_forbidden,
        "repo_paths_outside_src_touched": repo_non_src,
        "other_paths_outside_python_and_bundle": other,
        "forbidden_modules_imported": out["forbidden_modules"],
        "gidi_modules_loaded": out["loaded_gidi_modules"],
        "copied_bundle_predictions_match_in_process": predictions_match,
        "isolated": not touched_forbidden
        and not repo_non_src
        and not out["forbidden_modules"]
        and predictions_match,
    }


# ---------------------------------------------------------------------------------------------
# reproducibility


def reproducibility_section(cases: list[dict], int8: Any, fp32: Any) -> dict:
    from gidi.inference import GidiPredictor

    freeze_script = ROOT / "scripts" / "freeze_deployment.py"
    freeze: dict[str, Any] = {"script_exists": freeze_script.is_file()}
    if freeze_script.is_file():
        cmd = [
            "uv",
            "run",
            "python",
            "scripts/freeze_deployment.py",
            "--out",
            BUNDLE.relative_to(ROOT).as_posix(),
            "--check",
            *FREEZE_PROTOCOL_ARGS,
        ]
        proc = subprocess.run(
            cmd, cwd=ROOT, capture_output=True, text=True, check=False, timeout=1800
        )
        freeze.update(
            command=" ".join(cmd),
            exit_code=proc.returncode,
            stdout_tail=proc.stdout[-1500:],
            stderr_tail=proc.stderr[-1500:],
        )
    freeze["ok"] = freeze.get("exit_code") == 0

    from gidi.export.onnx_export import quantize_int8

    bundle_hash = sha256_file(BUNDLE / "model.int8.onnx")
    with tempfile.TemporaryDirectory() as tmp:
        out_path = Path(tmp) / "model.int8.onnx"
        quantize_int8(FP32_ONNX, out_path)
        requant_hash = sha256_file(out_path)
        identical = requant_hash == bundle_hash
        requant: dict[str, Any] = {
            "source_fp32": str(FP32_ONNX.relative_to(ROOT)),
            "bundle_int8_sha256": bundle_hash,
            "requantized_sha256": requant_hash,
            "byte_identical": identical,
        }
        if not identical:
            other = GidiPredictor.from_bundle(BUNDLE, model_path=out_path)
            max_diff = 0.0
            pred_mismatch = []
            for case in cases:
                a, b = int8.run(case["text"]), other.run(case["text"])
                max_diff = max(
                    max_diff,
                    float(np.abs(a.type_logits - b.type_logits).max()),
                    float(np.abs(a.tag_logits - b.tag_logits).max()),
                    float(np.abs(a.value_logits - b.value_logits).max()) if HAS_VALUE else 0.0,
                )
                if pred_key(int8.predict(case["text"]).to_dict()) != pred_key(
                    other.predict(case["text"]).to_dict()
                ):
                    pred_mismatch.append(case["id"])
            requant.update(
                golden_output_max_abs_logit_diff=max_diff,
                golden_prediction_mismatches=pred_mismatch,
                output_identical=max_diff == 0.0 and not pred_mismatch,
            )
        else:
            requant["output_identical"] = True
    reproducible = freeze["ok"] and requant["output_identical"]
    return {"freeze_check": freeze, "requantize_source_fp32": requant, "reproducible": reproducible}


# ---------------------------------------------------------------------------------------------


def versions() -> dict[str, str]:
    import importlib.metadata as md

    out = {"python": platform.python_version(), "platform": platform.platform()}
    for name in ("onnxruntime", "tokenizers", "numpy", "torch", "transformers", "onnx"):
        try:
            out[name] = md.version(name)
        except md.PackageNotFoundError:
            out[name] = "n/a"
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--protocol",
        type=Path,
        default=None,
        help="deployment protocol JSON of the version to verify "
        "(default: experiments/deployment-v1/protocol.json, the gidi-finance-v1 setup)",
    )
    parser.add_argument(
        "--out", type=Path, default=None, help="default: <protocol dir>/verification.json"
    )
    parser.add_argument("--skip-reproducibility", action="store_true")
    args = parser.parse_args()
    if args.protocol is not None:
        configure(args.protocol)
    if args.out is None:
        args.out = DEPLOY_DIR / "verification.json"

    from gidi.inference import GidiPredictor, verify_bundle

    cases = read_jsonl(DEPLOY_DIR / "golden-suite.jsonl")
    robust = read_jsonl(DEPLOY_DIR / "robustness-inputs.jsonl")
    golden_texts = [c["text"] for c in cases]

    bundle_hashes = verify_bundle(BUNDLE)
    int8 = GidiPredictor.from_bundle(BUNDLE)
    fp32 = GidiPredictor.from_bundle(BUNDLE, model_path=FP32_ONNX)
    torch_ref = TorchReference()
    vv = env = None
    if HAS_VALUE:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import verify_value_deployment as vv  # noqa: PLC0415 (a value-head version only)

        env = vv.Env(torch_ref, int8, fp32, DEPLOY_DIR, CHECKPOINT, BUNDLE, FP32_ONNX)

    t0 = time.perf_counter()
    golden, _ = golden_section(cases, torch_ref, fp32, int8)
    split_texts = [
        r["text"] for s in ("train", "validation") for r in read_jsonl(SPLITS / f"{s}.jsonl")
    ]
    if vv is not None:
        split_texts += vv.extra_tokenizer_texts()
    tok = tokenizer_section(
        torch_ref,
        int8,
        golden_texts + [r["text"] for r in robust if r["text"].strip()] + split_texts,
    )
    offsets, robustness = offsets_and_robustness(robust, cases, int8, fp32, torch_ref)
    performance = performance_section(golden_texts, int8, fp32)
    expected_preds = [int8.predict(t).to_dict() for t in golden_texts]
    isolation = isolation_section(golden_texts, expected_preds)
    repro = (
        {
            "freeze_check": None,
            "requantize_source_fp32": None,
            "reproducible": False,
            "skipped": True,
        }
        if args.skip_reproducibility
        else reproducibility_section(cases, int8, fp32)
    )

    tvf, fvi = golden["torch_vs_fp32"], golden["fp32_vs_int8"]
    size_mb = performance["sizes"]["model_int8_mb"]
    acceptance = {
        "reproducible": bool(repro["reproducible"]),
        "tokenizer_compatible": bool(
            tok["identical"]
            and tvf["input_ids_identical"]
            and robustness["torch_vs_fp32_ids_equal_all"]
        ),
        "int8_parity": bool(
            fvi["exact_agreement"] >= INT8_MIN_EXACT_AGREEMENT
            and fvi["type_agreement"] >= INT8_MIN_EXACT_AGREEMENT
        ),
        "offsets_stable": bool(offsets["stable"]),
        "golden_pass": bool(
            tvf["exact_agreement"] == 1.0
            and tvf["type_agreement"] == 1.0
            and tvf["span_agreement"] == 1.0
            and tvf.get("value_span_agreement", 1.0) == 1.0
            and max(
                tvf["max_abs_logit_diff_type"],
                tvf["max_abs_logit_diff_tag"],
                tvf.get("max_abs_logit_diff_value", 0.0),
            )
            <= FP32_LOGIT_TOL
        ),
        "robustness_pass": bool(robustness["passed"]),
        "runtime_isolated": bool(isolation["isolated"]),
    }
    if INT8_SIZE_MB is not None:
        acceptance["size_ok"] = bool(abs(size_mb - INT8_SIZE_MB) <= INT8_SIZE_TOL_MB)
    if HAS_VALUE:
        acceptance["value_head_int8_parity"] = bool(
            fvi["value_span_agreement"] >= INT8_MIN_EXACT_AGREEMENT
        )
    acceptance["all_pass"] = all(acceptance.values())

    report = {
        "experiment": EXPERIMENT,
        "model_version": MODEL_VERSION,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "environment": versions(),
        "thresholds": {
            "fp32_logit_tolerance": FP32_LOGIT_TOL,
            "int8_min_exact_agreement": INT8_MIN_EXACT_AGREEMENT,
            "int8_size_mb": INT8_SIZE_MB,
            "int8_size_tolerance_mb": INT8_SIZE_TOL_MB,
        },
        "bundle_files_sha256": bundle_hashes,
        "acceptance": acceptance,
        "golden": golden,
        "tokenizer": tok,
        "offsets": offsets,
        "robustness": robustness,
        "performance": performance,
        "isolation": isolation,
        "reproducibility": repro,
        "verify_wall_s_golden_to_robustness": time.perf_counter() - t0,
    }
    exit_ok = acceptance["all_pass"]
    if vv is not None:
        # The value-head version is judged by its own protocol's gate, not the v1 thresholds above.
        extra = vv.run_checks(sys.modules[__name__], env, report, args.out.parent)
        report.update(extra["sections"])
        report["legacy_acceptance_v1_thresholds"] = report.pop("acceptance")
        report["acceptance"] = extra["acceptance"]
        report["summary"] = extra["summary"]
        exit_ok = extra["acceptance"]["all_pass"]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(dumps_json(report, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(report["acceptance"], indent=2))
    print(
        f"golden: {golden['n_cases']} cases; torch==fp32 exact {tvf['exact_agreement']:.4f}; "
        f"fp32==int8 exact {fvi['exact_agreement']:.4f} ({fvi['n_disagreements']} disagreements)"
    )
    print(f"wrote {args.out}")
    return 0 if exit_ok else 1


if __name__ == "__main__":
    sys.exit(main())
