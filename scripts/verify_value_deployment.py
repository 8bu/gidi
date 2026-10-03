"""Value-head (``gidi-finance-v2``) deployment checks, run by ``verify_deployment.py``.

``verify_deployment.py --protocol <value-head protocol>`` calls :func:`run_checks` after its own
sections. Everything here only *measures*: no threshold is applied to a model number except the
literal gate items of ``protocol.json`` (``acceptance_gate``), which are evaluated mechanically
into ``acceptance`` and listed with the raw numbers they come from.

Sections added to ``verification.json``:

- ``parity``: PyTorch vs the FP32 runtime per set (golden, test, probe, robustness). Path A (type
  + target), path B (value emissions, Viterbi tags, value span), combined runtime output vs the
  PyTorch reference dict; every mismatch listed.
- ``int8_vs_fp32``: per set, every field, confidence drift, Viterbi tags, offsets, disagreeing
  notes with both outputs. ``path_a_invariance``: v2 vs the v1 bundle / v1 PyTorch / v1 FP32.
- ``value_quality``: the frozen value-span-v7 criteria code (``gidi.evaluation.value_gate``) on
  test / test-v1 / test-targeted / probe for PyTorch, FP32 runtime and INT8 runtime.
- ``hardening``: ``hardening-cases.jsonl`` + robustness inputs + seeded text fuzz through the INT8
  runtime, and seeded emission/transition fuzz through ``decode_value_crf``.
- ``performance_v2``: v1 vs v2 latency / cold start / memory / sizes, component timings, single
  tokenization evidence.
- ``summary``: the compact numbers of all of the above.

Per-note mismatches of every comparison are also written to ``mismatches.jsonl`` next to the report.
"""

from __future__ import annotations

import itertools
import json
import math
import re
import statistics
import subprocess
import sys
import time
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

V1_BUNDLE = ROOT / "models" / "gidi-finance-v1"
V1_FP32_ONNX = ROOT / "models/compression-v3-onnx/pos32-vocabB8000-ffn2048-4x768-seed1/model.onnx"
EVAL_DIR = ROOT / "datasets" / "annotation-v2" / "training-v1"
V7_COMPARISON = ROOT / "experiments" / "value-span-v7-dual-encoder" / "comparison.json"
EVAL_SETS = ("test", "test-v1", "test-targeted", "probe")
PARITY_SETS = ("golden", "test", "probe", "robustness")
# Every non-confidence field of ``Prediction.to_dict()``.
SEMANTIC = (
    "type",
    "target",
    "target_span",
    "value_text",
    "value_span",
    "truncated",
    "model_version",
)
V1_FIELDS = ("type", "target", "target_span")
MAX_LENGTH = 32
INT8_FIELD_MIN_AGREEMENT = 0.99  # protocol acceptance_gate: test and probe, each field
FUZZ_SEED = 20260503
FUZZ_DECODE_CASES = 4000
FUZZ_TEXT_CASES = 400
COMPONENT_CALLS = 2000
COMPONENT_WARMUP = 300


@dataclass
class Env:
    torch_ref: Any  # verify_deployment.TorchReference of the version under test
    int8: Any  # GidiPredictor on the bundle INT8
    fp32: Any  # GidiPredictor on the FP32 ONNX
    deploy_dir: Path
    checkpoint: Path
    bundle: Path  # bundle under test
    fp32_onnx: Path


# ---------------------------------------------------------------------------------------------
# small helpers


def maxabs(a: Any, b: Any) -> float:
    return float(np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)).max())


def to_original(text: str, pred: dict) -> dict:
    """Reference dict (spans in the NFC string) with spans mapped into the caller's string."""
    if unicodedata.is_normalized("NFC", text):
        return pred
    from gidi.inference.text import normalize_nfc

    norm = normalize_nfc(text)
    out = dict(pred)
    for span_key, text_key in (("target_span", "target"), ("value_span", "value_text")):
        span = pred.get(span_key)
        if span is not None:
            start, end = norm.span_to_original(*span)
            out[span_key], out[text_key] = [start, end], text[start:end]
    return out


def rate(ms: list[dict], key: str) -> float | None:
    vals = [m[key] for m in ms if key in m]
    return sum(vals) / len(vals) if vals else None


def peak(ms: list[dict], key: str) -> float | None:
    vals = [m[key] for m in ms if m.get(key) is not None]
    return max(vals) if vals else None


def mean_abs(ms: list[dict], key: str) -> float | None:
    vals = [m[key] for m in ms if m.get(key) is not None]
    return statistics.fmean(vals) if vals else None


def _trim(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


class Mismatches:
    """Collects one JSON row per (comparison, set, note) that differs; written as JSONL."""

    def __init__(self) -> None:
        self.rows: list[dict] = []

    def add(self, comparison: str, set_name: str, note_id: str, text: str, fields: list, **sides):
        self.rows.append(
            {
                "comparison": comparison,
                "set": set_name,
                "id": note_id,
                "text": text,
                "fields": fields,
                **sides,
            }
        )


# ---------------------------------------------------------------------------------------------
# evidence: every backend's output for every note, computed once


class Evidence:
    def __init__(self, env: Env, vd: Any) -> None:
        from gidi.inference import GidiPredictor

        self.env = env
        self.vd = vd
        self.v1_ref = vd.TorchReference(
            checkpoint=vd.ROOT / self.base_checkpoint(env), has_value=False
        )
        self.v1_int8 = GidiPredictor.from_bundle(V1_BUNDLE)
        self.v1_fp32 = GidiPredictor.from_bundle(V1_BUNDLE, model_path=V1_FP32_ONNX)
        self._rows: dict[str, dict] = {}

    @staticmethod
    def base_checkpoint(env: Env) -> str:
        protocol = json.loads((env.deploy_dir / "protocol.json").read_text(encoding="utf-8"))
        return protocol["base_checkpoint"]

    def row(self, text: str) -> dict:
        row = self._rows.get(text)
        if row is None:
            row = self._rows[text] = self._compute(text)
        return row

    def _compute(self, text: str) -> dict:
        e = self.env
        row: dict[str, Any] = {}
        for name, ref in (("torch", e.torch_ref), ("v1_torch", self.v1_ref)):
            try:
                row[name] = ref.predict(text)
            except Exception as exc:  # the HF path rejecting an input is itself a finding
                row[name] = None
                row[f"{name}_error"] = f"{type(exc).__name__}: {exc}"
        for name, predictor in (
            ("fp32", e.fp32),
            ("int8", e.int8),
            ("v1_int8", self.v1_int8),
            ("v1_fp32", self.v1_fp32),
        ):
            row[f"{name}_raw"] = predictor.run(text)
            row[name] = predictor.predict(text).to_dict()
        return row


# ---------------------------------------------------------------------------------------------
# sets


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def extra_tokenizer_texts() -> list[str]:
    """Texts of annotation-v2 training-v1 (all four files) for the tokenizer identity check."""
    texts = []
    for name in ("train", "validation", "test", "probe-v1-eval-only"):
        texts += [r["text"] for r in read_jsonl(EVAL_DIR / f"{name}.jsonl")]
    return texts


def parity_sets(env: Env) -> dict[str, list[tuple[str, str]]]:
    """``{set: [(id, text)]}`` of golden suite, held-out test, probe and robustness inputs."""
    robust = read_jsonl(env.deploy_dir / "robustness-inputs.jsonl")
    robust_items = [(r["id"], r["text"]) for r in robust if not r["expect_empty_error"]]
    robust_items += [
        ("robust-lone-surrogate", "ăn sáng \ud83d 35k"),
        ("robust-lone-surrogate-low", "\udc00 mượn anh Nam 2 triệu"),
    ]
    return {
        "golden": [(c["id"], c["text"]) for c in read_jsonl(env.deploy_dir / "golden-suite.jsonl")],
        "test": [(r["id"], r["text"]) for r in read_jsonl(EVAL_DIR / "test.jsonl")],
        "probe": [(r["id"], r["text"]) for r in read_jsonl(EVAL_DIR / "probe-v1-eval-only.jsonl")],
        "robustness": robust_items,
    }


# ---------------------------------------------------------------------------------------------
# Phase 3: PyTorch vs FP32 ONNX


def torch_vs_fp32_note(text: str, row: dict, crf: Any) -> dict:
    t_pred, t_raw = row["torch"]
    f_raw, f_pred = row["fp32_raw"], row["fp32"]
    expected = to_original(text, t_pred)
    m: dict[str, Any] = {
        "ids_equal": list(f_raw.input_ids) == t_raw["ids"],
        "offsets_equal": [tuple(o) for o in f_raw.normalized_offsets]
        == [tuple(o) for o in t_raw["offsets"]],
        "special_equal": [bool(s) for s in f_raw.special_tokens_mask] == t_raw["special"],
    }
    if m["ids_equal"]:
        real = np.array([not s for s in t_raw["special"]], dtype=bool)
        m["type_logit_diff"] = maxabs(t_raw["type_logits"], f_raw.type_logits)
        m["tag_logit_diff"] = maxabs(t_raw["tag_logits"], f_raw.tag_logits)
        m["value_logit_diff"] = maxabs(t_raw["value_logits"], f_raw.value_logits)
        t_tags, f_tags = t_raw["tag_logits"].argmax(-1), f_raw.tag_logits.argmax(-1)
        m["tag_tokens"] = int(real.sum())
        m["tag_tokens_equal"] = int((t_tags[real] == f_tags[real]).sum())
        m["tags_equal"] = m["tag_tokens_equal"] == m["tag_tokens"]
        t_vit, f_vit = np.asarray(t_raw["value_tags"]), np.asarray(f_raw.value_tags)
        m["viterbi_tokens_equal"] = int((t_vit[real] == f_vit[real]).sum())
        m["viterbi_equal"] = bool(np.array_equal(t_vit, f_vit))
    else:
        m.update(tags_equal=False, viterbi_equal=False)
    m["type_equal"] = expected["type"] == f_pred["type"]
    m["target_equal"] = (expected["target"], expected["target_span"]) == (
        f_pred["target"],
        f_pred["target_span"],
    )
    m["value_equal"] = (expected["value_text"], expected["value_span"]) == (
        f_pred["value_text"],
        f_pred["value_span"],
    )
    m["semantic_equal"] = all(expected[k] == f_pred[k] for k in SEMANTIC)
    for key in ("type", "target", "value"):
        name = f"{key}_confidence"
        m[f"{key}_conf_diff"] = abs(expected[name] - f_pred[name])
    return m


def parity_section(ev: Evidence, sets: dict[str, list[tuple[str, str]]], mm: Mismatches) -> dict:
    crf = ev.env.torch_ref.crf
    out: dict[str, Any] = {}
    for name in PARITY_SETS:
        ms, skipped = [], []
        for note_id, text in sets[name]:
            row = ev.row(text)
            if row["torch"] is None:
                skipped.append({"id": note_id, "text": text, "error": row["torch_error"]})
                continue
            m = torch_vs_fp32_note(text, row, crf)
            ms.append(m)
            bad = [
                k
                for k in (
                    "ids_equal",
                    "offsets_equal",
                    "special_equal",
                    "type_equal",
                    "tags_equal",
                    "target_equal",
                    "viterbi_equal",
                    "value_equal",
                    "semantic_equal",
                )
                if not m[k]
            ]
            if bad:
                mm.add(
                    "torch_vs_fp32",
                    name,
                    note_id,
                    text,
                    bad,
                    torch=to_original(text, row["torch"][0]),
                    fp32=row["fp32"],
                )
        n = len(ms)
        tag_tokens = sum(m.get("tag_tokens", 0) for m in ms)
        out[name] = {
            "n_notes": n,
            "n_skipped_torch_reference_failed": len(skipped),
            "skipped": skipped,
            "input_ids_identical": all(m["ids_equal"] and m["offsets_equal"] for m in ms),
            "path_a": {
                "type_logits_max_abs_diff": peak(ms, "type_logit_diff"),
                "type_prediction_agreement": rate(ms, "type_equal"),
                "target_logits_max_abs_diff": peak(ms, "tag_logit_diff"),
                "target_bio_tag_agreement_notes": rate(ms, "tags_equal"),
                "target_bio_tag_agreement_tokens": sum(m.get("tag_tokens_equal", 0) for m in ms)
                / max(1, tag_tokens),
                "target_span_agreement": rate(ms, "target_equal"),
                "target_confidence_max_abs_diff": peak(ms, "target_conf_diff"),
                "type_confidence_max_abs_diff": peak(ms, "type_conf_diff"),
            },
            "path_b": {
                "value_emissions_max_abs_diff": peak(ms, "value_logit_diff"),
                "viterbi_tag_agreement_notes": rate(ms, "viterbi_equal"),
                "viterbi_tag_agreement_tokens": sum(m.get("viterbi_tokens_equal", 0) for m in ms)
                / max(1, tag_tokens),
                "value_span_agreement": rate(ms, "value_equal"),
                "value_confidence_max_abs_diff": peak(ms, "value_conf_diff"),
            },
            "combined_runtime_vs_reference": {
                "fields": list(SEMANTIC),
                "semantic_agreement": rate(ms, "semantic_equal"),
                "n_mismatches": sum(not m["semantic_equal"] for m in ms),
            },
            "n_mismatching_notes_any_check": sum(
                not all(
                    m[k]
                    for k in (
                        "ids_equal",
                        "type_equal",
                        "tags_equal",
                        "target_equal",
                        "viterbi_equal",
                        "value_equal",
                        "semantic_equal",
                    )
                )
                for m in ms
            ),
        }
    return out


# ---------------------------------------------------------------------------------------------
# Phase 4: INT8 vs FP32, path A invariance


def int8_vs_fp32_note(row: dict) -> dict:
    f, i, fr, ir = row["fp32"], row["int8"], row["fp32_raw"], row["int8_raw"]
    fields = [k for k in SEMANTIC if f[k] != i[k]]
    real = np.array([not s for s in fr.special_tokens_mask], dtype=bool)
    f_tags, i_tags = fr.tag_logits.argmax(-1), ir.tag_logits.argmax(-1)
    m: dict[str, Any] = {
        "fields_differing": fields,
        "type_equal": f["type"] == i["type"],
        "target_equal": (f["target"], f["target_span"]) == (i["target"], i["target_span"]),
        "value_equal": (f["value_text"], f["value_span"]) == (i["value_text"], i["value_span"]),
        "full_equal": not fields,
        "target_tags_equal": bool(np.array_equal(f_tags[real], i_tags[real])),
        "viterbi_equal": tuple(fr.value_tags) == tuple(ir.value_tags),
        "viterbi_tokens": int(real.sum()),
        "viterbi_tokens_equal": int(
            (np.asarray(fr.value_tags)[real] == np.asarray(ir.value_tags)[real]).sum()
        ),
        "offsets_and_ids_equal": fr.offsets == ir.offsets
        and fr.input_ids == ir.input_ids
        and fr.normalized_offsets == ir.normalized_offsets,
        "type_logit_diff": maxabs(fr.type_logits, ir.type_logits),
        "tag_logit_diff": maxabs(fr.tag_logits, ir.tag_logits),
        "value_logit_diff": maxabs(fr.value_logits, ir.value_logits),
    }
    for key in ("type", "target", "value"):
        name = f"{key}_confidence"
        m[f"{key}_conf_diff"] = abs(f[name] - i[name])
    return m


def int8_vs_fp32_section(
    ev: Evidence, sets: dict[str, list[tuple[str, str]]], mm: Mismatches
) -> dict:
    out: dict[str, Any] = {}
    for name in PARITY_SETS:
        ms, disagree = [], []
        for note_id, text in sets[name]:
            row = ev.row(text)
            m = int8_vs_fp32_note(row)
            ms.append(m)
            if m["fields_differing"] or not m["viterbi_equal"] or not m["target_tags_equal"]:
                entry = {
                    "id": note_id,
                    "text": text,
                    "fields_differing": m["fields_differing"],
                    "viterbi_tags_differ": not m["viterbi_equal"],
                    "target_bio_tags_differ": not m["target_tags_equal"],
                    "fp32": row["fp32"],
                    "int8": row["int8"],
                }
                disagree.append(entry)
                mm.add(
                    "int8_vs_fp32",
                    name,
                    note_id,
                    text,
                    m["fields_differing"]
                    + (["viterbi_tags"] if not m["viterbi_equal"] else [])
                    + (["target_bio_tags"] if not m["target_tags_equal"] else []),
                    fp32=row["fp32"],
                    int8=row["int8"],
                )
        tokens = sum(m["viterbi_tokens"] for m in ms)
        span_diff = [
            d for d in disagree if {"target_span", "value_span"} & set(d["fields_differing"])
        ]
        out[name] = {
            "n_notes": len(ms),
            "type_agreement": rate(ms, "type_equal"),
            "target_exact_agreement": rate(ms, "target_equal"),
            "value_exact_agreement": rate(ms, "value_equal"),
            "full_output_agreement": rate(ms, "full_equal"),
            "full_output_fields": list(SEMANTIC),
            "target_bio_tag_agreement_notes": rate(ms, "target_tags_equal"),
            "viterbi_tag_agreement_notes": rate(ms, "viterbi_equal"),
            "viterbi_tag_agreement_tokens": sum(m["viterbi_tokens_equal"] for m in ms)
            / max(1, tokens),
            "token_offsets_and_ids_identical": all(m["offsets_and_ids_equal"] for m in ms),
            "confidence_drift": {
                key: {
                    "max_abs": peak(ms, f"{key}_conf_diff"),
                    "mean_abs": mean_abs(ms, f"{key}_conf_diff"),
                }
                for key in ("type", "target", "value")
            },
            "logit_max_abs_diff": {
                "type": peak(ms, "type_logit_diff"),
                "target": peak(ms, "tag_logit_diff"),
                "value_emissions": peak(ms, "value_logit_diff"),
            },
            "n_disagreeing_notes": sum(bool(d["fields_differing"]) for d in disagree),
            "n_notes_with_span_differences": len(span_diff),
            "n_notes_differing_only_in_tags_not_in_decoded_fields": sum(
                not d["fields_differing"] for d in disagree
            ),
            "disagreement_list_includes": "every note whose decoded fields, target BIO tags or "
            "Viterbi tags differ (fields_differing empty = tag-level difference only)",
            "disagreements": disagree,
        }
    return out


def _compare_a(a: dict, b: dict) -> dict:
    return {
        "type_equal": a["type"] == b["type"],
        "target_equal": a["target"] == b["target"],
        "target_span_equal": a["target_span"] == b["target_span"],
        "type_conf_diff": abs(a["type_confidence"] - b["type_confidence"]),
        "target_conf_diff": abs(a["target_confidence"] - b["target_confidence"]),
        "confidences_identical": a["type_confidence"] == b["type_confidence"]
        and a["target_confidence"] == b["target_confidence"],
    }


def path_a_section(ev: Evidence, sets: dict[str, list[tuple[str, str]]], mm: Mismatches) -> dict:
    """Path A is the frozen v1 model: v2 type/target must equal v1 under every backend pairing."""
    out: dict[str, Any] = {}
    for name in PARITY_SETS:
        pairs: dict[str, list[dict]] = {
            "v2_int8_vs_v1_int8_bundle": [],
            "v2_fp32_vs_v1_torch": [],
            "v2_fp32_vs_v1_fp32_onnx": [],
            "v1_int8_vs_v1_fp32_onnx": [],
        }
        listed: dict[str, list[dict]] = {k: [] for k in pairs}
        for note_id, text in sets[name]:
            row = ev.row(text)
            v1_torch = None if row["v1_torch"] is None else to_original(text, row["v1_torch"][0])
            v1_torch_raw = None if row["v1_torch"] is None else row["v1_torch"][1]
            combos = {
                "v2_int8_vs_v1_int8_bundle": (
                    row["int8"],
                    row["v1_int8"],
                    row["int8_raw"],
                    row["v1_int8_raw"],
                ),
                "v2_fp32_vs_v1_torch": (row["fp32"], v1_torch, row["fp32_raw"], v1_torch_raw),
                "v2_fp32_vs_v1_fp32_onnx": (
                    row["fp32"],
                    row["v1_fp32"],
                    row["fp32_raw"],
                    row["v1_fp32_raw"],
                ),
                "v1_int8_vs_v1_fp32_onnx": (
                    row["v1_int8"],
                    row["v1_fp32"],
                    row["v1_int8_raw"],
                    row["v1_fp32_raw"],
                ),
            }
            for key, (a, b, ra, rb) in combos.items():
                if b is None:
                    continue
                m = _compare_a(a, b)
                if isinstance(rb, dict):  # PyTorch reference raw
                    same_shape = list(ra.input_ids) == rb["ids"]
                    tl, gl = rb["type_logits"], rb["tag_logits"]
                else:
                    same_shape = ra.input_ids == rb.input_ids
                    tl, gl = rb.type_logits, rb.tag_logits
                if same_shape:
                    m["type_logit_diff"] = maxabs(ra.type_logits, tl)
                    m["tag_logit_diff"] = maxabs(ra.tag_logits, gl)
                    m["logits_bit_identical"] = bool(
                        np.array_equal(ra.type_logits, tl) and np.array_equal(ra.tag_logits, gl)
                    )
                else:
                    m["logits_bit_identical"] = False
                pairs[key].append(m)
                bad = [k for k in ("type_equal", "target_equal", "target_span_equal") if not m[k]]
                if bad:
                    side = {
                        "a": {
                            k: a[k] for k in (*V1_FIELDS, "type_confidence", "target_confidence")
                        },
                        "b": {
                            k: b[k] for k in (*V1_FIELDS, "type_confidence", "target_confidence")
                        },
                    }
                    listed[key].append({"id": note_id, "text": text, "fields": bad, **side})
                    mm.add(f"path_a:{key}", name, note_id, text, bad, **side)
        out[name] = {
            key: {
                "n_notes": len(ms),
                "type_agreement": rate(ms, "type_equal"),
                "target_text_agreement": rate(ms, "target_equal"),
                "target_span_agreement": rate(ms, "target_span_equal"),
                "type_confidence_max_abs_diff": peak(ms, "type_conf_diff"),
                "target_confidence_max_abs_diff": peak(ms, "target_conf_diff"),
                "confidences_bit_identical_rate": rate(ms, "confidences_identical"),
                "type_logits_max_abs_diff": peak(ms, "type_logit_diff"),
                "target_logits_max_abs_diff": peak(ms, "tag_logit_diff"),
                "logits_bit_identical_rate": rate(ms, "logits_bit_identical"),
                "n_mismatching_notes": len(listed[key]),
                "mismatches": listed[key],
            }
            for key, ms in pairs.items()
        }
    return out


# ---------------------------------------------------------------------------------------------
# held-out value quality per backend


def _deep_diff(a: Any, b: Any, path: str = "") -> list[str]:
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for k in sorted(set(a) | set(b), key=str):
            if k not in a or k not in b:
                out.append(f"{path}/{k}: missing on one side")
            else:
                out += _deep_diff(a[k], b[k], f"{path}/{k}")
        return out
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        return [] if a == b else [f"{path}: {a!r} != {b!r}"]
    return [] if a == b else [f"{path}: {a!r} != {b!r}"]


def value_quality_section(ev: Evidence) -> dict:
    import torch
    from compare_value_v1 import load_sets
    from evaluate_frozen_value import TRACKED_NOTES

    from gidi.evaluation.value_eval import evaluate_records, value_span_categories
    from gidi.evaluation.value_gate import (
        multi_number_errors,
        seen_value_texts,
        summarize_set,
        value_error_category,
        value_error_table,
        value_quality_criteria,
    )
    from gidi.evaluation.value_predict import torch_logits

    env = ev.env
    sets = load_sets(EVAL_DIR)
    train_records = read_jsonl(EVAL_DIR / "train.jsonl")
    seen = seen_value_texts(train_records)
    categories = value_span_categories()
    ref = env.torch_ref
    max_length = int(ref.meta["max_length"])
    device = torch.device("cpu")

    def runtime_logits(predictor: Any):
        def logits(data: Any):
            types, tags, values = [], [], []
            lengths = data.encoded.attention_mask.sum(dim=1).tolist()
            for record, n in zip(data.records, lengths, strict=True):
                raw = predictor.run(record["text"])
                if len(raw.input_ids) != n:
                    raise RuntimeError(f"{record['id']}: runtime has {len(raw.input_ids)} tokens")
                types.append(raw.type_logits)
                tags.append(raw.tag_logits)
                values.append(raw.value_logits)
            return np.stack(types), tags, values

        return logits

    backends = {
        "pytorch": (lambda data: torch_logits(ref.model, data, device), ref.crf),
        "onnx_fp32": (runtime_logits(env.fp32), env.fp32._crf),
        "onnx_int8": (runtime_logits(env.int8), env.int8._crf),
    }
    v7 = json.loads(V7_COMPARISON.read_text(encoding="utf-8"))["per_seed"]["1"]
    out: dict[str, Any] = {
        "criteria_code": "gidi.evaluation.value_gate.value_quality_criteria (frozen value-span-v7)",
        "decoder": "gidi.modeling.value_decode.decode_value(..., crf=) via "
        "gidi.evaluation.value_eval.evaluate_records; ONNX backends feed the runtime's batch-1 "
        "run() logits and the bundle config's CRF",
        "sets": {name: len(sets[name]) for name in EVAL_SETS},
    }
    all_rows: dict[str, dict[str, list]] = {}
    for backend, (logits_fn, crf) in backends.items():
        summaries, rows_by_set = {}, {}
        for name in EVAL_SETS:
            records = sets[name]
            metrics, rows, _ = evaluate_records(
                records,
                logits_fn,
                ref.tokenizer,
                max_length,
                train_records=train_records,
                categories=categories,
                crf=crf,
            )
            summaries[name] = summarize_set(
                metrics, rows, records, with_value=True, seen_texts=seen, regression_flags=None
            )
            rows_by_set[name] = rows
        all_rows[backend] = rows_by_set
        criteria = value_quality_criteria({1: summaries}, [1], backend)
        counts, errors = value_error_table({1: rows_by_set}, ("test", "probe"))
        out[backend] = {
            "headline": {name: _headline(summaries[name]) for name in EVAL_SETS},
            "summaries": summaries,
            "frozen_criteria": criteria,
            "frozen_criteria_all_pass": all(c["pass"] for c in criteria),
            "error_category_counts": {
                s: {c: v[1] for c, v in by_cat.items() if v[1]} for s, by_cat in counts.items()
            },
            "value_errors": [{k: v for k, v in e.items() if k != "seed"} for e in errors],
            "multi_number_errors": [
                {k: v for k, v in e.items() if k != "seed"}
                for name in ("test", "probe")
                for e in multi_number_errors(name, 1, rows_by_set[name])
            ],
            "tracked_notes": {
                text: [
                    {
                        "set": name,
                        "gold_value": r["gold_value"],
                        "pred_value": r["pred_value"],
                        "value_ok": r["value_ok"],
                        "category": value_error_category(
                            r["text"], r["gold_value"], r["pred_value"]
                        ),
                    }
                    for name in ("test", "probe")
                    for r in rows_by_set[name]
                    if r["text"] == text
                ]
                for text in TRACKED_NOTES
            },
        }
    diffs = {name: _deep_diff(out["pytorch"]["summaries"][name], v7[name]) for name in EVAL_SETS}
    out["pytorch_reproduces_value_span_v7_seed1"] = {
        "source": str(V7_COMPARISON.relative_to(ROOT)) + " per_seed.1",
        "differences": diffs,
        "identical": not any(diffs.values()),
    }
    out["note_level_value_changes"] = {
        f"{a}_to_{b}": [
            {
                "set": name,
                "id": ra["id"],
                "text": ra["text"],
                "gold_value": ra["gold_value"],
                a: ra["pred_value"],
                b: rb["pred_value"],
            }
            for name in EVAL_SETS
            if name in ("test", "probe")
            for ra, rb in zip(all_rows[a][name], all_rows[b][name], strict=True)
            if (ra["pred_value"] or {}).get("start") != (rb["pred_value"] or {}).get("start")
            or (ra["pred_value"] or {}).get("end") != (rb["pred_value"] or {}).get("end")
        ]
        for a, b in (("pytorch", "onnx_fp32"), ("onnx_fp32", "onnx_int8"), ("pytorch", "onnx_int8"))
    }
    return out


def _headline(s: dict) -> dict:
    v = s["value"]
    return {
        "n": s["n"],
        "value_n_complete": v["n"],
        "value_exact": v["exact"],
        "span_precision": v["span_precision"],
        "span_recall": v["span_recall"],
        "span_f1": v["span_f1"],
        "token_f1": v["token_f1"],
        "present_null_accuracy": v["present_accuracy"],
        "human_n": v["human"]["n"],
        "human_exact": v["human"]["exact"],
        "multi_number_n": s["value_slices"]["multi_number"]["n"],
        "multi_number_exact": s["value_slices"]["multi_number"]["exact"],
        "slices_exact": {
            k: {"n": b["n"], "exact": b["exact"]} for k, b in s["value_slices"].items()
        },
        "type_accuracy": s["type_accuracy"],
        "target_exact": s["target_exact"],
    }


# ---------------------------------------------------------------------------------------------
# Phase 5: hardening


def reference_viterbi(emissions: np.ndarray, start: Any, end: Any, trans: Any) -> list[int]:
    """Straightforward Viterbi (python loops, float64); independent of ``gidi.inference.crf``."""
    k = emissions.shape[1]
    if len(emissions) == 0:
        return []
    score = [float(start[j]) + float(emissions[0, j]) for j in range(k)]
    back: list[list[int]] = []
    for t in range(1, len(emissions)):
        new, ptr = [], []
        for cur in range(k):
            cands = [score[p] + float(trans[p][cur]) + float(emissions[t, cur]) for p in range(k)]
            best = max(range(k), key=lambda p: (cands[p], -p))  # ties: lowest previous tag
            ptr.append(best)
            new.append(cands[best])
        score, back = new, back + [ptr]
    final = [score[j] + float(end[j]) for j in range(k)]
    best = max(range(k), key=lambda j: (final[j], -j))
    path = [best]
    for ptr in reversed(back):
        best = ptr[best]
        path.append(best)
    return path[::-1]


def path_score(emissions: np.ndarray, tags: list[int], start: Any, end: Any, trans: Any) -> float:
    if not tags:
        return 0.0
    s = float(start[tags[0]]) + float(emissions[0, tags[0]])
    for t in range(1, len(tags)):
        s += float(trans[tags[t - 1]][tags[t]]) + float(emissions[t, tags[t]])
    return s + float(end[tags[-1]])


def bio_spans(tags: list[int], offsets: Any, special: Any, text: str) -> list[dict]:
    """Value spans of a BIO tag sequence (the protocol rule, written out independently)."""
    spans: list[dict] = []
    cur: dict | None = None
    for i, (tag, (s, e), sp) in enumerate(zip(tags, offsets, special, strict=True)):
        if sp:
            continue
        while s < e and text[s].isspace():
            s += 1
        while e > s and text[e - 1].isspace():
            e -= 1
        if s >= e:
            continue
        if tag == 1:
            if cur:
                spans.append(cur)
            cur = {"start": s, "end": e, "members": [i]}
        elif tag == 2:
            if cur is None:
                cur = {"start": s, "end": e, "members": [i]}
            else:
                cur["start"], cur["end"] = min(cur["start"], s), max(cur["end"], e)
                cur["members"].append(i)
        else:
            if cur:
                spans.append(cur)
            cur = None
    if cur:
        spans.append(cur)
    return spans


def derive_value(raw: Any) -> tuple[dict | None, float]:
    """Value span (NFC offsets) and confidence re-derived from ``raw.value_tags`` and the
    emissions per ``protocol.value_decoding_rule`` with plain numpy."""
    logits = np.asarray(raw.value_logits, dtype=np.float64)
    shifted = logits - logits.max(axis=-1, keepdims=True)
    logp = shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
    tags = list(raw.value_tags)
    special = list(raw.special_tokens_mask)
    spans = bio_spans(tags, raw.normalized_offsets, special, raw.normalized_text)
    if not spans:
        real = [i for i, s in enumerate(special) if not s]
        conf = math.exp(min(logp[i, 0] for i in real)) if real else 1.0
        return None, conf
    scored = [
        (math.exp(sum(logp[i, tags[i]] for i in span["members"]) / len(span["members"])), span)
        for span in spans
    ]
    best = 0
    for k, (c, _) in enumerate(scored):
        if c > scored[best][0]:
            best = k  # strictly greater: ties keep the earlier span
    return scored[best][1], scored[best][0]


class TokenCounter:
    """Independent token counts: HF training tokenizer and the checkpoint ``tokenizer.json``."""

    def __init__(self, torch_ref: Any, checkpoint: Path) -> None:
        from tokenizers import Tokenizer

        self.hf = torch_ref.tokenizer
        self.raw = Tokenizer.from_file(str(checkpoint / "tokenizer.json"))
        self.raw.no_truncation()
        self.raw.no_padding()
        self._surrogate = re.compile("[\ud800-\udfff]")

    def count(self, text: str) -> tuple[int | None, int, Any]:
        """(HF full count or None when HF rejects the text, raw-tokenizers full count, raw enc)."""
        nfc = unicodedata.normalize("NFC", text)
        enc = self.raw.encode(self._surrogate.sub("\ufffd", nfc))
        try:
            hf = len(self.hf(nfc, truncation=False)["input_ids"])
        except Exception:
            hf = None
        return hf, len(enc.ids), enc


def classify_cut(enc: Any, text_nfc: str, value_text: str | None) -> str | None:
    """Where ``value_text`` (first occurrence) lies relative to the 32-token cut."""
    if not value_text:
        return None
    start = text_nfc.find(unicodedata.normalize("NFC", value_text))
    if start < 0:
        return None
    end = start + len(unicodedata.normalize("NFC", value_text))
    real = [i for i, sp in enumerate(enc.special_tokens_mask) if not sp]
    kept = set(real[: MAX_LENGTH - 2])
    inside = [i for i in real if enc.offsets[i][0] < end and enc.offsets[i][1] > start]
    if not inside:
        return None
    flags = [i in kept for i in inside]
    if all(flags):
        return "at-end" if inside[-1] == real[: MAX_LENGTH - 2][-1] else "before"
    return "across" if any(flags) else "after"


def check_text(env: Env, vd: Any, counter: TokenCounter, item: dict) -> dict:
    """All per-input runtime invariants on the INT8 runtime; ``failures`` lists what broke."""
    from gidi.inference import EmptyInputError
    from gidi.inference.text import normalize_nfc

    text = item["text"]
    predictor = env.int8
    failures: list[str] = []
    row: dict[str, Any] = {
        "case": item["case"],
        "group": item["group"],
        "n_code_points": len(text),
        "text_repr": repr(text) if len(text) <= 100 else repr(text[:97]) + "...",
    }
    expect_empty = bool(item.get("expect_empty_error"))
    try:
        pred = predictor.predict(text)
    except EmptyInputError:
        row["raised"] = "EmptyInputError"
        if not expect_empty:
            failures.append("unexpected EmptyInputError")
        try:
            predictor.run(text)
            failures.append("run() did not raise EmptyInputError")
        except EmptyInputError:
            pass
        row["failures"] = failures
        return row
    except Exception as exc:
        row["failures"] = [f"crash: {type(exc).__name__}: {exc}"]
        return row
    if expect_empty:
        failures.append("expected EmptyInputError, got a prediction")
    d = pred.to_dict()
    row["prediction"] = d
    failures += [f"dict: {p}" for p in vd.validate_prediction_dict(text, d)]
    for key in ("type_confidence", "target_confidence", "value_confidence"):
        v = d[key]
        if not (isinstance(v, float) and math.isfinite(v) and 0.0 <= v <= 1.0):
            failures.append(f"{key}={v!r} not a finite float in [0, 1]")
    for label in ("target", "value"):
        span = d[f"{label}_span"]
        text_key = "target" if label == "target" else "value_text"
        if span is not None:
            s, e = span
            if not s < e:
                failures.append(f"{label} span {span} has start >= end")
            if text[s:e] != d[text_key]:
                failures.append(f"text[{s}:{e}] != returned {label}")
        elif d[text_key] is not None:
            failures.append(f"{label} text without span")
    try:
        encoded = json.dumps(d, allow_nan=False)
        if json.loads(encoded) != d:
            failures.append("json round trip changed the dict")
        json.dumps(d, allow_nan=False, ensure_ascii=False)
    except (ValueError, TypeError) as exc:
        failures.append(f"json: {exc}")

    raw = predictor.run(text)
    nfc = unicodedata.normalize("NFC", text)
    if raw.normalized_text != nfc:
        failures.append("normalized_text != unicodedata NFC")
    hf_count, raw_count, enc = counter.count(text)
    row["n_tokens_full"] = raw_count
    row["n_tokens_kept"] = len(raw.input_ids)
    if hf_count is not None and hf_count != raw_count:
        failures.append(f"independent tokenizers disagree: HF {hf_count} vs tokenizers {raw_count}")
    if d["truncated"] != (raw_count > MAX_LENGTH) or raw.truncated != d["truncated"]:
        failures.append(f"truncated={d['truncated']} but {raw_count} tokens (max {MAX_LENGTH})")
    if len(raw.input_ids) != min(raw_count, MAX_LENGTH):
        failures.append(f"kept {len(raw.input_ids)} tokens, expected {min(raw_count, MAX_LENGTH)}")
    # Viterbi tags
    tags = list(raw.value_tags)
    if len(tags) != len(raw.input_ids) or not all(
        isinstance(t, int) and not isinstance(t, bool) and t in (0, 1, 2) for t in tags
    ):
        failures.append(f"value_tags invalid: {tags}")
    else:
        if any(t != 0 for t, sp in zip(tags, raw.special_tokens_mask, strict=True) if sp):
            failures.append("special token tagged non-O")
        crf = predictor._crf
        real = [i for i, sp in enumerate(raw.special_tokens_mask) if not sp]
        ref = reference_viterbi(
            np.asarray(raw.value_logits, dtype=np.float64)[real],
            crf.start,
            crf.end,
            crf.transitions,
        )
        if [tags[i] for i in real] != ref:
            failures.append("Viterbi path differs from the independent reference Viterbi")
        span, conf = derive_value(raw)
        norm = normalize_nfc(text)
        if span is None:
            if d["value_span"] is not None:
                failures.append("value span returned but BIO tags contain none")
        else:
            want = list(norm.span_to_original(span["start"], span["end"]))
            if d["value_span"] != want:
                failures.append(f"value span {d['value_span']} != re-derived {want}")
        if abs(conf - d["value_confidence"]) > 1e-9:
            failures.append(f"value_confidence {d['value_confidence']} != re-derived {conf}")
    # spans stay inside the kept region (nothing past the cut)
    real_offsets = [o for o, sp in zip(raw.offsets, raw.special_tokens_mask, strict=True) if not sp]
    for label, span in (("target", d["target_span"]), ("value", d["value_span"])):
        if span is None:
            continue
        if (
            not real_offsets
            or span[0] < min(o[0] for o in real_offsets)
            or span[1] > max(o[1] for o in real_offsets)
        ):
            failures.append(f"{label} span {span} lies outside the kept tokens")
    if item.get("cut"):
        actual = classify_cut(enc, nfc, item.get("value_text"))
        row["value_vs_cut"] = actual
        if actual != item["cut"]:
            failures.append(f"case design: value is '{actual}' the cut, case says '{item['cut']}'")
    if predictor.predict(text).to_dict() != d:
        failures.append("second predict() differs")
    row["failures"] = failures
    return row


def fuzz_decode(crf: Any, cases: int = FUZZ_DECODE_CASES, seed: int = FUZZ_SEED) -> dict:
    """Seeded random emissions / transitions / offsets / masks through ``decode_value_crf``."""
    from gidi.inference.crf import CRFTransitions, viterbi, viterbi_masked
    from gidi.inference.decode import decode_value_crf

    rng = np.random.default_rng(seed)
    alphabet = list("abc 12  \t\n.,k") + ["triệu", " ", "đ", "\u00a0", "\u3000"]
    failures: list[str] = []
    stats: Counter[str] = Counter()
    for case in range(cases):
        n_chars = int(rng.integers(0, 60))
        text = "".join(alphabet[int(rng.integers(len(alphabet)))] for _ in range(n_chars))
        t = int(rng.integers(0, 40))
        # monotone, possibly empty / whitespace-only / overlapping token offsets
        cuts = sorted(int(rng.integers(0, n_chars + 1)) for _ in range(2 * t))
        offsets = [(cuts[2 * i], cuts[2 * i + 1]) for i in range(t)]
        mode = int(rng.integers(4))
        if mode == 0 and t >= 2:
            offsets[0], offsets[-1] = (0, 0), (0, 0)
            real = [i not in (0, t - 1) for i in range(t)]
        elif mode == 1:
            real = [bool(rng.integers(2)) for _ in range(t)]
        elif mode == 2:
            real = [False] * t
        else:
            real = [True] * t
        scale = float(rng.choice([0.0, 0.1, 1.0, 10.0, 100.0, 1e4]))
        emissions = (rng.normal(size=(t, 3)) * scale).astype(np.float32)
        cscale = float(rng.choice([0.0, 0.1, 1.0, 10.0]))
        params = CRFTransitions(
            (rng.normal(size=3) * cscale).astype(np.float32),
            (rng.normal(size=3) * cscale).astype(np.float32),
            (rng.normal(size=(3, 3)) * cscale).astype(np.float32),
        )
        try:
            span, conf = decode_value_crf(emissions, offsets, text, real, params)
            tags = viterbi_masked(emissions, real, params, fill=0)
        except Exception as exc:
            failures.append(f"case {case}: crash {type(exc).__name__}: {exc}")
            continue
        stats["cases"] += 1
        if not (isinstance(conf, float) and math.isfinite(conf) and 0.0 <= conf <= 1.0):
            failures.append(f"case {case}: confidence {conf!r}")
        if len(tags) != t or any(x not in (0, 1, 2) for x in tags):
            failures.append(f"case {case}: bad tags {tags}")
        if any(tag != 0 for tag, r in zip(tags, real, strict=True) if not r):
            failures.append(f"case {case}: non-real position tagged")
        if span is None:
            stats["no_span"] += 1
        else:
            stats["span"] += 1
            members = list(span.members)
            trimmed = [_trim(text, *offsets[i]) for i in members]
            ok = 0 <= span.start < span.end <= len(text)
            ok = ok and not text[span.start].isspace() and not text[span.end - 1].isspace()
            ok = ok and bool(members) and all(0 <= i < t and real[i] for i in members)
            ok = ok and all(s < e for s, e in trimmed)
            ok = ok and (span.start, span.end) == (
                min(s for s, _ in trimmed),
                max(e for _, e in trimmed),
            )
            if not ok:
                failures.append(f"case {case}: malformed span {span}")
        # Viterbi optimality against brute force on short real sequences
        idx = [i for i, r in enumerate(real) if r]
        if 0 < len(idx) <= 7:
            em = emissions[idx].astype(np.float64)
            got = [tags[i] for i in idx]
            best = max(
                path_score(em, list(p), params.start, params.end, params.transitions)
                for p in itertools.product(range(3), repeat=len(idx))
            )
            if abs(
                path_score(em, got, params.start, params.end, params.transitions) - best
            ) > 1e-6 * max(1.0, abs(best)):
                failures.append(f"case {case}: Viterbi path is not optimal")
            stats["brute_force_checked"] += 1
            if got != reference_viterbi(em, params.start, params.end, params.transitions):
                failures.append(f"case {case}: Viterbi differs from reference implementation")
        if idx and viterbi(emissions[idx], params) != [tags[i] for i in idx]:
            failures.append(f"case {case}: viterbi vs viterbi_masked")
    # the deployed CRF parameters too
    deployed_ok = True
    for case in range(500):
        t = int(rng.integers(1, 33))
        emissions = (rng.normal(size=(t, 3)) * float(rng.choice([0.1, 1, 10, 100]))).astype(
            np.float32
        )
        got = viterbi(emissions, crf)
        ref = reference_viterbi(emissions, crf.start, crf.end, crf.transitions)
        if got != ref:
            deployed_ok = False
            failures.append(f"deployed CRF case {case}: differs from reference Viterbi")
    return {
        "seed": seed,
        "cases": stats["cases"],
        "with_span": stats["span"],
        "without_span": stats["no_span"],
        "brute_force_optimality_checked": stats["brute_force_checked"],
        "deployed_crf_cases_vs_reference_viterbi": 500,
        "deployed_crf_ok": deployed_ok,
        "n_failures": len(failures),
        "failures": failures[:50],
        "passed": not failures,
    }


def random_texts(n: int, seed: int = FUZZ_SEED) -> list[str]:
    rng = np.random.default_rng(seed + 1)
    words = [
        "mượn", "chị", "Hạnh", "2", "triệu", "500", "nghìn", "50k", "1tr5", "5", "củ", "trả",
        "nợ", "anh", "Nam", "20/10", "cho", "vay", "ăn", "phở", "70", "ship", "t10", "hẹn",
        "muon", "chi", "Thao", "3", "xị", "đ", "VND", "lương", "15.500.000đ",
    ]  # fmt: skip
    noise = [
        "!!", "...", "\u201c", "\u201d", "\u2026", "\u2013", "\uff01", "\U0001f4b8", "\u0301",
        "\u0323", "\u200b", "\u3000", "\t", "\n", "  ", "(", ")", "/", "%", "#",
    ]  # fmt: skip
    out = []
    for _ in range(n):
        parts = []
        for _ in range(int(rng.integers(1, 45))):
            pool = noise if rng.random() < 0.15 else words
            parts.append(pool[int(rng.integers(len(pool)))])
        sep = " " if rng.random() < 0.8 else ""
        text = sep.join(parts)
        if rng.random() < 0.3:
            text = unicodedata.normalize("NFD", text)
        out.append(text)
    return out


def hardening_section(env: Env, vd: Any) -> dict:
    counter = TokenCounter(env.torch_ref, env.checkpoint)
    cases = read_jsonl(env.deploy_dir / "hardening-cases.jsonl")
    cases += [
        {
            "case": "lone-surrogate-high",
            "group": "unicode",
            "text": "ăn sáng \ud83d 35k",
            "expect_empty_error": False,
            "value_text": None,
        },
        {
            "case": "lone-surrogate-low",
            "group": "unicode",
            "text": "\udc00 mượn anh Nam 2 triệu",
            "expect_empty_error": False,
            "value_text": None,
        },
    ]
    robust = [
        {
            "case": r["id"],
            "group": "robustness-inputs",
            "text": r["text"],
            "expect_empty_error": r["expect_empty_error"],
            "value_text": None,
        }
        for r in read_jsonl(env.deploy_dir / "robustness-inputs.jsonl")
    ]
    fuzz_texts = [
        {
            "case": f"text-fuzz-{i:03d}",
            "group": "text-fuzz",
            "text": t,
            "expect_empty_error": False,
            "value_text": None,
        }
        for i, t in enumerate(random_texts(FUZZ_TEXT_CASES))
    ]
    results = [check_text(env, vd, counter, c) for c in cases + robust + fuzz_texts]
    by_group: dict[str, dict] = {}
    for r in results:
        g = by_group.setdefault(r["group"], {"n": 0, "passed": 0})
        g["n"] += 1
        g["passed"] += not r["failures"]
    failed = [r for r in results if r["failures"]]
    return {
        "bundle_runtime": "INT8 bundle runtime",
        "n_checks": len(results),
        "n_passed": len(results) - len(failed),
        "n_failed": len(failed),
        "by_group": by_group,
        "checked_per_input": [
            "no crash (EmptyInputError for empty / whitespace-only)",
            "spans in range, start < end, no edge whitespace",
            "text[start:end] == returned target / value (caller's original string)",
            "truncated == (independent token count > 32); kept tokens == min(count, 32)",
            "confidences finite floats in [0, 1]",
            "json.dumps(allow_nan=False) round trip",
            "Viterbi tags ints in {0,1,2}, specials O, equal to an independent Viterbi",
            "value span + confidence re-derived from the tags per the protocol rule",
            "spans inside the kept (untruncated) tokens",
            "repeat predict() identical",
        ],
        "failures": [
            {
                "case": r["case"],
                "group": r["group"],
                "text_repr": r["text_repr"],
                "failures": r["failures"],
            }
            for r in failed
        ],
        "decode_fuzz": fuzz_decode(env.int8._crf),
        "results": results,
    }


# ---------------------------------------------------------------------------------------------
# Phase 6: performance, single tokenization


def _stats_ms(times_ns: list[int]) -> dict:
    ms = [t / 1e6 for t in times_ns]
    ordered = sorted(ms)
    return {
        "p50_ms": statistics.median(ms),
        "p95_ms": ordered[min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))],
        "mean_ms": statistics.fmean(ms),
        "calls": len(ms),
    }


def _bench(fn: Any, n_texts: int) -> dict:
    for i in range(COMPONENT_WARMUP):
        fn(i % n_texts)
    times = []
    for i in range(COMPONENT_CALLS):
        t0 = time.perf_counter_ns()
        fn(i % n_texts)
        times.append(time.perf_counter_ns() - t0)
    return _stats_ms(times)


def component_timings(predictor: Any, texts: list[str]) -> dict:
    from gidi.inference import predictor as pmod
    from gidi.inference.crf import viterbi_masked
    from gidi.inference.decode import (
        decode_first_span,
        decode_value_crf,
        target_confidence,
        type_prediction,
    )

    tokenizer, runner, crf = predictor._tokenizer, predictor._runner, predictor._crf
    prepared = []
    for text in texts:
        norm = pmod.normalize_nfc(text)
        tok = tokenizer.encode(norm.text)
        type_logits, tag_logits, *rest = runner(tok.ids)
        prepared.append((text, norm, tok, type_logits, tag_logits, rest[0] if rest else None))
    n = len(prepared)
    out: dict[str, Any] = {
        "note": "each component timed alone on the golden texts: warm-up then timed calls, "
        "batch 1, 1 thread; components do not sum exactly to predict() (call overhead, slicing)",
        "normalize_nfc": _bench(lambda i: pmod.normalize_nfc(prepared[i][0]), n),
        "tokenizer_encode": _bench(lambda i: tokenizer.encode(prepared[i][1].text), n),
        "normalize_plus_tokenize": _bench(
            lambda i: tokenizer.encode(pmod.normalize_nfc(prepared[i][0]).text), n
        ),
        "onnx_run": _bench(lambda i: runner(prepared[i][2].ids), n),
    }

    def target_part(i: int) -> None:
        _, norm, tok, type_logits, tag_logits, _ = prepared[i]
        real = [not s for s in tok.special_tokens_mask]
        type_prediction(type_logits)
        tag_ids = np.argmax(tag_logits, axis=-1)
        span = decode_first_span(tok.offsets, tag_ids, norm.text)
        target_confidence(tag_logits, tag_ids, span, real)

    out["type_and_target_decode"] = _bench(target_part, n)
    if crf is not None:
        out["viterbi_only"] = _bench(
            lambda i: viterbi_masked(
                prepared[i][5], [not s for s in prepared[i][2].special_tokens_mask], crf, fill=0
            ),
            n,
        )
        out["value_crf_decode"] = _bench(
            lambda i: decode_value_crf(
                prepared[i][5],
                prepared[i][2].offsets,
                prepared[i][1].text,
                [not s for s in prepared[i][2].special_tokens_mask],
                crf,
            ),
            n,
        )
    out["predict_total"] = _bench(lambda i: predictor.predict(prepared[i][0]), n)
    out["predict_total_second_measurement"] = _bench(lambda i: predictor.predict(prepared[i][0]), n)
    parts = ["normalize_plus_tokenize", "onnx_run", "type_and_target_decode", "value_crf_decode"]
    out["sum_of_component_p50_ms"] = sum(out[p]["p50_ms"] for p in parts if p in out)
    return out


def single_tokenization(env: Env, texts: list[str]) -> dict:
    """One ``predict()`` must normalize once, tokenize once and run the ONNX session once."""
    from gidi.inference import predictor as pmod

    predictor = env.int8
    calls: Counter[str] = Counter()
    feeds: list[dict] = []
    encoded: list[Any] = []
    original_norm = pmod.normalize_nfc
    original_encode = predictor._tokenizer.encode
    inner = predictor._runner._session

    class CountingSession:
        def run(self, names: Any, feed: dict) -> Any:
            calls["session_run"] += 1
            feeds.append(feed)
            return inner.run(names, feed)

        def __getattr__(self, name: str) -> Any:
            return getattr(inner, name)

    def counted_norm(text: str) -> Any:
        calls["normalize_nfc"] += 1
        return original_norm(text)

    def counted_encode(text: str) -> Any:
        calls["tokenizer_encode"] += 1
        enc = original_encode(text)
        encoded.append(enc)
        return enc

    rows = []
    pmod.normalize_nfc = counted_norm
    predictor._tokenizer.encode = counted_encode
    predictor._runner._session = CountingSession()
    try:
        for text in texts:
            for label, fn in (("predict", predictor.predict), ("run", predictor.run)):
                calls.clear()
                feeds.clear()
                encoded.clear()
                fn(text)
                feed = feeds[0] if feeds else {}
                ids_match = bool(feeds) and feed["input_ids"][0].tolist() == list(encoded[0].ids)
                rows.append(
                    {
                        "entry_point": label,
                        "normalize_nfc": calls["normalize_nfc"],
                        "tokenizer_encode": calls["tokenizer_encode"],
                        "session_run": calls["session_run"],
                        "feed_keys": sorted(feed),
                        "feed_input_ids_equal_encoder_output": ids_match,
                        "attention_mask_all_ones_same_length": bool(feeds)
                        and feed["attention_mask"].shape == feed["input_ids"].shape
                        and bool((feed["attention_mask"] == 1).all()),
                    }
                )
    finally:
        pmod.normalize_nfc = original_norm
        predictor._tokenizer.encode = original_encode
        predictor._runner._session = inner
    once = all(r["normalize_nfc"] == r["tokenizer_encode"] == r["session_run"] == 1 for r in rows)
    return {
        "method": "monkeypatched counters on gidi.inference.predictor.normalize_nfc, "
        "BundleTokenizer.encode and the ORT session's run; each text through predict() and run()",
        "texts": len(texts),
        "calls_checked": len(rows),
        "exactly_once_each": once,
        "feed_keys_all": sorted({tuple(r["feed_keys"]) for r in rows}),
        "feed_is_the_tokenizer_output": all(r["feed_input_ids_equal_encoder_output"] for r in rows),
        "attention_mask_all_ones": all(r["attention_mask_all_ones_same_length"] for r in rows),
        "graph": graph_inspection(env),
    }


def graph_inspection(env: Env) -> dict:
    """Both encoders' word embeddings gather from the one ``input_ids`` graph input."""
    try:
        import onnx
    except ImportError:
        return {"available": False}
    import collections

    out: dict[str, Any] = {"available": True}
    for label, path in (("fp32", env.fp32_onnx), ("int8", env.bundle / "model.int8.onnx")):
        graph = onnx.load(str(path), load_external_data=False).graph
        init = {i.name: list(i.dims) for i in graph.initializer}
        consumers: dict[str, list] = collections.defaultdict(list)
        for node in graph.node:
            for name in node.input:
                consumers[name].append(node)
        word_gathers = [
            {
                "node": n.name,
                "table": n.input[0],
                "table_shape": init[n.input[0]],
                "indices": n.input[1],
            }
            for n in graph.node
            if n.op_type == "Gather" and "word_embeddings" in n.input[0] and n.input[0] in init
        ]
        position_gathers = [
            {"node": n.name, "table": n.input[0], "indices": n.input[1]}
            for n in graph.node
            if n.op_type == "Gather" and "position_embeddings" in n.input[0]
        ]
        out[label] = {
            "graph_inputs": [i.name for i in graph.input],
            "graph_outputs": [o.name for o in graph.output],
            "input_ids_consumers": [f"{n.op_type}:{n.name}" for n in consumers["input_ids"]],
            "attention_mask_consumers": [
                f"{n.op_type}:{n.name}" for n in consumers["attention_mask"]
            ],
            "word_embedding_gathers": word_gathers,
            "position_embedding_gathers": position_gathers,
            "both_word_embeddings_read_input_ids": len(word_gathers) == 2
            and all(g["indices"] == "input_ids" for g in word_gathers),
            "position_ids_shared": len({g["indices"] for g in position_gathers}) == 1,
        }
    return out


def performance_v2(
    vd: Any, env: Env, ev: Evidence, performance: dict, golden_texts: list[str]
) -> dict:
    v1_perf = vd.performance_section(golden_texts, ev.v1_int8, None, V1_BUNDLE)
    v2_int8 = env.int8
    sizes = performance["sizes"]
    v1_sizes = v1_perf["sizes"]
    fp32_bytes = env.fp32_onnx.stat().st_size
    config_like = {k: v for k, v in sizes["files_bytes"].items() if k not in ("model.int8.onnx",)}
    out = {
        **performance,
        "v1_bundle": v1_perf,
        "comparison_v1_vs_v2": {
            "same_harness": "vd.performance_section: cold start in fresh subprocesses, 1 thread, "
            "batch 1, warm-up then timed predict() calls over the v2 golden texts for both",
            "warm_p50_ms": {
                "v1_int8": v1_perf["warm_inference_int8"]["p50_ms"],
                "v2_int8": performance["warm_inference_int8"]["p50_ms"],
                "v2_fp32": performance["warm_inference_fp32_for_reference"]["p50_ms"],
            },
            "warm_p95_ms": {
                "v1_int8": v1_perf["warm_inference_int8"]["p95_ms"],
                "v2_int8": performance["warm_inference_int8"]["p95_ms"],
                "v2_fp32": performance["warm_inference_fp32_for_reference"]["p95_ms"],
            },
            "warm_mean_ms": {
                "v1_int8": v1_perf["warm_inference_int8"]["mean_ms"],
                "v2_int8": performance["warm_inference_int8"]["mean_ms"],
                "v2_fp32": performance["warm_inference_fp32_for_reference"]["mean_ms"],
            },
            "cold_start_median_s": {
                key: {
                    "v1": v1_perf["cold_start"][f"median_{key}"],
                    "v2": performance["cold_start"][f"median_{key}"],
                }
                for key in ("import_s", "load_s", "first_predict_s", "ready_s", "process_wall_s")
            },
            "peak_rss_mb_after_200_predicts": {
                "v1": v1_perf["memory"]["peak_mb_after_200_predicts_median"],
                "v2": performance["memory"]["peak_mb_after_200_predicts_median"],
            },
            "current_rss_mb_after_200_predicts": {
                "v1": v1_perf["memory"]["current_rss_mb_after_200_predicts_median"],
                "v2": performance["memory"]["current_rss_mb_after_200_predicts_median"],
            },
        },
        "components_v2_int8": component_timings(v2_int8, golden_texts),
        "components_v1_int8": component_timings(ev.v1_int8, golden_texts),
        "sizes_v2": {
            "bundle_total_bytes": sizes["bundle_total_bytes"],
            "bundle_total_mb": sizes["bundle_total_mb"],
            "int8_bytes": sizes["model_int8_bytes"],
            "int8_mb": sizes["model_int8_mb"],
            "fp32_bytes": fp32_bytes,
            "fp32_mb": fp32_bytes / 1e6,
            "int8_over_fp32": sizes["model_int8_bytes"] / fp32_bytes,
            "tokenizer_and_vocab_map_bytes": sizes["tokenizer_artifacts_bytes"],
            "other_bundle_files_bytes": config_like,
            "v1_bundle_total_bytes": v1_sizes["bundle_total_bytes"],
            "v1_int8_bytes": v1_sizes["model_int8_bytes"],
            "v2_over_v1_bundle": sizes["bundle_total_bytes"] / v1_sizes["bundle_total_bytes"],
        },
        "single_tokenization": single_tokenization(env, golden_texts[:20]),
    }
    return out


# ---------------------------------------------------------------------------------------------
# reproducibility / isolation of v1 (as v1 does)


def v1_freeze_check() -> dict:
    cmd = [
        "uv", "run", "python", "scripts/freeze_deployment.py",
        "--out", "models/gidi-finance-v1", "--check",
    ]  # fmt: skip
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, check=False, timeout=1800)
    return {
        "command": " ".join(cmd),
        "exit_code": proc.returncode,
        "ok": proc.returncode == 0,
        "stdout_tail": proc.stdout[-800:],
        "stderr_tail": proc.stderr[-800:],
    }


# ---------------------------------------------------------------------------------------------
# gate + summary


def protocol_gate(sections: dict, report: dict) -> dict:
    """The literal ``acceptance_gate`` items of protocol.json, evaluated mechanically."""
    parity, i8, pa = sections["parity"], sections["int8_vs_fp32"], sections["path_a_invariance"]
    vq, hard = sections["value_quality"], sections["hardening"]
    perf = sections["performance"]

    def zero_mismatch(name: str) -> bool:
        p = parity[name]
        a, b = p["path_a"], p["path_b"]
        return (
            p["input_ids_identical"]
            and a["type_prediction_agreement"] == 1.0
            and a["target_bio_tag_agreement_notes"] == 1.0
            and a["target_span_agreement"] == 1.0
            and b["viterbi_tag_agreement_notes"] == 1.0
            and b["value_span_agreement"] == 1.0
        )

    def invariant(name: str) -> bool:
        s = pa[name]
        return all(
            s[k][f] == 1.0
            for k in ("v2_int8_vs_v1_int8_bundle", "v2_fp32_vs_v1_torch")
            for f in ("type_agreement", "target_text_agreement", "target_span_agreement")
        )

    def field_ok(name: str) -> bool:
        s = i8[name]
        return all(
            s[k] >= INT8_FIELD_MIN_AGREEMENT
            for k in ("type_agreement", "target_exact_agreement", "value_exact_agreement")
        )

    g = i8["golden"]
    gate = {
        "bundle_reproducible_and_manifest_verified": bool(
            report["reproducibility"]["reproducible"]
        ),
        "bundle_tokenizer_identical_to_training_tokenizer": bool(
            report["tokenizer"]["identical"] and report["robustness"]["torch_vs_fp32_ids_equal_all"]
        ),
        "pytorch_vs_fp32_zero_mismatches_golden_test_probe_robustness": all(
            zero_mismatch(n) for n in PARITY_SETS
        ),
        "path_a_invariance_zero_mismatches_all_sets": all(invariant(n) for n in PARITY_SETS),
        "int8_vs_fp32_golden_zero_type_target_span_value_span_differences": (
            g["type_agreement"] == 1.0
            and g["target_exact_agreement"] == 1.0
            and g["value_exact_agreement"] == 1.0
        ),
        "int8_vs_fp32_test_and_probe_each_field_ge_99pct": field_ok("test") and field_ok("probe"),
        "int8_value_metrics_meet_frozen_v7_criteria": bool(
            vq["onnx_int8"]["frozen_criteria_all_pass"]
        ),
        "runtime_equals_pytorch_reference_decode_fp32": all(
            parity[n]["combined_runtime_vs_reference"]["semantic_agreement"] == 1.0
            for n in PARITY_SETS
        ),
        "offsets_truncation_confidences_null_states_robustness": bool(
            hard["n_failed"] == 0
            and hard["decode_fuzz"]["passed"]
            and report["offsets"]["stable"]
            and report["robustness"]["passed"]
            and perf["single_tokenization"]["exactly_once_each"]
        ),
        "runtime_reads_only_bundle_and_imports_only_numpy_onnxruntime_tokenizers": bool(
            report["isolation"]["isolated"]
        ),
        "int8_size_bundle_size_latency_measured": True,
    }
    gate["all_pass"] = all(gate.values())
    gate["note"] = (
        "mechanical reading of protocol.json acceptance_gate; INT8-vs-FP32 and value-quality "
        "items are reported as raw numbers in their sections, the verdict is the caller's"
    )
    return gate


def build_summary(sections: dict, report: dict) -> dict:
    parity, i8, pa = sections["parity"], sections["int8_vs_fp32"], sections["path_a_invariance"]
    vq, hard, perf = sections["value_quality"], sections["hardening"], sections["performance"]
    return {
        "torch_vs_fp32": {
            name: {
                "n": parity[name]["n_notes"],
                **{f"A_{k}": v for k, v in parity[name]["path_a"].items()},
                **{f"B_{k}": v for k, v in parity[name]["path_b"].items()},
                "combined_semantic_agreement": parity[name]["combined_runtime_vs_reference"][
                    "semantic_agreement"
                ],
                "n_mismatching_notes": parity[name]["n_mismatching_notes_any_check"],
            }
            for name in PARITY_SETS
        },
        "int8_vs_fp32": {
            name: {
                k: i8[name][k]
                for k in (
                    "n_notes",
                    "type_agreement",
                    "target_exact_agreement",
                    "value_exact_agreement",
                    "full_output_agreement",
                    "viterbi_tag_agreement_notes",
                    "n_disagreeing_notes",
                )
            }
            | {
                "conf_max_abs": {k: v["max_abs"] for k, v in i8[name]["confidence_drift"].items()},
                "conf_mean_abs": {
                    k: v["mean_abs"] for k, v in i8[name]["confidence_drift"].items()
                },
            }
            for name in PARITY_SETS
        },
        "path_a_invariance": {
            name: {
                key: {
                    "type": v["type_agreement"],
                    "target": v["target_text_agreement"],
                    "target_span": v["target_span_agreement"],
                    "conf_bit_identical_rate": v["confidences_bit_identical_rate"],
                    "n_mismatching_notes": v["n_mismatching_notes"],
                }
                for key, v in pa[name].items()
            }
            for name in PARITY_SETS
        },
        "value_quality": {
            backend: {
                "criteria_pass": {
                    c["id"]: [c["value"], c["pass"]] for c in vq[backend]["frozen_criteria"]
                },
                "all_frozen_criteria_pass": vq[backend]["frozen_criteria_all_pass"],
                "headline": {
                    name: {
                        k: h[k]
                        for k in (
                            "value_exact",
                            "span_f1",
                            "token_f1",
                            "present_null_accuracy",
                            "human_exact",
                            "multi_number_exact",
                        )
                    }
                    for name, h in vq[backend]["headline"].items()
                },
                "error_category_counts": vq[backend]["error_category_counts"],
                "tracked_notes": vq[backend]["tracked_notes"],
            }
            for backend in ("pytorch", "onnx_fp32", "onnx_int8")
        }
        | {
            "pytorch_reproduces_v7_seed1": vq["pytorch_reproduces_value_span_v7_seed1"]["identical"]
        },
        "hardening": {
            "n_checks": hard["n_checks"],
            "n_passed": hard["n_passed"],
            "n_failed": hard["n_failed"],
            "by_group": hard["by_group"],
            "decode_fuzz": {
                k: hard["decode_fuzz"][k]
                for k in ("cases", "brute_force_optimality_checked", "n_failures", "passed")
            },
        },
        "performance": {
            "warm_p50_ms": perf["comparison_v1_vs_v2"]["warm_p50_ms"],
            "warm_p95_ms": perf["comparison_v1_vs_v2"]["warm_p95_ms"],
            "cold_start_median_s": perf["comparison_v1_vs_v2"]["cold_start_median_s"],
            "peak_rss_mb_after_200_predicts": perf["comparison_v1_vs_v2"][
                "peak_rss_mb_after_200_predicts"
            ],
            "components_v2_p50_ms": {
                k: v["p50_ms"] for k, v in perf["components_v2_int8"].items() if isinstance(v, dict)
            },
            "sizes": perf["sizes_v2"],
            "single_tokenization_exactly_once": perf["single_tokenization"]["exactly_once_each"],
        },
    }


def run_checks(vd: Any, env: Env, report: dict, out_dir: Path) -> dict:
    """Run every value-head check; ``report`` holds the sections ``verify_deployment`` produced."""
    t0 = time.perf_counter()
    sets = parity_sets(env)
    mm = Mismatches()
    print("value-head checks: collecting evidence (PyTorch, FP32, INT8, v1) ...", flush=True)
    ev = Evidence(env, vd)
    for name in PARITY_SETS:
        for _, text in sets[name]:
            ev.row(text)
    print("  parity / int8 / path A", flush=True)
    sections: dict[str, Any] = {
        "parity": parity_section(ev, sets, mm),
        "int8_vs_fp32": int8_vs_fp32_section(ev, sets, mm),
        "path_a_invariance": path_a_section(ev, sets, mm),
    }
    print("  held-out value quality", flush=True)
    sections["value_quality"] = value_quality_section(ev)
    print("  hardening", flush=True)
    sections["hardening"] = hardening_section(env, vd)
    print("  performance v1 vs v2", flush=True)
    golden_texts = [t for _, t in sets["golden"]]
    sections["performance"] = performance_v2(vd, env, ev, report["performance"], golden_texts)
    sections["reproducibility_v1_freeze"] = v1_freeze_check()
    sections["tracked_notes_outputs"] = tracked_notes(ev, sections["value_quality"])
    sections["evaluation_sets"] = {name: {"n": len(items)} for name, items in sets.items()}
    (out_dir / "mismatches.jsonl").write_text(
        "".join(vd.dumps_json(r) + "\n" for r in mm.rows), encoding="utf-8"
    )
    sections["mismatches_file"] = {"path": "mismatches.jsonl", "rows": len(mm.rows)}
    gate = protocol_gate(sections, {**report, "reproducibility": report["reproducibility"]})
    sections["verify_value_wall_s"] = time.perf_counter() - t0
    return {
        "sections": sections,
        "acceptance": gate,
        "summary": build_summary(sections, report),
    }


def tracked_notes(ev: Evidence, vq: dict) -> dict:
    """Full runtime / reference outputs of the two tracked held-out notes, every backend."""
    from evaluate_frozen_value import TRACKED_NOTES

    out = {}
    for text in TRACKED_NOTES:
        row = ev.row(text)
        torch_pred = None if row["torch"] is None else to_original(text, row["torch"][0])
        out[text] = {
            "pytorch": torch_pred,
            "onnx_fp32": row["fp32"],
            "onnx_int8": row["int8"],
            "value_head_evaluation_rows": {
                b: vq[b]["tracked_notes"][text] for b in ("pytorch", "onnx_fp32", "onnx_int8")
            },
        }
    return out
