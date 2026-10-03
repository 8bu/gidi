"""Gold-label quality of the browser runtime next to the two Python INT8 sessions.

Scores the annotation-v2 test split (and its test-v1 / test-targeted parts) and the probe set with
the *frozen* evaluation code the v7 / deployment-v2 reports used
(``gidi.evaluation.value_eval.evaluate_records`` -> ``summarize_set`` ->
``value_quality_criteria``), once per backend, from the single-note logits each backend produced:

- ``python_default``: stock ``GidiPredictor`` (ORT default ``ORT_ENABLE_ALL``): the frozen v2;
- ``python_basic``: the same runtime with ``ORT_ENABLE_BASIC`` (the graph as written);
- ``web``: onnxruntime-web (``playground/scripts/parity.ts`` wrote ``web-logits.json``).

Decoding is the bundle config's CRF everywhere, so the three differ only in the logits. Writes
``playground/.parity/quality.json``, which ``pnpm parity`` merges into
``experiments/deployment-v2/web-parity.{md,json}``.

    uv run python scripts/dump_web_parity.py && pnpm -C playground parity \\
        && uv run python scripts/score_web_parity_quality.py && pnpm -C playground parity
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "src"))

from compare_value_v1 import load_sets  # noqa: E402
from verify_value_deployment import _headline  # noqa: E402

from gidi.evaluation.value_eval import evaluate_records, value_span_categories  # noqa: E402
from gidi.evaluation.value_gate import (  # noqa: E402
    seen_value_texts,
    summarize_set,
    value_quality_criteria,
)
from gidi.inference.bundle import load_config  # noqa: E402
from gidi.inference.crf import CRFTransitions  # noqa: E402
from gidi.modeling.value import load_value_checkpoint  # noqa: E402

RELEASE = ROOT / "dist" / "releases" / "gidi-finance-v2" / "2.0.2"
EVAL_DIR = ROOT / "datasets" / "annotation-v2" / "training-v1"
CHECKPOINT = ROOT / "models" / "value-span-v7-dual-encoder" / "seed1"
PARITY_DIR = ROOT / "playground" / ".parity"
EVAL_SETS = ("test", "test-v1", "test-targeted", "probe")
BACKENDS = ("python_default", "python_basic", "web")
PAIRS = (
    ("web", "python_default"),
    ("python_basic", "python_default"),
    ("web", "python_basic"),
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def load_crf() -> CRFTransitions:
    with tempfile.TemporaryDirectory() as bundle:
        (Path(bundle) / "config.json").symlink_to(RELEASE / "runtime" / "config.json")
        decoding = load_config(bundle).value_decoding
    assert decoding is not None
    return CRFTransitions(
        np.asarray(decoding.start, dtype=np.float64),
        np.asarray(decoding.end, dtype=np.float64),
        np.asarray(decoding.transitions, dtype=np.float64),
    )


def logits_by_backend(
    expected: list[dict[str, Any]], web: dict[str, Any]
) -> dict[str, dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    """``{backend: {"source/id": (type [8], tag [n,3], value [n,3])}}`` float32 like the runtime."""
    out: dict[str, dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]] = {b: {} for b in BACKENDS}
    for rec in expected:
        if rec["source"] not in ("test", "probe") or "trace" not in rec:
            continue
        key = f"{rec['source']}/{rec['id']}"

        def arrays(type_, tag, value):
            return (
                np.asarray(type_, dtype=np.float32),
                np.asarray(tag, dtype=np.float32).reshape(-1, 3),
                np.asarray(value, dtype=np.float32).reshape(-1, 3),
            )

        trace, default = rec["trace"], rec["default"]
        out["python_basic"][key] = arrays(
            trace["type_logits"], trace["tag_logits"], trace["value_logits"]
        )
        out["python_default"][key] = arrays(
            default["type_logits"], default["tag_logits"], default["value_logits"]
        )
        w = web[key]
        out["web"][key] = arrays(w["type"], w["tag"], w["value"])
    return out


def note_changes(
    records: list[dict[str, Any]],
    rows: dict[str, list[dict[str, Any]]],
    first: str,
    second: str,
) -> dict[str, Any]:
    """Notes whose gold correctness differs between two backends, per field."""
    counts = {f: {"gained": 0, "lost": 0} for f in ("type", "target", "value")}
    notes: list[dict[str, Any]] = []
    for i, record in enumerate(records):
        for field, ok_key, value_key in (
            ("type", "type_ok", "pred_type"),
            ("target", "target_ok", "pred_target"),
            ("value", "value_ok", "pred_value"),
        ):
            a, b = rows[first][i], rows[second][i]
            if bool(a[ok_key]) == bool(b[ok_key]):
                continue
            counts[field]["gained" if a[ok_key] else "lost"] += 1
            gold = {
                "type": record["type"],
                "target": record["target"],
                "value": record.get("value"),
            }[field]
            notes.append(
                {
                    "id": record["id"],
                    "text": record["text"],
                    "field": field,
                    "gold": gold,
                    second: b.get(value_key),
                    first: a.get(value_key),
                    f"{second}_ok": bool(b[ok_key]),
                    f"{first}_ok": bool(a[ok_key]),
                }
            )
    return counts | {"notes": notes}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, default=PARITY_DIR / "quality.json")
    args = parser.parse_args()

    expected = read_jsonl(PARITY_DIR / "expected.jsonl")
    web = json.loads((PARITY_DIR / "web-logits.json").read_text(encoding="utf-8"))
    logits = logits_by_backend(expected, web)

    sets = load_sets(EVAL_DIR)
    train_records = read_jsonl(EVAL_DIR / "train.jsonl")
    seen = seen_value_texts(train_records)
    categories = value_span_categories()
    _, tokenizer, meta = load_value_checkpoint(CHECKPOINT)
    max_length = int(meta["max_length"])
    crf = load_crf()

    headline: dict[str, Any] = {}
    criteria: dict[str, Any] = {}
    all_rows: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for backend in BACKENDS:
        summaries, rows_by_set = {}, {}
        for name in EVAL_SETS:
            source = "probe" if name == "probe" else "test"
            records = sets[name]

            def logits_fn(data: Any, backend: str = backend, source: str = source):
                types, tags, values = [], [], []
                lengths = data.encoded.attention_mask.sum(dim=1).tolist()
                for record, n in zip(data.records, lengths, strict=True):
                    type_, tag, value = logits[backend][f"{source}/{record['id']}"]
                    if len(tag) != n:
                        raise RuntimeError(f"{record['id']}: {len(tag)} logit rows, {n} tokens")
                    types.append(type_)
                    tags.append(tag)
                    values.append(value)
                return np.stack(types), tags, values

            metrics, rows, _ = evaluate_records(
                records,
                logits_fn,
                tokenizer,
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
        headline[backend] = {name: _headline(summaries[name]) for name in EVAL_SETS}
        checks = value_quality_criteria({1: summaries}, [1], backend)
        criteria[backend] = {"all_pass": all(c["pass"] for c in checks), "criteria": checks}

    note_level: dict[str, Any] = {}
    for first, second in PAIRS:
        note_level[f"{first}_vs_{second}"] = {
            name: note_changes(
                sets[name],
                {first: all_rows[first][name], second: all_rows[second][name]},
                first,
                second,
            )
            for name in EVAL_SETS
        }

    quality = {
        "evaluation_code": "gidi.evaluation.value_eval.evaluate_records + value_gate.summarize_set "
        "+ value_quality_criteria (frozen value-span-v7), tokenizer of "
        "models/value-span-v7-dual-encoder/seed1, bundle-config CRF",
        "backends": list(BACKENDS),
        "sets": {name: len(sets[name]) for name in EVAL_SETS},
        "headline": headline,
        "frozen_criteria": criteria,
        "note_level": note_level,
    }
    args.out.write_text(json.dumps(quality, indent=1) + "\n", encoding="utf-8")
    for backend in BACKENDS:
        t, p = headline[backend]["test"], headline[backend]["probe"]
        print(
            f"{backend:15s} test type {t['type_accuracy']:.4f} target {t['target_exact']:.4f} "
            f"value {t['value_exact']:.4f} | probe type {p['type_accuracy']:.4f} "
            f"target {p['target_exact']:.4f} value {p['value_exact']:.4f} | criteria "
            f"{'pass' if criteria[backend]['all_pass'] else 'FAIL'}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
