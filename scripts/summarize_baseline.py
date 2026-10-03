#!/usr/bin/env python
"""Summarize the baseline-v1 training grid into ``summary.json`` and ``summary.md``.

Usage:
    uv run python scripts/summarize_baseline.py
    uv run python scripts/summarize_baseline.py --runs-dir experiments/baseline-v1/runs \
        --out experiments/baseline-v1/summary.json --overwrite

Reads every ``<runs-dir>/<model-slug>/lr<lr>-seed<seed>/`` directory. A run counts only when
``config.json``, ``train_log.jsonl``, ``metrics.json``, ``predictions_test.jsonl`` and
``checkpoint.json`` all exist and parse; anything else (e.g. still training) is listed as
incomplete and left out of the statistics, so a partial grid can be summarized at any time.
The markdown goes next to ``--out`` (``.md`` suffix) unless ``--md-out`` is given. Existing
outputs are never replaced without ``--overwrite``.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from gidi.corpus.jsonl import read_jsonl
from gidi.evaluation import summary as S
from gidi.modeling.preprocessing import TYPES

DEFAULT_RUNS_DIR = "experiments/baseline-v1/runs"
DEFAULT_OUT = "experiments/baseline-v1/summary.json"
REQUIRED = (
    "config.json",
    "train_log.jsonl",
    "metrics.json",
    "predictions_test.jsonl",
    "checkpoint.json",
)
RUN_NAME = re.compile(r"^lr(?P<lr>.+)-seed(?P<seed>\d+)$")
SHORT = {
    "expense": "exp",
    "income": "inc",
    "borrow": "bor",
    "lend": "len",
    "repayment_in": "rep_in",
    "repayment_out": "rep_out",
    "transfer": "xfer",
    "refund": "ref",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs-dir", default=DEFAULT_RUNS_DIR)
    parser.add_argument("--out", default=DEFAULT_OUT, help="summary JSON path")
    parser.add_argument("--md-out", default=None, help="markdown path (default: --out with .md)")
    parser.add_argument("--overwrite", action="store_true", help="replace existing outputs")
    return parser.parse_args(argv)


# ----------------------------------------------------------------------------- loading


def load_runs(runs_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return ``(complete runs, incomplete run descriptions)``."""
    complete: list[dict[str, Any]] = []
    incomplete: list[dict[str, Any]] = []
    for model_dir in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
        for run_dir in sorted(p for p in model_dir.iterdir() if p.is_dir()):
            match = RUN_NAME.match(run_dir.name)
            info = {
                "model": model_dir.name,
                "lr": float(match["lr"]) if match else None,
                "seed": int(match["seed"]) if match else None,
                "dir": str(run_dir),
            }
            missing = [f for f in REQUIRED if not (run_dir / f).exists()]
            if missing:
                incomplete.append({**info, "problem": "missing " + ", ".join(missing)})
                continue
            try:
                config = json.loads((run_dir / "config.json").read_text())
                run = {
                    **info,
                    "model": config.get("model_slug", model_dir.name),
                    "model_name": config["model"],
                    "lr": config["lr"],
                    "seed": config["seed"],
                    "config": config,
                    "metrics": json.loads((run_dir / "metrics.json").read_text()),
                    "checkpoint": json.loads((run_dir / "checkpoint.json").read_text()),
                    "log": read_jsonl(run_dir / "train_log.jsonl"),
                }
            except (OSError, ValueError, KeyError) as exc:
                incomplete.append({**info, "problem": f"unreadable: {exc!r}"})
                continue
            complete.append(run)
    return complete, incomplete


# ----------------------------------------------------------------------------- summary


def build_summary(runs_dir: Path) -> dict[str, Any]:
    runs, incomplete = load_runs(runs_dir)
    grouped: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        grouped[(run["model"], run["lr"])].append(run)
    configs = [S.aggregate_config(rs) for _, rs in sorted(grouped.items())]
    best_cfgs = S.best_config_per_model(configs)

    best: dict[str, Any] = {}
    best_preds: dict[str, list[dict[str, Any]]] = {}
    for model, cfg in sorted(best_cfgs.items()):
        cfg_runs = grouped[(model, cfg["lr"])]
        top = S.best_run(cfg_runs)
        preds = read_jsonl(Path(top["dir"]) / "predictions_test.jsonl")
        best_preds[model] = preds
        n_seeds = {c["n_seeds"] for c in configs if c["model"] == model}
        best[model] = {
            "lr": cfg["lr"],
            "n_seeds": cfg["n_seeds"],
            "mean_val_score": cfg["val_score"],
            "seed_counts_differ_across_configs": len(n_seeds) > 1,
            "best_run": {
                "seed": top["seed"],
                "run_dir": top["dir"],
                "val_score": S.val_score(top),
                "checkpoint_path": top["checkpoint"].get("checkpoint_path"),
                "test": S.headline(top["metrics"]["test"]["overall"]),
            },
            "test_per_class": S.per_class_mean(cfg_runs, "test"),
            "test_confusion_matrix": {
                "labels": list(TYPES),
                "note": "rows gold, columns predicted, summed over seeds",
                "matrix": S.summed_confusion(cfg_runs, "test"),
            },
            "test_slices": S.slice_means(cfg_runs, "test"),
        }

    failures: dict[str, Any] = {
        "per_model": {
            m: {"run_dir": best[m]["best_run"]["run_dir"], **S.failure_cases(p)}
            for m, p in best_preds.items()
        },
        "shared": S.shared_failures(best_preds) if len(best_preds) >= 2 else None,
    }
    return {
        "runs_dir": str(runs_dir),
        "selection": "mean of validation type macro-F1 and validation span F1",
        "runs": {
            "complete": [
                {"model": r["model"], "lr": r["lr"], "seed": r["seed"], "dir": r["dir"]}
                for r in sorted(runs, key=lambda r: (r["model"], r["lr"], r["seed"]))
            ],
            "incomplete": incomplete,
        },
        "configs": configs,
        "best": best,
        "failures": failures,
    }


# ----------------------------------------------------------------------------- markdown


def _esc(text: Any) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


def _num(x: float | None, digits: int = 3) -> str:
    return "n/a" if x is None else f"{x:.{digits}f}"


def _spread(s: dict[str, Any], n_seeds: int) -> str:
    """``mean ± std [min, max]``; explicit when fewer values than seeds were averaged."""
    if s["n"] == 0:
        return "n/a"
    if s["n"] == 1:
        text = f"{_num(s['mean'])} (n=1)"
    else:
        text = f"{_num(s['mean'])} ± {_num(s['std'])} [{_num(s['min'])}, {_num(s['max'])}]"
    return text if s["n"] == n_seeds or s["n"] == 1 else f"{text} (n={s['n']}/{n_seeds})"


def _table(header: list[str], rows: list[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(_esc(c) for c in row) + " |" for row in rows]
    return lines + [""]


def _span(span: dict[str, Any] | None) -> str:
    return "∅" if span is None else f"`{span['text']}` [{span['start']}:{span['end']}]"


def render_markdown(s: dict[str, Any]) -> str:
    out: list[str] = ["# baseline-v1 summary", ""]
    complete, incomplete = s["runs"]["complete"], s["runs"]["incomplete"]
    out += [
        f"Runs dir `{s['runs_dir']}`: **{len(complete)} complete** run(s), "
        f"**{len(incomplete)} incomplete/unreadable** (excluded). Selection score = "
        f"{s['selection']}. Spread cells: mean ± sample std [min, max] across seeds; `n` = seeds.",
        "",
    ]
    by_cfg: dict[tuple[str, float], list[str]] = defaultdict(list)
    for r in complete:
        by_cfg[(r["model"], r["lr"])].append(f"s{r['seed']}")
    pending: dict[tuple[str, float | None], list[str]] = defaultdict(list)
    for r in incomplete:
        pending[(r["model"], r["lr"])].append(f"s{r['seed']} ({r['problem']})")
    keys = sorted(set(by_cfg) | set(pending), key=lambda k: (k[0], k[1] is None, k[1] or 0))
    out += _table(
        ["model", "lr", "complete seeds", "incomplete"],
        [
            [
                m,
                "?" if lr is None else f"{lr:g}",
                ", ".join(by_cfg.get((m, lr), [])) or "-",
                "; ".join(pending.get((m, lr), [])) or "-",
            ]
            for m, lr in keys
        ],
    )

    configs = s["configs"]
    if not configs:
        return "\n".join(out + ["No complete runs yet.", ""])

    out += ["## 1. Configs (model x lr)", ""]
    for split, title in (("validation", "Validation"), ("test", "Test")):
        n = configs[0]["split_n"][split]
        out += [
            f"### {title} (n={n['n']}, gold-target {n['n_gold_target']}, "
            f"gold-null {n['n_gold_null']})",
            "",
        ]
        header = [
            "model",
            "lr",
            "n",
            "type acc",
            "type macro-F1",
            "span F1",
            "span EM",
            "EM (gold target)",
            "null acc (gold null)",
        ]
        cols = [
            "type_accuracy",
            "type_macro_f1",
            "span_f1",
            "span_exact_match",
            "target_exact_match",
            "null_accuracy",
        ]
        rows = [
            [c["model"], f"{c['lr']:g}", str(c["n_seeds"])]
            + [_spread(c[split][k], c["n_seeds"]) for k in cols]
            for c in configs
        ]
        out += _table(header, rows)
    out += ["### Epochs, time, size", ""]
    out += _table(
        [
            "model",
            "lr",
            "n",
            "val score",
            "best/run epoch per seed",
            "wall s/run",
            "s/epoch",
            "params",
        ],
        [
            [
                c["model"],
                f"{c['lr']:g}",
                str(c["n_seeds"]),
                _spread(c["val_score"], c["n_seeds"]),
                ", ".join(
                    f"s{p['seed']}: {p['best_epoch']}/{p['epochs_run']}" for p in c["per_seed"]
                ),
                _spread_plain(c["wall_time_sec"], c["n_seeds"], 1),
                _spread_plain(c["sec_per_epoch"], c["n_seeds"], 2),
                "n/a" if c["param_count"] is None else f"{c['param_count']:,}",
            ]
            for c in configs
        ],
    )

    out += ["## 2. Best config per model", ""]
    best = s["best"]
    rows = []
    for m, b in best.items():
        r = b["best_run"]
        rows.append(
            [
                m,
                f"{b['lr']:g}",
                str(b["n_seeds"]),
                _spread(b["mean_val_score"], b["n_seeds"]),
                f"s{r['seed']}",
                _num(r["val_score"], 4),
                _num(r["test"]["type_macro_f1"]),
                _num(r["test"]["span_f1"]),
                str(r["checkpoint_path"]),
            ]
        )
    out += _table(
        [
            "model",
            "lr",
            "n",
            "mean val score",
            "best seed",
            "seed val score",
            "seed test type F1",
            "seed test span F1",
            "checkpoint",
        ],
        rows,
    )
    for m, b in best.items():
        if b["seed_counts_differ_across_configs"]:
            out += [
                f"> **{m}**: configs have different seed counts (partial grid); "
                "the comparison is provisional.",
                "",
            ]

    out += ["## 3. Best-config detail (test, across seeds)", ""]
    for m, b in best.items():
        out += [
            f"### {m} (lr {b['lr']:g}, {b['n_seeds']} seed(s))",
            "",
            "Per class (mean over seeds):",
            "",
        ]
        out += _table(
            ["class", "support", "P", "R", "F1", "mean n_pred"],
            [
                [
                    label,
                    str(v["support"]),
                    _num(v["precision"]),
                    "-" if not v["support"] else _num(v["recall"]),
                    "-" if not v["support"] else _num(v["f1"]),
                    _num(v["n_pred"], 1),
                ]
                for label, v in b["test_per_class"].items()
            ],
        )
        cm = b["test_confusion_matrix"]
        out += ["Confusion matrix summed over seeds (rows gold, columns predicted):", ""]
        heads = [SHORT.get(label, label[:6]) for label in cm["labels"]]
        out += _table(
            ["gold \\ pred", *heads],
            [
                [label, *map(str, row)]
                for label, row in zip(cm["labels"], cm["matrix"], strict=True)
            ],
        )
        out += ["Slices (mean over seeds):", ""]
        srows = []
        for name, values in b["test_slices"].items():
            for value, e in values.items():
                srows.append(
                    [
                        name,
                        value,
                        str(e["n"]),
                        _num(e["type_accuracy"]),
                        _num(e["type_macro_f1"]),
                        _num(e["span_f1"]),
                        _num(e["span_exact_match"]),
                    ]
                )
        out += _table(
            ["slice", "value", "n", "type acc", "type macro-F1", "span F1", "span EM"], srows
        )

    out += ["## 4. Failure cases (best run of each model, test)", ""]
    fail = s["failures"]
    for m, f in fail["per_model"].items():
        out += [
            f"### {m} — `{f['run_dir']}`: {len(f['type_errors'])} type error(s), "
            f"{len(f['span_errors'])} span error(s) of {f['n']}",
            "",
        ]
        if f["type_errors"]:
            out += _table(
                ["id", "text", "gold", "pred"],
                [[e["id"], e["text"], e["gold_type"], e["pred_type"]] for e in f["type_errors"]],
            )
        if f["span_errors"]:
            out += _table(
                ["id", "text", "gold span", "pred span"],
                [
                    [e["id"], e["text"], _span(e["gold_span"]), _span(e["pred_span"])]
                    for e in f["span_errors"]
                ],
            )
    shared = fail["shared"]
    if shared is None:
        out += ["Shared failures: need at least two models with complete runs.", ""]
    else:
        models = shared["models"]
        out += [
            f"### Both/all models err ({', '.join(models)}; {shared['n_common']} ids in common, "
            f"{shared['n_unmatched']} unmatched)",
            "",
            f"Type errors in all models: {len(shared['type_errors'])}",
            "",
        ]
        if shared["type_errors"]:
            out += _table(
                ["id", "text", "gold", *[f"pred {m}" for m in models]],
                [
                    [e["id"], e["text"], e["gold_type"], *[e["pred_type"][m] for m in models]]
                    for e in shared["type_errors"]
                ],
            )
        out += [f"Span errors in all models: {len(shared['span_errors'])}", ""]
        if shared["span_errors"]:
            out += _table(
                ["id", "text", "gold span", *[f"pred {m}" for m in models]],
                [
                    [
                        e["id"],
                        e["text"],
                        _span(e["gold_span"]),
                        *[_span(e["pred_span"][m]) for m in models],
                    ]
                    for e in shared["span_errors"]
                ],
            )
    return "\n".join(out)


def _spread_plain(s: dict[str, Any], n_seeds: int, digits: int) -> str:
    if s["n"] == 0:
        return "n/a"
    if s["n"] == 1:
        return f"{s['mean']:.{digits}f} (n=1)"
    text = f"{s['mean']:.{digits}f} ± {s['std']:.{digits}f}"
    return text if s["n"] == n_seeds else f"{text} (n={s['n']}/{n_seeds})"


# ----------------------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    runs_dir = Path(args.runs_dir)
    if not runs_dir.is_dir():
        print(f"error: runs dir {runs_dir} does not exist", file=sys.stderr)
        return 1
    out_json = Path(args.out)
    out_md = Path(args.md_out) if args.md_out else out_json.with_suffix(".md")
    existing = [p for p in (out_json, out_md) if p.exists()]
    if existing and not args.overwrite:
        names = ", ".join(map(str, existing))
        print(f"error: {names} exists; pass --overwrite to replace", file=sys.stderr)
        return 1

    summary = build_summary(runs_dir)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    out_md.write_text(render_markdown(summary) + "\n")
    runs = summary["runs"]
    print(
        f"{len(runs['complete'])} complete, {len(runs['incomplete'])} incomplete run(s); "
        f"wrote {out_json} and {out_md}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
