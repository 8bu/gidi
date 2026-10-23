#!/usr/bin/env python
"""Aggregate the bake-off runs and apply the pre-declared win rule of protocol.json.

    uv run python experiments/gidi-en-backbone-bakeoff/aggregate.py

Reads every run's ``results.json`` (roberta-base seed 1 is the existing v3 run), measures the
int8 op set and size of every run's int8 ONNX file, writes ``aggregate.json`` and ``tables.md``.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

import onnx

B = Path("experiments/gidi-en-backbone-bakeoff")
PROTOCOL = json.loads((B / "protocol.json").read_text(encoding="utf-8"))
SHIPPED_OPS = set(json.loads((B / "feasibility.json").read_text())["shipped_ops"])
V3 = Path("experiments/gidi-en-student-v3")
V2 = Path("experiments/gidi-en-student-v2")

# candidate -> (label, seeds, tokenizer class, web tokenizer cost)
CANDIDATES: dict[str, dict[str, Any]] = {
    "roberta-base": {
        "label": "roberta-base 4L/768 K2048 v9000 (control)",
        "seeds": [1, 2, 3],
        "tokenizer": "existing",
        "web_cost": "none: shipped byte-level BPE path",
    },
    "distilroberta": {
        "label": "distilroberta-base 6L/768 K832 v9000",
        "seeds": [1, 2],
        "tokenizer": "existing",
        "web_cost": "none: identical tokenizer.json vocab and merges as roberta-base",
    },
    "electra": {
        "label": "electra-base-discriminator 4L/768 K2048 v9000",
        "seeds": [1, 2],
        "tokenizer": "port",
        "web_cost": "new TS WordPiece tokenizer (see report)",
    },
    "minilm": {
        "label": "MiniLM-L12-H384 12L/384 K1536 v20500",
        "seeds": [1, 2],
        "tokenizer": "port",
        "web_cost": "new TS WordPiece tokenizer (see report)",
    },
}
SETS = ("primary", "secondary", "dev", "slang_probe", "genz_probe")
COMPOSITE_PARTS = (
    "secondary_type",
    "secondary_target",
    "slang_joint",
    "slang_value",
    "genz_joint",
)
GUARDS = ("primary_type", "primary_target")


def result_path(cand: str, seed: int) -> Path:
    if cand == "roberta-base" and seed == 1:
        return V3 / "results.json"
    return B / "runs" / cand / f"seed{seed}" / "results.json"


def onnx_path(cand: str, seed: int) -> Path:
    if cand == "roberta-base" and seed == 1:
        return Path("models/gidi-en-v3-onnx/seed1/model.int8.onnx")
    return Path(f"models/gidi-en-bakeoff-onnx/{cand}/seed{seed}/model.int8.onnx")


def run_metrics(results: dict[str, Any]) -> dict[str, float]:
    int8 = results["int8"]
    out: dict[str, float] = {}
    for key, short in (
        ("primary", "primary"),
        ("secondary", "secondary"),
        ("dev", "dev"),
        ("slang_probe", "slang"),
        ("genz_probe", "genz"),
    ):
        m = int8[key]
        out[f"{short}_type"] = m["type_macro_f1"]
        out[f"{short}_target"] = m["span_exact"]
        out[f"{short}_joint"] = m["joint"]
        out[f"{short}_value"] = m["value"]["exact"]
        out[f"{short}_full_joint"] = m["full_joint"]
    out["composite"] = sum(out[k] for k in COMPOSITE_PARTS) / len(COMPOSITE_PARTS)
    return out


def op_info(path: Path) -> dict[str, Any]:
    proto = onnx.load(str(path), load_external_data=False)
    ops = {n.op_type for n in proto.graph.node}
    return {
        "opset": max(o.version for o in proto.opset_import if o.domain in ("", "ai.onnx")),
        "ops": sorted(ops),
        "ops_not_in_shipped": sorted(ops - SHIPPED_OPS),
        "bytes": path.stat().st_size,
    }


def spread(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "min": min(values),
        "max": max(values),
        "std": statistics.pstdev(values),
        "n": len(values),
    }


def v2_runs() -> dict[str, dict[str, float]]:
    """The two existing runs of the v2 recipe (recorded seed 1 and the control re-run)."""
    out = {}
    for name, root, genz in (
        ("v2_recorded", V2, True),
        ("v2_control_rerun", V3 / "control-v2", False),
    ):
        res = json.loads((root / "results.json").read_text())["int8_snap"]
        slang = json.loads((root / "slang-probe-snap.json").read_text())["all"]
        row: dict[str, float] = {}
        for key in ("primary", "secondary", "dev"):
            row[f"{key}_type"] = res[key]["type_macro_f1"]
            row[f"{key}_target"] = res[key]["span_exact"]
            row[f"{key}_joint"] = res[key]["joint"]
        row["slang_type"] = slang["type_macro_f1"]
        row["slang_target"] = slang["span_exact"]
        row["slang_joint"] = slang["joint"]
        if genz:
            g = json.loads((V2 / "genz-probe-snap.json").read_text())["all"]
            row["genz_type"], row["genz_target"], row["genz_joint"] = (
                g["type_macro_f1"],
                g["span_exact"],
                g["joint"],
            )
        out[name] = row
    return out


def main() -> int:
    runs: dict[str, dict[int, dict[str, float]]] = {}
    ops: dict[str, dict[int, dict[str, Any]]] = {}
    missing = []
    for cand, spec in CANDIDATES.items():
        for seed in spec["seeds"]:
            path = result_path(cand, seed)
            if not path.exists():
                missing.append(str(path))
                continue
            runs.setdefault(cand, {})[seed] = run_metrics(json.loads(path.read_text()))
            ops.setdefault(cand, {})[seed] = op_info(onnx_path(cand, seed))
    if missing:
        raise SystemExit(f"missing results: {missing}")

    summary: dict[str, Any] = {}
    for cand, per_seed in runs.items():
        keys = next(iter(per_seed.values())).keys()
        summary[cand] = {k: spread([per_seed[s][k] for s in per_seed]) for k in keys}

    control = summary["roberta-base"]
    control_max = max(r["composite"] for r in runs["roberta-base"].values())
    margin = 0.01
    decisions: dict[str, Any] = {}
    for cand in CANDIDATES:
        if cand == "roberta-base":
            continue
        s = summary[cand]
        checks = {
            "composite_margin": {
                "challenger_mean": s["composite"]["mean"],
                "control_mean": control["composite"]["mean"],
                "delta": s["composite"]["mean"] - control["composite"]["mean"],
                "need": margin,
                "pass": s["composite"]["mean"] - control["composite"]["mean"] >= margin - 1e-12,
            }
        }
        for g in GUARDS:
            delta = s[g]["mean"] - control[g]["mean"]
            checks[f"no_regression_{g}"] = {"delta": delta, "pass": delta >= -margin - 1e-12}
        sizes = [o["bytes"] for o in ops[cand].values()]
        checks["int8_size"] = {"max_bytes": max(sizes), "pass": max(sizes) <= 30_000_000}
        extra = sorted({op for o in ops[cand].values() for op in o["ops_not_in_shipped"]})
        opsets = sorted({o["opset"] for o in ops[cand].values()})
        checks["ops_and_opset"] = {
            "ops_not_in_shipped": extra,
            "opsets": opsets,
            "pass": not extra and opsets == [17],
        }
        core = all(c["pass"] for c in checks.values())
        tokenizer = CANDIDATES[cand]["tokenizer"]
        verdict = "no" if not core else ("WIN" if tokenizer == "existing" else "WIN-WITH-PORT")
        low = min(r["composite"] for r in runs[cand].values())
        decisions[cand] = {
            "checks": checks,
            "tokenizer": tokenizer,
            "verdict": verdict,
            "separated": low > control_max,
            "challenger_min_composite": low,
            "control_max_composite": control_max,
        }
    winners = [c for c, d in decisions.items() if d["verdict"] != "no"]
    best = max(winners, key=lambda c: summary[c]["composite"]["mean"]) if winners else None
    out = {
        "per_run": {c: {str(s): m for s, m in r.items()} for c, r in runs.items()},
        "summary": summary,
        "ops": ops,
        "decisions": decisions,
        "decision": best or "roberta-base stays",
        "v2_runs": v2_runs(),
    }
    (B / "aggregate.json").write_text(json.dumps(out, indent=1) + "\n")
    (B / "tables.md").write_text(render(out))
    print(render(out))
    return 0


def fmt(s: dict[str, float]) -> str:
    return f"{s['mean']:.4f} ({s['min']:.4f}-{s['max']:.4f})"


def render(out: dict[str, Any]) -> str:
    lines = []
    cols = [
        ("composite", "composite"),
        ("secondary_type", "sec. type F1"),
        ("secondary_target", "sec. target"),
        ("slang_joint", "slang joint"),
        ("slang_value", "slang value"),
        ("genz_joint", "GenZ joint"),
        ("primary_type", "prim. type F1"),
        ("primary_target", "prim. target"),
    ]
    lines.append("| candidate | seeds | " + " | ".join(c[1] for c in cols) + " | int8 bytes |")
    lines.append("|" + "---|" * (len(cols) + 3))
    for cand, spec in CANDIDATES.items():
        s = out["summary"][cand]
        size = max(o["bytes"] for o in out["ops"][cand].values())
        lines.append(
            f"| {spec['label']} | {s['composite']['n']} | "
            + " | ".join(fmt(s[k]) for k, _ in cols)
            + f" | {size:,} |"
        )
    lines.append("")
    lines.append(
        "Per run (composite / sec type / sec target / slang joint / slang value / GenZ joint):"
    )
    lines.append("")
    for cand, runs in out["per_run"].items():
        for seed, m in runs.items():
            lines.append(
                f"- {cand} seed {seed}: "
                + " / ".join(
                    f"{m[k]:.4f}"
                    for k in (
                        "composite",
                        "secondary_type",
                        "secondary_target",
                        "slang_joint",
                        "slang_value",
                        "genz_joint",
                    )
                )
            )
    lines.append("")
    for cand, d in out["decisions"].items():
        lines.append(f"{cand}: **{d['verdict']}** (separated={d['separated']})")
        for name, c in d["checks"].items():
            detail = {k: v for k, v in c.items() if k != "pass"}
            lines.append(f"  - {name}: {'pass' if c['pass'] else 'FAIL'} {detail}")
    lines.append("")
    lines.append(f"Decision: {out['decision']}")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
