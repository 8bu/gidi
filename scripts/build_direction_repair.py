#!/usr/bin/env python3
"""Direction audit + repaired training set for value-span-v2-direction-repair.

Phase 1 (audit) reads TRAINING rows only (``training-v1/train.jsonl``) and counts lend/borrow
wording families per source batch. Phase 2 (repair) appends verbatim duplicates of existing,
already-approved name-first lend rows (``X mượn ...``) so that the name-first direction balance
returns to its pre-targeted-value-01 difference. No label is created or edited; validation, test
and probe are byte-identical copies of ``training-v1``.

Usage:
    uv run python scripts/build_direction_repair.py            # write audit + dataset
    uv run python scripts/build_direction_repair.py --check    # verify files on disk
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from gidi.annotation.value_span import fold
from gidi.corpus.jsonl import read_jsonl

SRC = Path("datasets/annotation-v2/training-v1")
OUT = Path("datasets/annotation-v2/training-v2-direction-repair")
EXP = Path("experiments/value-span-v2-direction-repair")
COPIED = ("validation.jsonl", "test.jsonl", "probe-v1-eval-only.jsonl")
TARGETED = "targeted-value-01"
DUP_SUFFIX = "#direction-repair-dup"
SALT = "value-span-v2-direction-repair"
DIRECTION_TYPES = ("borrow", "lend")

PREFIX = r"(ck|chuyen khoan|chuyen|gui|dua)"
SELF = r"(t|tao|minh|mk|e|em|toi|tui)"
LOAN = r"\b(muon|vay|ung)\b"

# Family -> meaning (A = lender-first borrow, B = subject-first lend, U = user-first, C = other).
FAMILIES = {
    "A_X_cho_muon": "name-first lender: X cho (minh) muon/vay ...",
    "A_duoc_X_cho": "duoc X cho muon/vay ...",
    "B_cho_X_muon": "cho X muon/vay ...",
    "B_prefix_cho_X_muon": "ck/chuyen/gui cho X muon/vay ...",
    "B_self_cho_X_muon": "t/em/minh cho X muon/vay ...",
    "B_X_muon": "name-first borrower: X muon/vay (cua minh) ...",
    "U_muon_X": "user-first: muon/vay X ...",
    "C_other": "other lend/borrow wording (ung luong, giai ngan, ...)",
}
# The confusable name-first pair: "X cho mượn" (borrow) vs "X mượn" (lend).
LENDER_FIRST = ("A_X_cho_muon", "A_duoc_X_cho")
BORROWER_FIRST = ("B_X_muon",)


def family(text: str) -> str:
    t = fold(text).strip()
    if re.match(rf"^{PREFIX}\s+cho\s", t) and re.search(LOAN, t):
        return "B_prefix_cho_X_muon"
    if re.match(r"^cho\s", t) and re.search(LOAN, t):
        return "B_cho_X_muon"
    if re.match(r"^duoc\s", t) and re.search(rf"\bcho\b.*{LOAN}", t):
        return "A_duoc_X_cho"
    if re.match(r"^(muon|vay)\s", t):
        return "U_muon_X"
    if re.match(r"^(cho|ung|duoc)\b", t):
        return "C_other"
    if re.match(rf"^{SELF}\s+cho\s", t) and re.search(LOAN, t):
        return "B_self_cho_X_muon"
    if re.search(rf"\bcho\s+(\S+\s+)?{LOAN}", t):
        return "A_X_cho_muon"
    if re.search(r"\b(muon|vay)\b", t):
        return "B_X_muon"
    return "C_other"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def audit(train: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [r for r in train if r["type"] in DIRECTION_TYPES]
    batches = sorted({r["source_batch"] for r in train})
    groups = {b: [r for r in rows if r["source_batch"] == b] for b in batches}
    groups["v1_rows (all but targeted-value-01)"] = [
        r for r in rows if r["source_batch"] != TARGETED
    ]
    groups["merged_train"] = rows
    out: dict[str, Any] = {
        "class_counts": {
            b: dict(Counter(r["type"] for r in train if r["source_batch"] == b)) for b in batches
        },
        "families": FAMILIES,
        "groups": {},
    }
    for name, recs in groups.items():
        fam: dict[str, Any] = {}
        for f in FAMILIES:
            members = [r for r in recs if family(r["text"]) == f]
            if not members:
                continue
            fam[f] = {
                "by_type": dict(Counter(r["type"] for r in members)),
                "accented": sum(r["accented"] for r in members),
                "unaccented": sum(not r["accented"] for r in members),
                "examples": [r["text"] for r in sorted(members, key=lambda r: r["id"])[:6]],
            }
        out["groups"][name] = {"n": len(recs), "by_type": dict(Counter(r["type"] for r in recs))}
        out["groups"][name]["families"] = fam
    return out


def name_first_balance(recs: list[dict[str, Any]]) -> dict[str, int]:
    a = sum(family(r["text"]) in LENDER_FIRST and r["type"] == "borrow" for r in recs)
    b = sum(family(r["text"]) in BORROWER_FIRST and r["type"] == "lend" for r in recs)
    return {"lender_first_borrow": a, "borrower_first_lend": b, "difference": a - b}


def near_duplicates(train: list[dict[str, Any]]) -> dict[str, Any]:
    """Repeated digit-masked folded texts, and targeted-value-01 `X cho mượn`/`cho X mượn` pairs."""
    rows = [r for r in train if r["type"] in DIRECTION_TYPES]
    masked = Counter(
        (r["source_batch"], re.sub(r"\d+([.,]\d+)*", "N", fold(r["text"])).strip(" ."))
        for r in rows
    )
    by_rest: dict[str, list[str]] = {}
    for r in rows:
        if r["source_batch"] == TARGETED and family(r["text"]) in ("A_X_cho_muon", "B_cho_X_muon"):
            key = re.sub(r"\b(cho|muon|vay)\b", "", fold(r["text"]))
            key = re.sub(r"\s+", " ", key).strip(" .")
            by_rest.setdefault(key, []).append(f"{r['type']}: {r['text']}")
    return {
        "repeated_masked_texts": [
            {"batch": b, "text": t, "n": n} for (b, t), n in masked.items() if n > 1
        ],
        "targeted_value_01_minimal_pairs": [v for v in by_rest.values() if len(v) > 1],
    }


def repair_rows(train: list[dict[str, Any]], pre: dict[str, int], post: dict[str, int]):
    n = max(0, post["difference"] - pre["difference"])
    pool = [
        r
        for r in train
        if r["source_batch"] != TARGETED
        and r["type"] == "lend"
        and family(r["text"]) in BORROWER_FIRST
        and r.get("value_status") == "complete"
    ]
    pool.sort(key=lambda r: hashlib.sha256(f"{SALT}:{r['id']}".encode()).hexdigest())
    chosen: list[dict[str, Any]] = []
    seen_values: set[str] = set()
    for r in pool:  # first pass: distinct value surfaces
        if len(chosen) < n and r["value"]["text"] not in seen_values:
            chosen.append(r)
            seen_values.add(r["value"]["text"])
    for r in pool:
        if len(chosen) < n and r not in chosen:
            chosen.append(r)
    if len(chosen) < n:
        raise SystemExit(f"pool has {len(pool)} rows, repair needs {n}")
    dups = []
    for r in chosen:
        d = dict(r)
        d["id"] = r["id"] + DUP_SUFFIX
        d["duplicate_of"] = r["id"]
        dups.append(d)
    return n, len(pool), dups


def jsonl_bytes(rows: list[dict[str, Any]]) -> bytes:
    return "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows).encode("utf-8")


def render_md(report: dict[str, Any]) -> str:
    lines = [
        "# value-span-v2-direction-repair: training-data direction audit",
        "",
        "Training rows only (`datasets/annotation-v2/training-v1/train.jsonl`). Test and probe "
        "were not read. Families are defined in `scripts/build_direction_repair.py`.",
        "",
        "## Class counts per batch (train)",
        "",
        "| batch | " + " | ".join(DIRECTION_TYPES) + " | all |",
        "|---|---|---|---|",
    ]
    for b, c in report["audit"]["class_counts"].items():
        lines.append(f"| {b} | {c.get('borrow', 0)} | {c.get('lend', 0)} | {sum(c.values())} |")
    lines += ["", "## Lend/borrow wording families (train)", ""]
    lines += [
        "| group | family | borrow | lend | accented | unaccented |",
        "|---|---|---|---|---|---|",
    ]
    for g, data in report["audit"]["groups"].items():
        for f, d in data["families"].items():
            bt = d["by_type"]
            lines.append(
                f"| {g} | {f} | {bt.get('borrow', 0)} | {bt.get('lend', 0)} | "
                f"{d['accented']} | {d['unaccented']} |"
            )
    lines += ["", "## Family examples (first 6 by id)", ""]
    for g, data in report["audit"]["groups"].items():
        if g in ("merged_train", "v1_rows (all but targeted-value-01)"):
            continue
        lines.append(f"### {g}")
        lines.append("")
        for f, d in data["families"].items():
            lines.append(f"- **{f}**: " + "; ".join(f"`{t}`" for t in d["examples"]))
        lines.append("")
    bal = report["name_first_balance"]
    lines += [
        "## Name-first direction balance",
        "",
        "`X cho mượn` (lender-first, borrow) vs `X mượn` (borrower-first, lend):",
        "",
        "| rows | lender-first borrow | borrower-first lend | difference |",
        "|---|---|---|---|",
    ]
    for k in ("v1_rows", "targeted_value_01", "merged", "repaired"):
        v = bal[k]
        lines.append(
            f"| {k} | {v['lender_first_borrow']} | {v['borrower_first_lend']} | {v['difference']} |"
        )
    nd = report["near_duplicates"]
    pairs = nd["targeted_value_01_minimal_pairs"]
    lines += ["", "## Generator patterns", ""]
    lines.append(f"targeted-value-01 minimal pairs `X cho mượn` / `cho X mượn`: {len(pairs)}")
    lines.append("")
    for p in pairs:
        lines.append("- " + " ↔ ".join(f"`{x}`" for x in p))
    lines += ["", "Repeated digit-masked lend/borrow texts within a batch:", ""]
    for d in nd["repeated_masked_texts"] or [{"batch": "none", "text": "-", "n": 0}]:
        lines.append(f"- {d['batch']}: `{d['text']}` × {d['n']}")
    rep = report["repair"]
    lines += [
        "",
        "## Repair (deterministic)",
        "",
        f"Rule: append verbatim duplicates of existing non-{TARGETED} `B_X_muon` lend rows with a "
        "complete value label until the name-first difference equals its pre-targeted-value-01 "
        f"value. Needed: {rep['n']} (pool {rep['pool']}). Order: sha256(`{SALT}:<id>`), distinct "
        "value surfaces first.",
        "",
        "| duplicate of | text | value | accented |",
        "|---|---|---|---|",
    ]
    for r in rep["rows"]:
        lines.append(f"| {r['duplicate_of']} | `{r['text']}` | `{r['value']}` | {r['accented']} |")
    lines.append("")
    return "\n".join(lines)


def build(root: Path) -> dict[Path, bytes]:
    train = read_jsonl(root / SRC / "train.jsonl")
    v1_rows = [r for r in train if r["source_batch"] != TARGETED]
    tv = [r for r in train if r["source_batch"] == TARGETED]
    pre, post = name_first_balance(v1_rows), name_first_balance(train)
    n, pool, dups = repair_rows(train, pre, post)
    repaired = train + dups
    report = {
        "experiment": "value-span-v2-direction-repair",
        "source": str(SRC / "train.jsonl"),
        "source_sha256": sha256_bytes((root / SRC / "train.jsonl").read_bytes()),
        "audit": audit(train),
        "name_first_balance": {
            "v1_rows": pre,
            "targeted_value_01": name_first_balance(tv),
            "merged": post,
            "repaired": name_first_balance(repaired),
        },
        "near_duplicates": near_duplicates(train),
        "repair": {
            "n": n,
            "pool": pool,
            "rows": [
                {
                    "id": d["id"],
                    "duplicate_of": d["duplicate_of"],
                    "text": d["text"],
                    "type": d["type"],
                    "value": d["value"]["text"],
                    "accented": d["accented"],
                }
                for d in dups
            ],
        },
    }
    out: dict[Path, bytes] = {OUT / "train.jsonl": jsonl_bytes(repaired)}
    for name in COPIED:
        out[OUT / name] = (root / SRC / name).read_bytes()
    manifest = {
        "dataset": str(OUT),
        "derived_from": str(SRC),
        "train": {"n": len(repaired), "base": len(train), "duplicates": n},
        "files": {str(p.name): sha256_bytes(b) for p, b in sorted(out.items())},
        "copied_unchanged": {
            name: sha256_bytes((root / SRC / name).read_bytes()) for name in COPIED
        },
    }
    out[OUT / "manifest.json"] = (
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    ).encode()
    out[EXP / "data-audit.json"] = (
        json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    ).encode()
    out[EXP / "data-audit.md"] = render_md(report).encode("utf-8")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--check", action="store_true", help="verify the files on disk")
    parser.add_argument("--force", action="store_true", help="replace existing outputs")
    args = parser.parse_args(argv)
    outputs = build(args.root)
    if args.check:
        bad = [
            p
            for p, b in outputs.items()
            if not (args.root / p).is_file() or (args.root / p).read_bytes() != b
        ]
        if bad:
            print("CHECK FAILED: " + ", ".join(map(str, bad)))
            return 1
        print(f"ok: {len(outputs)} files reproduce")
        return 0
    existing = [p for p in outputs if (args.root / p).exists()]
    if existing and not args.force:
        print(f"refusing to overwrite {existing[0]}; pass --force", file=sys.stderr)
        return 1
    for p, b in outputs.items():
        (args.root / p).parent.mkdir(parents=True, exist_ok=True)
        (args.root / p).write_bytes(b)
    print(f"wrote {len(outputs)} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
