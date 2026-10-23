#!/usr/bin/env python
"""Build the silver value-span labels of gidi-en (annotation-en-v1 ``value:`` section).

Three labelers, all silver (never human): A = the deterministic amount scan
(``gidi.value_parser.english``), B = Cloudflare Clef choosing among A's loose candidates
(``run_value_clef`` in ``experiments/gidi-en-student-v3/run_value_teacher.py``, run inside omp
``eval``), and Claude, who adjudicates only the notes where A and B disagree.

    uv run python scripts/build_en_value_labels.py prepare     # work/value/notes.jsonl
    # (eval)  %load experiments/gidi-en-student-v3/run_value_teacher.py ; await run_value_clef()
    uv run python scripts/build_en_value_labels.py disagreements   # compact list for adjudication
    uv run python scripts/build_en_value_labels.py build       # label files + agreement.json

Adjudications live in ``datasets/annotation-en-v1/work/value/adjudicated.jsonl``:
``{"key": "<set>:<id>", "decision": "A"|"B"|"N"|"U"|"=<substring>"}`` (A / B take that labeler's
span, N no amount, U uncertain, ``=text`` the last occurrence of ``text`` in the note).
Only ``complete`` notes (type labels) get a value label. The eval sets (test, dev, seed, the two
probes) and the training sets (silver-train, consistent silver-slang-train) are labelled the same
way; the files are separate from the type/target labels and never marked human.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from gidi.corpus.jsonl import read_jsonl, write_jsonl
from gidi.value_parser.english import choose, loose_candidates

D = Path("datasets/annotation-en-v1")
WORK = D / "work" / "value"
SETS = ("test", "dev", "seed", "probe-slang", "probe-genz", "train", "slang-train")
OUT = {
    "test": D / "value-labels-test.jsonl",
    "dev": D / "value-labels-dev.jsonl",
    "seed": D / "value-labels-seed.jsonl",
    "probe-slang": D / "value-labels-probe-slang.jsonl",
    "probe-genz": D / "value-labels-probe-genz.jsonl",
    "train": D / "value-silver-train.jsonl",
    "slang-train": D / "value-silver-slang-train.jsonl",
}
PROV_CONSENSUS = "silver-consensus"
PROV_ADJUDICATED = "silver-adjudicated"

CLEF_INSTRUCTIONS = (
    "A short, informal English personal-finance note (US, UK, Indian or Singapore usage). Pick "
    "the candidate that is the amount of money of the transaction: the sum paid, received, "
    "lent, borrowed, repaid, refunded or moved. It is NOT a count of items or people, a number "
    "of days, weeks or months, a date, a time, a percentage, a phone, account or reference "
    "number, an interest rate, or a threshold or balance mentioned in passing. When a note "
    "has a main amount plus a fee or tip, pick the main amount. Prefer the candidate that "
    "includes the currency sign or money word written with the number ('$20', '20 bucks', "
    "'rs 500', '£12', 'twenty quid', 'tenner'). If the note states no sum of money, pick "
    "'none'. Examples: 'coffee 4.50' -> '4.50'; 'lunch for 4 at nando's 62' -> '62'; 'sam paid "
    "20 bucks for 2 beers' -> '20 bucks'; 'rent due 1st, paid 800' -> '800'; 'paid the "
    "electric bill' -> none; 'dinner at 7:30 w sam' -> none; 'lent mate a fiver' -> 'fiver'."
)


def clef_questions(text: str, cands: list[dict[str, Any]]) -> dict[str, Any]:
    """The single ``judge`` question for one note: ``c<i>`` is the i-th loose candidate."""
    criteria = {f"c{i}": f"'{c['text']}'" for i, c in enumerate(cands)}
    criteria["none"] = "The note states no amount of money."
    return {"value": {"type": "choice", "instructions": CLEF_INSTRUCTIONS, "criteria": criteria}}


def _texts(split: str) -> dict[str, str]:
    return {r["id"]: r["text"] for r in read_jsonl(D / "splits" / f"{split}.jsonl")}


def collect() -> list[dict[str, str]]:
    """Every note to label as ``{set, id, text}``: complete type labels only."""
    rows: list[dict[str, str]] = []

    def add(name: str, texts: dict[str, str], labels: list[dict[str, Any]], keep=None) -> None:
        for lab in labels:
            if lab["annotation_status"] == "complete" and (keep is None or keep(lab)):
                rows.append({"set": name, "id": lab["id"], "text": texts[lab["id"]]})

    for name in ("test", "dev", "seed"):
        add(name, _texts(name), read_jsonl(D / f"labels-{name}.jsonl"))
    for name, probe in (("probe-slang", "probe-en-slang"), ("probe-genz", "probe-en-genz")):
        texts = {r["id"]: r["text"] for r in read_jsonl(Path("datasets") / probe / "notes.jsonl")}
        add(name, texts, read_jsonl(Path("datasets") / probe / "labels.jsonl"))
    add("train", _texts("train"), read_jsonl(D / "silver-train.jsonl"))
    slang = read_jsonl(D / "silver-slang-train.jsonl")
    add("slang-train", {r["id"]: r["text"] for r in slang}, slang, keep=lambda r: r["consistent"])
    keys = [f"{r['set']}:{r['id']}" for r in rows]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate (set, id) among the notes to label")
    return rows


def prepare() -> None:
    rows = []
    for r in collect():
        cands = loose_candidates(r["text"])
        pick = choose(cands)
        rows.append(
            {
                **r,
                "key": f"{r['set']}:{r['id']}",
                "regex": None if pick is None else [pick.start, pick.end],
                "cands": [{"start": c.start, "end": c.end, "text": c.text} for c in cands],
            }
        )
    write_jsonl(WORK / "notes.jsonl", rows, overwrite=True)
    print(f"{len(rows)} notes; {sum(1 for r in rows if r['cands'])} with candidates")


def _span(text: str, se: list[int] | None) -> dict[str, Any] | None:
    return None if se is None else {"text": text[se[0] : se[1]], "start": se[0], "end": se[1]}


def _clef_span(row: dict[str, Any], clef: dict[str, Any] | None) -> list[int] | None:
    if clef is None or clef["choice"] == "none":
        return None
    c = row["cands"][int(clef["choice"][1:])]
    return [c["start"], c["end"]]


def _load() -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, str]]:
    rows = read_jsonl(WORK / "notes.jsonl")
    clef = {r["key"]: r for r in read_jsonl(WORK / "clef.jsonl")}
    adj_path = WORK / "adjudicated.jsonl"
    adj = {r["key"]: r["decision"] for r in read_jsonl(adj_path)} if adj_path.exists() else {}
    return rows, clef, adj


def _disagreeing(rows, clef) -> list[dict[str, Any]]:
    out = []
    for r in rows:
        if not r["cands"]:
            continue  # nothing amount-shaped in the note: A and B agree on none
        b = _clef_span(r, clef.get(r["key"]))
        if r["key"] not in clef or b != r["regex"]:
            out.append(r)
    return out


def disagreements() -> None:
    rows, clef, _ = _load()
    bad = _disagreeing(rows, clef)
    missing = [r["key"] for r in rows if r["cands"] and r["key"] not in clef]
    print(f"# {len(bad)} disagreements ({len(missing)} notes without a Clef answer)")
    for i, r in enumerate(bad):
        a = _span(r["text"], r["regex"])
        b = _span(r["text"], _clef_span(r, clef.get(r["key"])))
        print(f"{i}|{r['text']}|A={a and a['text']}|B={b and b['text']}")
    write_jsonl(
        WORK / "disagreements.jsonl",
        ({"i": i, "key": r["key"]} for i, r in enumerate(bad)),
        overwrite=True,
    )


def _decide(row: dict[str, Any], clef_se, decision: str) -> tuple[list[int] | None, str]:
    text = row["text"]
    if decision == "A":
        return row["regex"], "complete"
    if decision == "B":
        return clef_se, "complete"
    if decision == "N":
        return None, "complete"
    if decision == "U":
        return None, "uncertain"
    if decision.startswith("="):
        want = decision[1:]
        at = text.rfind(want)
        if at < 0:
            raise ValueError(f"{row['key']}: {want!r} is not in {text!r}")
        return [at, at + len(want)], "complete"
    raise ValueError(f"{row['key']}: unknown decision {decision!r}")


def build() -> None:
    rows, clef, adj = _load()
    bad = {r["key"] for r in _disagreeing(rows, clef)}
    unresolved = sorted(bad - set(adj))
    if unresolved:
        raise SystemExit(f"{len(unresolved)} disagreements lack an adjudication: {unresolved[:3]}")
    by_set: dict[str, list[dict[str, Any]]] = {s: [] for s in SETS}
    stats: dict[str, Counter] = {s: Counter() for s in SETS}
    for r in rows:
        text = r["text"]
        b_se = _clef_span(r, clef.get(r["key"]))
        if r["key"] in bad:
            se, status = _decide(r, b_se, adj[r["key"]])
            prov = PROV_ADJUDICATED
        else:
            se, status = r["regex"], "complete"
            prov = PROV_CONSENSUS
        value = _span(text, se)
        rec = {
            "id": r["id"],
            "value": value,
            "value_status": status,
            "value_provenance": prov,
            "labelers": {
                "regex": (_span(text, r["regex"]) or {}).get("text"),
                "clef": (_span(text, b_se) or {}).get("text"),
            },
        }
        by_set[r["set"]].append(rec)
        s = stats[r["set"]]
        s["n"] += 1
        s["consensus" if prov == PROV_CONSENSUS else "adjudicated"] += 1
        s["uncertain"] += status == "uncertain"
        s["no_amount"] += status == "complete" and value is None
        s["agree_regex_clef"] += r["key"] not in bad
        s["clef_missing"] += r["key"] not in clef and bool(r["cands"])
    for name, recs in by_set.items():
        write_jsonl(OUT[name], recs, overwrite=True)
    total = Counter()
    for s in stats.values():
        total.update(s)
    report = {
        "per_set": {k: dict(v) for k, v in stats.items()},
        "total": dict(total),
        "agreement_regex_vs_clef": total["agree_regex_clef"] / total["n"],
        "adjudicated": total["adjudicated"],
    }
    (WORK / "agreement.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=1))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("command", choices=("prepare", "disagreements", "build"))
    args = ap.parse_args(argv)
    {"prepare": prepare, "disagreements": disagreements, "build": build}[args.command]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
