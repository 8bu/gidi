#!/usr/bin/env python
"""gidi-en silver labels for test/dev/seed: two-labeler agreement, adjudication, label files.

Usage:
    uv run python scripts/build_en_labels.py compare   # work/disagreements.jsonl + agreement.json
    uv run python scripts/build_en_labels.py build     # labels-{test,dev,seed}.jsonl, provenance

Labelers (independent, both guided by docs/annotation-en-v1.md / configs/annotation-en-v1.yaml):
``gemini`` (work/gem-out-*.jsonl) and ``clef`` zero-shot (work/clef-zero.jsonl). Agreement on type
and target text -> source ``consensus``. Every other note is adjudicated by Claude by reading the
guide (work/adjudicated.jsonl, source ``claude-adjudicated``). All labels are silver
(``label_provenance: silver-claude-clef``); nothing here is human.

``labels-*.jsonl`` follow the validator contract (no extra fields); the per-id ``source`` is in
``provenance.jsonl`` and in ``labels-*.full.jsonl`` (text, label, both labelers' raw answers).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from gidi.corpus.jsonl import read_jsonl, write_jsonl

ROOT = Path("datasets/annotation-en-v1")
WORK = ROOT / "work"
SPLITS = ("test", "dev", "seed")
CORPUS = Path("corpus/raw/en-baseline-01.jsonl")


def load() -> tuple[dict[str, dict], dict[str, dict], dict[str, dict], dict[str, str]]:
    split_of: dict[str, str] = {}
    texts: dict[str, str] = {}
    for s in SPLITS:
        for r in read_jsonl(ROOT / "splits" / f"{s}.jsonl"):
            split_of[r["id"]] = s
            texts[r["id"]] = r["text"]
    gem = {r["id"]: r for p in sorted(WORK.glob("gem-out-*.jsonl")) for r in read_jsonl(p)}
    clef = {}
    for r in read_jsonl(WORK / "clef-zero.jsonl"):
        if "error" in r:
            continue
        span = r.get("target")
        r["target_text"] = r["text"][span[0] : span[1]] if span else None
        clef[r["id"]] = r
    return gem, clef, texts, split_of


def compare() -> int:
    gem, clef, texts, split_of = load()
    missing = [i for i in texts if i not in gem]
    if missing:
        print(f"gemini is missing {len(missing)} ids", file=sys.stderr)
        return 1
    rows, dis = [], []
    for i, text in texts.items():
        g, c = gem[i], clef.get(i)
        valid = g.get("target") is None or g["target"] in text
        t_ok = c is not None and g["type"] == c["type"]
        s_ok = c is not None and g.get("target") == c["target_text"]
        ok = g.get("status") == "complete" and t_ok and s_ok and valid
        rows.append((i, g, c, t_ok, s_ok, ok))
        if not ok:
            dis.append(
                {
                    "id": i,
                    "text": text,
                    "gemini": {k: g.get(k) for k in ("type", "target", "status", "note")},
                    "clef": c and {"type": c["type"], "target": c["target_text"]},
                }
            )
    write_jsonl(WORK / "disagreements.jsonl", dis, overwrite=True)
    by_type: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0, 0])
    for _, g, _c, t_ok, s_ok, ok in rows:
        b = by_type[str(g["type"])]
        b[0] += 1
        b[1] += t_ok
        b[2] += s_ok
        b[3] += ok
    n = len(rows)
    both_target_any = [
        r for r in rows if r[1].get("target") is not None or (r[2] and r[2]["target_text"])
    ]
    po = sum(r[3] for r in rows) / n
    gc, cc = Counter(r[1]["type"] for r in rows), Counter(r[2]["type"] for r in rows if r[2])
    pe = sum(gc[t] * cc[t] for t in gc) / (n * n)
    stats = {
        "n": n,
        "type_agree": po,
        "type_kappa": (po - pe) / (1 - pe),
        "target_agree": sum(r[4] for r in rows) / n,
        "target_agree_on_notes_with_a_target": (
            sum(r[4] for r in both_target_any) / len(both_target_any)
        ),
        "notes_with_a_target": len(both_target_any),
        "consensus": sum(r[5] for r in rows),
        "to_adjudicate": len(dis),
        "by_gemini_type": {
            t: dict(zip(("n", "type_agree", "target_agree", "consensus"), v, strict=True))
            for t, v in sorted(by_type.items())
        },
    }
    (WORK / "agreement.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats, indent=2))
    return 0


def _span(text: str, target: str | None, clef: dict | None) -> dict | None:
    if target is None:
        return None
    if clef and clef.get("target") and clef["target_text"] == target:
        start, end = clef["target"]
    else:
        start = text.index(target)
        end = start + len(target)
    return {"text": target, "start": start, "end": end}


def build() -> int:
    gem, clef, texts, split_of = load()
    adj = {r["id"]: r for r in read_jsonl(WORK / "adjudicated.jsonl")}
    dis_ids = {r["id"] for r in read_jsonl(WORK / "disagreements.jsonl")}
    if dis_ids != set(adj):
        print(f"adjudications cover {len(set(adj) & dis_ids)}/{len(dis_ids)}", file=sys.stderr)
        return 1
    out: dict[str, list] = {s: [] for s in SPLITS}
    full: dict[str, list] = {s: [] for s in SPLITS}
    prov = []
    for i, text in texts.items():
        c, g = clef.get(i), gem[i]
        if i in adj:
            a, source = adj[i], "claude-adjudicated"
        else:
            a, source = g, "consensus"
        status = a.get("status", "complete")
        kind = None if status == "skipped" else a["type"]
        if status == "skipped" or kind == "transfer":
            target = None
        else:
            target = _span(text, a.get("target"), c)
        rec = {"id": i, "annotation_status": status, "type": kind, "target": target}
        if a.get("note"):
            rec["note"] = a["note"]
        out[split_of[i]].append(rec)
        full[split_of[i]].append(
            {
                **rec,
                "text": text,
                "source": source,
                "gemini": {k: g.get(k) for k in ("type", "target", "status")},
                "clef": c and {"type": c["type"], "target": c["target_text"]},
            }
        )
        prov.append(
            {
                "id": i,
                "split": split_of[i],
                "annotator": source,
                "label_provenance": "silver-claude-clef",
                "labelers": ["gemini-3.8-flash", "@cf/cloudflare/clef-zero-shot"],
                "source": source,
            }
        )
    for s in SPLITS:
        write_jsonl(ROOT / f"labels-{s}.jsonl", out[s], overwrite=True)
        write_jsonl(ROOT / f"labels-{s}.full.jsonl", full[s], overwrite=True)
    write_jsonl(ROOT / "provenance.jsonl", prov, overwrite=True)
    queue = [
        {"id": r["id"], "text": r["text"], "corpus": str(CORPUS), "split": split_of.get(r["id"])}
        for r in read_jsonl(CORPUS)
    ]
    write_jsonl(ROOT / "queue.jsonl", queue, overwrite=True)
    print({s: len(v) for s, v in out.items()}, Counter(p["source"] for p in prov))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("cmd", choices=("compare", "build"))
    args = parser.parse_args(argv)
    return compare() if args.cmd == "compare" else build()


if __name__ == "__main__":
    sys.exit(main())
