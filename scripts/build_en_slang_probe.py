#!/usr/bin/env python
"""Build ``datasets/probe-en-slang``: an evaluation-only English slang probe (never trained on).

Usage:
    uv run python scripts/build_en_slang_probe.py dedupe   # work/candidates.jsonl, gem-in.jsonl
    uv run python scripts/build_en_slang_probe.py compare  # work/disagreements.jsonl
    uv run python scripts/build_en_slang_probe.py build    # notes/labels/queue/provenance/manifest

Same two-labeler method as ``scripts/build_en_labels.py``: Gemini (``work/gem-out.jsonl``) and Clef
zero-shot (``work/clef-zero.jsonl``); agreement on type and target text is ``consensus``, the rest
is adjudicated by Claude (``work/adjudicated.jsonl``). All labels are silver, none human.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

from gidi.annotation.leakage import EVAL_THRESHOLD, NEAR_DUP_THRESHOLD, nearest, reference
from gidi.corpus.jsonl import read_jsonl, write_jsonl

OUT = Path("datasets/probe-en-slang")
WORK = OUT / "work"
SOURCE = Path("generated/en-slang-01.jsonl")
CORPUS = Path("corpus/raw/en-baseline-01.jsonl")
EN_SPLITS = Path("datasets/annotation-en-v1/splits")
PREFIX = "probe-en-slang-"
LABELERS = ["gemini-3.8-flash", "@cf/cloudflare/clef-zero-shot"]


def note_id(text: str) -> str:
    return PREFIX + hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def dedupe() -> int:
    refs = {r["text"]: r for r in read_jsonl(CORPUS)}
    for split in ("train", "dev", "test", "seed"):
        for r in read_jsonl(EN_SPLITS / f"{split}.jsonl"):
            refs.setdefault(r["text"], r)
    ref_list = [reference(t, None, "en") for t in refs]
    ref_norms = {r["norm"] for r in ref_list}
    kept: list[dict] = []
    kept_refs: list[dict] = []
    dropped: list[dict] = []
    for r in read_jsonl(SOURCE):
        text = r["text"].strip()
        ref = reference(text, None, "probe")
        reason = None
        if text in refs or ref["norm"] in ref_norms:
            reason = "exact"
        else:
            sim, i = nearest(ref["norm"], ref_list)
            if sim >= EVAL_THRESHOLD:
                reason = f"near_corpus {sim:.2f}: {ref_list[i]['text']!r}"
            else:
                sim, i = nearest(ref["norm"], kept_refs)
                if sim >= NEAR_DUP_THRESHOLD:
                    reason = f"near_batch {sim:.2f}: {kept_refs[i]['text']!r}"
        if reason:
            dropped.append({"text": text, "reason": reason})
        else:
            kept.append({"id": note_id(text), "text": text})
            kept_refs.append(ref)
    write_jsonl(WORK / "candidates.jsonl", kept, overwrite=True)
    write_jsonl(WORK / "dropped.jsonl", dropped, overwrite=True)
    for n, i in enumerate(range(0, len(kept), 400), 1):
        write_jsonl(WORK / f"gem-in-{n}.jsonl", kept[i : i + 400], overwrite=True)
    print(f"source {len(kept) + len(dropped)}: kept {len(kept)}, dropped {len(dropped)}")
    print(Counter(d["reason"].split(" ")[0] for d in dropped))
    return 0


def load() -> tuple[dict[str, str], dict[str, dict], dict[str, dict]]:
    texts = {r["id"]: r["text"] for r in read_jsonl(WORK / "candidates.jsonl")}
    gem = {r["id"]: r for p in sorted(WORK.glob("gem-out*.jsonl")) for r in read_jsonl(p)}
    clef = {}
    for r in read_jsonl(WORK / "clef-zero.jsonl"):
        if "error" in r:
            continue
        span = r.get("target")
        r["target_text"] = r["text"][span[0] : span[1]] if span else None
        clef[r["id"]] = r
    return texts, gem, clef


def compare() -> int:
    texts, gem, clef = load()
    missing = [i for i in texts if i not in gem]
    if missing:
        print(f"gemini is missing {len(missing)} ids", file=sys.stderr)
        return 1
    dis, agree = [], 0
    for i, text in texts.items():
        g, c = gem[i], clef.get(i)
        valid = g.get("target") is None or g["target"] in text
        ok = (
            g.get("status") == "complete"
            and c is not None
            and g["type"] == c["type"]
            and g.get("target") == c["target_text"]
            and valid
        )
        agree += ok
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
    stats = {"n": len(texts), "consensus": agree, "to_adjudicate": len(dis)}
    (WORK / "agreement.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(stats)
    return 0


def span(text: str, target: str | None, clef: dict | None) -> dict | None:
    if target is None:
        return None
    if clef and clef.get("target") and clef["target_text"] == target:
        start, end = clef["target"]
    else:
        start = text.index(target)
        end = start + len(target)
    return {"text": target, "start": start, "end": end}


def build() -> int:
    texts, gem, clef = load()
    adj = {r["id"]: r for r in read_jsonl(WORK / "adjudicated.jsonl")}
    dis_ids = {r["id"] for r in read_jsonl(WORK / "disagreements.jsonl")}
    if dis_ids != set(adj):
        print(f"adjudications cover {len(set(adj) & dis_ids)}/{len(dis_ids)}", file=sys.stderr)
        return 1
    notes, labels, prov, full, queue = [], [], [], [], []
    for pos, (i, text) in enumerate(texts.items(), 1):
        c, g = clef.get(i), gem[i]
        a, source = (adj[i], "claude-adjudicated") if i in adj else (g, "consensus")
        status = a.get("status", "complete")
        kind = None if status == "skipped" else a["type"]
        target = (
            None if status == "skipped" or kind == "transfer" else span(text, a.get("target"), c)
        )
        rec = {"id": i, "annotation_status": status, "type": kind, "target": target}
        if a.get("note"):
            rec["note"] = a["note"]
        notes.append({"id": i, "text": text, "source": "gemini-slang", "prompt_id": "en-slang-01"})
        labels.append(rec)
        queue.append({"id": i, "text": text, "corpus": str(OUT / "notes.jsonl"), "position": pos})
        prov.append(
            {
                "id": i,
                "annotator": source,
                "label_provenance": "silver-claude-clef",
                "labelers": LABELERS,
                "source": source,
            }
        )
        full.append(
            {
                **rec,
                "text": text,
                "source": source,
                "gemini": {k: g.get(k) for k in ("type", "target", "status")},
                "clef": c and {"type": c["type"], "target": c["target_text"]},
            }
        )
    write_jsonl(OUT / "notes.jsonl", notes, overwrite=True)
    write_jsonl(OUT / "labels.jsonl", labels, overwrite=True)
    write_jsonl(OUT / "labels.full.jsonl", full, overwrite=True)
    write_jsonl(OUT / "provenance.jsonl", prov, overwrite=True)
    write_jsonl(OUT / "queue.jsonl", queue, overwrite=True)
    manifest = {
        "name": "probe-en-slang",
        "evaluation_only": True,
        "never_train": True,
        "purpose": "diagnostic probe: slang-heavy English notes (texting slang, abbreviations, "
        "5k/grand/quid/lakh, emoji) that the annotation-en-v1 test set barely covers",
        "annotation_version": "annotation-en-v1",
        "label_provenance": "silver-claude-clef (Gemini + Clef zero-shot, Claude adjudication)",
        "source_file": str(SOURCE),
        "id_prefix": PREFIX,
        "total": len(labels),
        "status_counts": dict(Counter(r["annotation_status"] for r in labels)),
        "type_counts": dict(Counter(str(r["type"]) for r in labels)),
        "source_counts": dict(Counter(p["source"] for p in prov)),
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(manifest)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("cmd", choices=("dedupe", "compare", "build"))
    cmd = parser.parse_args(argv).cmd
    return {"dedupe": dedupe, "compare": compare, "build": build}[cmd]()


if __name__ == "__main__":
    sys.exit(main())
