#!/usr/bin/env python
"""Data for gidi-en v2: silver slang training notes and the eval-only Gen Z probe.

Usage:
    uv run python scripts/build_en_slang_v2_data.py dedupe
    uv run python scripts/build_en_slang_v2_data.py silver   # after the teacher run (see below)
    uv run python scripts/build_en_slang_v2_data.py probe    # after Gemini + Clef + adjudication

``dedupe`` reads the generated slang files (``generated/en-slang-02{a,b}.jsonl`` and
``generated/en-slang-03*-genz.jsonl``), drops exact and near duplicates of the English corpus, every
English split, ``datasets/probe-en-slang`` and ``generated/en-slang-01.jsonl`` (its source), and of
each other, then holds 100 Gen Z notes (by id hash order) out as ``datasets/probe-en-genz``
(evaluation only, never trained on). Training candidates that are near duplicates of a held-out
probe note are dropped too. Work files go to ``datasets/annotation-en-v1/work/slang-v2`` and
``datasets/probe-en-genz/work``.

``silver`` writes ``datasets/annotation-en-v1/silver-slang-train.jsonl`` from the Clef kNN teacher
output (``experiments/gidi-en-student-v2/run_teacher_slang.py``; pool = seed notes only).
``probe`` builds ``datasets/probe-en-genz`` with the two-labeler method of the English probe
(Gemini + Clef zero-shot; disagreements adjudicated by Claude). Labels are silver, none human.
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

DATA = Path("datasets/annotation-en-v1")
WORK = DATA / "work/slang-v2"
PROBE = Path("datasets/probe-en-genz")
PWORK = PROBE / "work"
GEN = Path("generated")
SOURCES = {
    "en-slang-02": [GEN / "en-slang-02a.jsonl", GEN / "en-slang-02b.jsonl"],
    "en-genz-03": sorted(GEN.glob("en-slang-03*-genz.jsonl")),
}
N_PROBE = 100
LABELERS = ["gemini-3.8-flash", "@cf/cloudflare/clef-zero-shot"]


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def references() -> tuple[set[str], list[dict]]:
    texts: set[str] = set()
    for path in [Path("corpus/raw/en-baseline-01.jsonl"), GEN / "en-slang-01.jsonl"]:
        texts |= {r["text"] for r in read_jsonl(path)}
    for split in ("train", "dev", "test", "seed"):
        texts |= {r["text"] for r in read_jsonl(DATA / "splits" / f"{split}.jsonl")}
    texts |= {r["text"] for r in read_jsonl(Path("datasets/probe-en-slang/notes.jsonl"))}
    return texts, [reference(t, None, "en") for t in sorted(texts)]


def dedupe() -> int:
    texts, ref_list = references()
    ref_norms = {r["norm"] for r in ref_list}
    kept: list[dict] = []
    kept_refs: list[dict] = []
    dropped: list[dict] = []
    for prompt_id, files in SOURCES.items():
        for path in files:
            for r in read_jsonl(path):
                text = r["text"].strip()
                ref = reference(text, None, prompt_id)
                reason = None
                if text in texts or ref["norm"] in ref_norms:
                    reason = "exact"
                else:
                    sim, i = nearest(ref["norm"], ref_list)
                    if sim >= EVAL_THRESHOLD:
                        reason = f"near_corpus {sim:.2f}"
                    else:
                        sim, i = nearest(ref["norm"], kept_refs)
                        if sim >= NEAR_DUP_THRESHOLD or ref["norm"] in {
                            k["norm"] for k in kept_refs
                        }:
                            reason = f"near_batch {sim:.2f}"
                if reason:
                    dropped.append({"text": text, "prompt_id": prompt_id, "reason": reason})
                else:
                    kept.append({"text": text, "prompt_id": prompt_id, "file": path.name})
                    kept_refs.append(ref)
    genz = sorted(
        (k for k in kept if k["prompt_id"] == "en-genz-03"), key=lambda k: digest(k["text"])
    )
    probe, genz_train = genz[:N_PROBE], genz[N_PROBE:]
    probe_refs = [reference(k["text"], None, "probe") for k in probe]
    leaks = []
    for k in genz_train + [k for k in kept if k["prompt_id"] == "en-slang-02"]:
        ref = reference(k["text"], None, "x")
        sim, i = nearest(ref["norm"], probe_refs)
        if sim >= EVAL_THRESHOLD:
            leaks.append(k["text"])
            dropped.append(
                {"text": k["text"], "prompt_id": k["prompt_id"], "reason": "near_genz_probe"}
            )
    drop_texts = set(leaks)
    train = [k for k in kept if k["prompt_id"] == "en-slang-02" and k["text"] not in drop_texts] + [
        k for k in genz_train if k["text"] not in drop_texts
    ]
    for k in train:
        k["id"] = f"{k['prompt_id']}-{digest(k['text'])}"
    for k in probe:
        k["id"] = f"probe-en-genz-{digest(k['text'])}"
    write_jsonl(WORK / "train-candidates.jsonl", train, overwrite=True)
    write_jsonl(WORK / "dropped.jsonl", dropped, overwrite=True)
    write_jsonl(PWORK / "candidates.jsonl", probe, overwrite=True)
    print(
        f"kept {len(kept)}: train {len(train)} "
        f"(slang-02 {sum(k['prompt_id'] == 'en-slang-02' for k in train)}, "
        f"genz {sum(k['prompt_id'] == 'en-genz-03' for k in train)}), probe {len(probe)}"
    )
    print("dropped", Counter(d["reason"].split(" ")[0] for d in dropped))
    return 0


def silver() -> int:
    recs = read_jsonl(DATA / "work/slang-v2/teacher-out.jsonl")
    # notes rejected by scripts/check_note_similarity.py (eval_bag_match), one text per line
    leak_path = DATA / "work/slang-v2/leak-drops.txt"
    leaks = set(leak_path.read_text(encoding="utf-8").splitlines()) if leak_path.exists() else set()
    out = []
    for r in recs:
        if r["text"] in leaks:
            continue
        target = r["target"]
        flags = []
        if target is not None and r["text"][target["start"] : target["end"]] != target["text"]:
            continue  # substring validation
        if r["type"] == "transfer" and target is not None:
            r = {**r, "target": None}
            flags.append("transfer_target_dropped")
        out.append({**r, "flags": flags, "annotation_status": "complete"})
    write_jsonl(DATA / "silver-slang-train.jsonl", out, overwrite=True)
    consistent = sum(r["consistent"] for r in out)
    print(f"silver-slang-train: {len(out)} complete, {consistent} self-consistent")
    print(Counter(r["type"] for r in out if r["consistent"]))
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


def probe() -> int:
    texts = {r["id"]: r["text"] for r in read_jsonl(PWORK / "candidates.jsonl")}
    gem = {r["id"]: r for r in read_jsonl(PWORK / "gem-out.jsonl")}
    clef = {}
    for r in read_jsonl(PWORK / "clef-zero.jsonl"):
        if "error" in r:
            continue
        sp = r.get("target")
        r["target_text"] = r["text"][sp[0] : sp[1]] if sp else None
        clef[r["id"]] = r
    adj_path = PWORK / "adjudicated.jsonl"
    adj = {r["id"]: r for r in read_jsonl(adj_path)} if adj_path.exists() else {}
    missing = [i for i in texts if i not in gem]
    if missing:
        print(f"gemini is missing {len(missing)} ids", file=sys.stderr)
        return 1
    dis = []
    for i, text in texts.items():
        g, c = gem[i], clef.get(i)
        ok = (
            g.get("status") == "complete"
            and c is not None
            and g["type"] == c["type"]
            and g.get("target") == c["target_text"]
            and (g.get("target") is None or g["target"] in text)
        )
        if not ok and i not in adj:
            dis.append(
                {
                    "id": i,
                    "text": text,
                    "gemini": {k: g.get(k) for k in ("type", "target", "status", "note")},
                    "clef": c and {"type": c["type"], "target": c["target_text"]},
                }
            )
    if dis:
        write_jsonl(PWORK / "disagreements.jsonl", dis, overwrite=True)
        print(f"{len(dis)} disagreements need adjudication ({PWORK}/disagreements.jsonl)")
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
        notes.append({"id": i, "text": text, "source": "gemini-genz", "prompt_id": "en-genz-03"})
        labels.append(rec)
        queue.append({"id": i, "text": text, "corpus": str(PROBE / "notes.jsonl"), "position": pos})
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
    for name, rows in (
        ("notes", notes),
        ("labels", labels),
        ("labels.full", full),
        ("provenance", prov),
        ("queue", queue),
    ):
        write_jsonl(PROBE / f"{name}.jsonl", rows, overwrite=True)
    manifest = {
        "name": "probe-en-genz",
        "evaluation_only": True,
        "never_train": True,
        "purpose": "diagnostic probe: current Gen Z English slang finance notes, held out of "
        "gidi-en v2 training; reported as an extra, not part of any pass/fail",
        "annotation_version": "annotation-en-v1",
        "label_provenance": "silver-claude-clef (Gemini + Clef zero-shot, Claude adjudication)",
        "source_file": "generated/en-slang-03*-genz.jsonl",
        "id_prefix": "probe-en-genz-",
        "total": len(labels),
        "status_counts": dict(Counter(r["annotation_status"] for r in labels)),
        "type_counts": dict(Counter(str(r["type"]) for r in labels)),
        "source_counts": dict(Counter(p["source"] for p in prov)),
    }
    (PROBE / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(manifest)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("stage", choices=["dedupe", "silver", "probe"])
    args = ap.parse_args(argv)
    return {"dedupe": dedupe, "silver": silver, "probe": probe}[args.stage]()


if __name__ == "__main__":
    raise SystemExit(main())
