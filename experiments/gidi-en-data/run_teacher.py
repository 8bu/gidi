"""gidi-en few-shot kNN Clef teacher runner. Run inside omp ``eval`` (py) where ``judge`` exists:

    %load experiments/gidi-en-data/run_teacher.py
    await run_dev()      # pool = seed labels; scores against labels-dev; writes dev-teacher.jsonl
    await run_train()    # pool = seed labels; labels train notes -> silver-train.jsonl

Pass A: 16 nearest seed shots. Pass B (self-consistency): 12 nearest shots, order reversed. A note
is ``consistent`` when both passes agree on type and target. The test split is never read here.
Labels are silver (Claude + Clef), never human.
"""

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, "scripts")
import clef_teacher as ct  # noqa: E402

D = Path("datasets/annotation-en-v1")
SEM = asyncio.Semaphore(24)
TEACHER = "@cf/cloudflare/clef (omp judge role)"


def _jl(path):
    return [json.loads(line) for line in open(path) if line.strip()]


def seed_pool():
    texts = {r["id"]: r["text"] for r in _jl(D / "splits/seed.jsonl")}
    return [
        {"text": texts[r["id"]], "type": r["type"], "target": r["target"]}
        for r in _jl(D / "labels-seed.jsonl")
        if r["annotation_status"] == "complete"
    ]


async def _ask(text, state):
    async with SEM:
        try:
            ans = await judge(state, ct.questions(text, lang="en"))  # noqa: F821
            return ct.decode(text, ans)
        except Exception as exc:  # noqa: BLE001
            return {"error": str(exc)}


async def _label(row, pool):
    state_a = pool.state(row["text"], k=16)
    state_b = pool.state(row["text"], k=12)
    state_b["labelled_examples_following_the_same_annotation_rules"].reverse()
    a, b = await asyncio.gather(_ask(row["text"], state_a), _ask(row["text"], state_b))
    return row, a, b


def _span(text, pred):
    t = pred["target"]
    if t is None:
        return None
    return {"text": text[t[0] : t[1]], "start": t[0], "end": t[1]}


def _record(row, a, b):
    if "error" in a or "error" in b:
        return None
    ta, tb = _span(row["text"], a), _span(row["text"], b)
    return {
        "id": row["id"],
        "text": row["text"],
        "type": a["type"],
        "target": ta,
        "type_prob": round(a["type_probs"][a["type"]], 4),
        "target_prob": round(a["target_prob"], 4),
        "consistent": a["type"] == b["type"] and ta == tb,
        "type_b": b["type"],
        "target_b": tb,
        "provenance": {
            "annotator": "clef",
            "teacher": TEACHER,
            "prompt": "scripts/clef_teacher.py lang=en, 16-shot kNN (pool = seed)",
            "label_provenance": "silver-claude-clef",
        },
    }


async def run(split, out):
    pool = ct.ShotPool(seed_pool())
    rows = _jl(D / f"splits/{split}.jsonl")
    res = await asyncio.gather(*[_label(r, pool) for r in rows])
    recs, bad = [], 0
    for row, a, b in res:
        rec = _record(row, a, b)
        if rec is None:
            bad += 1
            continue
        if (
            rec["target"]
            and rec["text"][rec["target"]["start"] : rec["target"]["end"]]
            != (rec["target"]["text"])
        ):  # pragma: no cover - candidates are substrings by construction
            bad += 1
            continue
        recs.append(rec)
    with open(out, "w") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{split}: {len(recs)} labelled, {bad} dropped (judge error)")
    return recs


async def run_dev():
    return await run("dev", D / "work/dev-teacher.jsonl")


async def run_train():
    return await run("train", D / "silver-train.jsonl")
