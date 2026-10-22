"""Silver-label the new slang training notes with the Clef kNN teacher (pool = seed only).

Run inside omp ``eval`` (py) where ``judge`` exists:

    %load experiments/gidi-en-student-v2/run_teacher_slang.py
    await run_slang()

Same teacher, shots and consistency rule as ``experiments/gidi-en-data/run_teacher.py``
(pass A = 16 nearest seed shots, pass B = 12 nearest reversed). Input
``datasets/annotation-en-v1/work/slang-v2/train-candidates.jsonl`` (written by
``scripts/build_en_slang_v2_data.py dedupe``); output ``work/slang-v2/teacher-out.jsonl``.
"""

import asyncio
import json
import sys

sys.path.insert(0, "scripts")
sys.path.insert(0, "experiments/gidi-en-data")
import clef_teacher as ct  # noqa: E402
import run_teacher as rt  # noqa: E402

rt.judge = judge  # noqa: F821


async def run_slang():
    pool = ct.ShotPool(rt.seed_pool())
    rows = rt._jl(rt.D / "work/slang-v2/train-candidates.jsonl")
    res = await asyncio.gather(*[rt._label(r, pool) for r in rows])
    recs, bad = [], 0
    for row, a, b in res:
        rec = rt._record(row, a, b)
        if rec is None:
            bad += 1
            continue
        rec["prompt_id"] = row["prompt_id"]
        recs.append(rec)
    with open(rt.D / "work/slang-v2/teacher-out.jsonl", "w") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{len(recs)} labelled, {bad} dropped (judge error)")
