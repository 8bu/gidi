"""Labeler B of the gidi-en value labels: Clef picks the amount among A's loose candidates.

Run inside omp ``eval`` (py) where ``judge`` exists, after ``build_en_value_labels.py prepare``:

    %load experiments/gidi-en-student-v3/run_value_teacher.py
    await run_value_clef()     # writes datasets/annotation-en-v1/work/value/clef.jsonl

Zero-shot with a fixed instruction and six hand-written examples (build_en_value_labels.py); no
dataset note is shown to the model. Notes without any amount-shaped token are not sent (labelers
agree on ``none``). Labels are silver (Clef), never human.
"""

import asyncio
import json
import sys

sys.path.insert(0, "scripts")
import build_en_value_labels as bv  # noqa: E402

SEM = asyncio.Semaphore(24)


async def _ask(row):
    async with SEM:
        try:
            ans = await judge(row["text"], bv.clef_questions(row["text"], row["cands"]))  # noqa: F821
            a = ans["value"]
            return {
                "key": row["key"],
                "choice": a["choice"],
                "prob": round(a["probabilities"].get(a["choice"], 0.0), 4),
            }
        except Exception as exc:  # noqa: BLE001
            return {"key": row["key"], "error": str(exc)}


async def run_value_clef():
    rows = [json.loads(x) for x in (bv.WORK / "notes.jsonl").read_text().splitlines() if x.strip()]
    todo = [r for r in rows if r["cands"]]
    res = await asyncio.gather(*[_ask(r) for r in todo])
    ok = [r for r in res if "error" not in r]
    with open(bv.WORK / "clef.jsonl", "w") as f:
        for r in ok:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{len(ok)} answered, {len(res) - len(ok)} errors of {len(todo)}")
    return [r for r in res if "error" in r][:3]
