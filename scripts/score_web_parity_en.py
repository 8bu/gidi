"""Gold-label quality of the browser runtime next to the two Python INT8 sessions (gidi-en).

Scores the silver-labelled English evaluation sets of ``scripts/evaluate_en_value.py`` (test
primary/secondary, dev, slang probe, Gen Z probe) once per backend, from the decisions each
produced for the same bundle:

- ``python_default``: stock ``GidiPredictor`` (ORT default ``ORT_ENABLE_ALL``), the session the
  bake-off scored;
- ``python_basic``: the same with ``ORT_ENABLE_BASIC`` (the graph as written;
  ``expected.jsonl`` of ``dump_web_parity_en.py``);
- ``web``: onnxruntime-web (``pnpm -C playground parity:en`` wrote ``web-predictions.jsonl``).

Type macro-F1, target exact (whole-word snapped, as the bundle does) and value exact (snapped)
are computed with the frozen evaluation code. Writes ``quality.json`` into the dump directory.
Read only: never trains and never selects.

    uv run python scripts/dump_web_parity_en.py && pnpm -C playground parity:en \\
        && uv run python scripts/score_web_parity_en.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "src"))

from evaluate_en_student import evaluate_set  # noqa: E402
from evaluate_en_value import eval_sets, value_block  # noqa: E402

from gidi.inference import GidiPredictor  # noqa: E402

SOURCE = {
    "primary": "en-test",
    "secondary": "en-test",
    "dev": "en-dev",
    "slang_probe": "probe-en-slang",
    "genz_probe": "probe-en-genz",
}
BACKENDS = ("python_default", "python_basic", "web")


def span(text: str, found: list[int] | tuple[int, int] | None) -> dict[str, Any] | None:
    if found is None:
        return None
    return {"text": text[found[0] : found[1]], "start": found[0], "end": found[1]}


def score(records: list[dict[str, Any]], preds: list[dict[str, Any]]) -> dict[str, float]:
    types = [p["type"] for p in preds]
    targets = [span(r["text"], p["target_span"]) for r, p in zip(records, preds, strict=True)]
    values = [span(r["text"], p["value_span"]) for r, p in zip(records, preds, strict=True)]
    base = evaluate_set(records, types, targets)
    value = value_block(records, values, None)
    return {
        "n": len(records),
        "type_macro_f1": base["type_macro_f1"],
        "target_exact": base["span_exact"],
        "joint": base["joint"],
        "value_exact": value["exact"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--bundle", type=Path, default=ROOT / "models" / "gidi-en-v4-bundle")
    parser.add_argument("--dump", type=Path, default=None, help="dump dir (default by bundle)")
    args = parser.parse_args()
    dump = args.dump or ROOT / "playground" / ".parity" / args.bundle.name
    basic = {}
    for line in (dump / "expected.jsonl").read_text(encoding="utf-8").split("\n"):
        rec = json.loads(line) if line else {}
        if "expected" in rec:
            basic[(rec["source"], rec["id"])] = rec["expected"]
    web = {}
    for line in (dump / "web-predictions.jsonl").read_text(encoding="utf-8").split("\n"):
        if line:
            rec = json.loads(line)
            web[(rec["source"], rec["id"])] = rec["prediction"]
    default = GidiPredictor.from_bundle(args.bundle)

    sets = eval_sets()
    results: dict[str, dict[str, Any]] = {b: {} for b in BACKENDS}
    cache: dict[tuple[str, str], dict[str, Any]] = {}
    for name, records in sets.items():
        keys = [(SOURCE[name], r["id"]) for r in records]
        for key, record in zip(keys, records, strict=True):
            if key not in cache:
                cache[key] = default.predict(record["text"]).to_dict()
        for backend, table in (("python_basic", basic), ("web", web)):
            results[backend][name] = score(records, [table[k] for k in keys])
        results["python_default"][name] = score(records, [cache[k] for k in keys])

    for backend in BACKENDS:
        r = results[backend]
        r["composite"] = (
            r["secondary"]["type_macro_f1"]
            + r["secondary"]["target_exact"]
            + r["slang_probe"]["joint"]
            + r["slang_probe"]["value_exact"]
            + r["genz_probe"]["joint"]
        ) / 5
    (dump / "quality.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print("backend         composite  " + "  ".join(f"{s[:11]:>11}" for s in sets))
    for metric in ("type_macro_f1", "target_exact", "value_exact"):
        print(metric)
        for backend in BACKENDS:
            row = "  ".join(f"{results[backend][s][metric]:11.4f}" for s in sets)
            print(f"  {backend:<14}{results[backend]['composite']:9.4f}  {row}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
