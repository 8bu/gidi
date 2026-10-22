"""Dump the Python INT8 runtime's outputs that the browser must reproduce for ``gidi-en-v2``.

The English counterpart of ``dump_web_parity.py``. ``GidiPredictor`` on the INT8 model of the
frozen ``models/gidi-en-v2-bundle`` (type + target from the encoder, whole-word target snap; no
value) is the reference for ``playground/src/runtime``, which ``pnpm -C playground parity:en``
compares token by token, logit by logit and span by span.

Outputs in ``--out`` (default ``playground/.parity/en``, gitignored):

- ``expected.jsonl``: per input ``Prediction.to_dict()`` plus the tokenization, raw logits and
  target tags of the portable reference (INT8 graph as written, ``ORT_ENABLE_BASIC``).
  Inputs: the annotation-en-v1 splits and the two English probes (read only: scoring and parity,
  never training), the app's English examples and smoke notes, and seeded adversarial strings.
- ``sweep.jsonl``: tokenizer + NFC-map reference for ``a<c> <c>b`` over the same code points as
  the Vietnamese sweep (no model).
- ``meta.json``: bundle hashes and library versions.

``--fixture`` also writes the small committed fixture
``playground/src/runtime/english.fixture.json`` that ``pnpm -C playground test`` checks without
any model: tokenizer ids/offsets and word-snap cases (``gidi.inference.decode.snap_to_words``).

    uv run python scripts/dump_web_parity_en.py --fixture
"""

from __future__ import annotations

import argparse
import json
import sys
import unicodedata
from importlib import metadata
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "src"))

from dump_web_parity import (  # noqa: E402
    graph_level_predictor,
    json_line,
    read_jsonl,
    sha256_file,
    stress_inputs,
    sweep_record,
    sweep_texts,
)

from gidi.inference import GidiPredictor  # noqa: E402
from gidi.inference.bundle import CONFIG_FILE, MODEL_FILE, TOKENIZER_FILE  # noqa: E402
from gidi.inference.decode import snap_to_words  # noqa: E402
from gidi.inference.schema import EmptyInputError  # noqa: E402
from gidi.inference.text import normalize_nfc  # noqa: E402
from gidi.inference.tokenizer import BundleTokenizer  # noqa: E402

BUNDLE = ROOT / "models" / "gidi-en-v2-bundle"
DEFAULT_OUT = ROOT / "playground" / ".parity" / "en"
FIXTURE = ROOT / "playground" / "src" / "runtime" / "english.fixture.json"

# The English examples of the app (playground/src/app/i18n.ts) and the notes of the smoke test.
APP_NOTES = [
    "venmo'd mom 50",
    "bro paid me back 20",
    "got paid",
    "lent sam 30 for pizza",
    "uber to the airport 42.50",
    "split dinner w/ the guys, owe Mike 23.40",
    "landlord refunded my deposit 1,200",
    "cashapp'd jess for the concert tix",
    "Mom sent me $200 for books",
    "paid back dave",
    "sold my old bike for 150",
    "amazon refund for the headphones",
]

SNAP_CASES = [
    ("lunch w ader 15", "der"),
    ("paid osa back 20", "os"),
    ("venmo'd mom 50", "mo"),
    ("(vpbank), transfer 5", "pban"),
    ("owe Sarah's brother 15", "ara"),
    ("send $ to jo-anne now", "anne"),
    ("rent to bob_smith!!", "smit"),
    ("lent Dr. Lee 40", "Lee"),
    ("coffee for mike—thanks", "ike"),
    ("uber w/ jake&amy 12", "jake"),
    ("cash to \u201cDan\u201d for gas", "Da"),
    ("ok", "ok"),
    ("  padded   words here  ", "dde"),
]


def english_inputs() -> list[tuple[str, str, str]]:
    rows: list[tuple[str, str, str]] = []
    for split in ("test", "dev", "seed", "train"):
        for r in read_jsonl(ROOT / "datasets" / "annotation-en-v1" / "splits" / f"{split}.jsonl"):
            rows.append((f"en-{split}", r["id"], r["text"]))
    for probe in ("probe-en-slang", "probe-en-genz"):
        for r in read_jsonl(ROOT / "datasets" / probe / "notes.jsonl"):
            rows.append((probe, r["id"], r["text"]))
    rows += [("app", f"app-{i}", t) for i, t in enumerate(APP_NOTES)]
    rows += [("stress", f"stress-{i:04d}", t) for i, t in enumerate(stress_inputs())]
    rows += [
        ("hardening", "roberta-space", " ".join(["it's"] * 20)),
        ("hardening", "contractions", "didn't pay y'all's share, i'll pay mike's 20 tmrw"),
        ("hardening", "long", "paid " + "x " * 200),
        ("hardening", "empty", "   "),
    ]
    return rows


def reference_record(graph: GidiPredictor, source: str, ident: str, text: str) -> dict[str, Any]:
    record: dict[str, Any] = {"source": source, "id": ident, "text": text}
    try:
        prediction = graph.predict(text).to_dict()
        raw = graph.run(text)
    except EmptyInputError:
        record["error"] = "EmptyInputError"
        return record
    normalized = normalize_nfc(text)
    record["expected"] = prediction
    record["trace"] = {
        "nfc": raw.normalized_text,
        "nfc_start_map": None if normalized.start_map is None else list(normalized.start_map),
        "nfc_end_map": None if normalized.end_map is None else list(normalized.end_map),
        "ids": list(raw.input_ids),
        "attention_mask": [1] * len(raw.input_ids),
        "offsets": [list(o) for o in raw.normalized_offsets],
        "original_offsets": [list(o) for o in raw.offsets],
        "special": list(raw.special_tokens_mask),
        "type_logits": [float(x) for x in raw.type_logits],
        "tag_logits": [[float(x) for x in row] for row in raw.tag_logits],
        "target_tags": [int(i) for i in raw.tag_logits.argmax(axis=-1)],
    }
    return record


def fixture(tokenizer: BundleTokenizer) -> dict[str, Any]:
    """Small committed reference: tokenization of the app notes and the snap of fixed spans."""
    tokens = []
    for text in APP_NOTES + [
        "didn't pay y'all's share",
        "caf\u00e9 \u2603 \U0001f355 12",
        "a" * 80,
        # <mask> is lstrip: it takes the whitespace run (Unicode White_Space) before it.
        "a  <mask> b",
        "ab \u00a0<mask> \u2003 <mask>x",
        "x <mask> <unk> y",
        "a\ufeff <mask>",
        "<s> a <pad>",
    ]:
        normalized = normalize_nfc(text)
        encoded = tokenizer.encode(normalized.text)
        tokens.append(
            {
                "text": text,
                "ids": list(encoded.ids),
                "offsets": [list(o) for o in encoded.offsets],
                "special": list(encoded.special_tokens_mask),
                "truncated": encoded.truncated,
            }
        )
    snaps = []
    for text, piece in SNAP_CASES:
        start = text.index(piece)
        lo, hi = snap_to_words(text, start, start + len(piece))
        snaps.append({"text": text, "start": start, "end": start + len(piece), "snapped": [lo, hi]})
    return {
        "tokenizer_sha256": sha256_file(BUNDLE / TOKENIZER_FILE),
        "max_length": 32,
        "tokens": tokens,
        "snaps": snaps,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--fixture", action="store_true", help="also write the committed fixture")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    graph = graph_level_predictor(BUNDLE)
    inputs = english_inputs()
    records = [reference_record(graph, *item) for item in inputs]
    tokenizer = BundleTokenizer(BUNDLE / TOKENIZER_FILE, 32)

    sweep_count = 0
    with (args.out / "sweep.jsonl").open("w", encoding="utf-8") as sweep:
        for text in sweep_texts():
            sweep.write(json_line(sweep_record(tokenizer, text)) + "\n")
            sweep_count += 1
    (args.out / "expected.jsonl").write_text(
        "".join(json_line(r) + "\n" for r in records), encoding="utf-8"
    )
    meta = {
        "bundle": str(BUNDLE.relative_to(ROOT)),
        "files": {
            name: sha256_file(BUNDLE / name) for name in (CONFIG_FILE, TOKENIZER_FILE, MODEL_FILE)
        },
        "python": sys.version.split()[0],
        "unicode": unicodedata.unidata_version,
        "versions": {n: metadata.version(n) for n in ("onnxruntime", "tokenizers", "numpy")},
        "inputs": len(records),
        "by_source": {
            s: sum(1 for r in records if r["source"] == s)
            for s in dict.fromkeys(r["source"] for r in records)
        },
        "sweep": sweep_count,
    }
    (args.out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    if args.fixture:
        FIXTURE.write_text(json.dumps(fixture(tokenizer), indent=1, ensure_ascii=True) + "\n")
    print(f"wrote {len(records)} inputs + {sweep_count} sweep strings to {args.out}")
    print(json.dumps(meta["by_source"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
