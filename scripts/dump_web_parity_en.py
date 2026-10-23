"""Dump the Python INT8 runtime's outputs that the browser must reproduce for ``gidi-en``.

The English counterpart of ``dump_web_parity.py``. ``GidiPredictor`` on the INT8 model of a frozen
English bundle (``--bundle``, default ``models/gidi-en-v4-bundle``: type + target + value from the
encoder, whole-word snaps; ``models/gidi-en-v2-bundle`` has no value) is the reference for
``playground/src/runtime``, which ``pnpm -C playground parity:en`` compares token by token, logit
by logit and span by span.

Outputs in ``--out`` (default ``playground/.parity/<bundle name>``, gitignored):

- ``expected.jsonl``: per input ``Prediction.to_dict()`` plus the tokenization, raw logits, target
  and value tags of the portable reference (INT8 graph as written, ``ORT_ENABLE_BASIC``).
  Inputs: the annotation-en-v1 splits and the two English probes (read only: scoring and parity,
  never training), the app's English examples and smoke notes, and seeded adversarial strings.
- ``sweep.jsonl``: tokenizer + NFC-map reference for ``a<c> <c>b`` over the same code points as
  the Vietnamese sweep (no model).
- ``meta.json``: bundle hashes and library versions.

``--fixture`` also writes the small committed fixture that ``pnpm -C playground test`` checks
without any model: for a BPE bundle (gidi-en-v2) ``playground/src/runtime/english.fixture.json``
(tokenizer ids/offsets and word-snap cases, ``gidi.inference.decode.snap_to_words``); for a
WordPiece bundle (gidi-en-v4) ``playground/src/runtime/wordpiece.fixture.json``: ids, offsets and
masks of every test, dev and probe note and of seeded fuzz strings.

    uv run python scripts/dump_web_parity_en.py --fixture
"""

from __future__ import annotations

import argparse
import json
import random
import re
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
from gidi.inference.bundle import (  # noqa: E402
    CONFIG_FILE,
    MODEL_FILE,
    TOKENIZER_FILE,
    TOKENIZER_TYPE_WORDPIECE,
    load_config,
)
from gidi.inference.decode import snap_to_words  # noqa: E402
from gidi.inference.schema import EmptyInputError  # noqa: E402
from gidi.inference.text import normalize_nfc  # noqa: E402
from gidi.inference.tokenizer import BundleTokenizer  # noqa: E402

DEFAULT_BUNDLE = ROOT / "models" / "gidi-en-v4-bundle"
PARITY_ROOT = ROOT / "playground" / ".parity"
FIXTURE = ROOT / "playground" / "src" / "runtime" / "english.fixture.json"
WORDPIECE_FIXTURE = ROOT / "playground" / "src" / "runtime" / "wordpiece.fixture.json"
FUZZ_SEED = 20261024
FUZZ_CASES = 600

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
    # amount shapes of the value head
    "got paid 2400",
    "lent bro 20",
    "coffee 5k",
    "2 grand for the car",
    "twenty bucks to sam",
    "paid rent",
    "rent 1.2k, split 3 ways",
    "$5 coffee",
    "owe jake $12.50 for lunch",
    "sold the couch for 1,200 usd",
    "s$30 grab",
    "a tenner for the cab",
    "50 cents",
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
    if raw.value_logits is not None:
        record["trace"]["value_logits"] = [[float(x) for x in row] for row in raw.value_logits]
        record["trace"]["value_tags"] = [int(i) for i in raw.value_tags or ()]
    return record


def token_record(tokenizer: BundleTokenizer, text: str) -> dict[str, Any]:
    encoded = tokenizer.encode(normalize_nfc(text).text)
    return {
        "text": text,
        "ids": list(encoded.ids),
        "offsets": [list(o) for o in encoded.offsets],
        "special": list(encoded.special_tokens_mask),
        "truncated": encoded.truncated,
    }


def snap_records() -> list[dict[str, Any]]:
    snaps = []
    for text, piece in SNAP_CASES:
        start = text.index(piece)
        lo, hi = snap_to_words(text, start, start + len(piece))
        snaps.append({"text": text, "start": start, "end": start + len(piece), "snapped": [lo, hi]})
    return snaps


def bpe_fixture(bundle: Path, tokenizer: BundleTokenizer) -> dict[str, Any]:
    """Small committed reference of the BPE bundle: tokenization of app notes, snap of spans."""
    texts = APP_NOTES + [
        "didn't pay y'all's share",
        "caf\u00e9 \u2603 \U0001f355 12",
        "a" * 80,
        # <mask> is lstrip: it takes the whitespace run (Unicode White_Space) before it.
        "a  <mask> b",
        "ab \u00a0<mask> \u2003 <mask>x",
        "x <mask> <unk> y",
        "a\ufeff <mask>",
        "<s> a <pad>",
    ]
    return {
        "tokenizer_sha256": sha256_file(bundle / TOKENIZER_FILE),
        "max_length": 32,
        "tokens": [token_record(tokenizer, text) for text in texts],
        "snaps": snap_records(),
    }


def wordpiece_fuzz_inputs() -> list[str]:
    """Seeded strings aimed at the places a WordPiece port can differ from the Rust crate."""
    rng = random.Random(FUZZ_SEED)
    words = [
        "venmo'd", "mom", "Sam", "lent", "borrowed", "PAID", "Uber", "Caf\u00e9", "na\u00efve",
        "r\u00e9sum\u00e9", "Zo\u00eb", "\u00d1o\u00f1o", "stra\u00dfe", "\u0130stanbul",
        "\u03a3\u03af\u03c3\u03c5\u03c6\u03bf\u03c2",
        "cashapp", "amazon", "x" * 101, "ab" * 60, "unbelievably", "tmrw", "y'all's", "mike's",
        "2400", "5k", "1.2m", "2.5K", "1,200", "12.50", "$20", "$5.99", "\u20ac50", "\u00a345",
        "\u20b9500", "s$30", "150s$", "20bucks", "twenty", "a hundred", "fiver", "3rd", "5pm",
        "7:30", "3/5", "http://a.co/x?y=1&z=2", "https://venmo.com/u/sam", "me@x.io", "#tag",
        "@sam", "50%", "i18n", "x2", "e-mail", "o'clock", "w/", "b/c", "&", "+", "=", "<", ">",
        "~", "^", "`", "|", "_", "\u2014", "\u2013", "\u2026", "\u201csmart\u201d",
        "\u2018quotes\u2019",
    ]  # fmt: skip
    seps = [" ", " ", " ", "  ", "", ",", ".", "!", "?", ";", ":", "(", ")", "-", "/", "\t", "\n"]
    specials = ["[CLS]", "[SEP]", "[PAD]", "[UNK]", "[MASK]", "[cls]", "[ SEP ]", "[CLS", "SEP]"]
    odd = [
        "\U0001f4b8", "\U0001f1fa\U0001f1f8", "\U0001f468\u200d\U0001f469\u200d\U0001f467",
        "\U0001f44d\U0001f3fd", "\U0001d554", "\U0010ffff", "\ud800", "\udfff", "\ufb03",
        "\u00c5", "\u212b", "\u01c6", "\ud55c", "\u4e2d\u6587", "\u3042\u3044", "\u0e01\u0e33",
        "\u00f6", "o\u0308", "\u01df", "a\u0308\u0304",
        "\x00", "\x7f", "\x1c", "\x85", "\xa0", "\u200b", "\u200d", "\ufeff", "\u3000", "\u2003",
        "\u2028", "\ufffd", "\u0301", "\u0323\u0301", "e\u0323\u0302\u0301", "\u1112\u1161\u11ab",
        "\uff21\uff22\uff23", "\u2460", "\u33cf", "\U00020000", "\U0002f800", "\U000e0001",
    ]  # fmt: skip
    pieces = words + seps + specials + odd
    out: list[str] = []
    for _ in range(FUZZ_CASES):
        parts = [rng.choice(pieces) for _ in range(rng.randint(1, 14))]
        glue = rng.choice([" ", "", " ", "  "])
        text = glue.join(parts)
        mode = rng.choice(["nfc", "nfc", "nfd", "mixed"])
        if mode == "nfd":
            text = unicodedata.normalize("NFD", text)
        elif mode == "mixed":
            text = "".join(
                unicodedata.normalize("NFD", ch) if rng.random() < 0.5 else ch for ch in text
            )
        # Lone surrogates must not be adjacent: JS joins a high+low pair into one code point.
        out.append(re.sub("([\ud800-\udbff])(?=[\udc00-\udfff])", r"\1x", text))
    out += [
        "x" * 3000,
        "paid " + "x " * 200,
        "a" * 99,
        "a" * 100,
        "a" * 101,
        "[CLS] a [SEP] b [MASK] c [PAD] d [UNK]",
        "a" * 29 + " b",
        " ".join(["tok"] * 31),
        " ".join(["tok"] * 30),
        "Mom sent me $200 for books, thx",
        "\U0001f600" * 40,
        "...,,,!!!???\u2026\u3002\uff0c\u3001\u0964\u06d4\u060c",
    ]
    return out


def split_notes() -> list[tuple[str, str]]:
    """Every test, dev and probe note (read only: scoring and parity, never training)."""
    rows: list[tuple[str, str]] = []
    for split in ("test", "dev"):
        path = ROOT / "datasets" / "annotation-en-v1" / "splits" / f"{split}.jsonl"
        rows += [(f"en-{split}", r["text"]) for r in read_jsonl(path)]
    for probe in ("probe-en-slang", "probe-en-genz"):
        rows += [(probe, r["text"]) for r in read_jsonl(ROOT / "datasets" / probe / "notes.jsonl")]
    return rows


def wordpiece_fixture(bundle: Path, tokenizer: BundleTokenizer) -> dict[str, Any]:
    """Committed reference of the WordPiece bundle: every eval-split note and the fuzz strings."""
    texts = list(dict.fromkeys(APP_NOTES + [t for _, t in split_notes()] + wordpiece_fuzz_inputs()))
    return {
        "tokenizer_sha256": sha256_file(bundle / TOKENIZER_FILE),
        "max_length": 32,
        "tokens": [token_record(tokenizer, text) for text in texts],
        "snaps": snap_records(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--fixture", action="store_true", help="also write the committed fixture")
    args = parser.parse_args()
    bundle: Path = args.bundle.resolve()
    out: Path = args.out or PARITY_ROOT / bundle.name
    out.mkdir(parents=True, exist_ok=True)
    config = load_config(bundle)
    wordpiece = config.tokenizer_type == TOKENIZER_TYPE_WORDPIECE

    graph = graph_level_predictor(bundle)
    inputs = english_inputs()
    if wordpiece:
        inputs += [("fuzz", f"fuzz-{i:04d}", t) for i, t in enumerate(wordpiece_fuzz_inputs())]
    records = [reference_record(graph, *item) for item in inputs]
    tokenizer = BundleTokenizer(bundle / TOKENIZER_FILE, config.max_length)

    sweep_count = 0
    with (out / "sweep.jsonl").open("w", encoding="utf-8") as sweep:
        for text in sweep_texts():
            sweep.write(json_line(sweep_record(tokenizer, text)) + "\n")
            sweep_count += 1
    (out / "expected.jsonl").write_text(
        "".join(json_line(r) + "\n" for r in records), encoding="utf-8"
    )
    meta = {
        "bundle": str(bundle.relative_to(ROOT)),
        "files": {
            name: sha256_file(bundle / name) for name in (CONFIG_FILE, TOKENIZER_FILE, MODEL_FILE)
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
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    if args.fixture:
        if wordpiece:
            lines = wordpiece_fixture(bundle, tokenizer)
            rows = ",\n".join(
                json.dumps(row, ensure_ascii=True, separators=(",", ":")) for row in lines["tokens"]
            )
            head = json.dumps(
                {k: v for k, v in lines.items() if k not in ("tokens", "snaps")},
                ensure_ascii=True,
                separators=(",", ":"),
            )[:-1]
            snaps = json.dumps(lines["snaps"], ensure_ascii=True, separators=(",", ":"))
            WORDPIECE_FIXTURE.write_text(
                f'{head},"snaps":{snaps},"tokens":[\n{rows}\n]}}\n', encoding="utf-8"
            )
        else:
            FIXTURE.write_text(
                json.dumps(bpe_fixture(bundle, tokenizer), indent=1, ensure_ascii=True) + "\n"
            )
    print(f"wrote {len(records)} inputs + {sweep_count} sweep strings to {out}")
    print(json.dumps(meta["by_source"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
