"""Dump the Python INT8 runtime's outputs that the browser runtime must reproduce.

The in-browser playground (``playground/src/runtime``) is a port of ``gidi.inference`` and
``gidi.value_parser``. This script runs the reference, ``GidiPredictor`` on the INT8 model of the
``gidi-finance-v3`` bundle (type + target from the encoder, whole-word target snap, value from the
rule parser), over a fixed set of inputs and writes everything the TS side needs to compare:

- ``expected.jsonl``: one record per input with ``Prediction.to_dict()`` (or the error), the
  tokenization (ids, NFC offsets, special mask) and the raw logits / target tags, from the
  *portable* reference (INT8 graph as written, ``ORT_ENABLE_BASIC``; see ``graph_level_predictor``),
  plus ``default``: the stock ``GidiPredictor`` (ORT's default fused kernels) for comparison.
  Inputs: the deployment-v2 golden suite, robustness inputs and hardening cases, the annotation-v2
  test / train / validation / probe texts, human-value-01 and human-value-02, the production
  regressions, the playground presets, and a seeded set of adversarial strings (special-token text,
  separators, whitespace variants, NFD, combining orders, astral, lone surrogates, long inputs).
- ``parser.jsonl``: parser-only reference, ``gidi.value_parser.parse_value`` on every distinct text
  of the corpus, datasets, regressions and presets plus seeded parser-oriented fuzz strings.
- ``sweep.jsonl``: tokenizer + NFC-map reference for ``a<c> <c>b`` over every BMP code point, all of
  plane 1 and a stride through the remaining planes (no model; catches Unicode class differences).
- ``meta.json``: release hashes, library versions and the Unicode range tables the TS side checks.

``pnpm -C playground parity`` consumes these files. The default output directory is gitignored.

    uv run python scripts/dump_web_parity.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import unicodedata
from importlib import metadata
from pathlib import Path
from typing import Any

import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "src"))

from gen_runtime_unicode_tables import fold_pairs, python_unicode_ranges  # noqa: E402

from gidi.inference import GidiPredictor  # noqa: E402
from gidi.inference.bundle import CONFIG_FILE, MODEL_FILE, TOKENIZER_FILE  # noqa: E402
from gidi.inference.schema import EmptyInputError  # noqa: E402
from gidi.inference.text import normalize_nfc  # noqa: E402
from gidi.inference.tokenizer import BundleTokenizer  # noqa: E402
from gidi.value_parser import parse_value  # noqa: E402

BUNDLE = ROOT / "models" / "gidi-finance-v3"
DEPLOY_V2 = ROOT / "experiments" / "deployment-v2"
EVAL_DIR = ROOT / "datasets" / "annotation-v2" / "training-v1"
HV01_DIR = ROOT / "datasets" / "annotation-v2" / "human-value-01"
HV02_DIR = ROOT / "datasets" / "annotation-v3" / "human-value-02"
PRESETS_TS = ROOT / "playground" / "src" / "lib" / "presets.ts"
PRODUCTION_REGRESSIONS = ROOT / "tests" / "data" / "production-regressions.jsonl"
DEFAULT_OUT = ROOT / "playground" / ".parity"
STRESS_SEED = 20261003
STRESS_CASES = 900
PARSER_FUZZ_CASES = 6000


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def release_files() -> dict[str, Path]:
    """The three bundle files the runtime needs, verified against the bundle manifest."""
    listed = json.loads((BUNDLE / "manifest.json").read_text(encoding="utf-8"))["files"]
    paths: dict[str, Path] = {}
    for name in (CONFIG_FILE, TOKENIZER_FILE, MODEL_FILE):
        path = BUNDLE / name
        entry = listed[name]
        if entry["sha256"] != sha256_file(path) or entry["bytes"] != path.stat().st_size:
            raise SystemExit(f"{path} does not match the bundle manifest")
        paths[name] = path
    return paths


def playground_presets() -> list[str]:
    source = PRESETS_TS.read_text(encoding="utf-8")
    block = re.search(r"PRESETS = \[(.*?)\] as const", source, re.S)
    if block is None:
        raise SystemExit(f"cannot find PRESETS in {PRESETS_TS}")
    return json.loads("[" + block.group(1).rstrip().rstrip(",") + "]")


def stress_inputs() -> list[str]:
    """Seeded adversarial strings aimed at the places a port can differ from the Rust crates."""
    rng = random.Random(STRESS_SEED)
    words = [
        "cho", "mượn", "trả", "nợ", "anh", "chị", "Thảo", "Hùng", "Nguyễn", "phở", "cà phê",
        "1tr5", "500k", "2 triệu", "80.000đ", "1,2tr", "$20", "xăng", "tiền", "điện", "ck", "Lan",
    ]  # fmt: skip
    seps = ["(", ")", "|", ".", ",", "!", "?", "…", "。", "，", "、", "।", "۔", "،", ";", ":", "-"]
    specials = ["<s>", "</s>", "<pad>", "<unk>", "<mask>", "<S>", "< s>", "<s", "s>"]
    spaces = [
        " ", "  ", "\t", "\n", "\r\n", "\x0b", "\x0c", "\x1c", "\x1d", "\x1e", "\x1f", "\x85",
        "\xa0", "\u1680", "\u2000", "\u2003", "\u200a", "\u200b", "\u2028", "\u2029", "\u202f",
        "\u205f", "\u3000", "\ufeff", "\u180e", "\u200d",
    ]  # fmt: skip
    marks = ["\u0300", "\u0301", "\u0303", "\u0309", "\u031b", "\u0323", "\u0302", "\u0306"]
    odd = [
        "💸", "🇻🇳", "👨\u200d👩\u200d👧", "👍🏽", "𝕔", "\U00010348", "\U0010ffff",
        "\ud800", "\udfff",
        "ﬃ", "Å", "Å", "ǆ", "한", "\u1112\u1161\u11ab", "ｶﾞ", "①", "㎏", "\x00", "\x7f", "\u0e01",
        "ก\u0e33", "ö", "o\u0308", "ǟ", "a\u0308\u0304",
    ]  # fmt: skip
    pieces = words + seps + specials + spaces + marks + odd

    def vietnamese_variant(text: str) -> str:
        mode = rng.choice(["nfc", "nfd", "mixed", "reordered"])
        if mode == "nfd":
            return unicodedata.normalize("NFD", text)
        if mode == "mixed":
            return "".join(
                unicodedata.normalize("NFD", ch) if rng.random() < 0.5 else ch for ch in text
            )
        if mode == "reordered":
            # Dot below before the tone mark / horn: canonical reordering must fix it.
            decomposed = unicodedata.normalize("NFD", text)
            return re.sub("([\u0300\u0301\u0303\u0309])(\u0323)", r"\2\1", decomposed)
        return text

    out: list[str] = []
    for _ in range(STRESS_CASES):
        count = rng.randint(1, 14)
        parts = [rng.choice(pieces) for _ in range(count)]
        glue = rng.choice([" ", "", " ", "  "])
        text = vietnamese_variant(glue.join(parts))
        # Lone surrogates must not be adjacent: JS joins a high+low pair into one code point.
        text = re.sub("([\ud800-\udbff])(?=[\udc00-\udfff])", r"\1x", text)
        out.append(text)
    out += [
        "<s>",
        "</s> cho Lan vay 500k <s>",
        "a" * 3000,
        "ă" * 700,
        "cho Lan vay 500k " + "x " * 200,
        " ".join(["mượn anh Nam 2 triệu"] * 40),
        "\u0323\u0301 ăn sáng 35k",
        "e\u0323\u0302\u0301 cho Hùng vay 1tr",
        "Hàn Quốc 한국 trả 1tr",
        "😀" * 40,
        "ăn sáng 35k\ud83d",
        "\ud83dăn sáng 35k",
        "...,,,!!!???…。，、।۔،",
        "(((cho)))|||Lan|||vay(500k)",
    ]
    return out


def collect_inputs() -> list[tuple[str, str, str, dict[str, bool] | None]]:
    """``(source, id, text, gold)`` in a stable order. ``gold`` is ``{target_null, value_null}``."""
    rows: list[tuple[str, str, str, dict[str, bool] | None]] = []

    def gold_flags(target: Any, value: Any) -> dict[str, bool]:
        return {"target_null": target is None, "value_null": value is None}

    for r in read_jsonl(DEPLOY_V2 / "golden-suite.jsonl"):
        gold = gold_flags(r["gold"]["target"], r["gold"]["value"])
        rows.append(("golden", r["id"], r["text"], gold))
    for r in read_jsonl(DEPLOY_V2 / "robustness-inputs.jsonl"):
        rows.append(("robustness", r["id"], r["text"], None))
    for r in read_jsonl(DEPLOY_V2 / "hardening-cases.jsonl"):
        rows.append(("hardening", r["case"], r["text"], None))
    for split in ("test", "validation", "train"):
        for r in read_jsonl(EVAL_DIR / f"{split}.jsonl"):
            rows.append((split, r["id"], r["text"], gold_flags(r["target"], r.get("value"))))
    for r in read_jsonl(EVAL_DIR / "probe-v1-eval-only.jsonl"):
        rows.append(("probe", r["id"], r["text"], gold_flags(r["target"], r.get("value"))))
    labels01 = {r["id"]: r for r in read_jsonl(HV01_DIR / "labels.jsonl")}
    for r in read_jsonl(HV01_DIR / "review-queue.jsonl"):
        label = labels01[r["id"]]
        rows.append(
            ("human-value-01", r["id"], r["text"], gold_flags(label["target"], label["value"]))
        )
    labels02 = {r["id"]: r for r in read_jsonl(HV02_DIR / "labels.jsonl")}
    for r in read_jsonl(HV02_DIR / "review-queue-all.jsonl"):
        label = labels02[r["id"]]
        rows.append(
            ("human-value-02", r["id"], r["text"], gold_flags(label["target"], label["value"]))
        )
    rows += [("preset", f"preset-{i}", t, None) for i, t in enumerate(playground_presets())]
    for r in read_jsonl(PRODUCTION_REGRESSIONS):
        rows.append(("regression", r["id"], r["text"], None))
    rows += [("stress", f"stress-{i:04d}", t, None) for i, t in enumerate(stress_inputs())]
    return rows


_NUMBER = re.compile(r"\d[\d.,]*")
_BARE_NUMBER = re.compile(
    r"^\s*\$?\d[\d.,]*\s*(k|tr|đ|d|vnd|nghìn|ngàn|triệu|củ|xị)?\s*$", re.IGNORECASE
)
_SLANG_TOKEN = re.compile(
    r"^(\d+(k|tr\d*|m|cu|xị|củ)|ck|ib|tt|xị|củ|cu|tr|k|vnd|đ)$", re.IGNORECASE
)


def coverage_tags(
    text: str, prediction: dict[str, Any], n_tokens: int, gold: dict[str, bool] | None
) -> list[str]:
    """Simple text/output predicates for the coverage report (never fed to the model)."""
    nfd = unicodedata.normalize("NFD", text)
    accented = any(unicodedata.combining(c) for c in nfd) or "đ" in text.lower()
    has_letter = any(c.isalpha() for c in text)
    tags = {
        "accented": accented,
        "unaccented": has_letter and not accented and text.isascii(),
        "bare_number": bool(_BARE_NUMBER.match(text)),
        "slang": any(_SLANG_TOKEN.match(t) for t in re.split(r"[\s,;:]+", text) if t),
        "multi_number": len(_NUMBER.findall(text)) >= 2,
        "long_32_tokens": n_tokens >= 32,
        "truncated": bool(prediction["truncated"]),
        "punctuation": any(unicodedata.category(c).startswith("P") for c in text),
        "null_target": prediction["target"] is None,
        "null_value": prediction["value_text"] is None,
        "nfd_input": text != unicodedata.normalize("NFC", text),
        "astral": any(ord(c) > 0xFFFF for c in text),
        "gold_null_target": bool(gold and gold["target_null"]),
        "gold_null_value": bool(gold and gold["value_null"]),
    }
    return [name for name, hit in tags.items() if hit]


def json_line(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=True, allow_nan=False, separators=(",", ":"))


def graph_level_predictor(bundle: str | Path) -> GidiPredictor:
    """``GidiPredictor`` whose ORT session applies only the basic graph optimizations.

    ``OnnxRunner`` uses ORT's default level (``ORT_ENABLE_ALL``). On this graph that level fuses
    ``DynamicQuantizeLinear`` + ``MatMulInteger`` into ``com.microsoft.DynamicQuantizeMatMul`` /
    ``MatMulIntegerToFloat`` whose native kernel is platform specific (on arm64 ORT >= 1.22 it
    runs KleidiAI with different activation-quantization numerics). onnxruntime-web executes the
    ONNX graph as written, so the portable reference is the unfused graph: same session options
    as ``OnnxRunner`` but ``ORT_ENABLE_BASIC``.
    """
    predictor = GidiPredictor.from_bundle(bundle)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    predictor._runner._session = ort.InferenceSession(  # noqa: SLF001
        str(Path(bundle) / MODEL_FILE), options, providers=["CPUExecutionProvider"]
    )
    return predictor


def reference_record(
    graph: GidiPredictor,
    default: GidiPredictor,
    source: str,
    ident: str,
    text: str,
    gold: dict[str, bool] | None,
) -> dict[str, Any]:
    """``expected``/``trace``: the portable reference (the gate). ``default``: stock runtime."""
    record: dict[str, Any] = {"source": source, "id": ident, "text": text}
    try:
        prediction = graph.predict(text).to_dict()
        raw = graph.run(text)
    except EmptyInputError:
        record["error"] = "EmptyInputError"
        record["tags"] = ["empty_error"]
        return record
    normalized = normalize_nfc(text)
    record["expected"] = prediction
    record["tags"] = coverage_tags(text, prediction, len(raw.input_ids), gold)
    record["trace"] = {
        "nfc": raw.normalized_text,
        "nfc_start_map": None if normalized.start_map is None else list(normalized.start_map),
        "nfc_end_map": None if normalized.end_map is None else list(normalized.end_map),
        "ids": list(raw.input_ids),
        # OnnxRunner feeds ones (single sequence, no padding).
        "attention_mask": [1] * len(raw.input_ids),
        "offsets": [list(o) for o in raw.normalized_offsets],
        "original_offsets": [list(o) for o in raw.offsets],
        "special": list(raw.special_tokens_mask),
        "type_logits": [float(x) for x in raw.type_logits],
        "tag_logits": [[float(x) for x in row] for row in raw.tag_logits],
        "target_tags": [int(i) for i in raw.tag_logits.argmax(axis=-1)],
    }
    default_raw = default.run(text)
    record["default"] = {
        "expected": default.predict(text).to_dict(),
        "type_logits": [float(x) for x in default_raw.type_logits],
        "tag_logits": [[float(x) for x in row] for row in default_raw.tag_logits],
    }
    return record


def sweep_texts() -> list[str]:
    code_points = [c for c in range(0x10000) if not 0xD800 <= c <= 0xDFFF]
    code_points += range(0x10000, 0x20000)
    code_points += range(0x20000, 0x110000, 251)
    return [f"a{chr(c)} {chr(c)}b" for c in code_points]


def sweep_record(tokenizer: BundleTokenizer, text: str) -> dict[str, Any]:
    normalized = normalize_nfc(text)
    tokens = tokenizer.encode(normalized.text)
    record: dict[str, Any] = {
        "text": text,
        "nfc": normalized.text,
        "ids": list(tokens.ids),
        "offsets": [list(o) for o in tokens.offsets],
        "truncated": tokens.truncated,
    }
    if normalized.start_map is not None:
        record["start_map"] = list(normalized.start_map)
        record["end_map"] = list(normalized.end_map or ())
    return record


def parser_fuzz_inputs() -> list[str]:
    """Seeded strings built from the parser's own vocabulary: numbers, units, context words."""
    rng = random.Random(STRESS_SEED + 1)
    numbers = [
        "1", "5", "10", "70", "150", "612", "1500", "20000", "80.000", "1.250.000", "1,5",
        "12,500,000",
        "0912 345 678", "0912345678", "1 000 000", "20/10", "9/12/2024", "10:30", "5%", "2024",
        "2019",
        "6", "8", "3", "12", "250", "1tr5", "2tr", "500k", "15 K", "80.000đ", "250.000 VND",
        "350.000 đồng", "5 củ rưỡi", "1 triệu 2", "2 củ", "3 xị", "4 chai", "5 lít",
        "hai trăm nghìn",
        "bốn triệu", "nửa củ", "một triệu rưỡi", "ba chục", "t10", "q4", "iphone15", "10kg", "5h",
    ]  # fmt: skip
    words = [
        "tháng", "kỳ", "đợt", "lần", "quý", "năm", "Năm", "Lan", "Thu", "thứ", "tuần", "ngày",
        "mùng",
        "lớp", "tầng", "phòng", "chia", "số", "hàng", "tô", "vé", "người", "cái", "xăng", "sữa",
        "trà",
        "bia", "nước", "cafe", "đồng", "đóng", "dong", "ck", "mua", "trả", "cho", "vay", "mượn",
        "anh", "chị", "Nam", "Hùng", "đ", "d", "vnd", "₫", "k", "tr", "trieu", "ty", "tỷ", "nghìn",
        "ngàn", "rưỡi", "ruoi", "kg", "m2", "gb", "phút", "giờ", "tuổi", "TỔNG", "ĐỒNG", "NĂM",
    ]  # fmt: skip
    seps = [" ", " ", " ", "", ".", ",", "-", "/", ":", "%", "(", ")", " - ", "  "]
    out: list[str] = []
    for _ in range(PARSER_FUZZ_CASES):
        parts = [
            rng.choice(numbers if rng.random() < 0.45 else words) for _ in range(rng.randint(1, 7))
        ]
        text = ""
        for part in parts:
            text += part + rng.choice(seps)
        mode = rng.random()
        if mode < 0.15:
            text = unicodedata.normalize("NFD", text)
        elif mode < 0.25:
            text = text.upper()
        elif mode < 0.3:
            text = text.lower()
        out.append(text)
    return out


def corpus_texts(extra: list[str]) -> list[str]:
    """Every distinct note text of ``corpus/``, ``datasets/`` and ``tests/data``, then ``extra``."""
    seen: dict[str, None] = {}
    for base in (ROOT / "corpus", ROOT / "datasets", ROOT / "tests" / "data"):
        for path in sorted(base.rglob("*.jsonl")):
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.startswith("{"):
                    continue
                row = json.loads(line)
                if isinstance(row, dict) and isinstance(row.get("text"), str):
                    seen.setdefault(row["text"])
    for text in extra:
        seen.setdefault(text)
    return list(seen)


def parser_record(text: str) -> dict[str, Any]:
    value = parse_value(text)
    return {"text": text, "value": None if value is None else [value.start, value.end]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    files = release_files()
    inputs = collect_inputs()
    args.out.mkdir(parents=True, exist_ok=True)

    default = GidiPredictor.from_bundle(BUNDLE)
    graph = graph_level_predictor(BUNDLE)
    records = [reference_record(graph, default, *item) for item in inputs]
    sweep_tokenizer = BundleTokenizer(BUNDLE / TOKENIZER_FILE, 32)

    sweep_count = 0
    with (args.out / "sweep.jsonl").open("w", encoding="utf-8") as sweep:
        for text in sweep_texts():
            sweep.write(json_line(sweep_record(sweep_tokenizer, text)) + "\n")
            sweep_count += 1

    parser_texts = corpus_texts([text for _, _, text, _ in inputs] + parser_fuzz_inputs())
    (args.out / "parser.jsonl").write_text(
        "".join(json_line(parser_record(t)) + "\n" for t in parser_texts), encoding="utf-8"
    )

    (args.out / "expected.jsonl").write_text(
        "".join(json_line(r) + "\n" for r in records), encoding="utf-8"
    )
    meta = {
        "bundle": str(BUNDLE.relative_to(ROOT)),
        "files": {name: sha256_file(path) for name, path in files.items()},
        "python": sys.version.split()[0],
        "unicode": unicodedata.unidata_version,
        "versions": {
            name: metadata.version(name) for name in ("onnxruntime", "tokenizers", "numpy")
        },
        "inputs": len(records),
        "by_source": {
            s: sum(1 for r in records if r["source"] == s)
            for s in dict.fromkeys(r["source"] for r in records)
        },
        "sweep": sweep_count,
        "parser_texts": len(parser_texts),
        "unicode_ranges": python_unicode_ranges(),
        "fold_pairs": fold_pairs(),
    }
    (args.out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(
        f"wrote {len(records)} inputs + {sweep_count} sweep strings + "
        f"{len(parser_texts)} parser texts to {args.out}"
    )
    print(json.dumps(meta["by_source"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
