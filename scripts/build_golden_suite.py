"""Build the deployment golden suite and the runtime robustness inputs for gidi-finance-v1.

Golden suite (``experiments/deployment-v1/golden-suite.jsonl``)
    Deterministic selection from approved annotation-v1 records in ``splits/train.jsonl`` and
    ``splits/validation.jsonl`` plus a few ``annotation_status: skipped`` records from
    ``labels.jsonl`` (their text lives in ``queue.jsonl``). ``splits/test.jsonl`` and
    ``datasets/probe-v1/`` are never read. There is no randomness: candidates for a rule are
    ordered by (human-provenance first, sha1 of the id) and picked greedily, skipping
    near-duplicate templates (lowercased text with digits and the target masked). A rule is only
    topped up while the cases selected so far satisfy fewer than its ``quota`` records, so cases
    chosen for one rule count towards the others. The ``categories`` of a case are computed from
    every rule afterwards, not just the rule that picked it.

    The expected outputs of the suite are the PyTorch reference predictions of the deployed
    checkpoint (computed by ``scripts/verify_deployment.py``); the human label stored here is for
    information only.

Robustness inputs (``experiments/deployment-v1/robustness-inputs.jsonl``)
    Hand-written runtime edge cases, not drawn from any dataset.

Usage: ``uv run python scripts/build_golden_suite.py``

Another version (e.g. a value-head ``gidi-finance-v2``): ``--protocol <protocol.json>`` takes the
tokenizer from its ``source_checkpoint`` and writes next to it; ``--annotation-dir`` points at the
annotation dataset and ``--with-value`` keeps ``value`` / ``value_status`` in the gold and adds
value-span selection rules (``VALUE_RULES``). Without these flags the output is byte-identical to
the committed v1 suite.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPLIT_DIR = ROOT / "datasets" / "annotation-v1" / "splits"
LABELS = ROOT / "datasets" / "annotation-v1" / "labels.jsonl"
QUEUE = ROOT / "datasets" / "annotation-v1" / "queue.jsonl"
PROVENANCE = ROOT / "datasets" / "annotation-v1" / "provenance.jsonl"
OUT_DIR = ROOT / "experiments" / "deployment-v1"
SELECTION_SPLITS = ("train", "validation")  # NEVER "test"
SALT = "golden-v1"

TYPES = (
    "expense",
    "income",
    "borrow",
    "lend",
    "repayment_in",
    "repayment_out",
    "transfer",
    "refund",
)

# ---- category rules -------------------------------------------------------------------------

TITLES = r"(?:anh|chị|chi|cô|co|chú|chu|bác|bac|ông|ong|bà|ba|cậu|cau|dì|di|em|bạn|ban)"
FAMILY = r"(?:bố|mẹ|ba|má|me|bo|ma|bà|ông|ngoại|nội|noi|cô|dì|chú|bác|con)"
GIFT_WORDS = r"(?:lì xì|li xi|biếu|bieu|tặng|tang|quà|mừng tuổi|mung tuoi)"
SLANG = (
    r"(?:(?<!\w)(?:ck|tk|stk|ko|k|cty|vs|củ|lít|lit|dc|đc|mn|ae)(?!\w)"
    r"|\d+\s*cu(?!\w)|\d+tr\d)"
)
MERCHANTS = {
    "circle k",
    "winmart",
    "shopee",
    "highlands",
    "fahasa",
    "long châu",
    "long chau",
    "fpt",
    "grab food",
    "grabfood",
    "grab",
    "spotify",
    "netflix",
    "be bike",
    "tiki",
    "baemin",
    "lazada",
    "cgv",
    "momo",
    "spaylater",
}
CHANNEL = r"(?:momo|zalopay|chuyển khoản|chuyen khoan|(?<!\w)ck(?!\w)|thẻ|the tin dung|visa)"


def _tgt(rec: dict) -> str | None:
    return rec["gold"]["target"]["text"] if rec["gold"]["target"] else None


def _title_excluded(rec: dict) -> bool:
    """A title word directly precedes a target that does not include it ('chị Hạnh')."""
    t = rec["gold"]["target"]
    if not t:
        return False
    before = rec["text"][: t["start"]].rstrip()
    return bool(re.search(rf"(?<!\w){TITLES}$", before, re.I))


def _title_included(rec: dict) -> bool:
    """The target span itself starts with a title word ('chú Hai')."""
    tgt = _tgt(rec)
    return bool(tgt and re.match(rf"{TITLES}\s+\S", tgt, re.I))


@dataclass(frozen=True)
class Rule:
    name: str
    quota: int
    match: Callable[[dict], bool]
    doc: str


def _is_type(name: str) -> Callable[[dict], bool]:
    return lambda rec: rec["gold"]["type"] == name


def _rx(pattern: str) -> Callable[[dict], bool]:
    compiled = re.compile(pattern, re.I)
    return lambda rec: bool(compiled.search(rec["text"]))


RULES: tuple[Rule, ...] = (
    Rule("insurance", 2, _rx(r"bảo hiểm|bao hiem|bhyt|bhxh|bhnt"), "insurance keywords"),
    Rule(
        "lender_first_cho_muon",
        3,
        lambda r: (
            r["gold"]["type"] == "lend"
            and bool(re.match(r"cho .+ (mượn|muon|vay)", r["text"], re.I))
        ),
        "type lend, text 'cho X mượn/vay ...' (lender-first phrasing)",
    ),
    Rule(
        "borrow_with_cho_vay_phrase",
        2,
        lambda r: (
            r["gold"]["type"] == "borrow"
            and bool(re.search(r"cho (mượn|muon|vay)", r["text"], re.I))
        ),
        "type borrow although the text contains 'cho mượn/vay' (counterparty lends to the user)",
    ),
    Rule(
        "borrow_direction",
        3,
        lambda r: (
            r["gold"]["type"] == "borrow" and bool(re.match(r"(mượn|muon|vay)\b", r["text"], re.I))
        ),
        "type borrow, text starts with the verb mượn/vay",
    ),
    Rule(
        "repayment_in",
        5,
        lambda r: r["gold"]["type"] == "repayment_in",
        "type repayment_in",
    ),
    Rule(
        "repayment_out",
        5,
        lambda r: r["gold"]["type"] == "repayment_out",
        "type repayment_out",
    ),
    Rule(
        "family_gift",
        4,
        lambda r: bool(
            re.search(GIFT_WORDS, r["text"], re.I)
            and re.search(rf"(?<!\w){FAMILY}(?!\w)", r["text"], re.I)
        ),
        "gift word (lì xì/biếu/tặng/quà) and a family word",
    ),
    Rule(
        "refund_cashback",
        4,
        lambda r: (
            r["gold"]["type"] == "refund"
            and bool(re.search(r"hoàn|hoan|cashback", r["text"], re.I))
        ),
        "type refund with hoàn tiền / cashback wording",
    ),
    Rule(
        "title_excluded_from_target", 4, _title_excluded, "title word precedes a bare-name target"
    ),
    Rule("title_included_in_target", 4, _title_included, "target span starts with the title word"),
    Rule(
        "merchant_target",
        5,
        lambda r: (_tgt(r) or "").lower() in MERCHANTS,
        "target text is a known merchant / brand",
    ),
    Rule(
        "payment_channel_null_target",
        4,
        lambda r: r["gold"]["target"] is None and bool(re.search(CHANNEL, r["text"], re.I)),
        "no target although a payment channel (momo/ck/chuyển khoản/thẻ) is mentioned",
    ),
    Rule(
        "slang_shorthand",
        8,
        _rx(SLANG),
        "shorthand tokens: ck tk stk ko cty vs củ dc mn, 3 cu, 1tr5",
    ),
    Rule("amount_separator", 3, _rx(r"\d[.,]\d"), "amount written with '.' or ',' separators"),
    Rule(
        "multiword_target",
        3,
        lambda r: bool(_tgt(r) and " " in _tgt(r)),
        "target spans more than one word",
    ),
    Rule("long_text", 2, lambda r: len(r["text"]) >= 45, "text of at least 45 characters"),
    *(Rule(f"type_{t}", 9, _is_type(t), f"type {t}") for t in TYPES),
    Rule("accented", 12, lambda r: r["accented"], "text carries Vietnamese diacritics"),
    Rule("unaccented", 12, lambda r: not r["accented"], "text has no diacritics"),
    Rule("target_present", 12, lambda r: r["gold"]["target"] is not None, "target span present"),
    Rule("target_null", 12, lambda r: r["gold"]["target"] is None, "no target span"),
)


def _value(rec: dict) -> dict | None:
    return rec["gold"].get("value")


def _value_rx(pattern: str) -> Callable[[dict], bool]:
    compiled = re.compile(pattern, re.I)
    return lambda rec: _value(rec) is not None and bool(compiled.search(_value(rec)["text"]))


# Extra rules for annotation-v2 records (``build_golden(with_value=True)``): value-span cases.
VALUE_RULES: tuple[Rule, ...] = (
    Rule(
        "value_present",
        12,
        lambda r: r["gold"].get("value_status") == "complete" and _value(r) is not None,
        "a complete value span",
    ),
    Rule(
        "value_no_amount",
        4,
        lambda r: r["gold"].get("value_status") == "complete" and _value(r) is None,
        "complete, no amount in the note (no_amount)",
    ),
    Rule(
        "value_uncertain",
        3,
        lambda r: r["gold"].get("value_status") == "uncertain",
        "value status uncertain (loss masked in training)",
    ),
    Rule(
        "value_not_first_number",
        6,
        lambda r: _value(r) is not None and bool(re.search(r"\d", r["text"][: _value(r)["start"]])),
        "another number precedes the value span (quantity, month, date)",
    ),
    Rule(
        "value_unit_or_slang",
        8,
        _value_rx(
            r"\d\s*(?:k|tr|củ|cu|xị|xi|chai|lít|lit|nghìn|nghin|ngàn|ngan|triệu|trieu|tỷ|ty)"
            r"(?!\w)|\dtr\d"
        ),
        "value carries a unit or slang (50k, 1tr5, 5 xị, 2 củ)",
    ),
    Rule("value_separator", 3, _value_rx(r"\d[.,]\d"), "value written with '.' or ',' separators"),
    Rule(
        "value_currency_marker",
        3,
        _value_rx(r"(?:đ|vnd|vnđ|đồng)$"),
        "value includes a trailing currency marker",
    ),
    Rule("value_bare_number", 4, _value_rx(r"^\d+$"), "value is a bare number (100)"),
)

SKIPPED_QUOTA = 5


def _strip_accents(text: str) -> str:
    folded = unicodedata.normalize("NFD", text.replace("đ", "d").replace("Đ", "D"))
    return "".join(c for c in folded if not unicodedata.combining(c))


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _order_key(rec: dict) -> tuple[int, str]:
    digest = hashlib.sha1(f"{SALT}:{rec['id']}".encode()).hexdigest()
    return (0 if rec["annotator"] == "human" else 1, digest)


def _template(rec: dict) -> str:
    text = rec["text"].lower()
    tgt = rec["gold"]["target"]
    if tgt:
        text = text.replace(tgt["text"].lower(), "§")
    return re.sub(r"\d+", "#", text)


def load_pool(split_dir: Path = SPLIT_DIR, with_value: bool = False) -> list[dict]:
    pool = []
    for split in SELECTION_SPLITS:
        for raw in _read_jsonl(split_dir / f"{split}.jsonl"):
            if unicodedata.normalize("NFC", raw["text"]) != raw["text"]:
                raise ValueError(f"non-NFC text in {raw['id']}")
            gold = {"type": raw["type"], "target": raw["target"]}
            if with_value:
                complete = raw.get("value_status") == "complete"
                gold["value"] = raw.get("value") if complete else None
                gold["value_status"] = raw.get("value_status")
            pool.append(
                {
                    "id": raw["id"],
                    "text": raw["text"],
                    "gold": gold,
                    "accented": bool(raw["accented"]),
                    "split": split,
                    "annotator": raw["provenance"]["annotator"],
                    "skipped": False,
                }
            )
    return pool


def load_skipped(
    labels: Path = LABELS,
    queue: Path = QUEUE,
    provenance_file: Path = PROVENANCE,
    with_value: bool = False,
) -> list[dict]:
    provenance = {r["id"]: r for r in _read_jsonl(provenance_file)}
    texts = {r["id"]: r["text"] for r in _read_jsonl(queue)}
    out = []
    for label in _read_jsonl(labels):
        if label["annotation_status"] != "skipped":
            continue
        text = texts[label["id"]]
        if unicodedata.normalize("NFC", text) != text:
            raise ValueError(f"non-NFC text in {label['id']}")
        gold = {"type": None, "target": None}
        if with_value:
            gold.update(value=None, value_status=None)
        out.append(
            {
                "id": label["id"],
                "text": text,
                "gold": gold,
                "accented": text != _strip_accents(text),
                "split": "labels.jsonl:skipped",
                "annotator": provenance[label["id"]]["annotator"],
                "skipped": True,
            }
        )
    return out


def select(
    pool: list[dict], skipped: list[dict], rules: tuple[Rule, ...] = RULES
) -> tuple[list[dict], list[dict]]:
    chosen: list[dict] = []
    seen_ids: set[str] = set()
    seen_templates: set[str] = set()
    shortfalls: list[dict] = []

    def add(rec: dict) -> None:
        chosen.append(rec)
        seen_ids.add(rec["id"])
        seen_templates.add(_template(rec))

    for rule in rules:
        have = sum(1 for c in chosen if rule.match(c))
        candidates = sorted((r for r in pool if rule.match(r)), key=_order_key)
        for rec in candidates:
            if have >= rule.quota:
                break
            if rec["id"] in seen_ids or _template(rec) in seen_templates:
                continue
            add(rec)
            have += 1
        if have < rule.quota:
            shortfalls.append({"rule": rule.name, "quota": rule.quota, "have": have})

    # Skipped records (pipeline-level out-of-taxonomy case): 3 accented + 2 unaccented, human first.
    skipped_sorted = sorted(skipped, key=_order_key)
    for want_accented, quota in ((True, (SKIPPED_QUOTA + 1) // 2), (False, SKIPPED_QUOTA // 2)):
        n = 0
        for rec in (s for s in skipped_sorted if s["accented"] is want_accented):
            if n >= quota:
                break
            if _template(rec) in seen_templates:
                continue
            add(rec)
            n += 1
    return chosen, shortfalls


def categories_for(rec: dict, rules: tuple[Rule, ...] = RULES) -> list[str]:
    if rec["skipped"]:
        cats = ["skipped", "target_null", "accented" if rec["accented"] else "unaccented"]
        cats += [r.name for r in rules if r.name in ("slang_shorthand",) and r.match(rec)]
        return sorted(set(cats))
    return [r.name for r in rules if r.match(rec)]


def build_golden(
    annotation_dir: Path | None = None, with_value: bool = False
) -> tuple[list[dict], dict]:
    """Select the golden cases from ``annotation_dir`` (default: annotation-v1).

    ``with_value`` is for annotation-v2 records: the gold keeps ``value`` / ``value_status`` and
    ``VALUE_RULES`` join the selection and the categories.

    ``annotation_dir`` holds either ``splits/{train,validation}.jsonl`` plus ``labels.jsonl`` /
    ``queue.jsonl`` / ``provenance.jsonl`` (annotation-v1 layout, with skipped records), or
    ``{train,validation}.jsonl`` directly (a ``training-v1`` directory: approved records only,
    no skipped ones).
    """
    rules = (*RULES, *VALUE_RULES) if with_value else RULES
    if annotation_dir is None:
        pool = load_pool(with_value=with_value)
        skipped = load_skipped(with_value=with_value)
    else:
        splits = annotation_dir / "splits"
        pool = load_pool(splits if splits.is_dir() else annotation_dir, with_value)
        has_skipped = (annotation_dir / "labels.jsonl").is_file()
        skipped = (
            load_skipped(
                annotation_dir / "labels.jsonl",
                annotation_dir / "queue.jsonl",
                annotation_dir / "provenance.jsonl",
                with_value,
            )
            if has_skipped
            else []
        )
    chosen, shortfalls = select(pool, skipped, rules)
    chosen.sort(
        key=lambda r: (
            r["skipped"],
            TYPES.index(r["gold"]["type"]) if r["gold"]["type"] else 99,
            r["id"],
        )
    )
    cases = []
    for i, rec in enumerate(chosen, start=1):
        cases.append(
            {
                "id": f"golden-{i:03d}",
                "text": rec["text"],
                "gold": rec["gold"],
                "categories": categories_for(rec, rules),
                "source": {
                    "record_id": rec["id"],
                    "split": rec["split"],
                    "annotator": rec["annotator"],
                },
            }
        )
    counts = Counter(c for case in cases for c in case["categories"])
    return cases, {
        "n_cases": len(cases),
        "category_counts": dict(sorted(counts.items())),
        "shortfalls": shortfalls,
        "annotators": dict(Counter(c["source"]["annotator"] for c in cases)),
    }


# ---- robustness inputs ---------------------------------------------------------------------


def robustness_inputs() -> list[dict]:
    nfc = "Chị Thảo trả lại 1tr tiền ăn phở"
    nfd = unicodedata.normalize("NFD", nfc)
    nfc2 = "trả nợ chú Hùng 1tr5 tiền sửa nhà"
    nfd2 = unicodedata.normalize("NFD", nfc2)
    items: list[tuple[str, str, str, bool, str | None]] = [
        # (id, text, group, expect_empty_error, nfc_pair)
        ("empty", "", "empty", True, None),
        ("space-only", "   ", "empty", True, None),
        ("tab-newline-only", "\t \n\r\n ", "empty", True, None),
        ("nbsp-only", "\u00a0\u00a0", "empty", True, None),
        ("ideographic-space-only", "\u3000", "empty", True, None),
        ("single-char-k", "k", "short", False, None),
        ("single-word-an", "ăn", "short", False, None),
        ("single-digit", "5", "short", False, None),
        ("two-words", "ăn sáng", "short", False, None),
        ("leading-trailing-space", "  mượn anh Nam 2 triệu  ", "whitespace", False, None),
        ("double-inner-space", "cho  Lan   vay 500k", "whitespace", False, None),
        ("tabs-newlines", "ăn trưa\t\tvới Hùng\n150k\r\nchia đôi", "whitespace", False, None),
        ("newline-between", "mượn anh Nam\n2 triệu", "whitespace", False, None),
        ("control-chars", "ăn sáng\x00\x07 35k\x1b[0m", "control", False, None),
        ("zero-width", "chị\u200b Thảo\u200d trả lại\ufeff 1tr", "control", False, None),
        ("rtl-arabic", "مرحبا cho Lan vay 500k", "script", False, None),
        ("rtl-mark", "\u202eăn sáng 35k\u202c", "script", False, None),
        ("cjk", "午餐 mượn anh Nam 2 triệu 谢谢", "script", False, None),
        ("thai-korean", "ขอบคุณ 점심 trả nợ chị Mai 1tr", "script", False, None),
        (
            "punct-heavy",
            "!!! ăn sáng ... 35k ??? (phở) [bò] {tái} ~~~ ###",
            "punctuation",
            False,
            None,
        ),
        ("punct-only", "!?.,;:-_=+*/\\|()[]{}<>@#$%^&~`'\"", "punctuation", False, None),
        ("punct-quotes", "“cho” ‘Lan’ «vay» 500k…", "punctuation", False, None),
        ("emoji-basic", "ăn sáng 35k 🍜", "emoji", False, None),
        ("emoji-astral", "𝕔ho Lan vay 500k 💸🎉", "emoji", False, None),
        ("emoji-zwj-family", "lì xì 👨‍👩‍👧‍👦 mẹ 500k", "emoji", False, None),
        ("emoji-zwj-flag-skin", "cà phê ☕ 🇻🇳 👍🏽 vs Hùng 45k", "emoji", False, None),
        ("emoji-only", "💰🍜🍜", "emoji", False, None),
        ("mixed-en-vi", "grab food lunch with Linh 120k, paid by momo", "mixed", False, None),
        ("mixed-en-vi-2", "Chị Thảo pay back 1tr for dinner", "mixed", False, None),
        ("mixed-en-only", "refund from Amazon order 25 dollars", "mixed", False, None),
        ("amount-1tr5", "1tr5", "numbers", False, None),
        ("amount-200k", "200k", "numbers", False, None),
        ("amount-dotted-dong", "1.500.000đ", "numbers", False, None),
        ("amount-dollar", "$20", "numbers", False, None),
        ("amount-dollar-note", "mua sách $20 trên amazon", "numbers", False, None),
        ("amount-comma", "ck anh Nam 1,2tr", "numbers", False, None),
        ("amount-words", "mượn chị Lan hai triệu rưỡi", "numbers", False, None),
        ("amount-big", "lương tháng này về 1.234.567.890 đồng", "numbers", False, None),
        ("digits-only-long", "0123456789012345678901234567890123456789", "numbers", False, None),
        ("unknown-merchant", "xyzzy corp quux 77k", "unknown", False, None),
        ("unknown-name", "mượn Đặng Nguyễn Thiên Phúc Bảo 2 triệu", "unknown", False, None),
        ("unknown-name-foreign", "cho Kwame Nkrumah vay 500k", "unknown", False, None),
        ("gibberish", "asdfghjkl qwertyuiop zxcvbnm", "unknown", False, None),
        ("all-caps", "CHO LAN VAY 500K", "case", False, None),
        ("all-lower-unaccented", "cho lan vay 500k", "case", False, None),
        ("accent-stripped", "tra no chi Hoa 1tr", "case", False, None),
        ("nfc-form-1", nfc, "nfc-nfd", False, "nfc-nfd-1"),
        ("nfd-form-1", nfd, "nfc-nfd", False, "nfc-nfd-1"),
        ("nfc-form-2", nfc2, "nfc-nfd", False, "nfc-nfd-2"),
        ("nfd-form-2", nfd2, "nfc-nfd", False, "nfc-nfd-2"),
        (
            "nfd-mixed-target",
            "mượn " + unicodedata.normalize("NFD", "Hạnh") + " 2tr",
            "nfc-nfd",
            False,
            None,
        ),
        (
            "nfd-leading",
            unicodedata.normalize("NFD", "Ăn sáng với Hùng 40k"),
            "nfc-nfd",
            False,
            None,
        ),
        ("combining-orphan", "\u0301 ăn sáng 35k", "nfc-nfd", False, None),
        ("ascii-d-name", "tra no chu Dung 1tr", "case", False, None),
        ("fullwidth-latin", "ｃｈｏ Ｌａｎ ｖａｙ ５００ｋ", "script", False, None),
    ]
    out = [
        {
            "id": f"robust-{name}",
            "text": text,
            "group": group,
            "expect_empty_error": expect_empty,
            "nfc_nfd_pair": pair,
        }
        for name, text, group, expect_empty, pair in items
    ]

    # Length boundary notes: built to land on exactly 32 / 33 / many tokens. The token counts are
    # asserted by scripts/verify_deployment.py with the bundle tokenizer, not here.
    out.append(
        {
            "id": "robust-long-300-words",
            "text": " ".join(f"w{i}" for i in range(300)),
            "group": "length",
            "expect_empty_error": False,
            "nfc_nfd_pair": None,
            "length_probe": "very_long",
        }
    )
    out.append(
        {
            "id": "robust-long-300-words-vi",
            "text": " ".join(["chuyển khoản trả nợ anh Nam tiền ăn trưa 1tr5"] * 30),
            "group": "length",
            "expect_empty_error": False,
            "nfc_nfd_pair": None,
            "length_probe": "very_long",
        }
    )
    out.append(
        {
            "id": "robust-long-single-token-run",
            "text": "a" * 5000,
            "group": "length",
            "expect_empty_error": False,
            "nfc_nfd_pair": None,
            "length_probe": "very_long",
        }
    )
    out.append(
        {
            "id": "robust-long-target-after-cut",
            "text": " ".join(["ăn"] * 40) + " trả nợ anh Nam 1tr",
            "group": "length",
            "expect_empty_error": False,
            "nfc_nfd_pair": None,
            "length_probe": "very_long",
        }
    )
    out.append(
        {
            "id": "robust-trailing-whitespace-long",
            "text": "ăn sáng 35k" + " " * 2000,
            "group": "length",
            "expect_empty_error": False,
            "nfc_nfd_pair": None,
        }
    )
    return out


def length_boundary_inputs(token_count: Callable[[str], int]) -> list[dict]:
    """Notes of exactly 32 and 33 full tokens (including <s> and </s>), built with a tokenizer.

    ``token_count`` returns the full (untruncated) token count of a text. Words are appended to a
    base note until the count hits the target.
    """
    base = "mượn anh Nam 2 triệu"
    out = []
    for label, target in (("exactly-32-tokens", 32), ("just-over-33-tokens", 33)):
        text = base
        guard = 0
        while token_count(text) < target:
            text += " a"
            guard += 1
            if guard > 100:
                raise RuntimeError("could not reach the token target")
        if token_count(text) != target:
            raise RuntimeError(f"{label}: got {token_count(text)} tokens, wanted {target}")
        out.append(
            {
                "id": f"robust-{label}",
                "text": text,
                "group": "length",
                "expect_empty_error": False,
                "nfc_nfd_pair": None,
                "length_probe": label,
            }
        )
    return out


def _bundle_or_checkpoint_tokenizer_count(checkpoint: Path | None = None) -> Callable[[str], int]:
    """Token counter from a checkpoint's tokenizer.json (needs only ``tokenizers``).

    Default: the gidi-finance-v1 (compression-v3 seed-1) checkpoint.
    """
    from tokenizers import Tokenizer

    path = (
        ROOT / "models/compression-v3/supervised/"
        "student-4x768-pretrained-pos32-vocab-B-rank-8000-ffn-combined-K2048/seed1/tokenizer.json"
        if checkpoint is None
        else checkpoint / "tokenizer.json"
    )
    tok = Tokenizer.from_file(str(path))
    tok.no_truncation()
    tok.no_padding()
    return lambda text: len(tok.encode(unicodedata.normalize("NFC", text)).ids)


# ---- hardening cases (value-head runtime) ----------------------------------------------------


def _case(
    name: str,
    text: str,
    group: str,
    *,
    empty: bool = False,
    value_text: str | None = None,
    cut: str | None = None,
) -> dict:
    out = {
        "case": name,
        "group": group,
        "text": text,
        "expect_empty_error": empty,
        "value_text": value_text,
    }
    if cut:
        out["cut"] = cut  # where `value_text` sits relative to the 32-token cut
    return out


def hardening_cases(token_count: Callable[[str], int]) -> list[dict]:
    """Hand-written runtime hardening notes for a bundle with a value head (no dataset text).

    ``token_count`` is the full (untruncated) token count of a text; the truncation cases are
    built with it so the value lies before, across, or after the 32-token cut by construction.
    """
    nfd = lambda s: unicodedata.normalize("NFD", s)  # noqa: E731
    base = "cho Nam vay 2 triệu"
    filler = "hôm nay trời mưa nên ở nhà nấu cơm cho cả nhà ăn"
    cases = [
        # original-string offsets: the spans must index the caller's string, not a normalized copy
        _case("offsets-plain", "cho anh Nam vay 2 triệu", "offsets", value_text="2 triệu"),
        _case("offsets-leading-ws", "   cho anh Nam vay 2 triệu", "offsets", value_text="2 triệu"),
        _case("offsets-trailing-ws", "cho anh Nam vay 2 triệu   ", "offsets", value_text="2 triệu"),
        _case(
            "offsets-nfd-all",
            nfd("Trả nợ chị Thảo 1 triệu rưỡi"),
            "offsets",
            value_text=nfd("1 triệu rưỡi"),
        ),
        _case(
            "offsets-nfd-target-only",
            nfd("Mượn chị Hạnh") + " 3tr",
            "offsets",
            value_text="3tr",
        ),
        _case(
            "offsets-nfd-value-only",
            "cho Nam vay " + nfd("5 triệu") + " tháng",
            "offsets",
            value_text=nfd("5 triệu"),
        ),
        _case("offsets-emoji", "💸 trả Nam 200k 🙏", "offsets", value_text="200k"),
        _case("offsets-astral-before", "𝒜𝒷 mượn chị Lan 2tr5", "offsets", value_text="2tr5"),
        # accented / unaccented
        _case(
            "accented", "mượn chị Hạnh 3 triệu 500 nghìn", "script", value_text="3 triệu 500 nghìn"
        ),
        _case(
            "unaccented",
            "muon chi Hanh 3 trieu 500 nghin",
            "script",
            value_text="3 trieu 500 nghin",
        ),
        _case("accented-upper", "MƯỢN CHỊ HẠNH 3 TRIỆU", "script", value_text="3 TRIỆU"),
        _case("unaccented-upper", "TRA NO ANH NAM 2TR", "script", value_text="2TR"),
        _case("mixed-accent", "tra nợ chị Thảo 1tr5", "script", value_text="1tr5"),
        # Unicode punctuation and width variants
        _case(
            "quotes-curly", "cho \u201cNam\u201d vay 1 triệu\u2026", "unicode", value_text="1 triệu"
        ),
        _case(
            "apostrophe-curly", "cho \u2018Nam\u2019 vay 1 triệu", "unicode", value_text="1 triệu"
        ),
        _case("en-dash", "chuyển khoản \u2013 Lan \u2013 500k", "unicode", value_text="500k"),
        _case("em-dash", "chuyển khoản \u2014 Lan \u2014 500k", "unicode", value_text="500k"),
        _case("ellipsis", "ăn phở\u2026 70k\u2026", "unicode", value_text="70k"),
        _case(
            "fullwidth-punct", "Nam vay 5 triệu\uff01\uff0c hẹn tháng sau\uff08trả\uff09", "unicode"
        ),
        _case("fullwidth-letters", "\uff21\uff22\uff23 vay 50k", "unicode", value_text="50k"),
        _case("fullwidth-digits", "ăn phở \uff15\uff10k", "unicode"),
        _case(
            "ideographic-space",
            "cho\u3000Nam\u3000vay\u30002 triệu",
            "unicode",
            value_text="2 triệu",
        ),
        _case("nbsp", "cho\u00a0Nam\u00a0vay\u00a02\u00a0triệu", "unicode"),
        _case("zero-width-space", "cho Nam\u200b vay 2\u200b triệu", "unicode"),
        _case("bom-prefix", "\ufeffcho Nam vay 2 triệu", "unicode", value_text="2 triệu"),
        _case("cjk", "借 Nam 2 triệu 谢谢", "unicode"),
        _case("rtl-and-control", "cho Nam vay 2tr \u202e\u0007", "unicode", value_text="2tr"),
        _case("combining-only", "\u0301\u0323\u0300", "unicode"),
        # lone surrogates cannot be stored in a UTF-8 JSONL file: verify_deployment adds them
        # empty and whitespace-only: EmptyInputError, never a crash or a prediction
        _case("empty", "", "empty", empty=True),
        _case("space-only", "   ", "empty", empty=True),
        _case("tab-newline-only", "\t\n\r\n\t", "empty", empty=True),
        _case("ideographic-space-only", "\u3000\u3000", "empty", empty=True),
        _case("nbsp-only", "\u00a0\u2003", "empty", empty=True),
        # extra whitespace
        _case("ws-leading", "    mượn Nam 2 triệu", "whitespace", value_text="2 triệu"),
        _case("ws-trailing", "mượn Nam 2 triệu    ", "whitespace", value_text="2 triệu"),
        _case(
            "ws-internal", "mượn    Nam     2      triệu", "whitespace", value_text="2      triệu"
        ),
        _case("ws-tabs", "mượn\tNam\t2\ttriệu", "whitespace", value_text="2\ttriệu"),
        _case("ws-newlines", "mượn\nNam\n2 triệu\n", "whitespace", value_text="2 triệu"),
        _case("ws-crlf", "mượn Nam\r\n2 triệu\r\n", "whitespace", value_text="2 triệu"),
        _case("ws-mixed", " \t\n mượn \t Nam \n\n 2 triệu \t ", "whitespace", value_text="2 triệu"),
        # punctuation-heavy
        _case("punct-only", "!!!???...,,,;;;---", "punctuation"),
        _case(
            "punct-brackets",
            "((( Nam ))) --- 1.000.000đ !!!",
            "punctuation",
            value_text="1.000.000đ",
        ),
        _case("punct-slashes", "///\\\\\\ 20/10 *** 2tr ###", "punctuation", value_text="2tr"),
        _case("punct-quotes", "\"'Nam'\" vay \"'2tr'\"", "punctuation"),
        _case("punct-single", ".", "punctuation"),
        _case("digits-only", "1234567890", "punctuation"),
        _case("single-letter", "a", "punctuation"),
        # long input (far beyond the cut)
        _case("long-words", " ".join(["ăn trưa 50k với đồng nghiệp"] * 60), "long"),
        _case("long-chars", "ăn trưa 50k " * 2000, "long"),
        _case("long-one-token-run", "x" * 5000 + " vay 2tr", "long"),
        _case("long-spaces", "cho Nam vay 2 triệu" + " " * 3000 + "hẹn trả", "long"),
        # task categories
        _case("no-target", "cà phê sáng 35k", "task", value_text="35k"),
        _case("no-value", "trả nợ anh Nam", "task"),
        _case("no-target-no-value", "hôm nay trời đẹp", "task"),
        _case("bare-number-amount", "ăn phở 70", "task", value_text="70"),
        _case("bare-number-lend", "cho Nam mượn 500", "task", value_text="500"),
        _case("slang-cu", "Thắng vay 5 củ", "task", value_text="5 củ"),
        _case("slang-chai", "mua giày 2 chai", "task", value_text="2 chai"),
        _case("slang-xi", "nợ anh Tú 3 xị", "task", value_text="3 xị"),
        _case("slang-k", "ship đồ ăn 50k", "task", value_text="50k"),
        _case("slang-tr", "trả góp 1tr5", "task", value_text="1tr5"),
        _case("multi-number-qty", "ăn 2 tô phở 70", "task", value_text="70"),
        _case("multi-number-unit", "mua 3 vé 150k", "task", value_text="150k"),
        _case("multi-number-date", "20/10 cho Nam vay 2 triệu", "task", value_text="2 triệu"),
        _case("multi-number-installment", "trả góp kỳ 3 1tr5", "task", value_text="1tr5"),
        _case("multi-number-rate", "cho vay 5 triệu 3 tháng lãi 2%", "task", value_text="5 triệu"),
        _case("multi-number-two-amounts", "ăn sáng 35k cà phê 25k", "task"),
        _case(
            "multi-token-amount",
            "mượn Nam 1 triệu 500 nghìn",
            "task",
            value_text="1 triệu 500 nghìn",
        ),
        _case(
            "multi-token-words", "trả nợ Nam một triệu rưỡi", "task", value_text="một triệu rưỡi"
        ),
        _case("multi-token-long-words", "Nam vay một trăm năm mươi nghìn", "task"),
        _case("separators", "lương tháng này 15.500.000đ", "task", value_text="15.500.000đ"),
        _case("comma-separator", "nhận lương 15,5 triệu", "task", value_text="15,5 triệu"),
        _case("currency-marker", "mua sách 120.000 đồng", "task", value_text="120.000 đồng"),
        _case("target-and-value", "trả Nam 2 triệu", "task", value_text="2 triệu"),
        _case("target-null-value-present", "cafe 35k", "task", value_text="35k"),
        _case("value-null-target-present", "chuyển khoản cho Lan", "task"),
        _case("value-null", "mượn tiền", "task"),
        _case(
            "tracked-style-date-first",
            "Hùng vay 3 triệu, hẹn t11 trả",
            "task",
            value_text="3 triệu",
        ),
    ]

    # max-length truncation: the value before, at, across and after the 32-token cut. `pad(n)`
    # is n single-token words, so the number of real tokens in front of the value is exact
    # (the cut keeps 30 real tokens plus <s> and </s>).
    def pad(n: int) -> str:
        text = " ".join(["a"] * n)
        if token_count(text) != n + 2:
            raise RuntimeError(f"pad({n}) is {token_count(text)} tokens, not {n + 2}")
        return text

    value_tokens = token_count("2 triệu") - 2
    cases += [
        _case(
            "trunc-value-before",
            f"{base} {pad(60)}",
            "truncation",
            value_text="2 triệu",
            cut="before",
        ),
        _case(
            "trunc-value-before-filler",
            f"{base} {' '.join([filler] * 4)}",
            "truncation",
            value_text="2 triệu",
            cut="before",
        ),
        _case(
            "trunc-value-ends-at-cut",
            f"{pad(30 - value_tokens)} 2 triệu hẹn trả",
            "truncation",
            value_text="2 triệu",
            cut="at-end",
        ),
        _case(
            "trunc-value-straddles",
            f"{pad(29)} 1 triệu 500 nghìn",
            "truncation",
            value_text="1 triệu 500 nghìn",
            cut="across",
        ),
        _case(
            "trunc-target-at-cut",
            f"{pad(28)} cho Nam vay 1 triệu 500 nghìn",
            "truncation",
            value_text="1 triệu 500 nghìn",
            cut="after",
        ),
        _case(
            "trunc-value-after",
            f"{pad(34)} {base}",
            "truncation",
            value_text="2 triệu",
            cut="after",
        ),
        _case(
            "trunc-value-after-far",
            f"{pad(120)} {base}",
            "truncation",
            value_text="2 triệu",
            cut="after",
        ),
        _case(
            "trunc-value-before-and-after",
            f"ăn sáng 35k {pad(40)} cà phê 25k",
            "truncation",
            value_text="35k",
            cut="before",
        ),
        _case(
            "trunc-only-value-after",
            f"cho Nam vay {pad(60)} 2 triệu",
            "truncation",
            value_text="2 triệu",
            cut="after",
        ),
        _case(
            "trunc-unaccented",
            f"{pad(31)} muon anh Nam 2 trieu",
            "truncation",
            value_text="2 trieu",
            cut="after",
        ),
        _case(
            "trunc-nfd",
            nfd(f"mượn chị Hạnh 3 triệu {pad(40)}"),
            "truncation",
            value_text=nfd("3 triệu"),
            cut="before",
        ),
    ]
    return cases


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--protocol",
        type=Path,
        default=None,
        help="deployment protocol JSON of another version: its source_checkpoint supplies the "
        "tokenizer and, unless --out-dir is given, its directory receives the files "
        "(default: the gidi-finance-v1 setup)",
    )
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument(
        "--annotation-dir",
        type=Path,
        default=None,
        help="annotation dataset with splits/, labels.jsonl, queue.jsonl and provenance.jsonl "
        "(default: datasets/annotation-v1)",
    )
    parser.add_argument(
        "--with-value",
        action="store_true",
        help="records carry value / value_status (annotation-v2): keep them in the gold and add "
        "the value-span selection rules",
    )
    args = parser.parse_args()

    checkpoint = None
    out_dir = args.out_dir
    if args.protocol is not None:
        protocol_path = ROOT / args.protocol
        checkpoint = (
            ROOT / json.loads(protocol_path.read_text(encoding="utf-8"))["source_checkpoint"]
        )
        out_dir = out_dir or protocol_path.parent
    out_dir = out_dir or OUT_DIR

    cases, summary = build_golden(args.annotation_dir, args.with_value)
    write_jsonl(out_dir / "golden-suite.jsonl", cases)
    count = _bundle_or_checkpoint_tokenizer_count(checkpoint)
    robust = robustness_inputs()
    robust[:0] = length_boundary_inputs(count)
    write_jsonl(out_dir / "robustness-inputs.jsonl", robust)
    if args.with_value:
        hardening = hardening_cases(count)
        write_jsonl(out_dir / "hardening-cases.jsonl", hardening)
        print(f"hardening cases: {len(hardening)}")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(
        f"robustness inputs: {len(robust)}  (expect EmptyInputError: "
        f"{sum(r['expect_empty_error'] for r in robust)})"
    )


if __name__ == "__main__":
    main()
