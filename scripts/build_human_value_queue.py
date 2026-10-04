#!/usr/bin/env python
r"""Build the independent human-labelled validation queue ``human-value-01`` (phase 1).

Usage:
    uv run python scripts/build_human_value_queue.py [--root R] [--out-dir D] [--check]

Purpose: a held-out, human-labelled set for the final V7-vs-V8 value-span comparison. It must be
independent of everything V8's rules were designed on, of the frozen evaluation sets and of the
earlier value batches. This script only *samples* notes; the labels come from a human in Quet.

Hard rules (enforced by construction, grep-verifiable):

* no model, value parser, rule proposer or LLM is imported or called: the script is stdlib only;
* difficulty comes from observable surface structure (regexes and word lists below) used for
  stratification only. A surface feature never emits a value span, and the queue carries no value
  candidate, proposal or prefill;
* no synthetic notes: the pool is the Quet-*approved* reviewed corpus. ``corpus/raw`` batches
  without a Quet approval are not used.

Pipeline:

1. Pool: ``corpus/reviewed/baseline-01.jsonl`` rows with ``quet.status == "approved"``.
2. Exclusion against the *hard* references: V8 rule-design input (``training-v1/train.jsonl``),
   validation, frozen test, ``probe-v1`` (+ eval-only copy) and ``targeted-value-01`` (every
   ``*.jsonl`` under its dataset dir plus its raw batch). Rules, in order, each note counted once
   under the first rule that fires:

   ``exact``      same NFC + strip + lowercase text;
   ``folded``     same text after NFC, lowercase, accent stripping (``đ`` -> ``d``), digit runs
                  masked to ``#`` and whitespace collapsed (a different amount alone does not
                  make a note different);
   ``sequence``   ``difflib`` ratio of the folded texts >= 0.90 (the repo-wide near-duplicate
                  threshold, ``gidi.annotation.leakage.NEAR_DUP_THRESHOLD``);
   ``token``      content-word set Jaccard of the folded texts >= 0.80 with >= 4 words on both
                  sides (amount units and digits ignored; catches reordered or one-word-swapped
                  longer notes the sequence ratio can miss);
   ``char3``      folded character-3-gram Jaccard >= 0.80.
3. Prior-usage disclosure: the same rules against the *encoder-seen* references (training sets
   gidi-finance-v1/V7 type/target training may have used). Seen-near notes are dropped because
   enough unseen notes exist; counts are reported.
4. Dedup inside the pool with the same rules (best surface-difficulty representative kept).
5. Stratified selection (overlap allowed) by surface features, down-weighting trivial single
   ``500k`` notes, then fill to ``TARGET_TOTAL`` by the same difficulty score.
6. Verification: every selected note is re-checked against all hard references and against every
   other selected note; the build fails on any hit.

Writes into ``--out-dir`` (``datasets/annotation-v2/human-value-01``):

* ``review-queue.jsonl``  Quet queue ``{id, text, strata, review_group}`` (shuffled by seed);
* ``candidate-provenance.jsonl``  per queued note: source, prior usage (NOT given to Quet);
* ``manifest.json``  inputs + sha256, seed, exclusion/strata/prior-usage counts, the Quet schema
  (``configs/annotation-v2.quet.yaml``) and its sha256, label fields, phases.

``annotation-guide.md`` is hand-written in the same directory and hashed into the manifest.
The human pass is ONE combined Quet pass (type + target span + value span, Quet multi-span);
``labels.jsonl`` is written only by Quet. ``--check`` rebuilds in memory and compares the files
on disk byte for byte.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
import sys
import unicodedata
from collections import Counter
from collections.abc import Iterable
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

NAME = "human-value-01"
SEED = "human-value-01:v1"
DEFAULT_OUT_DIR = Path("datasets/annotation-v2/human-value-01")
REVIEW_GROUP = "primary"

POOL = Path("corpus/reviewed/baseline-01.jsonl")
UNAPPROVED_RAW = (
    Path("corpus/raw/targeted-annotation-v1-01.jsonl"),
    Path("corpus/raw/targeted-value-01.jsonl"),
)

# Hard exclusions: V8 rule-design input, frozen evaluation sets, probes, the earlier value batch.
HARD_FILES = (
    Path("datasets/annotation-v2/training-v1/train.jsonl"),
    Path("datasets/annotation-v2/training-v1/validation.jsonl"),
    Path("datasets/annotation-v2/training-v1/test.jsonl"),
    Path("datasets/annotation-v2/training-v1/probe-v1-eval-only.jsonl"),
    Path("datasets/probe-v1/notes.jsonl"),
    Path("corpus/raw/targeted-value-01.jsonl"),
)
HARD_DIRS = (Path("datasets/annotation-v2/targeted-value-01"),)

# Encoder-seen references: sets whose notes the gidi-finance-v1/V7 type/target encoder may have
# been trained on (disclosure; seen-near notes are dropped when enough unseen notes exist).
SEEN_FILES = (
    Path("datasets/annotation-v1/distillation-v1/train.jsonl"),
    Path("datasets/annotation-v1/distillation-v1/validation.jsonl"),
    Path("datasets/annotation-v1/training-v2/train.jsonl"),
    Path("datasets/annotation-v1/training-v3/train.jsonl"),
    Path("datasets/annotation-v2/training-v2-direction-repair/train.jsonl"),
    Path("datasets/annotation-v1/targeted-02/notes.jsonl"),
    Path("datasets/annotation-v1/targeted-03/notes.jsonl"),
)
ANNOTATION_V1_QUEUE = Path("datasets/annotation-v1/queue.jsonl")
ANNOTATION_V1_LABELS = Path("datasets/annotation-v1/labels.jsonl")
V1_ENCODER_TRAIN = Path("datasets/annotation-v1/distillation-v1/train.jsonl")
GUIDE_FILE = "annotation-guide.md"
QUET_SCHEMA = Path("configs/annotation-v2.quet.yaml")
LABEL_FIELDS = ["id", "annotation_status", "type", "target", "value", "span_status", "note"]
QUEUE_FILE = "review-queue.jsonl"
PROVENANCE_FILE = "candidate-provenance.jsonl"
MANIFEST_FILE = "manifest.json"

SEQ_THRESHOLD = 0.90
TOKEN_THRESHOLD = 0.80
TOKEN_MIN = 4
CHAR3_THRESHOLD = 0.80
RULES = ("exact", "folded", "sequence", "token", "char3")

TARGET_TOTAL = 150
MIN_TOTAL, MAX_TOTAL = 120, 180
STRATUM_TARGETS = {
    "bare_number": 25,
    "multi_number": 30,
    "date_month_year_amount": 15,
    "quantity_amount": 15,
    "installment_index_amount": 10,
    "slang_compact_money": 15,
    "unaccented": 10,
    "unusual_whitespace_punct": 10,
    "long_noisy": 10,
    "null_no_number_ambiguous": 10,
}
FILL_STRATUM = "general"

PHASES = [
    {
        "phase": "phase-1-awaiting-annotation",
        "what": "this build: candidate pool, dedup, stratification, queue, guide. A human labels "
        "type, target and value of the queue in one combined Quet pass (labels.jsonl is created "
        "by Quet only).",
    },
    {
        "phase": "phase-2-second-pass",
        "what": "second human pass over every note that is uncertain (status or value status), "
        "multi_number or has a null value plus a random sample of the remaining notes; "
        "disagreements are resolved and the resolution recorded.",
    },
    {
        "phase": "phase-3-freeze-and-run",
        "what": "freeze the labels with sha256 and stats, then run V7 and V8 each exactly once "
        "on the frozen set: compare V7 vs V8 on the value span, and also report the human "
        "type/target accuracy of the shared v1 path. No labelling or rule changes after the "
        "first model run.",
    },
]


# --------------------------------------------------------------------------------------------
# normalisation and near-duplicate rules
# --------------------------------------------------------------------------------------------


def exact_key(text: str) -> str:
    return unicodedata.normalize("NFC", text).strip().lower()


def fold(text: str) -> str:
    """NFC, lowercase, accents stripped, digit runs masked to ``#``, whitespace collapsed."""
    text = unicodedata.normalize("NFC", text).lower().replace("đ", "d")
    text = "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")
    return " ".join(re.sub(r"\d+", "#", text).split())


_AMOUNT_UNITS = frozenset(
    ["k", "tr", "trieu", "nghin", "ngan", "ty", "cu", "xi", "chai", "lit", "d", "dong", "vnd"]
)


def _tokens(folded: str) -> frozenset[str]:
    """Content words of a folded note: letter runs of 2+ chars without amount units."""
    return frozenset(w for w in re.findall(r"[^\W\d_]{2,}", folded) if w not in _AMOUNT_UNITS)


def _char3(folded: str) -> frozenset[str]:
    padded = f" {folded} "
    return frozenset(padded[i : i + 3] for i in range(len(padded) - 2))


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    union = len(a | b)
    return len(a & b) / union if union else 0.0


class Reference:
    """Precomputed normal forms of one note."""

    __slots__ = ("exact", "folded", "tokens", "char3")

    def __init__(self, text: str) -> None:
        self.exact = exact_key(text)
        self.folded = fold(text)
        self.tokens = _tokens(self.folded)
        self.char3 = _char3(self.folded)


def rule_between(a: Reference, b: Reference) -> str | None:
    """The first near-duplicate rule that ties ``a`` to ``b`` (``None`` when they differ)."""
    if a.exact == b.exact:
        return "exact"
    if a.folded == b.folded:
        return "folded"
    matcher = SequenceMatcher(None, a.folded, b.folded)
    if (
        matcher.real_quick_ratio() >= SEQ_THRESHOLD
        and matcher.quick_ratio() >= SEQ_THRESHOLD
        and matcher.ratio() >= SEQ_THRESHOLD
    ):
        return "sequence"
    if (
        len(a.tokens) >= TOKEN_MIN
        and len(b.tokens) >= TOKEN_MIN
        and _jaccard(a.tokens, b.tokens) >= TOKEN_THRESHOLD
    ):
        return "token"
    if _jaccard(a.char3, b.char3) >= CHAR3_THRESHOLD:
        return "char3"
    return None


class ReferenceSet:
    """A deduplicated set of reference notes with a fast exact/folded pre-check."""

    def __init__(self, texts: Iterable[str]) -> None:
        self.refs: list[Reference] = []
        self._exact: set[str] = set()
        self._folded: set[str] = set()
        for text in texts:
            self.add(text)

    def add(self, text: str) -> None:
        self.add_ref(Reference(text))

    def has_exact(self, ref: Reference) -> bool:
        return ref.exact in self._exact

    def add_ref(self, ref: Reference) -> None:
        if ref.exact in self._exact:
            return
        self._exact.add(ref.exact)
        self._folded.add(ref.folded)
        self.refs.append(ref)

    def match(self, ref: Reference) -> str | None:
        if ref.exact in self._exact:
            return "exact"
        if ref.folded in self._folded:
            return "folded"
        for other in self.refs:
            rule = rule_between(ref, other)
            if rule is not None:
                return rule
        return None


# --------------------------------------------------------------------------------------------
# surface features (stratification only; nothing here yields a value span)
# --------------------------------------------------------------------------------------------

_TOKEN = re.compile(r"(?<!\d)(\d+(?:[.,:/]\d+)*)([a-zà-ỹ]*)(\d*)")
_SEPARATED = re.compile(r"^\d{1,3}(?:[.,]\d{3})+$")
_DECIMAL = re.compile(r"[.,]\d{1,2}$")
_PREV_WORD = re.compile(r"([a-zà-ỹ]+)\s?$")
_NEXT_WORD = re.compile(r"\s?([a-zà-ỹ]+)")

MONEY_TAILS = frozenset(
    [
        "k",
        "tr",
        "trieu",
        "triệu",
        "nghin",
        "nghìn",
        "ngan",
        "ngàn",
        "ty",
        "tỷ",
        "cu",
        "củ",
        "d",
        "đ",
        "dong",
        "đồng",
        "vnd",
        "vnđ",
        "xi",
        "xị",
        "chai",
        "lit",
        "lít",
    ]
)
MONEY_NEXT = MONEY_TAILS - {"k", "d", "đ"}
SLANG_UNITS = frozenset(["cu", "củ", "xi", "xị", "chai", "lit", "lít"])
MEASURE_TAILS = frozenset(["kg", "g", "gb", "mb", "tb", "ml", "cm", "l", "m", "inch"])
PERIOD_WORDS = frozenset(
    ["tháng", "thang", "ngày", "ngay", "mùng", "mung", "thứ", "thu", "năm", "nam", "quý", "quy"]
)
INSTALLMENT_WORDS = frozenset(
    [
        "kỳ",
        "ky",
        "đợt",
        "dot",
        "lần",
        "lan",
        "vòng",
        "vong",
        "lượt",
        "luot",
        "phiên",
        "tuần",
        "tuan",
    ]
)
PERIOD_NEXT = frozenset(
    [
        "tháng",
        "thang",
        "th",
        "tuần",
        "tuan",
        "năm",
        "nam",
        "ngày",
        "ngay",
        "tiếng",
        "tieng",
        "giờ",
        "gio",
    ]
)
QUANTITY_NEXT = frozenset(
    [
        "người",
        "nguoi",
        "ng",
        "vé",
        "ve",
        "ly",
        "tô",
        "to",
        "phần",
        "phan",
        "cái",
        "cai",
        "thùng",
        "thung",
        "chỉ",
        "chi",
        "chiếc",
        "suất",
        "suat",
        "hộp",
        "hop",
        "gói",
        "goi",
        "lọ",
        "lo",
        "bó",
        "bo",
    ]
)
ID_WORDS = frozenset(["stk", "tk", "sđt", "sdt", "mã", "ma", "đơn", "don", "cccd", "cmnd"])
INSTALLMENT_CONTEXT = re.compile(r"trả góp|tra gop|góp|\bgop\b")
NUMBER_WORDS = frozenset(
    [
        "một",
        "mot",
        "hai",
        "ba",
        "bốn",
        "bon",
        "năm",
        "nam",
        "sáu",
        "sau",
        "bảy",
        "bay",
        "tám",
        "tam",
        "chín",
        "chin",
        "mười",
        "muoi",
        "trăm",
        "tram",
        "rưỡi",
        "ruoi",
        "nửa",
        "nua",
    ]
)
UNACCENTED_HINTS = frozenset(
    [
        "tien",
        "tra",
        "nap",
        "chuyen",
        "mua",
        "ban",
        "rut",
        "gui",
        "tiet",
        "kiem",
        "thang",
        "hom",
        "nay",
        "no",
        "muon",
        "cho",
        "vay",
        "luong",
        "thuong",
        "phong",
        "dien",
        "nuoc",
        "an",
        "com",
        "cat",
        "quy",
        "de",
        "danh",
        "the",
        "tin",
        "dung",
        "trieu",
        "nghin",
        "ngan",
        "toi",
    ]
)

MONEY_ROLES = frozenset({"money_unit", "money_sep", "money_long", "bare"})


def _has_diacritics(text: str) -> bool:
    nfd = unicodedata.normalize("NFD", text.lower())
    return "đ" in nfd or any(unicodedata.category(c) == "Mn" for c in nfd)


def numeric_expressions(text: str) -> list[dict[str, Any]]:
    """Numeric expressions of ``text`` with a coarse surface role, left to right.

    A number written straight after a money unit (``1 triệu 2``) continues that expression and is
    not counted separately. Roles: ``date``, ``time``, ``percent``, ``money_unit``, ``money_sep``,
    ``money_long``, ``bare``, ``period``, ``installment``, ``quantity``, ``id``, ``other``.
    """
    text = unicodedata.normalize("NFC", text).lower()
    found: list[dict[str, Any]] = []
    prev_unit_end = -1
    for m in _TOKEN.finditer(text):
        num, tail, post = m.group(1), m.group(2), m.group(3)
        before, after = text[: m.start()], text[m.end() :]
        prefix = re.search(r"[a-zà-ỹ]+$", before)
        prev = _PREV_WORD.search(before)
        prev_word = prev.group(1) if prev else ""
        nxt = _NEXT_WORD.match(after) if not tail and not post else None
        next_word = nxt.group(1) if nxt else ""
        if m.start() == prev_unit_end:
            prev_unit_end = m.end()
            continue
        role, slang = "other", False
        if "/" in num:
            role = "date"
        elif ":" in num or tail == "h" or next_word in {"giờ", "gio"}:
            role = "time"
        elif after.startswith("%"):
            role = "percent"
        elif tail in MONEY_TAILS:
            role, slang = "money_unit", tail in SLANG_UNITS or bool(post)
            slang = slang or bool(_DECIMAL.search(num))
        elif tail in MEASURE_TAILS:
            role = "quantity"
        elif tail in {"th", "t"} and len(num) <= 2:
            role = "period"
        elif tail:
            role = "other"
        elif prefix:
            role = "period" if prefix.group(0) in {"t", "q", "hk"} else "other"
        elif _SEPARATED.match(num):
            role = "money_sep"
        elif next_word in MONEY_NEXT:
            role, slang = "money_unit", next_word in SLANG_UNITS
            nxt_match = _NEXT_WORD.match(after)
            prev_unit_end = m.end() + (nxt_match.end() if nxt_match else 0) + 1
            slang = slang or bool(_DECIMAL.search(num))
        elif prev_word in INSTALLMENT_WORDS:
            role = "installment"
        elif prev_word in PERIOD_WORDS:
            role = "period"
        elif prev_word in ID_WORDS or (num.isdigit() and len(num) >= 8):
            role = "id"
        elif next_word in PERIOD_NEXT:
            role = "period"
        elif next_word in QUANTITY_NEXT:
            role = "quantity"
        elif num.isdigit() and 5 <= len(num) <= 7:
            role = "money_long"
        elif num.isdigit() and len(num) <= 4:
            role = "bare"
        found.append({"role": role, "slang": slang, "num": num})
    return found


def surface_features(text: str) -> dict[str, Any]:
    nfc = unicodedata.normalize("NFC", text)
    low = nfc.lower()
    exprs = numeric_expressions(text)
    roles = [e["role"] for e in exprs]
    money_like = [r for r in roles if r in MONEY_ROLES]
    unit_money = [r for r in money_like if r != "bare"]
    # a bare number next to a unit amount is a context number, not a second money candidate
    competing = len(unit_money) if unit_money else len(money_like)
    words = re.findall(r"[a-zà-ỹ]+", low)
    return {
        "n_expr": len(exprs),
        "roles": roles,
        "n_money_like": competing,
        "has_money_like": bool(money_like),
        "slang": any(e["slang"] for e in exprs) or "rưỡi" in low or "ruoi" in words,
        "unaccented": not _has_diacritics(nfc)
        and len(words) >= 3
        and any(w in UNACCENTED_HINTS for w in words),
        "n_tokens": len(nfc.split()),
        "length": len(nfc),
        "trivial": (
            len(exprs) == 1
            and exprs[0]["role"] == "money_unit"
            and re.search(r"(?<!\d)\d+k(?![\w])", low) is not None
            and len(nfc) <= 25
        ),
    }


def unusual_whitespace_punct(text: str) -> bool:
    if text != text.strip() or "  " in text or "\t" in text or "\n" in text:
        return True
    if unicodedata.normalize("NFC", text) != text:
        return True
    for ch in text:
        cat = unicodedata.category(ch)
        if ch in "~+*()[]{}<>!?;\"'…_=@&|\\^#" or (cat in {"Cf", "So", "Zs"} and ch != " "):
            return True
        if ord(ch) > 0x2000 and cat.startswith(("S", "P")):
            return True
    return bool(re.search(r"[.,!:;-]\s*$|\.\.|,,|\s[,.]|\+\s*\+| - ", text))


def assign_strata(text: str) -> tuple[list[str], dict[str, Any]]:
    f = surface_features(text)
    roles = set(f["roles"])
    amount = f["has_money_like"]
    strata: list[str] = []
    if f["n_expr"] >= 2:
        strata.append("multi_number")
    if "bare" in roles and not roles & {"money_unit", "money_sep", "money_long"}:
        strata.append("bare_number")
    if roles & {"date", "period"} and amount:
        strata.append("date_month_year_amount")
    if "quantity" in roles and amount:
        strata.append("quantity_amount")
    if amount and (
        "installment" in roles or ("period" in roles and INSTALLMENT_CONTEXT.search(text.lower()))
    ):
        strata.append("installment_index_amount")
    if f["slang"]:
        strata.append("slang_compact_money")
    if f["unaccented"]:
        strata.append("unaccented")
    if unusual_whitespace_punct(text):
        strata.append("unusual_whitespace_punct")
    if f["length"] >= 36 or f["n_tokens"] >= 8:
        strata.append("long_noisy")
    if f["n_expr"] == 0 or f["n_money_like"] >= 2 or (f["n_expr"] >= 1 and not amount):
        strata.append("null_no_number_ambiguous")
    return strata, f


def difficulty(strata: list[str], f: dict[str, Any]) -> int:
    """Context-sensitivity score: more surface traps up, a lone plain ``500k`` down."""
    weight = {
        "multi_number": 3,
        "bare_number": 3,
        "null_no_number_ambiguous": 3,
        "date_month_year_amount": 2,
        "quantity_amount": 2,
        "installment_index_amount": 2,
        "slang_compact_money": 2,
        "unusual_whitespace_punct": 2,
        "long_noisy": 1,
        "unaccented": 1,
    }
    return sum(weight[s] for s in strata) - (2 if f["trivial"] else 0)


# --------------------------------------------------------------------------------------------
# io
# --------------------------------------------------------------------------------------------


def read_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text("utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha_key(*parts: str) -> str:
    return hashlib.sha256(":".join((SEED, *parts)).encode("utf-8")).hexdigest()


def dump_jsonl(rows: list[dict[str, Any]]) -> bytes:
    return "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows).encode("utf-8")


def dump_json(obj: Any) -> bytes:
    return (json.dumps(obj, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def texts_of(
    root: Path, files: Iterable[Path], dirs: Iterable[Path] = ()
) -> tuple[list[str], list[Path]]:
    paths = list(files)
    for d in dirs:
        paths.extend(sorted(p.relative_to(root) for p in (root / d).glob("*.jsonl")))
    texts: list[str] = []
    for rel in paths:
        texts.extend(r["text"] for r in read_rows(root / rel) if isinstance(r.get("text"), str))
    return texts, paths


def quet_record_count(db: Path) -> int | None:
    if not db.exists():
        return None
    con = sqlite3.connect(f"file:{db}?mode=ro&immutable=1", uri=True)
    try:
        return con.execute("select count(*) from records where status = 'approved'").fetchone()[0]
    finally:
        con.close()


# --------------------------------------------------------------------------------------------
# build
# --------------------------------------------------------------------------------------------


def select(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Stratified greedy selection (overlap allowed), then fill by difficulty to TARGET_TOTAL."""
    ranked = sorted(candidates, key=lambda c: (-c["score"], c["hash"]))
    chosen: dict[str, dict[str, Any]] = {}
    members = {s: [c for c in ranked if s in c["strata"]] for s in STRATUM_TARGETS}
    for stratum in sorted(STRATUM_TARGETS, key=lambda s: (len(members[s]), s)):
        have = sum(1 for c in chosen.values() if stratum in c["strata"])
        for cand in members[stratum]:
            if have >= STRATUM_TARGETS[stratum]:
                break
            if cand["id"] not in chosen:
                chosen[cand["id"]] = cand
                have += 1
    for cand in ranked:
        if len(chosen) >= TARGET_TOTAL:
            break
        if cand["id"] not in chosen:
            chosen[cand["id"]] = cand
    return sorted(chosen.values(), key=lambda c: c["hash"])


def build(root: Path, out_dir: Path) -> dict[str, bytes]:
    pool_rows = read_rows(root / POOL)
    inputs: dict[str, str] = {str(POOL): sha256_file(root / POOL)}

    hard_texts, hard_paths = texts_of(root, HARD_FILES, HARD_DIRS)
    seen_texts, seen_paths = texts_of(root, SEEN_FILES)
    for rel in (
        *hard_paths,
        *seen_paths,
        *UNAPPROVED_RAW,
        ANNOTATION_V1_QUEUE,
        ANNOTATION_V1_LABELS,
    ):
        if (root / rel).exists():
            inputs[str(rel)] = sha256_file(root / rel)
    hard = ReferenceSet(hard_texts)
    seen = ReferenceSet(seen_texts)
    v1_train = ReferenceSet(texts_of(root, [V1_ENCODER_TRAIN])[0])

    annotation_queue = {r["id"] for r in read_rows(root / ANNOTATION_V1_QUEUE)}
    annotation_status = {
        r["id"]: r["annotation_status"] for r in read_rows(root / ANNOTATION_V1_LABELS)
    }

    stats: dict[str, Any] = {
        "pool_rows": len(pool_rows),
        "not_approved": 0,
        "dropped_hard_exclusion": dict.fromkeys(RULES, 0),
        "dropped_encoder_seen": dict.fromkeys(RULES, 0),
        "dropped_pool_duplicate": dict.fromkeys(RULES, 0),
    }
    survivors: list[dict[str, Any]] = []
    for row in pool_rows:
        if (row.get("quet") or {}).get("status") != "approved":
            stats["not_approved"] += 1
            continue
        ref = Reference(row["text"])
        rule = hard.match(ref)
        if rule:
            stats["dropped_hard_exclusion"][rule] += 1
            continue
        rule = seen.match(ref)
        if rule:
            stats["dropped_encoder_seen"][rule] += 1
            continue
        strata, feats = assign_strata(row["text"])
        survivors.append(
            {
                "row": row,
                "id": row["id"],
                "ref": ref,
                "strata": strata,
                "score": difficulty(strata, feats),
                "hash": sha_key(row["id"]),
                "feats": feats,
            }
        )
    stats["after_exclusion"] = len(survivors)
    stats["pool_exact_in_v1_encoder_train"] = sum(
        1 for r in pool_rows if v1_train.has_exact(Reference(r["text"]))
    )
    stats["unapproved_raw_not_used"] = {}
    for rel in UNAPPROVED_RAW:
        raw_rows = read_rows(root / rel)
        stats["unapproved_raw_not_used"][str(rel)] = {
            "rows": len(raw_rows),
            "quet_approved_records": quet_record_count(root / f"{rel}.quet.db"),
            "not_excluded_by_hard_rules": sum(
                1 for r in raw_rows if hard.match(Reference(r["text"])) is None
            ),
        }

    kept: list[dict[str, Any]] = []
    kept_set = ReferenceSet([])
    for cand in sorted(survivors, key=lambda c: (-c["score"], c["hash"])):
        rule = kept_set.match(cand["ref"])
        if rule:
            stats["dropped_pool_duplicate"][rule] += 1
            continue
        kept_set.add_ref(cand["ref"])
        kept.append(cand)
    stats["after_dedup"] = len(kept)

    chosen = select(kept)
    if not MIN_TOTAL <= len(chosen) <= MAX_TOTAL:
        raise SystemExit(f"selected {len(chosen)} notes, outside {MIN_TOTAL}-{MAX_TOTAL}")

    # verification: no selected note ties to a hard reference or to another selected note
    for i, cand in enumerate(chosen):
        hit = hard.match(cand["ref"])
        if hit:
            raise SystemExit(f"{cand['id']} overlaps an excluded set ({hit})")
        for other in chosen[:i]:
            hit = rule_between(cand["ref"], other["ref"])
            if hit:
                raise SystemExit(f"{cand['id']} duplicates {other['id']} ({hit})")

    queue = [
        {
            "id": c["id"],
            "text": c["row"]["text"],
            "strata": c["strata"] or [FILL_STRATUM],
            "review_group": REVIEW_GROUP,
        }
        for c in chosen
    ]
    provenance, usage = [], Counter()
    for c in chosen:
        rid = c["id"]
        in_queue = rid in annotation_queue
        status = annotation_status.get(rid)
        seen_exact = v1_train.has_exact(c["ref"])
        usage["annotation_v1_queued"] += in_queue
        usage[f"annotation_v1_status_{status or 'never_annotated'}"] += 1
        usage["seen_by_v1_encoder_type_target_training"] += seen_exact
        provenance.append(
            {
                "id": rid,
                "source_corpus": str(POOL),
                "source_batch": c["row"].get("prompt_id"),
                "source": c["row"].get("source"),
                "quet_review_status": c["row"]["quet"]["status"],
                "prior_usage": {
                    "annotation_v1_queued": in_queue,
                    "annotation_v1_status": status,
                    "seen_by_v1_encoder_type_target_training": seen_exact,
                },
                "strata": c["strata"] or [FILL_STRATUM],
                "surface_score": c["score"],
            }
        )

    achieved = {s: sum(1 for c in chosen if s in c["strata"]) for s in STRATUM_TARGETS}
    achieved[FILL_STRATUM] = sum(1 for c in chosen if not c["strata"])
    available = {s: sum(1 for c in kept if s in c["strata"]) for s in STRATUM_TARGETS}
    strata_table = {
        s: {
            "target": STRATUM_TARGETS[s],
            "achieved": achieved[s],
            "available_after_dedup": available[s],
            "shortfall": max(0, STRATUM_TARGETS[s] - achieved[s]),
        }
        for s in STRATUM_TARGETS
    }

    queue_bytes = dump_jsonl(queue)
    guide = root / out_dir / GUIDE_FILE
    manifest = {
        "name": NAME,
        "annotation_version": "annotation-v2",
        "role": "independent human-labelled validation set (type, target and value in one Quet "
        "pass; V7-vs-V8 value comparison)",
        "phase": PHASES[0]["phase"],
        "seed": SEED,
        "selection": {
            "pool": str(POOL),
            "pool_rule": "Quet-approved reviewed notes only; no synthetic notes",
            "near_duplicate_rules": {
                "exact": "NFC + strip + lowercase",
                "folded": "NFC, lowercase, accents stripped, digits masked, whitespace collapsed",
                "sequence": f"difflib ratio of folded texts >= {SEQ_THRESHOLD}",
                "token": f"folded token Jaccard >= {TOKEN_THRESHOLD} (>= {TOKEN_MIN} tokens)",
                "char3": f"folded char-3-gram Jaccard >= {CHAR3_THRESHOLD}",
            },
            "hard_exclusion_sources": [str(p) for p in hard_paths],
            "encoder_seen_sources": [str(p) for p in seen_paths],
            "target_total": TARGET_TOTAL,
            "allowed_range": [MIN_TOTAL, MAX_TOTAL],
        },
        "counts": {
            "queue": len(queue),
            **stats,
            "dropped_hard_exclusion_total": sum(stats["dropped_hard_exclusion"].values()),
            "dropped_encoder_seen_total": sum(stats["dropped_encoder_seen"].values()),
            "dropped_pool_duplicate_total": sum(stats["dropped_pool_duplicate"].values()),
        },
        "verification": {
            "overlap_with_hard_exclusions": 0,
            "near_duplicate_pairs_inside_queue": 0,
        },
        "strata": strata_table,
        "strata_fill_notes": achieved[FILL_STRATUM],
        "strata_shortfalls": {s: v["shortfall"] for s, v in strata_table.items() if v["shortfall"]},
        "prior_usage": dict(sorted(usage.items())),
        "inputs": dict(sorted(inputs.items())),
        "outputs": {
            QUEUE_FILE: hashlib.sha256(queue_bytes).hexdigest(),
            **({GUIDE_FILE: sha256_file(guide)} if guide.exists() else {}),
        },
        "queue_sha256": hashlib.sha256(queue_bytes).hexdigest(),
        "planned_phases": PHASES,
        "quet_schema": {"path": str(QUET_SCHEMA), "sha256": sha256_file(root / QUET_SCHEMA)},
        "quet_command": (
            f"quet annotate {out_dir / QUEUE_FILE} --schema {QUET_SCHEMA} "
            f"--out {out_dir / 'labels.jsonl'}"
        ),
        "label_fields": LABEL_FIELDS,
        "field_mapping": {
            "type": "Quet `type` (annotation-v1 taxonomy, docs/annotation-v1.md)",
            "target": "Quet span `target` {text, start, end} or null (annotation-v1 target rule)",
            "value": "Quet span `value` {text, start, end} or null: the minimal amount expression",
            "span_status": "Quet `span_status` {value: complete | uncertain} (value span only)",
            "annotation_status": "Quet status `complete` / `uncertain` / `skipped`",
            "annotation_note": "Quet `note` (required for an uncertain status or value)",
            "offsets": "code points into the NFC note, text == note[start:end]",
        },
        "validate_command": (
            f"uv run python scripts/validate_annotations.py {out_dir / 'labels.jsonl'} "
            f"--config {QUET_SCHEMA} --queue {out_dir / QUEUE_FILE}"
        ),
    }
    return {
        QUEUE_FILE: queue_bytes,
        PROVENANCE_FILE: dump_jsonl(provenance),
        MANIFEST_FILE: dump_json(manifest),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--check", action="store_true", help="verify files on disk, write nothing")
    args = parser.parse_args(argv)

    files = build(args.root, args.out_dir)
    out = args.root / args.out_dir
    if args.check:
        bad = [
            n
            for n, data in files.items()
            if not (out / n).exists() or (out / n).read_bytes() != data
        ]
        if bad:
            print(f"MISMATCH: {', '.join(bad)}", file=sys.stderr)
            return 1
        print(f"{NAME}: files match a fresh build")
        return 0
    labels = out / "labels.jsonl"
    if labels.exists() and (out / QUEUE_FILE).read_bytes() != files[QUEUE_FILE]:
        print("labels.jsonl exists and the queue would change; refusing", file=sys.stderr)
        return 1
    out.mkdir(parents=True, exist_ok=True)
    for name, data in files.items():
        (out / name).write_bytes(data)
    manifest = json.loads(files[MANIFEST_FILE])
    print(json.dumps({k: manifest[k] for k in ("counts", "strata", "prior_usage")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
