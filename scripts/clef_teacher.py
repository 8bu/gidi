"""Clef-as-teacher helpers: question schema and scoring against human labels.

Clef is a decision model (``judge`` role in omp): it takes a state and typed questions and returns a
probability for every option. Type is a ``choice`` over the 8 types. The target span is a ``choice``
over the candidate word n-grams of the note plus ``none``. The model calls themselves run in omp's
``judge_batch``; this module only builds the questions and decodes/scores the answers, so it never
writes into a dataset.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "annotation-v3.yaml"
# The contract is the single source of truth: Clef sees the definitions humans labelled by.
TYPES = {
    name: " ".join(text.split())
    for name, text in yaml.safe_load(CONFIG.read_text())["types"].items()
}
TYPE_INSTRUCTIONS = (
    "A short, informal Vietnamese personal-finance note written by the user about their own money "
    "(often without diacritics; amounts like 'k', 'tr', 'củ', 'lít'). Classify the money movement "
    "from the WRITER's point of view. 'mượn/vay/ứng/nợ <person>' = the writer borrows from them "
    "(borrow); '<person> mượn/vay' or 'cho <person> mượn' = they borrow from the writer (lend). "
    "'trả nợ/trả lại <person>' = the writer repays (repayment_out); '<person> trả nợ/trả lại' = "
    "repayment_in. Everyday purchases, bills, food, fees, gifts given and money spent are expense."
)
TARGET_INSTRUCTIONS = (
    "Which candidate is the external counterparty of the transaction: the person, shop, platform "
    "or organisation the writer pays, receives from, lends to, borrows from or is repaid by? Pick "
    "the "
    "shortest span that names them, without a kinship or title prefix such as 'anh', 'chị', 'bé' "
    "when a proper name follows; keep a kinship or role noun that is the only identifier (mẹ, "
    "khách, cty, chủ nhà, sếp). Items, dishes, bills, fees, activities, payment apps or banks used "
    "only as the channel, companions after 'với/vs', and beneficiaries of a purchase are NOT "
    "counterparties. If the note names no counterparty, pick the 'none' candidate."
)
RULES_V4 = (
    "annotation-v4 labelling rules. Each applies ONLY to a note that matches its pattern; for "
    "every other note ignore them and classify as usual (a purchase, food, ride, bill, ticket, "
    "shop or service payment is expense). "
    "R1 '<X> trả ...' (who pays decides): X a person (name or kinship term: Long, long, chị "
    "Linh, mẹ) and the note names no goods, service or work -> repayment_in ('Long tra 5 lit', "
    "'chị Linh trả 2tr'). X an employer, company, client or customer, OR the note says what the "
    "money pays for (wages, a month, an order, goods, work) -> income ('cong ty tra tien t10 5 "
    "lit', 'khach tra tien 500k', 'anh Minh tra tien cong son nha 4tr'). An explicit debt word "
    "wins: '<X> trả nợ / trả lại / trả tiền mượn' -> repayment_in. "
    "R2 paying on someone's behalf: 'trả giùm/trả hộ/ứng giùm/ứng hộ/mua hộ/đóng hộ <X> ...' "
    "means the writer pays or buys FOR X and X owes it back -> lend, never expense; the return "
    "side ('<X> trả lại tiền mình trả giùm', '<X> gửi lại tiền cơm') -> repayment_in. "
    "R3 sending money to other PEOPLE ('chuyển/gửi/biếu cho mẹ, họ hàng, bạn, ba ...') -> "
    "expense, or lend when a loan is stated; transfer is ONLY between the writer's own accounts, "
    "wallets, cash, savings or investments ('chuyển sang ví momo 1tr', 'gửi tiết kiệm 5tr'). "
    "R4 salary: 'nhận lương', 'lãnh lương', 'lương về', 'có lương', 'mới nhận lương' (also "
    "without diacritics) -> income; but a salary ADVANCE ('ứng lương', 'tạm ứng', 'ứng trước "
    "lương') is borrow (the writer owes it back), never income."
)
TARGET_RULES_V4 = (
    " annotation-v4 target rules: an employer, company or client noun that is the only name IS "
    "the counterparty ('cong ty tra tien t10 5 lit' -> 'cong ty', 'khach tra tien 500k' -> "
    "'khach'); a kinship or relative noun that is the only name is kept ('chuyển cho mẹ 2 củ' -> "
    "'mẹ', 'chuyển cho họ hàng 2 củ' -> 'họ hàng'); a person's name without its kinship prefix "
    "('chị Linh trả 2tr' -> 'Linh'); for 'trả giùm/hộ <X>', 'ứng giùm <X>', 'mua hộ <X>' the "
    "counterparty is X; salary notes ('nhận lương', 'lương về') name no counterparty -> 'none'."
)
_WORD = re.compile(r"\w+", re.UNICODE)


def candidates(text: str, max_words: int = 4) -> list[tuple[int, int]]:
    """Word n-gram character spans (start, end) of ``text``, up to ``max_words`` words."""
    words = [(m.start(), m.end()) for m in _WORD.finditer(text)]
    return [
        (words[i][0], words[j][1])
        for i in range(len(words))
        for j in range(i, min(i + max_words, len(words)))
    ]


def questions(text: str, rules: bool = False) -> dict:
    """The ``judge`` questions for one note; ``c<i>`` is the i-th entry of ``candidates``.

    ``rules=True`` is the annotation-v4 teacher: the type and target instructions additionally
    state R1-R4 (``RULES_V4``, ``TARGET_RULES_V4``); the type definitions stay the v3 ones.
    """
    criteria = {f"c{i}": f"'{text[s:e]}'" for i, (s, e) in enumerate(candidates(text))}
    criteria["none"] = "No counterparty is named."
    type_instructions, target_instructions = TYPE_INSTRUCTIONS, TARGET_INSTRUCTIONS
    if rules:
        type_instructions = f"{TYPE_INSTRUCTIONS} {RULES_V4}"
        target_instructions = TARGET_INSTRUCTIONS + TARGET_RULES_V4
    return {
        "type": {"type": "choice", "instructions": type_instructions, "criteria": TYPES},
        "target": {"type": "choice", "instructions": target_instructions, "criteria": criteria},
    }


def decode(text: str, answers: dict) -> dict:
    """Clef answers -> ``{type, type_probs, target: (start, end) | None, target_prob}``."""
    spans = candidates(text)
    choice = answers["target"]["choice"]
    return {
        "type": answers["type"]["choice"],
        "type_probs": answers["type"]["probabilities"],
        "type_conf": answers["type"].get("confidence"),
        "target": None if choice == "none" else spans[int(choice[1:])],
        "target_prob": answers["target"]["probabilities"].get(choice),
    }


def score(rows: list[dict], preds: list[dict]) -> dict:
    """Type / target / joint exact match against human labels, with the confusion list."""
    n = len(rows)
    type_ok = target_ok = joint_ok = covered = 0
    confusion: Counter[tuple[str, str]] = Counter()
    for row, p in zip(rows, preds, strict=True):
        gold = row.get("target")
        gt = (gold["start"], gold["end"]) if gold else None
        covered += gt is None or gt in set(candidates(row["text"]))
        t_ok, g_ok = p["type"] == row["type"], p["target"] == gt
        type_ok += t_ok
        target_ok += g_ok
        joint_ok += t_ok and g_ok
        if not t_ok:
            confusion[(row["type"], p["type"])] += 1
    return {
        "n": n,
        "type": type_ok / n,
        "target": target_ok / n,
        "joint": joint_ok / n,
        "gold_target_coverage": covered / n,
        "confusion": confusion.most_common(12),
    }


def _fold(text: str) -> str:
    return unicodedata.normalize("NFD", text.lower()).encode("ascii", "ignore").decode()


class ShotPool:
    """k-nearest labelled notes (char n-gram TF-IDF, diacritics folded) to use as Clef few-shots.

    The pool must hold human-labelled training notes only: never probe, test or human-value sets.
    """

    def __init__(self, pool: list[dict]) -> None:
        self._pool = pool
        self._vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), preprocessor=_fold)
        self._x = self._vec.fit_transform([r["text"] for r in pool])

    def state(self, text: str, k: int = 16) -> dict:
        sims = (self._x @ self._vec.transform([text]).T).toarray().ravel()
        shots = [
            {
                "note": self._pool[i]["text"],
                "type": self._pool[i]["type"],
                "counterparty": (self._pool[i].get("target") or {}).get("text"),
            }
            for i in np.argsort(-sims)[:k]
        ]
        return {
            "labelled_examples_following_the_same_annotation_rules": shots,
            "note_to_label": text,
        }
