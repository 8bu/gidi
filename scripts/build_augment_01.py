"""Build augment-01: deterministic, label-preserving variants of training-v5 notes.

Reads only datasets/annotation-v3/training-v5/train.jsonl. No model calls.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "datasets/annotation-v3/training-v5/train.jsonl"
OUT_DIR = ROOT / "datasets/annotation-v3/augment-01"
BATCH = "augment-01"
MAX_PER_NOTE = 2
MAX_TOTAL = 1200
AMOUNT_RE = re.compile(r"(?<![\w.,])(\d+(?:[.,]\d+)?) (tr|k|cu|củ|lít|lit)(?!\w)", re.IGNORECASE)


def strip_dia(s: str) -> str:
    s = s.replace("đ", "d").replace("Đ", "D")
    return "".join(c for c in unicodedata.normalize("NFD", s) if not unicodedata.combining(c))


def has_dia(s: str) -> bool:
    return strip_dia(unicodedata.normalize("NFC", s)) != unicodedata.normalize("NFC", s)


def is_person_target(rec: dict) -> bool:
    t = rec["target"]
    return bool(t) and t["text"] != t["text"].lower()


def _parts(rec: dict) -> tuple[str, str, str] | None:
    t = rec["target"]
    if not t:
        return None
    text = rec["text"]
    return text[: t["start"]], text[t["start"] : t["end"]], text[t["end"] :]


def _finish(pre: str, tgt: str | None, post: str) -> tuple[str, dict | None]:
    if tgt is None:
        return pre, None
    return pre + tgt + post, {"text": tgt, "start": len(pre), "end": len(pre) + len(tgt)}


def _map_all(rec: dict, fn) -> tuple[str, dict | None]:
    p = _parts(rec)
    if p is None:
        return fn(rec["text"]), None
    return _finish(fn(p[0]), fn(p[1]), fn(p[2]))


def t_lower_target(rec):
    p = _parts(rec)
    if p is None or p[1] == p[1].lower():
        return None
    return _finish(p[0], p[1].lower(), p[2])


def t_strip_diacritics(rec):
    if not has_dia(rec["text"]):
        return None
    return _map_all(rec, strip_dia)


def t_strip_and_lower(rec):
    if not is_person_target(rec) or not has_dia(rec["text"]):
        return None
    return _map_all(rec, lambda s: strip_dia(s).lower())


def _glue(s: str) -> str:
    return AMOUNT_RE.sub(lambda m: m.group(1) + strip_dia(m.group(2)), s)


def t_glue_amount(rec):
    if not AMOUNT_RE.search(rec["text"]):
        return None
    p = _parts(rec)
    if p is None:
        return _glue(rec["text"]), None
    if AMOUNT_RE.search(p[1]):
        return None
    # a match straddling the span boundary would change the span; reject
    if any(m.start() < len(p[0]) < m.end() or m.start() < len(p[0]) + len(p[1]) < m.end()
           for m in AMOUNT_RE.finditer(rec["text"])):  # fmt: skip
        return None
    return _finish(_glue(p[0]), p[1], _glue(p[2]))


def t_no_diacritic_name_only(rec):
    p = _parts(rec)
    if p is None or not has_dia(p[1]):
        return None
    return _finish(p[0], strip_dia(p[1]), p[2])


TRANSFORMS = {
    "lower_target": t_lower_target,
    "strip_diacritics": t_strip_diacritics,
    "strip_and_lower": t_strip_and_lower,
    "glue_amount": t_glue_amount,
    "no_diacritic_name_only": t_no_diacritic_name_only,
}


def sha(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def build() -> tuple[list[dict], dict]:
    raw = SOURCE.read_bytes()
    recs = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
    seen = {r["text"] for r in recs}
    cands = []  # (order key, rec, transform name, text, target)
    for r in recs:
        opts = []
        for name, fn in TRANSFORMS.items():
            res = fn(r)
            if res is None:
                continue
            text, target = res
            if text == r["text"]:
                continue
            opts.append((sha(f"{r['id']}|{name}"), name, text, target))
        opts.sort()
        chosen, local = [], set()
        for key, name, text, target in opts:
            if text in seen or text in local:
                continue
            local.add(text)
            chosen.append((key, name, text, target))
            if len(chosen) == MAX_PER_NOTE:
                break
        for key, name, text, target in chosen:
            cands.append((key, r, name, text, target))
    cands.sort(key=lambda c: c[0] + c[1]["id"])
    out, used = [], set()
    for _, r, name, text, target in cands:
        if len(out) >= MAX_TOTAL:
            break
        if text in seen or text in used:
            continue
        used.add(text)
        if target is not None:
            assert text[target["start"] : target["end"]] == target["text"], (r["id"], name)
        out.append(
            {
                "id": f"{BATCH}-{sha(r['id'] + '|' + name)[:12]}",
                "text": text,
                "type": r["type"],
                "target": target,
                "source_batch": BATCH,
                "accented": has_dia(text),
                "provenance": {
                    "annotator": "augment",
                    "derived_from": r["id"],
                    "transform": name,
                    "source_batch": BATCH,
                },
                "group": r["group"],
            }
        )
    out.sort(key=lambda x: x["id"])
    manifest = {
        "version": BATCH,
        "source": str(SOURCE.relative_to(ROOT)),
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "source_records": len(recs),
        "records": len(out),
        "max_per_note": MAX_PER_NOTE,
        "max_total": MAX_TOTAL,
        "per_transform": dict(sorted(Counter(r["provenance"]["transform"] for r in out).items())),
        "per_type": dict(sorted(Counter(r["type"] for r in out).items())),
    }
    return out, manifest


def render(out: list[dict], manifest: dict) -> dict[str, str]:
    return {
        "train.jsonl": "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out),
        "manifest.json": json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    files = render(*build())
    if args.check:
        bad = [n for n, c in files.items() if not (OUT_DIR / n).exists()
               or (OUT_DIR / n).read_text() != c]  # fmt: skip
        if bad:
            print(f"MISMATCH: {bad}")
            return 1
        print("augment-01 OK")
        return 0
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for n, c in files.items():
        (OUT_DIR / n).write_text(c)
    print(files["manifest.json"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
