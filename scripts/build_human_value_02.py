#!/usr/bin/env python
r"""Build the human-labelled TEST set ``datasets/annotation-v3/human-value-02`` (queue only).

Usage:
    uv run python scripts/build_human_value_02.py [--root R] [--out-dir D] [--check]

``candidates.txt`` holds LLM-written Vietnamese finance notes (one per line, ``## weak`` / ``##
general`` section headers): the weak strata of human-value-01 batch 2 (bare number, multi
number, quantity, installment index, date, no amount, unusual whitespace / punctuation) and
general notes over all eight types (debt-only notes both ways, shops, gifts, banks, kinship
targets). They were written from ``docs/annotation-v1.md``, ``docs/annotation-v3.md`` and the
reviewed-corpus style only, never from training data, so the set stays independent.

The builder drops every candidate that is an exact or near duplicate (the rules of
``scripts/build_human_value_queue.py``: exact, folded, sequence >= 0.90, token, char3) of a text in
any reference set or of an earlier candidate: human-value-01, the frozen test splits, probe-v1,
the annotation-v3 batches and training sets up to ``training-v3``, the annotation-v1 / v2
datasets and the whole ``corpus/`` tree. It then writes, deterministically:

* ``review-queue.jsonl``: ``{id, text, strata, review_group}``; ``id`` is ``hv02-<sha256(NFC
  text)[:12]>``, the order a seeded shuffle;
* ``manifest.json``: counts, strata, leakage drops, hashes, provenance.

``annotation-guide.md`` is written by hand and only hashed. There is NO proposals file and no
model or rule output: the labels (type, target, value, annotation-v3 combined pass) come from
the user in Quet, and the set is never used for training. ``--check`` rebuilds in memory and
compares the files on disk byte for byte.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

SCRIPTS = Path(__file__).resolve().parent


def _load(name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(name, module)
    spec.loader.exec_module(module)
    return module


hv = _load("build_human_value_queue")

NAME = "human-value-02"
SEED = "human-value-02:v1"
DEFAULT_OUT_DIR = Path("datasets/annotation-v3/human-value-02")
SCHEMA = Path("configs/annotation-v3.quet.yaml")
CONTRACT = Path("configs/annotation-v3.yaml")
DOC = Path("docs/annotation-v3.md")
QUET_PROJECT = "gidi-hv02"
QUET_NAME = "gidi human-value-02 · test set"
REVIEW_GROUP = "primary"
FILL_STRATUM = "general"

CANDIDATES_FILE = "candidates.txt"
GUIDE_FILE = "annotation-guide.md"
QUEUE_FILE = "review-queue.jsonl"
MANIFEST_FILE = "manifest.json"
MIN_TOTAL, MAX_TOTAL = 120, 180
TARGET_WEAK, TARGET_GENERAL = 70, 80

# Weak strata of human-value-01 batch 2 (shortfall against its targets) the weak section aims at.
WEAK_TARGETS = {
    "bare_number": 23,
    "multi_number": 11,
    "quantity_amount": 11,
    "unusual_whitespace_punct": 8,
    "installment_index_amount": 7,
    "null_no_number_ambiguous": 6,
    "date_month_year_amount": 5,
}

# Leakage reference groups in priority order; a drop is counted under the first that fires.
# Sets newer than training-v3 are not references here: they must gate against this set instead.
GROUPS = (
    ("human-value-01", ("datasets/annotation-v2/human-value-01",)),
    (
        "frozen-test",
        (
            "datasets/annotation-v2/training-v1/test.jsonl",
            "datasets/annotation-v2/training-v2-direction-repair/test.jsonl",
            "datasets/annotation-v1/splits/test.jsonl",
            "datasets/annotation-v1/distillation-v1/test.jsonl",
            "datasets/annotation-v1/training-v2/test.jsonl",
            "datasets/annotation-v1/training-v3/test.jsonl",
            "datasets/annotation-v3/training-v1/test.jsonl",
            "datasets/annotation-v3/training-v2/test.jsonl",
            "datasets/annotation-v3/training-v3/test.jsonl",
        ),
    ),
    (
        "probe-v1",
        (
            "datasets/probe-v1",
            "datasets/annotation-v2/training-v1/probe-v1-eval-only.jsonl",
            "datasets/annotation-v2/training-v2-direction-repair/probe-v1-eval-only.jsonl",
        ),
    ),
    (
        "annotation-v3",
        (
            "datasets/annotation-v3/debt-01",
            "datasets/annotation-v3/contrast-01",
            "datasets/annotation-v3/contrast-02",
            "datasets/annotation-v3/training-v1",
            "datasets/annotation-v3/training-v2",
            "datasets/annotation-v3/training-v3",
        ),
    ),
    ("annotation-v1-v2", ("datasets/annotation-v1", "datasets/annotation-v2")),
    ("corpus", ("corpus",)),
)


def _nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def note_id(text: str) -> str:
    return f"hv02-{_sha256(_nfc(text).encode('utf-8'))[:12]}"


def order_key(record_id: str) -> str:
    return _sha256(f"{SEED}:{record_id}".encode())


def _dump_jsonl(rows: list[dict[str, Any]]) -> bytes:
    return "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows).encode("utf-8")


def read_candidates(path: Path) -> list[tuple[str, str]]:
    """``(section, NFC text)`` per line; blank lines and duplicate lines are errors."""
    out: list[tuple[str, str]] = []
    section = None
    seen: set[str] = set()
    for raw in path.read_text("utf-8").splitlines():
        if raw.startswith("## "):
            section = raw[3:].strip()
            continue
        if not raw.strip():
            raise ValueError(f"{path.name}: blank line")
        if section not in {"weak", "general"}:
            raise ValueError(f"{path.name}: note before a '## weak' / '## general' header")
        text = _nfc(raw)
        if text != text.strip():
            raise ValueError(f"{path.name}: leading/trailing whitespace in {raw!r}")
        if text.lower() in seen:
            raise ValueError(f"{path.name}: duplicate note {raw!r}")
        seen.add(text.lower())
        out.append((section, text))
    return out


def _texts(path: Path) -> list[str]:
    texts: list[str] = []
    for line in path.read_text("utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            return []
        if isinstance(row, dict) and isinstance(row.get("text"), str):
            texts.append(row["text"])
    return texts


def load_reference_groups(
    root: Path, out_dir: Path
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """Texts and file names per leakage group (every ``*.jsonl``; sidecar databases are skipped)."""
    own = (root / out_dir).resolve()
    texts: dict[str, list[str]] = {name: [] for name, _ in GROUPS}
    files: dict[str, list[str]] = {name: [] for name, _ in GROUPS}
    claimed: set[Path] = set()
    for name, entries in GROUPS:
        for rel in entries:
            base = root / rel
            paths = [base] if base.is_file() else sorted(base.rglob("*.jsonl"))
            for path in paths:
                resolved = path.resolve()
                if resolved in claimed or own in resolved.parents:
                    continue
                claimed.add(resolved)
                found = _texts(path)
                if found:
                    texts[name] += found
                    files[name].append(str(path.relative_to(root)))
    return texts, files


def leakage_filter(
    candidates: list[tuple[str, str]], groups: dict[str, list[str]]
) -> tuple[list[tuple[str, str]], list[dict[str, str]], dict[str, int]]:
    sets = {name: hv.ReferenceSet(texts) for name, texts in groups.items()}
    intra = hv.ReferenceSet([])
    kept: list[tuple[str, str]] = []
    dropped: list[dict[str, str]] = []
    for section, text in candidates:
        ref = hv.Reference(text)
        hit: tuple[str, str] | None = None
        for name, _ in GROUPS:
            rule = sets[name].match(ref)
            if rule:
                hit = (name, rule)
                break
        if hit is None:
            rule = intra.match(ref)
            if rule:
                hit = ("earlier-candidate", rule)
        if hit:
            dropped.append({"text": text, "section": section, "group": hit[0], "rule": hit[1]})
        else:
            kept.append((section, text))
            intra.add_ref(ref)
    return kept, dropped, {name: len(s.refs) for name, s in sets.items()}


def has_diacritics(text: str) -> bool:
    nfd = unicodedata.normalize("NFD", text.lower())
    return "đ" in nfd or any(unicodedata.category(c) == "Mn" for c in nfd)


def select(kept: list[tuple[str, str]]) -> tuple[list[tuple[str, str]], list[str]]:
    """Seeded pick of ~150 survivors: the weak section fills ``WEAK_TARGETS`` first (a note that
    adds to a stratum still under its target), then up to ``TARGET_WEAK`` weak notes and
    ``TARGET_GENERAL`` general notes, always in seeded-hash order. Returns ``(chosen, surplus)``.
    """
    ordered = sorted(kept, key=lambda c: order_key(note_id(c[1])))
    weak = [c for c in ordered if c[0] == "weak"]
    general = [c for c in ordered if c[0] == "general"]
    chosen_weak: list[tuple[str, str]] = []
    have: Counter[str] = Counter()
    for cand in weak:
        strata, _ = hv.assign_strata(cand[1])
        if any(have[s] < WEAK_TARGETS.get(s, 0) for s in strata):
            chosen_weak.append(cand)
            have.update(strata)
    for cand in weak:
        if len(chosen_weak) >= TARGET_WEAK:
            break
        if cand not in chosen_weak:
            chosen_weak.append(cand)
    chosen = chosen_weak + general[:TARGET_GENERAL]
    taken = {c[1] for c in chosen}
    return chosen, sorted(c[1] for c in ordered if c[1] not in taken)


def build(root: Path, out_dir: Path) -> dict[str, bytes]:
    """All generated files as ``{file name: bytes}`` (nothing is written)."""
    base = root / out_dir
    candidates = read_candidates(base / CANDIDATES_FILE)
    texts, files = load_reference_groups(root, out_dir)
    survivors, dropped, ref_sizes = leakage_filter(candidates, texts)
    kept, surplus = select(survivors)
    if not MIN_TOTAL <= len(kept) <= MAX_TOTAL:
        raise SystemExit(f"{len(kept)} notes survive the gate, outside {MIN_TOTAL}-{MAX_TOTAL}")

    rows: list[dict[str, Any]] = []
    section_of: dict[str, str] = {}
    for section, text in kept:
        strata, _ = hv.assign_strata(text)
        rid = note_id(text)
        section_of[rid] = section
        rows.append(
            {
                "id": rid,
                "text": text,
                "strata": strata or [FILL_STRATUM],
                "review_group": REVIEW_GROUP,
            }
        )
    if len({r["id"] for r in rows}) != len(rows):
        raise SystemExit("duplicate queue ids")
    rows.sort(key=lambda r: order_key(r["id"]))
    queue_bytes = _dump_jsonl(rows)

    strata_counts = Counter(s for r in rows for s in r["strata"])
    guide = base / GUIDE_FILE
    manifest = {
        "name": NAME,
        "annotation_version": "annotation-v3",
        "role": "independent human-labelled TEST set: type + target + value in one combined Quet "
        "pass; scores V8 (encoder 1 + rule value parser) end to end under annotation-v3",
        "never_train": True,
        "labels_by": "the user, in Quet (annotation-v3 combined pass); no proposals, no model or "
        "rule labels",
        "provenance": {
            "source": "llm-written",
            "writer": "an LLM (the assistant), by hand, from docs/annotation-v1.md, "
            "docs/annotation-v3.md and the reviewed-corpus style only; no training-data file and "
            "no human-value-01 label was read when writing",
            "candidates_file": CANDIDATES_FILE,
            "sections": {"weak": "weak strata of human-value-01 batch 2", "general": "all types"},
        },
        "seed": SEED,
        "id_rule": "hv02-<sha256(NFC text)[:12]>; order = sha256(seed:id)",
        "counts": {
            "candidates": len(candidates),
            "queue": len(rows),
            "dropped_by_leakage_gate": len(dropped),
            "survived_gate": len(survivors),
            "not_selected_surplus": len(surplus),
            "queue_by_section": dict(sorted(Counter(section_of.values()).items())),
            "no_diacritics_share": round(
                sum(1 for r in rows if not has_diacritics(r["text"])) / len(rows), 3
            ),
        },
        "leakage": {
            "rules": {
                "exact": "NFC + strip + lowercase",
                "folded": "NFC, lowercase, accents stripped, digits masked, whitespace collapsed",
                "sequence": f"difflib ratio of folded texts >= {hv.SEQ_THRESHOLD}",
                "token": f"folded token Jaccard >= {hv.TOKEN_THRESHOLD} (>= {hv.TOKEN_MIN} tokens)",
                "char3": f"folded char-3-gram Jaccard >= {hv.CHAR3_THRESHOLD}",
            },
            "reference_groups_in_priority_order": [name for name, _ in GROUPS],
            "reference_unique_texts": ref_sizes,
            "reference_files": {name: len(f) for name, f in files.items()},
            "dropped_by_group": dict(Counter(d["group"] for d in dropped)),
            "dropped_by_rule": dict(Counter(d["rule"] for d in dropped)),
            "dropped": dropped,
            "surplus_not_selected": surplus,
            "note": "sets newer than training-v3 are not references here; they must gate against "
            "this queue",
        },
        "strata": dict(sorted(strata_counts.items())),
        "weak_stratum_targets": WEAK_TARGETS,
        "outputs": {
            QUEUE_FILE: _sha256(queue_bytes),
            **({GUIDE_FILE: _sha256(guide.read_bytes())} if guide.exists() else {}),
        },
        "inputs": {CANDIDATES_FILE: _sha256((base / CANDIDATES_FILE).read_bytes())},
        "queue_sha256": _sha256(queue_bytes),
        "proposals": None,
        "quet_schema": {"path": str(SCHEMA), "sha256": _sha256((root / SCHEMA).read_bytes())},
        "quet_command": (
            f"quet annotate {out_dir / QUEUE_FILE} --schema {SCHEMA} "
            f"--out {out_dir / 'labels.jsonl'}"
        ),
        "quet_web_push": (
            f"quet web push {out_dir / QUEUE_FILE} --schema {SCHEMA} --project {QUET_PROJECT} "
            f'--name "{QUET_NAME} ({len(rows)})"'
        ),
        "validate_command": (
            f"uv run python scripts/validate_annotations.py {out_dir / 'labels.jsonl'} "
            f"--config {SCHEMA} --queue {out_dir / QUEUE_FILE}"
        ),
    }
    return {
        QUEUE_FILE: queue_bytes,
        MANIFEST_FILE: (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8"),
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
    if (out / "labels.jsonl").exists() and (out / QUEUE_FILE).read_bytes() != files[QUEUE_FILE]:
        print("labels.jsonl exists and the queue would change; refusing", file=sys.stderr)
        return 1
    for name, data in files.items():
        (out / name).write_bytes(data)
    manifest = json.loads(files[MANIFEST_FILE])
    summary = {k: manifest[k] for k in ("counts", "strata")}
    summary["dropped"] = [
        (d["text"], d["group"], d["rule"]) for d in manifest["leakage"]["dropped"]
    ]
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
