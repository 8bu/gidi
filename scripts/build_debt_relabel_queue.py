#!/usr/bin/env python
r"""Build the annotation-v3 debt-only relabel batch ``datasets/annotation-v3/debt-01``.

Usage:
    uv run python scripts/build_debt_relabel_queue.py [--root R] [--out-dir D] [--check]

annotation-v3 changes one rule: a debt-only note (an existing debt, no money moves now) is
``borrow`` when the user owes and ``lend`` when the other party owes the user, not ``skipped``.
This batch is the human re-label pass for that rule. It holds:

* the 14 annotation-v1 ``skipped`` notes that are NOT in the frozen held-out set
  ``datasets/annotation-v2/human-value-01`` (the other 9 are excluded everywhere, by id), source
  ``annotation-v1-skipped``, v1 ids kept;
* the hand-written debt-only notes of ``generated.jsonl`` (the generator is the author, not a
  model call) that survive the leakage filter, source ``debt-01-generated``. Ids are
  ``debt-01-<sha256(NFC text)[:12]>``.

Inputs written by hand (never by this script): ``generated.jsonl`` (``text`` plus the proposed
label and a Vietnamese ``reason``) and ``v1-skipped-relabel.jsonl`` (proposed label per kept v1
id). Outputs, all deterministic:

* ``queue.jsonl``       ``{id, text, source}``, ordered by sha256(``SEED:id``);
* ``proposals.jsonl``   one suggestion per queue note (``reason`` starts "Đề xuất từ LLM:"),
                        validated against ``configs/annotation-v3.quet.yaml``; a human labels in
                        Quet, proposals are advisory only;
* ``provenance.jsonl``  per queue note: source, v1 id/status;
* ``manifest.json``     counts, leakage report, excluded ids, sha256 of every file.

Leakage: a generated note is dropped when it is an exact/near duplicate (the rules of
``scripts/build_human_value_queue.py``: exact, folded, sequence >= 0.90, token, char3) of any note
in human-value-01, the frozen test split, probe-v1, the 14 kept v1 notes, the existing corpus
(every other ``text`` under ``corpus/`` and ``datasets/``) or an earlier generated note. Each drop
is counted once under the first group that fires. In a labelling pass of this batch ``skipped``
means "reject this note" (it is not a debt-only note after all). ``--check`` rebuilds in memory
and compares the files on disk byte for byte.
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

NAME = "debt-01"
SEED = "debt-01:v1"
DEFAULT_OUT_DIR = Path("datasets/annotation-v3/debt-01")
# Sets built from this batch (they hold its texts) or derived from it: not leakage references.
DERIVED_DIRS = (
    Path("datasets/annotation-v3/training-v1"),
    Path("datasets/annotation-v3/contrast-01"),
    Path("datasets/annotation-v3/training-v2"),
    Path("datasets/annotation-v3/contrast-02"),
    Path("datasets/annotation-v3/contrast-03"),
    Path("datasets/annotation-v3/training-v4"),
    Path("datasets/annotation-v3/gift-relabel-01"),
    Path("datasets/annotation-v3/contrast-04"),
    Path("datasets/annotation-v3/training-v5"),
    # human-value-02 is a test set: the new builders gate it as an explicit group, so earlier
    # batches (frozen before it existed) keep rebuilding byte for byte.
    Path("datasets/annotation-v3/human-value-02"),
    Path("datasets/annotation-v3/training-v3"),
)
SCHEMA = Path("configs/annotation-v3.quet.yaml")
CONTRACT = Path("configs/annotation-v3.yaml")
DOC = Path("docs/annotation-v3.md")
QUET_PROJECT = "gidi-annotation-v3-debt-01"

V1_QUEUE = Path("datasets/annotation-v1/combined/queue.jsonl")
V1_LABELS = Path("datasets/annotation-v1/combined/labels.jsonl")
HUMAN_QUEUE = Path("datasets/annotation-v2/human-value-01/review-queue.jsonl")

GENERATED_FILE = "generated.jsonl"
RELABEL_FILE = "v1-skipped-relabel.jsonl"
QUEUE_FILE = "queue.jsonl"
PROPOSALS_FILE = "proposals.jsonl"
PROVENANCE_FILE = "provenance.jsonl"
MANIFEST_FILE = "manifest.json"
HAND_FILES = (GENERATED_FILE, RELABEL_FILE)
HASHED_FILES = (*HAND_FILES, QUEUE_FILE, PROPOSALS_FILE, PROVENANCE_FILE)

SRC_V1 = "annotation-v1-skipped"
SRC_GEN = "debt-01-generated"
REASON_PREFIX = "Đề xuất từ LLM:"

# Leakage reference groups in priority order; a drop is counted under the first that fires.
HUMAN_FILES = (
    "datasets/annotation-v2/human-value-01/review-queue.jsonl",
    "datasets/annotation-v2/human-value-01/recheck-01-queue.jsonl",
    "datasets/annotation-v2/human-value-01/review-02-queue.jsonl",
)
TEST_FILES = (
    "datasets/annotation-v2/training-v1/test.jsonl",
    "datasets/annotation-v1/splits/test.jsonl",
)
PROBE_FILES = (
    "datasets/annotation-v2/training-v1/probe-v1-eval-only.jsonl",
    "datasets/probe-v1/notes.jsonl",
    "datasets/probe-v1/queue.jsonl",
)
CORPUS_DIRS = ("corpus", "datasets")
GROUPS = ("human-value-01", "test-split", "probe-v1", "v1-skipped-kept", "existing-corpus")


def _load_hv() -> Any:
    """The near-duplicate rules of the human-value queue builder (a script, not a package)."""
    path = Path(__file__).resolve().with_name("build_human_value_queue.py")
    spec = importlib.util.spec_from_file_location("build_human_value_queue", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("build_human_value_queue", module)
    spec.loader.exec_module(module)
    return module


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text("utf-8").splitlines() if line.strip()]


def _dump(rows: list[dict[str, Any]]) -> str:
    return "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _guide_bytes(root: Path) -> bytes:
    """``docs/annotation-v3.md`` up to the section after this batch's own (later sections of the
    document, e.g. the human-value-02 test set, do not belong to the debt-01 guide)."""
    text = (root / DOC).read_text("utf-8")
    own = text.find("\n## Re-label batch `debt-01`")
    end = text.find("\n## ", own + 1) if own >= 0 else -1
    return (text if end < 0 else text[:end]).encode("utf-8")


def _nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def note_id(text: str) -> str:
    return f"{NAME}-{_sha256(_nfc(text).encode('utf-8'))[:12]}"


def order_key(record_id: str) -> str:
    return _sha256(f"{SEED}:{record_id}".encode())


def _texts(path: Path) -> list[str]:
    try:
        rows = _read_jsonl(path)
    except (OSError, json.JSONDecodeError):
        return []
    return [r["text"] for r in rows if isinstance(r, dict) and isinstance(r.get("text"), str)]


def load_reference_groups(root: Path, out_dir: Path, v1_kept: list[str]) -> dict[str, list[str]]:
    """Texts per leakage group; the v3 batch itself and sidecar databases are not references."""
    groups: dict[str, list[str]] = {name: [] for name in GROUPS}
    claimed: set[str] = set()
    for name, files in (
        ("human-value-01", HUMAN_FILES),
        ("test-split", TEST_FILES),
        ("probe-v1", PROBE_FILES),
    ):
        for rel in files:
            groups[name] += _texts(root / rel)
            claimed.add(str((root / rel).resolve()))
    groups["v1-skipped-kept"] = list(v1_kept)
    # The batch itself and the training set derived from it (it holds this batch's texts).
    own = {out_dir.resolve(), *((root / p).resolve() for p in DERIVED_DIRS)}
    for top in CORPUS_DIRS:
        for path in sorted((root / top).rglob("*.jsonl")):
            resolved = path.resolve()
            if str(resolved) in claimed or own & set(path.absolute().parents) | own & set(
                resolved.parents
            ):
                continue
            groups["existing-corpus"] += _texts(path)
    return groups


def leakage_filter(
    hv: Any, candidates: list[str], groups: dict[str, list[str]]
) -> tuple[list[str], list[dict[str, str]], dict[str, int]]:
    """Keep the candidates that are not (near) duplicates of any reference or earlier kept note."""
    sets = {name: hv.ReferenceSet(texts) for name, texts in groups.items()}
    intra = hv.ReferenceSet([])
    kept: list[str] = []
    dropped: list[dict[str, str]] = []
    for text in candidates:
        ref = hv.Reference(text)
        hit: tuple[str, str] | None = None
        for name in GROUPS:
            rule = sets[name].match(ref)
            if rule:
                hit = (name, rule)
                break
        if hit is None:
            rule = intra.match(ref)
            if rule:
                hit = ("earlier-generated", rule)
        if hit:
            dropped.append({"text": text, "group": hit[0], "rule": hit[1]})
        else:
            kept.append(text)
            intra.add_ref(ref)
    sizes = {name: len(ref_set.refs) for name, ref_set in sets.items()}
    return kept, dropped, sizes


def span(text: str, piece: str | None, what: str) -> dict[str, Any] | None:
    """``{text, start, end}`` of the first occurrence of ``piece`` (code-point offsets)."""
    if piece is None:
        return None
    start = text.find(piece)
    if start < 0:
        raise ValueError(f"{what} {piece!r} not found in {text!r}")
    return {"text": piece, "start": start, "end": start + len(piece)}


def proposal(record_id: str, text: str, hand: dict[str, Any]) -> dict[str, Any]:
    reason = hand["reason"]
    if not reason.startswith(REASON_PREFIX):
        raise ValueError(f"{record_id}: reason must start with {REASON_PREFIX!r}")
    out: dict[str, Any] = {
        "id": record_id,
        "annotation_status": hand["annotation_status"],
        "type": hand["type"],
        "target": span(text, hand.get("target"), "target"),
        "value": span(text, hand.get("value"), "value"),
        "span_status": {"value": "complete"},
    }
    if hand.get("note"):
        out["note"] = hand["note"]
    out["reason"] = reason
    return out


def build(root: Path, out_dir: Path) -> dict[str, str]:
    """All output files of the batch as ``{file name: content}`` (nothing is written)."""
    hv = _load_hv()
    v1_texts = {r["id"]: r["text"] for r in _read_jsonl(root / V1_QUEUE)}
    v1_skipped = [
        r["id"] for r in _read_jsonl(root / V1_LABELS) if r["annotation_status"] == "skipped"
    ]
    human_ids = {r["id"] for r in _read_jsonl(root / HUMAN_QUEUE)}
    excluded = sorted(i for i in v1_skipped if i in human_ids)
    v1_kept = [i for i in v1_skipped if i not in human_ids]

    relabel = {r["id"]: r for r in _read_jsonl(out_dir / RELABEL_FILE)}
    if set(relabel) != set(v1_kept):
        raise ValueError(f"{RELABEL_FILE}: ids must be exactly the {len(v1_kept)} kept v1 ids")
    generated = _read_jsonl(out_dir / GENERATED_FILE)
    by_text = {_nfc(row["text"]): row for row in generated}
    if len(by_text) != len(generated):
        raise ValueError(f"{GENERATED_FILE}: duplicate or non-distinct texts")
    for text in by_text:
        if text != text.strip() or not text:
            raise ValueError(f"{GENERATED_FILE}: bad text {text!r}")

    groups = load_reference_groups(root, out_dir, [v1_texts[i] for i in v1_kept])
    kept, dropped, ref_sizes = leakage_filter(hv, list(by_text), groups)

    queue: list[dict[str, str]] = []
    proposals: dict[str, dict[str, Any]] = {}
    provenance: dict[str, dict[str, Any]] = {}
    for rid in v1_kept:
        text = _nfc(v1_texts[rid])
        queue.append({"id": rid, "text": text, "source": SRC_V1})
        proposals[rid] = proposal(rid, text, relabel[rid])
        provenance[rid] = {"id": rid, "source": SRC_V1, "v1_status": "skipped", "generator": None}
    for text in kept:
        rid = note_id(text)
        queue.append({"id": rid, "text": text, "source": SRC_GEN})
        proposals[rid] = proposal(rid, text, by_text[text])
        provenance[rid] = {
            "id": rid,
            "source": SRC_GEN,
            "v1_status": None,
            "generator": "hand-written by the assistant (generated.jsonl)",
        }
    if len({q["id"] for q in queue}) != len(queue):
        raise ValueError("duplicate queue ids")
    queue.sort(key=lambda q: order_key(q["id"]))
    ordered_ids = [q["id"] for q in queue]
    problems = validate_proposals(root, queue, [proposals[i] for i in ordered_ids])
    if problems:
        raise ValueError("proposals invalid:\n" + "\n".join(problems))

    files = {
        QUEUE_FILE: _dump(queue),
        PROPOSALS_FILE: _dump([proposals[i] for i in ordered_ids]),
        PROVENANCE_FILE: _dump([provenance[i] for i in ordered_ids]),
    }
    manifest = build_manifest(
        root,
        out_dir,
        files,
        queue=queue,
        proposals=[proposals[i] for i in ordered_ids],
        generated=len(by_text),
        dropped=dropped,
        ref_sizes=ref_sizes,
        excluded=excluded,
        v1_kept=len(v1_kept),
    )
    files[MANIFEST_FILE] = json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    return files


def validate_proposals(
    root: Path, queue: list[dict[str, str]], proposals: list[dict[str, Any]]
) -> list[str]:
    """Contract violations of the proposals (without ``reason``) against the v3 Quet schema."""
    from gidi.annotation.combined import load_combined_config, validate_combined_label

    config = load_combined_config(root / SCHEMA)
    texts = {q["id"]: q["text"] for q in queue}
    problems = []
    for prop in proposals:
        label = {k: v for k, v in prop.items() if k != "reason"}
        problems += [
            f"{prop['id']}: {p}" for p in validate_combined_label(label, texts[prop["id"]], config)
        ]
    return problems


def build_manifest(
    root: Path,
    out_dir: Path,
    files: dict[str, str],
    *,
    queue: list[dict[str, str]],
    proposals: list[dict[str, Any]],
    generated: int,
    dropped: list[dict[str, str]],
    ref_sizes: dict[str, int],
    excluded: list[str],
    v1_kept: int,
) -> dict[str, Any]:
    sha: dict[str, str] = {}
    for name in HASHED_FILES:
        data = files[name].encode("utf-8") if name in files else (out_dir / name).read_bytes()
        sha[name] = _sha256(data)
    new_kept = sum(q["source"] == SRC_GEN for q in queue)
    return {
        "name": NAME,
        "annotation_version": "annotation-v3",
        "role": "re-label of debt-only notes under the annotation-v3 rule: borrow when the user "
        "owes, lend when the other party owes the user (annotation-v1/v2 labelled them skipped)",
        "seed": SEED,
        "schema": str(SCHEMA),
        "schema_sha256": _sha256((root / SCHEMA).read_bytes()),
        "contract": str(CONTRACT),
        "contract_sha256": _sha256((root / CONTRACT).read_bytes()),
        "guide": str(DOC),
        "guide_sha256": _sha256(_guide_bytes(root)),
        "counts": {
            "queue": len(queue),
            "old_v1_skipped": v1_kept,
            "new_generated": new_kept,
            "generated_written": generated,
            "generated_dropped": len(dropped),
            "proposals": len(proposals),
            "proposal_type": dict(sorted(Counter(p["type"] or "null" for p in proposals).items())),
            "proposal_status": dict(
                sorted(Counter(p["annotation_status"] for p in proposals).items())
            ),
            "proposal_value_null": sum(p["value"] is None for p in proposals),
            "proposal_target_null": sum(p["target"] is None for p in proposals),
            "source": dict(sorted(Counter(q["source"] for q in queue).items())),
        },
        "leakage": {
            "rules": {
                "exact": "NFC + strip + lowercase",
                "folded": "NFC, lowercase, accents stripped, digits masked, whitespace collapsed",
                "sequence": "difflib ratio of folded texts >= 0.9",
                "token": "folded token Jaccard >= 0.8 (>= 4 tokens)",
                "char3": "folded char-3-gram Jaccard >= 0.8",
                "source": "scripts/build_human_value_queue.py",
            },
            "group_order": [*GROUPS, "earlier-generated"],
            "reference_texts": ref_sizes,
            "dropped_by_group": dict(sorted(Counter(d["group"] for d in dropped).items())),
            "dropped_by_rule": dict(sorted(Counter(d["rule"] for d in dropped).items())),
            "dropped": dropped,
        },
        "excluded_human_value_01_ids": excluded,
        "excluded_note": "the 9 annotation-v1 skipped notes inside the frozen held-out set "
        "datasets/annotation-v2/human-value-01: never relabelled, never trained on",
        "labelling_instruction": {
            "skipped": "in this pass `skipped` means 'reject this note': it is not a debt-only "
            "note after all and is dropped from the batch",
            "uncertain": "debt-only note whose direction (who owes whom) is unclear; needs a note",
            "proposals": "advisory (LLM) suggestions; only the reviewer's decision counts",
        },
        "quet_web": {
            "project": QUET_PROJECT,
            "name": f"gidi annotation-v3 · debt-only notes ({len(queue)})",
            "collaborator": "8bu",
            "show_proposals": True,
        },
        "files_sha256": sha,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--check", action="store_true", help="compare, do not write")
    args = parser.parse_args(argv)
    out_dir = args.root / args.out_dir
    files = build(args.root, out_dir)
    if args.check:
        stale = [
            name
            for name, content in files.items()
            if not (out_dir / name).is_file() or (out_dir / name).read_text("utf-8") != content
        ]
        if stale:
            print(f"STALE: {', '.join(stale)}", file=sys.stderr)
            return 1
        print(f"ok: {len(files)} files match ({NAME})")
        return 0
    for name, content in files.items():
        (out_dir / name).write_text(content, encoding="utf-8")
    manifest = json.loads(files[MANIFEST_FILE])
    print(json.dumps(manifest["counts"], ensure_ascii=False, indent=2))
    print("leakage dropped:", json.dumps(manifest["leakage"]["dropped_by_group"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
