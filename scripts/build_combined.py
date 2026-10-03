#!/usr/bin/env python
"""Rebuild (or verify) the frozen combined annotation-v1 dataset from its canonical sources.

Usage:
    uv run python scripts/build_combined.py [--overwrite]
    uv run python scripts/build_combined.py --check

Concatenates the baseline batch (``datasets/annotation-v1/{queue,labels,provenance}.jsonl``) and
the targeted batch (``datasets/annotation-v1/targeted-01/...``), in queue order, into
``datasets/annotation-v1/combined/{queue,labels,provenance}.jsonl`` and writes
``combined/manifest.json`` (counts and sha256 of every input and output; no timestamps, so it is
byte-reproducible). Labels are copied verbatim; nothing under the sources is ever modified.

The build fails on duplicate ids, queue/labels/provenance id mismatch, or any validator error.
Existing outputs are refused unless ``--overwrite`` is given. ``--check`` recomputes everything
and fails if the files on disk differ from the recomputation (freeze verification).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gidi.annotation.schema import AnnotationConfig, load_config, load_texts_by_id, validate_file
from gidi.annotation.split import accented

ANNOTATION_DIR = Path("datasets/annotation-v1")
CONFIG = Path("configs/annotation-v1.yaml")
DOC = Path("docs/annotation-v1.md")
KINDS = ("queue", "labels", "provenance")
ANNOTATORS = ("human", "ai")
MANIFEST = "manifest.json"


@dataclass(frozen=True)
class Source:
    name: str  # the ``source_batch`` value
    directory: Path  # directory holding queue/labels/provenance.jsonl

    def path(self, kind: str) -> Path:
        return self.directory / f"{kind}.jsonl"


def default_sources() -> list[Source]:
    return [
        Source("baseline-01", ANNOTATION_DIR),
        Source("targeted-annotation-v1-01", ANNOTATION_DIR / "targeted-01"),
    ]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _lines(path: Path) -> list[str]:
    return [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def _relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _validate(source: Source, config: AnnotationConfig) -> list[str]:
    texts = load_texts_by_id(source.path("queue"))
    report = validate_file(source.path("labels"), texts, config)
    return [f"{source.name}: {issue}" for issue in report.errors]


def build_contents(
    sources: list[Source], config_path: Path, doc_path: Path, root: Path
) -> dict[str, str]:
    """Return ``{filename: text}`` for the combined queue/labels/provenance and the manifest.

    Raises ``ValueError`` on any inconsistency.
    """
    config = load_config(config_path)
    errors: list[str] = []
    queue_out: list[str] = []
    labels_out: list[str] = []
    prov_out: list[str] = []
    seen: dict[str, str] = {}
    batches: dict[str, dict[str, Any]] = {}
    complete_rows: list[tuple[str, str, str, str]] = []  # batch, type, accent flag, annotator

    for source in sources:
        errors.extend(_validate(source, config))
        queue = [json.loads(ln) for ln in _lines(source.path("queue"))]
        label_lines = _lines(source.path("labels"))
        labels = [json.loads(ln) for ln in label_lines]
        provenance = [json.loads(ln) for ln in _lines(source.path("provenance"))]
        q_ids = [q["id"] for q in queue]
        for kind, ids in (
            ("labels", [r["id"] for r in labels]),
            ("provenance", [r["id"] for r in provenance]),
        ):
            if len(set(ids)) != len(ids) or set(ids) != set(q_ids):
                errors.append(f"{source.name}: {kind} ids differ from the queue ids")
        for rid in q_ids:
            if rid in seen:
                errors.append(f"duplicate id {rid} in {source.name} (first in {seen[rid]})")
            seen[rid] = source.name
        if errors:
            continue

        text_of = {q["id"]: q["text"] for q in queue}
        prov_of = {p["id"]: p for p in provenance}
        stats: dict[str, Any] = {
            "records": len(queue),
            "by_status": Counter(),
            "trainable": 0,
            "annotators": Counter(),
        }
        for q in queue:
            if q.get("source_batch", source.name) != source.name:
                errors.append(
                    f"{q['id']}: queue source_batch {q['source_batch']!r} != {source.name}"
                )
            queue_out.append(json.dumps({**q, "source_batch": source.name}, ensure_ascii=False))
        # Quet saves labels in annotation order, not queue order: labels and provenance keep the
        # order of the source labels file (line-aligned), so labels are copied verbatim.
        for label_line, label in zip(label_lines, labels, strict=True):
            p_out = dict(prov_of[label["id"]])
            if p_out.setdefault("source_batch", source.name) != source.name:
                errors.append(f"{label['id']}: provenance source_batch != {source.name}")
            annotator = p_out.get("annotator")
            if annotator not in ANNOTATORS:
                errors.append(f"{label['id']}: annotator {annotator!r} not in {ANNOTATORS}")
            labels_out.append(label_line)
            prov_out.append(json.dumps(p_out, ensure_ascii=False, separators=(",", ":")))
            status = label["annotation_status"]
            stats["by_status"][status] += 1
            stats["annotators"][annotator] += 1
            if status in config.trainable_statuses:
                stats["trainable"] += 1
                complete_rows.append(
                    (
                        source.name,
                        label["type"],
                        "accented" if accented(text_of[label["id"]]) else "unaccented",
                        str(annotator),
                    )
                )
        batches[source.name] = stats

    if errors:
        raise ValueError("\n".join(errors))

    contents = {
        "queue.jsonl": "".join(f"{ln}\n" for ln in queue_out),
        "labels.jsonl": "".join(f"{ln}\n" for ln in labels_out),
        "provenance.jsonl": "".join(f"{ln}\n" for ln in prov_out),
    }
    manifest = _manifest(
        config, sources, batches, complete_rows, contents, config_path, doc_path, root
    )
    contents[MANIFEST] = json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    return contents


def _manifest(
    config: AnnotationConfig,
    sources: list[Source],
    batches: dict[str, dict[str, Any]],
    complete_rows: list[tuple[str, str, str, str]],
    contents: dict[str, str],
    config_path: Path,
    doc_path: Path,
    root: Path,
) -> dict[str, Any]:
    by_status: Counter[str] = Counter()
    annotators: Counter[str] = Counter()
    for stats in batches.values():
        by_status.update(stats["by_status"])
        annotators.update(stats["annotators"])

    types = Counter(t for _, t, _, _ in complete_rows)
    accents = Counter(a for _, _, a, _ in complete_rows)

    def accent_counts(rows: list[tuple[str, str, str, str]]) -> dict[str, int]:
        tally = Counter(a for _, _, a, _ in rows)
        return {"accented": tally["accented"], "unaccented": tally["unaccented"]}

    per_batch = {}
    for name, stats in batches.items():
        rows = [r for r in complete_rows if r[0] == name]
        per_batch[name] = {
            "records": stats["records"],
            "by_status": dict(sorted(stats["by_status"].items())),
            "trainable": stats["trainable"],
            "human_reviewed": stats["annotators"]["human"],
            "ai_accepted": stats["annotators"]["ai"],
            "types": {t: sum(1 for r in rows if r[1] == t) for t in config.types},
            "accent": accent_counts(rows),
        }

    def sha(path: Path) -> str:
        return sha256_file(path)

    sources_sha = {_relative(s.path(k), root): sha(s.path(k)) for s in sources for k in KINDS}
    canonical_sha = {
        f"combined/{name}": hashlib.sha256(text.encode("utf-8")).hexdigest()
        for name, text in contents.items()
    }
    return {
        "annotation_version": config.version,
        "frozen": True,
        "records": {
            "total": sum(by_status.values()),
            "by_status": dict(sorted(by_status.items())),
            "trainable": sum(s["trainable"] for s in batches.values()),
        },
        "types": {t: types[t] for t in config.types},
        "accent": {
            "overall": {"accented": accents["accented"], "unaccented": accents["unaccented"]},
            "by_type": {
                t: accent_counts([r for r in complete_rows if r[1] == t]) for t in config.types
            },
        },
        "annotators": {"human_reviewed": annotators["human"], "ai_accepted": annotators["ai"]},
        "source_batches": per_batch,
        "sha256": {
            "combined": canonical_sha,
            "sources": sources_sha,
            "config": {_relative(config_path, root): sha(config_path)},
            "docs": {_relative(doc_path, root): sha(doc_path)},
        },
    }


def write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def check(out_dir: Path, contents: dict[str, str]) -> list[str]:
    """Names of the files whose on-disk bytes differ from the recomputation."""
    stale = []
    for name, text in contents.items():
        path = out_dir / name
        if not path.exists() or path.read_bytes() != text.encode("utf-8"):
            stale.append(name)
    return stale


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=Path("."), help="repository root")
    parser.add_argument("--out-dir", type=Path, default=ANNOTATION_DIR / "combined")
    parser.add_argument("--check", action="store_true", help="verify the freeze; write nothing")
    parser.add_argument("--overwrite", action="store_true", help="replace existing outputs")
    args = parser.parse_args(argv)

    root: Path = args.root
    sources = [Source(s.name, root / s.directory) for s in default_sources()]
    out_dir = args.out_dir if args.out_dir.is_absolute() else root / args.out_dir
    try:
        contents = build_contents(sources, root / CONFIG, root / DOC, root)
    except ValueError as exc:
        raise SystemExit(f"build failed:\n{exc}") from exc

    if args.check:
        stale = check(out_dir, contents)
        if stale:
            raise SystemExit(f"freeze check FAILED: differs from recomputation: {', '.join(stale)}")
        total = json.loads(contents[MANIFEST])["records"]["total"]
        print(f"freeze check OK: {total} records; {len(contents)} files match the recomputation")
        return 0

    existing = [out_dir / n for n in contents if (out_dir / n).exists()]
    if existing and not args.overwrite:
        raise SystemExit(f"refusing to overwrite existing file: {existing[0]} (pass --overwrite)")
    for name, text in contents.items():
        write_atomic(out_dir / name, text)
    manifest = json.loads(contents[MANIFEST])
    print(f"wrote {manifest['records']['total']} records to {out_dir}")
    print(f"  by status: {manifest['records']['by_status']}")
    print(f"  trainable: {manifest['records']['trainable']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
