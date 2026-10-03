#!/usr/bin/env python
r"""Build the risk-based review plan and small Quet queue for the targeted-value-01 batch.

Usage:
    uv run python scripts/build_targeted_value_queue.py [--root R] [--out-dir D] [--report-dir D]
                                                        [--tokenizer DIR] [--force] [--check]

Works from the raw batch ``corpus/raw/targeted-value-01.jsonl`` directly (no Quet corpus review or
export step), the generation sidecar ``generation-notes.jsonl`` and the advisory
``type-proposals.jsonl``. Each record lands in one group (rules and seed in
``gidi.annotation.risk_review``; the draws are deterministic):

* ``must_review``: any objective risk signal (proposer ``multiple_money_candidates``,
  ``no_candidate``, ``compound_amount``, ``unusual_punctuation``; type proposal not complete or
  disagreeing with the generation intent; value proposal disagreeing with the intent; a
  suggestion failing validation; duplicate/near duplicate of existing text; length out of
  bounds; a number dropped only by elimination);
* ``hard_sample``: 3 per hard-pattern stratum (``multi_number``, ``bare_number``, each slang unit);
* ``clean_audit``: ~20 stratified clean records;
* ``auto_accept``: everything else, labelled ``synthetic-auto`` (never ``human``).

Writes, into ``--out-dir`` (``datasets/annotation-v2/targeted-value-01``):

* ``review-queue.jsonl``: Quet queue ``{id, text, review_group, stratum, reasons}``, must-review
  first; used for both Quet passes;
* ``review-type-proposals.jsonl`` / ``review-value-proposals.jsonl``: advisory Quet proposals for
  the queued ids (``--proposals``); value proposals carry the generation intent in ``reason``;
* ``auto-labels.jsonl`` (type/target) and ``value-auto-labels.jsonl`` for auto-accepted records;
* ``review-provenance.jsonl``: per record group, stratum, reasons, ``provenance``;
* ``review-manifest.json``: counts, reason counts, input/output sha256, seed and the predeclared
  escalation rule;

and into ``--report-dir`` (``experiments/value-span-v1``)
``targeted-value-01-review-plan.{md,json}``.

The human label files (``labels.jsonl``, ``value-labels.jsonl``) are written only by Quet. After
labelling, ``scripts/score_value_review.py --dataset targeted-value-01`` applies the escalation
rule. Existing outputs are only replaced with ``--force`` (never if that would orphan an id a
human already labelled); ``--check`` rebuilds in memory and verifies the files on disk.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import unicodedata
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

from gidi.annotation import risk_review as rr
from gidi.annotation.schema import load_config
from gidi.annotation.value_span import load_value_config, propose_value
from gidi.corpus.jsonl import read_jsonl

NAME = rr.NAME
DEFAULT_OUT_DIR = Path("datasets/annotation-v2/targeted-value-01")
DEFAULT_REPORT_DIR = Path("experiments/value-span-v1")
DEFAULT_TOKENIZER = Path("models/baseline-v1/bamibert/lr2e-05-seed1")
RAW = Path("corpus/raw/targeted-value-01.jsonl")
CONFIGS = (
    Path("configs/annotation-v1.yaml"),
    Path("configs/annotation-v2.yaml"),
    Path("configs/annotation-v2-value.quet.yaml"),
)
REPORT_STEM = "targeted-value-01-review-plan"
HUMAN_LABELS = ("labels.jsonl", "value-labels.jsonl")

QUEUE_FILE = "review-queue.jsonl"
TYPE_PROPOSALS_FILE = "review-type-proposals.jsonl"
VALUE_PROPOSALS_FILE = "review-value-proposals.jsonl"
AUTO_FILE = "auto-labels.jsonl"
VALUE_AUTO_FILE = "value-auto-labels.jsonl"
PROVENANCE_FILE = "review-provenance.jsonl"
MANIFEST_FILE = "review-manifest.json"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def jsonl_bytes(rows: list[dict[str, Any]]) -> bytes:
    return "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows).encode("utf-8")


def json_bytes(obj: Any) -> bytes:
    return (json.dumps(obj, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def make_token_counter(tokenizer_dir: Path) -> tuple[Callable[[str], int], str]:
    """Token counter (special tokens included) of the v1 student's tokenizer, and its sha256."""
    file = tokenizer_dir / "tokenizer.json"
    if not file.exists():
        raise SystemExit(
            f"{file} not found: pass --tokenizer DIR with the v1 student's tokenizer "
            f"(default {DEFAULT_TOKENIZER})"
        )
    from gidi.modeling.tokenization import load_tokenizer

    tokenizer = load_tokenizer(str(tokenizer_dir))
    return lambda text: len(tokenizer(text)["input_ids"]), sha256_bytes(file.read_bytes())


def load_batch(root: Path, out_dir: Path) -> tuple[list[dict], dict[str, dict], dict[str, dict]]:
    """Raw records, generation notes and type proposals, checked to be one-to-one."""
    raw = read_jsonl(root / RAW)
    notes = {n["id"]: n for n in read_jsonl(root / out_dir / "generation-notes.jsonl")}
    proposals = {p["id"]: p for p in read_jsonl(root / out_dir / "type-proposals.jsonl")}
    problems = []
    raw_ids = [r["id"] for r in raw]
    if len(set(raw_ids)) != len(raw_ids):
        problems.append("duplicate ids in the raw batch")
    for label, ids in (("generation-notes", set(notes)), ("type-proposals", set(proposals))):
        if ids != set(raw_ids):
            problems.append(f"{label} ids differ from the raw batch")
    for r in raw:
        text = unicodedata.normalize("NFC", r["text"])
        if r["id"] in notes and notes[r["id"]]["text"] != text:
            problems.append(f"{r['id']}: text differs between raw batch and generation-notes")
    if problems:
        raise SystemExit("inconsistent batch inputs:\n  " + "\n  ".join(problems))
    return raw, notes, proposals


def intended_span(note: dict) -> str | None:
    value = note.get("intended_value")
    return value["text"] if value else None


def coverage(
    notes: dict[str, dict],
    spans: dict[str, str | None],
    decisions: dict[str, rr.Decision],
    columns: dict[str, Callable[[rr.Decision], bool]],
) -> dict[str, dict[str, dict[str, int]]]:
    """Counts per dimension value and column: pattern, amount form, type, accent, slang unit."""
    dims: dict[str, Callable[[dict], str]] = {
        "pattern": lambda n: n["pattern"],
        "amount_form": lambda n: n["amount_form"],
        "type": lambda n: n["intended_type"],
        "accent": lambda n: "accented" if n["accented"] else "unaccented",
        "slang_unit": lambda n: rr.slang_unit(spans[n["id"]] or intended_span(n)) or "none",
    }
    out: dict[str, dict[str, dict[str, int]]] = {}
    for dim, fn in dims.items():
        values = sorted({fn(n) for n in notes.values()})
        out[dim] = {
            v: {
                col: sum(1 for i, n in notes.items() if fn(n) == v and pred(decisions[i]))
                for col, pred in columns.items()
            }
            for v in values
        }
    return out


def quet_commands(out_dir: Path) -> list[str]:
    d = out_dir.as_posix()
    return [
        f"quet annotate {d}/review-queue.jsonl \\\n"
        f"  --schema configs/annotation-v1.yaml \\\n"
        f"  --out {d}/labels.jsonl \\\n"
        f"  --proposals {d}/review-type-proposals.jsonl",
        f"quet annotate {d}/review-queue.jsonl \\\n"
        f"  --schema configs/annotation-v2-value.quet.yaml \\\n"
        f"  --out {d}/value-labels.jsonl \\\n"
        f"  --proposals {d}/review-value-proposals.jsonl",
    ]


def build(
    root: Path,
    out_dir: Path,
    report_dir: Path,
    token_count: Callable[[str], int],
    tokenizer_sha: str,
    tokenizer_label: str,
) -> dict[Path, bytes]:
    """All outputs as ``path -> bytes`` (paths relative to ``root``)."""
    raw, notes, type_proposals = load_batch(root, out_dir)
    type_config = load_config(root / "configs" / "annotation-v1.yaml")
    value_config = load_value_config(root / "configs" / "annotation-v2.yaml")

    # Training sets built downstream from this batch are derived outputs, not prior references;
    # scanning them would make this build depend on later steps.
    derived = sorted((root / "datasets" / "annotation-v2").glob("training-*"))
    ref_texts, ref_files = rr.load_reference_texts(root, set(notes), [root / out_dir, *derived])
    batch = [notes[r["id"]] for r in raw]
    dups = rr.scan_duplicates(batch, ref_texts)

    value_proposals = {n["id"]: propose_value(n["text"]) for n in batch}
    spans = {
        n["id"]: value_proposals[n["id"]].chosen.text if value_proposals[n["id"]].chosen else None
        for n in batch
    }
    facts, reasons_by_id = [], {}
    for n in batch:
        i = n["id"]
        value = value_proposals[i]
        reasons = rr.assess_record(
            n, type_proposals[i], value, dups[i], token_count(n["text"]), type_config, value_config
        )
        reasons_by_id[i] = reasons
        facts.append(
            rr.Facts(
                id=i,
                reasons=tuple(reasons),
                hard_stratum=rr.hard_stratum(value.categories, spans[i]),
                type=n["intended_type"],
                accented=n["accented"],
                form=n["amount_form"],
                target_present=n["intended_target"] is not None,
                length=len(n["text"]),
                group=n["group"],
            )
        )
    decisions = rr.plan_review(facts)
    order = rr.queue_order(decisions)
    queued = set(order)

    queue = [
        {
            "id": i,
            "text": notes[i]["text"],
            "review_group": decisions[i].group,
            "stratum": decisions[i].stratum,
            "reasons": list(decisions[i].reasons),
        }
        for i in order
    ]
    review_type = [type_proposals[i] for i in order]
    review_value = [rr.value_proposal_row(notes[i], value_proposals[i]) for i in order]
    auto_ids = [n["id"] for n in batch if decisions[n["id"]].group == rr.AUTO]
    auto_labels = [rr.type_label(type_proposals[i]) for i in auto_ids]
    value_auto = [rr.value_auto_label(i, value_proposals[i]) for i in auto_ids]
    provenance = [
        {
            "id": n["id"],
            "review_group": decisions[n["id"]].group,
            "stratum": decisions[n["id"]].stratum,
            "reasons": list(decisions[n["id"]].reasons),
            "provenance": (rr.PROVENANCE_QUEUED if n["id"] in queued else rr.PROVENANCE_AUTO),
        }
        for n in batch
    ]

    group_count = Counter(d.group for d in decisions.values())
    reason_counts = Counter(r for rs in reasons_by_id.values() for r in rs)
    sims = {i: max(d.corpus_sim, d.batch_sim) for i, d in dups.items()}
    watch = rr.NEAR_WATCH_THRESHOLD
    near = rr.NEAR_DUP_THRESHOLD
    strata = {}
    for s in rr.HARD_STRATA:
        in_stratum = (rr.HARD, rr.AUTO)
        members = [i for i, d in decisions.items() if d.stratum == s and d.group in in_stratum]
        strata[s] = {
            "members": len(members),
            "sampled": sorted(i for i in members if decisions[i].group == rr.HARD),
            "auto_accepted": sorted(i for i in members if decisions[i].group == rr.AUTO),
        }
    columns: dict[str, Callable[[rr.Decision], bool]] = {
        rr.MUST: lambda d: d.group == rr.MUST,
        rr.HARD: lambda d: d.group == rr.HARD,
        rr.AUDIT: lambda d: d.group == rr.AUDIT,
        "queue": lambda d: d.queued,
        rr.AUTO: lambda d: d.group == rr.AUTO,
        "all": lambda d: True,
    }
    cover = coverage(notes, spans, decisions, columns)

    def sha(rel: Path) -> str:
        return sha256_bytes((root / rel).read_bytes())

    outputs = {
        out_dir / QUEUE_FILE: jsonl_bytes(queue),
        out_dir / TYPE_PROPOSALS_FILE: jsonl_bytes(review_type),
        out_dir / VALUE_PROPOSALS_FILE: jsonl_bytes(review_value),
        out_dir / AUTO_FILE: jsonl_bytes(auto_labels),
        out_dir / VALUE_AUTO_FILE: jsonl_bytes(value_auto),
        out_dir / PROVENANCE_FILE: jsonl_bytes(provenance),
    }
    counts = {
        "total": len(batch),
        "auto_accepted": group_count[rr.AUTO],
        "must_review": group_count[rr.MUST],
        "hard_sample": group_count[rr.HARD],
        "clean_audit": group_count[rr.AUDIT],
        "queue": len(queue),
    }
    manifest = {
        "name": NAME,
        "annotation_version": "annotation-v2",
        "role": "risk-based review of a synthetic batch (no full manual review)",
        "seed": rr.SEED,
        "counts": counts,
        "queue_over_80": len(queue) > 80,
        "reason_counts": {r: reason_counts[r] for r in rr.REASON_ORDER if reason_counts[r]},
        "must_review_first_reason_counts": dict(
            Counter(d.stratum for d in decisions.values() if d.group == rr.MUST)
        ),
        "strata": {
            s: {
                "members": info["members"],
                "sampled": len(info["sampled"]),
                "auto_accepted": len(info["auto_accepted"]),
            }
            for s, info in strata.items()
        },
        "parameters": {
            "hard_sample_per_stratum": rr.HARD_SAMPLE_PER_STRATUM,
            "clean_audit_size": rr.CLEAN_AUDIT_SIZE,
            "hard_strata_order": list(rr.HARD_STRATA),
            "length_chars": [rr.MIN_CHARS, rr.MAX_CHARS],
            "max_tokens": rr.MAX_TOKENS,
            "near_duplicate_threshold": near,
            "watch_threshold": watch,
        },
        "duplicates": {
            "reference_texts": len(ref_texts),
            "reference_files": len(ref_files),
            "normalised_matches_corpus": sum(d.corpus_norm_match for d in dups.values()),
            "normalised_matches_batch": sum(d.batch_norm_match for d in dups.values()),
            "similarity_ge_0.90": sum(s >= near for s in sims.values()),
            "similarity_0.85_to_0.90_not_forced": sum(watch <= s < near for s in sims.values()),
            "max_corpus_similarity": max(d.corpus_sim for d in dups.values()),
            "max_batch_similarity": max(d.batch_sim for d in dups.values()),
        },
        "not_evaluated": {
            "unnatural_wording": (
                "no objective signal exists in the inputs (the batch passes corpus validation "
                "with 0 errors and 0 warnings); none is invented, so wording is covered only by "
                "the hard-pattern sample and the clean audit"
            )
        },
        "escalation_rule": rr.ESCALATION_RULE,
        "provenance": {
            "auto_accept": rr.PROVENANCE_AUTO,
            "queued": rr.PROVENANCE_QUEUED,
            "note": "auto-accepted labels are never human labels; only Quet writes those",
        },
        "inputs": {
            str(RAW): sha(RAW),
            str(out_dir / "generation-notes.jsonl"): sha(out_dir / "generation-notes.jsonl"),
            str(out_dir / "type-proposals.jsonl"): sha(out_dir / "type-proposals.jsonl"),
            **{str(c): sha(c) for c in CONFIGS},
            "tokenizer": {"dir": tokenizer_label, "tokenizer.json": tokenizer_sha},
            "reference_texts_sha256": sha256_bytes("\n".join(ref_texts).encode("utf-8")),
        },
        "files": {p.name: sha256_bytes(data) for p, data in outputs.items()},
    }
    outputs[out_dir / MANIFEST_FILE] = json_bytes(manifest)

    plan = {
        "name": NAME,
        "seed": rr.SEED,
        "counts": counts,
        "reason_counts": manifest["reason_counts"],
        "strata": strata,
        "coverage": cover,
        "escalation_rule": rr.ESCALATION_RULE,
        "duplicates": manifest["duplicates"],
        "quet_commands": quet_commands(out_dir),
        "manifest": str(out_dir / MANIFEST_FILE),
    }
    outputs[report_dir / f"{REPORT_STEM}.json"] = json_bytes(plan)
    outputs[report_dir / f"{REPORT_STEM}.md"] = render_markdown(plan, out_dir).encode("utf-8")
    return outputs


def _table(header: list[str], rows: list[list[Any]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return lines


def render_markdown(plan: dict[str, Any], out_dir: Path) -> str:
    c = plan["counts"]
    d = plan["duplicates"]
    lines = [
        "# targeted-value-01: risk-based review plan",
        "",
        "Generated by `scripts/build_targeted_value_queue.py` (deterministic; `--check` verifies).",
        "Replaces full manual review of the 253 synthetic notes. Nothing here is a human label:",
        f"auto-accepted records carry `{rr.PROVENANCE_AUTO}` provenance; only Quet writes human",
        "labels (`labels.jsonl`, `value-labels.jsonl`). Seed: " + f"`{plan['seed']}`.",
        "",
        "## Groups",
        "",
        *_table(
            ["group", "records", "rule"],
            [
                ["must_review", c["must_review"], "every record with an objective risk signal"],
                ["hard_sample", c["hard_sample"], "3 per hard-pattern stratum (rest auto)"],
                ["clean_audit", c["clean_audit"], "stratified over the remaining clean records"],
                ["**queue**", f"**{c['queue']}**", "must_review + hard_sample + clean_audit"],
                ["auto_accept", c["auto_accepted"], "`synthetic-auto`, passes every gate"],
                ["total", c["total"], ""],
            ],
        ),
        "",
        "## Must-review reasons (records can have several)",
        "",
        *_table(["reason", "records"], [[r, n] for r, n in plan["reason_counts"].items()]),
        "",
        f"Duplicate scan: {d['reference_texts']} existing texts from {d['reference_files']} files "
        f"(`corpus/raw`, `corpus/reviewed`, `datasets`, `generated`; the batch itself excluded). "
        f"Normalised matches: {d['normalised_matches_corpus']} corpus, "
        f"{d['normalised_matches_batch']} batch; similarity >= 0.90: "
        f"{d['similarity_ge_0.90']}; 0.85-0.90 (reported, not forced to review): "
        f"{d['similarity_0.85_to_0.90_not_forced']}; max corpus similarity "
        f"{d['max_corpus_similarity']}.",
        "",
        "Unnatural wording is not evaluated: no objective signal exists in the inputs, and none is "
        "invented.",
        "",
        "## Hard-pattern strata",
        "",
        *_table(
            ["stratum", "members (not must-review)", "sampled", "auto-accepted"],
            [
                [s, i["members"], len(i["sampled"]), len(i["auto_accepted"])]
                for s, i in plan["strata"].items()
            ],
        ),
        "",
        "A record counts once, in its first matching stratum (`multi_number`, `bare_number`, "
        "then "
        "the slang unit of its value span).",
        "",
        "## Coverage per group",
        "",
    ]
    cols = ["must_review", "hard_sample", "clean_audit", "queue", "auto_accept", "all"]
    for dim, values in plan["coverage"].items():
        lines += [
            f"### {dim}",
            "",
            *_table([dim, *cols], [[v, *(row[c] for c in cols)] for v, row in values.items()]),
            "",
        ]
    lines += [
        "## Escalation rule (predeclared)",
        "",
        f"- Error: {plan['escalation_rule']['error']}.",
        f"- (a) {plan['escalation_rule']['hard_stratum']}.",
        f"- (b) Clean audit: {plan['escalation_rule']['clean_audit']['accept']}; "
        f"{plan['escalation_rule']['clean_audit']['expand']}.",
        f"- (c) Must-review: {plan['escalation_rule']['must_review']}.",
        "",
        "Apply it with `uv run python scripts/score_value_review.py --dataset targeted-value-01`",
        "once both Quet passes exist; it writes the verdicts into",
        "`experiments/value-span-v1/review-score.json`.",
        "",
        "## Quet commands",
        "",
        "Two passes over the same small queue (type/target, then value):",
        "",
        "```sh",
        *plan["quet_commands"][0].splitlines(),
        "",
        *plan["quet_commands"][1].splitlines(),
        "```",
        "",
        "`--out` is strict: labels must be ids of the queue. Proposals are advisory; only the",
        "reviewer's confirmed decision is saved. Afterwards:",
        "",
        "```sh",
        f"uv run python scripts/validate_annotations.py {out_dir.as_posix()}/labels.jsonl \\",
        f"  --queue {out_dir.as_posix()}/{QUEUE_FILE}",
        "uv run python scripts/score_value_review.py --dataset targeted-value-01",
        "```",
        "",
    ]
    return "\n".join(lines)


def human_ids(root: Path, out_dir: Path) -> set[str]:
    ids: set[str] = set()
    for name in HUMAN_LABELS:
        path = root / out_dir / name
        if path.exists():
            ids |= {r["id"] for r in read_jsonl(path)}
    return ids


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--force", action="store_true", help="replace existing outputs")
    parser.add_argument("--check", action="store_true", help="verify the files on disk")
    args = parser.parse_args(argv)

    root = args.root
    token_count, tokenizer_sha = make_token_counter(
        args.tokenizer if args.tokenizer.is_absolute() else root / args.tokenizer
    )
    outputs = build(
        root, args.out_dir, args.report_dir, token_count, tokenizer_sha, str(args.tokenizer)
    )

    if args.check:
        bad = [
            f"{p}: {'missing' if not (root / p).exists() else 'differs from a fresh build'}"
            for p, data in outputs.items()
            if not (root / p).exists() or (root / p).read_bytes() != data
        ]
        if bad:
            print("CHECK FAILED", *bad, sep="\n  ", file=sys.stderr)
            return 1
        print(f"ok: {len(outputs)} files reproduce")
        return 0

    existing = [p for p in outputs if (root / p).exists()]
    if existing and not args.force:
        print(f"refusing to overwrite {existing[0]} (pass --force)", file=sys.stderr)
        return 1
    new_queue = {
        json.loads(line)["id"] for line in outputs[args.out_dir / QUEUE_FILE].decode().splitlines()
    }
    old_path = root / args.out_dir / QUEUE_FILE
    old_queue = {r["id"] for r in read_jsonl(old_path)} if old_path.exists() else set()
    orphaned = sorted((human_ids(root, args.out_dir) & old_queue) - new_queue)
    if orphaned:
        print(
            f"refusing to replace the queue: {len(orphaned)} human-labelled id(s) would fall "
            f"outside it (e.g. {orphaned[0]})",
            file=sys.stderr,
        )
        return 1
    for rel, data in outputs.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    manifest = json.loads(outputs[args.out_dir / MANIFEST_FILE])
    print(f"wrote {len(outputs)} files; counts {manifest['counts']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
