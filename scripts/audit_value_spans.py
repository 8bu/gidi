#!/usr/bin/env python
"""Pre-training data audit of the annotation-v2 value spans.

Usage:
    uv run python scripts/audit_value_spans.py [--mode pre-review|training]
        [--training-dir datasets/annotation-v2/training-v1] [--out-dir experiments/value-span-v1]
        [--tokenizer models/gidi-finance-v1] [--no-tokens]

Two modes, same report:

* ``pre-review`` (default, available as soon as ``build_value_queue.py`` has run): audits the
  rule *proposals* for the in-scope records. Nothing is human-reviewed yet, so every number is
  labelled PRE-REVIEW. Writes ``data-audit.json`` / ``data-audit.md``. When the human Quet output
  ``datasets/annotation-v2/value-labels.jsonl`` exists, a proposer-vs-human section is added.
* ``training``: audits built training data (``build_training_v2.py`` output): value spans as
  labelled, provenance, masked records. Writes ``data-audit-training.json`` / ``.md``.

Reports: total / with span / null span / auto-proposed / needing review / multi-number notes
(the ``multi_number`` tag: 2+ numeric expressions, one amount or date counts once);
the frequency of each surface pattern (digits masked, e.g. ``<num>k``, ``<num> xi``,
``<num>tr<num>``, ``<dot3>``), overall and accented vs unaccented notes; which patterns are seen
or unseen in train for validation / test / probe; span length in characters and in v1 tokens
(``models/gidi-finance-v1``; also whether tokens align with span edges and whether truncation at
32 tokens loses a span); and an explicit coverage table for the amount forms the model must
handle.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path

from gidi.annotation import risk_review as rr
from gidi.annotation.value_span import (
    load_value_config,
    pattern_signature,
    propose_value,
    validate_label_file,
)
from gidi.corpus.jsonl import read_jsonl

V2 = Path("datasets/annotation-v2")
SPLITS = ("train", "validation", "test", "probe")

# (label, signatures that count as this form, required by the assignment?)
COVERAGE_FORMS: tuple[tuple[str, tuple[str, ...], bool], ...] = (
    ("100", ("<num>",), True),
    ("50k", ("<num>k",), True),
    ("50K", ("<num>K",), False),
    ("50 K", ("<num> K", "<num> k"), True),
    ("1tr", ("<num>tr",), True),
    ("1tr5", ("<num>tr<num>",), True),
    ("1.5tr", ("<dec.>tr",), True),
    ("1,5tr", ("<dec,>tr",), True),
    ("2 triệu", ("<num> trieu",), True),
    ("2 củ", ("<num> cu",), True),
    ("5 xị", ("<num> xi",), True),
    ("5 chai", ("<num> chai",), True),
    ("5 lít", ("<num> lit",), True),
    ("500 nghìn", ("<num> nghin",), True),
    ("500 ngàn", ("<num> ngan",), True),
    ("1.500.000", ("<dot3>",), True),
    ("1500000", ("<long>",), True),
    ("80.000đ", ("<dot3>đ",), False),
    ("12,500,000", ("<com3>",), False),
    ("2 tỷ", ("<num> ty",), False),
    ("2 triệu rưỡi", ("<num> trieu ruoi",), False),
    ("1 triệu 2", ("<num> trieu <num>",), False),
)


def describe(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    ordered = sorted(values)
    p95 = ordered[min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))]
    return {
        "n": len(values),
        "mean": round(statistics.fmean(values), 2),
        "median": statistics.median(values),
        "p95": p95,
        "max": ordered[-1],
        "histogram": dict(sorted(Counter(int(v) for v in values).items())),
    }


def load_items(args: argparse.Namespace) -> tuple[list[dict], dict]:
    """Uniform items: id, text, split, accented, span (dict|None), state, categories."""
    v2 = args.root / V2
    meta: dict = {"mode": args.mode}
    items: list[dict] = []
    if args.mode == "pre-review":
        queue = read_jsonl(v2 / "value-queue.jsonl")
        review = {r["id"]: r for r in read_jsonl(v2 / "value-review-queue.jsonl")}
        for r in queue:
            p = propose_value(r["text"])
            if p.auto_accept:
                state = "auto_proposed"
            elif p.chosen is not None:
                state = "proposed_needs_review"
            elif "multiple_money_candidates" in p.categories:
                state = "multiple_candidates_needs_review"
            else:
                state = "no_candidate_needs_review"
            items.append(
                {
                    "id": r["id"],
                    "text": r["text"],
                    "split": r["split"],
                    "source_batch": r["source_batch"],
                    "span": p.span,
                    "state": state,
                    "categories": list(p.categories),
                    "review_reasons": list(p.review_reasons),
                    "review_group": review.get(r["id"], {}).get("review_group"),
                }
            )
        meta["review_queue"] = len(review)
        meta["review_queue_by_group"] = dict(Counter(r["review_group"] for r in review.values()))
    else:
        directory = args.root / args.training_dir
        validation_ids = {r["id"] for r in read_jsonl(directory / "validation.jsonl")}
        files = {
            "train": "train.jsonl",
            "validation": "validation.jsonl",
            "test": "test.jsonl",
            "probe": "probe-v1-eval-only.jsonl",
        }
        for split, name in files.items():
            path = directory / name
            if not path.exists():
                continue
            for r in read_jsonl(path):
                if split == "train" and r["id"] in validation_ids:
                    continue  # in-sample copy; counted under validation
                if r["value_status"] != "complete":
                    state = "masked_uncertain"
                elif r["value_provenance"] == "human":
                    state = "human"
                else:
                    state = "rule"
                p = propose_value(r["text"])
                items.append(
                    {
                        "id": r["id"],
                        "text": r["text"],
                        "split": split,
                        "source_batch": r["source_batch"],
                        "span": r["value"],
                        "state": state,
                        "categories": list(p.categories),
                        "review_reasons": [],
                    }
                )
        meta["training_dir"] = str(args.training_dir)
    for item in items:
        item["accented"] = any(ord(c) > 127 for c in item["text"])
        item["signature"] = pattern_signature(item["span"]["text"]) if item["span"] else None
    return items, meta


def pattern_tables(items: list[dict]) -> dict:
    with_span = [i for i in items if i["span"]]
    by_sig = Counter(i["signature"] for i in with_span)
    by_sig_acc = {
        "accented": Counter(i["signature"] for i in with_span if i["accented"]),
        "unaccented": Counter(i["signature"] for i in with_span if not i["accented"]),
    }
    train = {i["signature"] for i in with_span if i["split"] == "train"}
    train_surface = {i["span"]["text"] for i in with_span if i["split"] == "train"}
    seen: dict = {}
    for split in ("validation", "test", "probe"):
        rows = [i for i in with_span if i["split"] == split]
        unseen_sig = sorted({i["signature"] for i in rows if i["signature"] not in train})
        seen[split] = {
            "spans": len(rows),
            "pattern_seen_in_train": sum(i["signature"] in train for i in rows),
            "pattern_unseen_in_train": sum(i["signature"] not in train for i in rows),
            "unseen_patterns": {
                sig: [i["span"]["text"] for i in rows if i["signature"] == sig][:5]
                for sig in unseen_sig
            },
            "exact_span_seen_in_train": sum(i["span"]["text"] in train_surface for i in rows),
            "exact_span_unseen_in_train": sum(i["span"]["text"] not in train_surface for i in rows),
        }
    per_split = {
        split: dict(Counter(i["signature"] for i in with_span if i["split"] == split).most_common())
        for split in SPLITS
    }
    return {
        "patterns": dict(by_sig.most_common()),
        "patterns_by_note_accent": {k: dict(v.most_common()) for k, v in by_sig_acc.items()},
        "patterns_by_split": per_split,
        "seen_unseen_vs_train": seen,
    }


def coverage_table(items: list[dict]) -> list[dict]:
    rows = []
    with_span = [i for i in items if i["span"]]
    for label, sigs, required in COVERAGE_FORMS:
        hits = [i for i in with_span if i["signature"] in sigs]
        per = {s: sum(i["split"] == s for i in hits) for s in SPLITS}
        rows.append(
            {
                "form": label,
                "required": required,
                "signatures": list(sigs),
                "total": len(hits),
                **{f"n_{s}": per[s] for s in SPLITS},
                "note_accented": sum(i["accented"] for i in hits),
                "note_unaccented": sum(not i["accented"] for i in hits),
                "span_with_diacritics": sum(
                    any(ord(c) > 127 for c in i["span"]["text"]) for i in hits
                ),
                "examples": [i["span"]["text"] for i in hits[:2]],
                "status": (
                    "covered in train"
                    if per["train"]
                    else "eval only (unseen in train)"
                    if sum(per.values())
                    else "ABSENT"
                ),
            }
        )
    return rows


def token_stats(items: list[dict], tokenizer_path: Path) -> dict:
    from gidi.modeling.preprocessing import encode
    from gidi.modeling.tokenization import load_tokenizer

    with_span = [i for i in items if i["span"]]
    tokenizer = load_tokenizer(str(tokenizer_path))
    enc = encode(
        tokenizer, [i["text"] for i in with_span], [i["span"] for i in with_span], 32, strict=False
    )
    counts = [int(((row == 1) | (row == 2)).sum()) for row in enc.tag_labels]
    by_sig: dict[str, list[int]] = {}
    for item, n in zip(with_span, counts, strict=True):
        by_sig.setdefault(item["signature"], []).append(n)
    misaligned = [
        {"id": i["id"], "text": i["text"], "span": i["span"]["text"]}
        for i, bad in zip(with_span, enc.boundary_mismatch, strict=True)
        if bad
    ]
    return {
        "tokenizer": str(tokenizer_path),
        "max_length": 32,
        "tokens_per_span": describe(counts),
        "mean_tokens_by_pattern": {
            sig: round(statistics.fmean(v), 2) for sig, v in sorted(by_sig.items())
        },
        "notes_truncated_at_32": enc.n_truncated,
        "spans_losing_tokens_to_truncation": enc.n_span_truncated,
        "spans_with_token_boundary_mismatch": enc.n_boundary_mismatch,
        "boundary_mismatch_examples": misaligned[:10],
    }


def human_vs_proposer(args: argparse.Namespace, items: list[dict]) -> dict | None:
    path = args.root / V2 / "value-labels.jsonl"
    if args.mode != "pre-review" or not path.exists():
        return None
    config = load_value_config(args.root / "configs" / "annotation-v2.yaml")
    texts = {i["id"]: i["text"] for i in items}
    report = validate_label_file(path, texts, config)
    if not report.ok:
        return {"error": report.errors[:10]}
    by_id = {i["id"]: i for i in items}
    stats: dict = {"reviewed": len(report.labels)}
    buckets = {"all": [], "clean_audit": []}
    disagreements = []
    for rid, label in report.labels.items():
        item = by_id[rid]
        human = label["target"]["text"] if label["target"] else None
        prop = item["span"]["text"] if item["span"] else None
        agree = label["annotation_status"] == "complete" and human == prop
        buckets["all"].append(agree)
        if item["review_group"] == rr.AUDIT and item["state"] == "auto_proposed":
            buckets["clean_audit"].append(agree)
        if not agree:
            disagreements.append(
                {
                    "id": rid,
                    "text": item["text"],
                    "proposed": prop,
                    "human": human,
                    "status": label["annotation_status"],
                }
            )
    for name, rows in buckets.items():
        stats[name] = {
            "n": len(rows),
            "agree": sum(rows),
            "agreement": round(sum(rows) / len(rows), 4) if rows else None,
        }
    stats["clean_audit_precision"] = stats["clean_audit"]["agreement"]
    stats["disagreements"] = disagreements
    return stats


def build_report(args: argparse.Namespace) -> dict:
    items, meta = load_items(args)
    pre = args.mode == "pre-review"
    by_split = {s: [i for i in items if i["split"] == s] for s in SPLITS}
    trainable = [i for i in items if i["split"] != "probe"]
    with_span = [i for i in items if i["span"]]
    states = Counter(i["state"] for i in items)
    report: dict = {
        "title": "value-span data audit",
        "status": "PRE-REVIEW: rule proposals, no human review yet" if pre else "training data",
        **meta,
        "totals": {
            "records": len(items),
            "trainable_records_excluding_probe": len(trainable),
            "records_by_split": {s: len(v) for s, v in by_split.items()},
            "with_value_span": len(with_span),
            "null_value_span": len(items) - len(with_span),
            "states": dict(states),
            "auto_proposed_spans": states.get("auto_proposed", 0) + states.get("rule", 0),
            "human_labelled_spans": states.get("human", 0),
            "needing_review_or_masked": sum(
                states.get(k, 0)
                for k in (
                    "proposed_needs_review",
                    "multiple_candidates_needs_review",
                    "no_candidate_needs_review",
                    "masked_uncertain",
                )
            ),
            "multi_number_notes": sum("multi_number" in i["categories"] for i in items),
            "notes_by_category": dict(
                Counter(c for i in items for c in i["categories"]).most_common()
            ),
            "accented_notes": sum(i["accented"] for i in items),
            "unaccented_notes": sum(not i["accented"] for i in items),
        },
        "span_length_chars": describe([len(i["span"]["text"]) for i in with_span]),
        **pattern_tables(items),
        "coverage": coverage_table(items),
    }
    if pre:
        report["review_reasons"] = dict(
            Counter(i["review_reasons"][0] for i in items if i["review_reasons"])
        )
    if args.no_tokens:
        report["tokens"] = {"skipped": "--no-tokens"}
    else:
        try:
            report["tokens"] = token_stats(items, args.root / args.tokenizer)
        except Exception as e:  # report, do not hide, a tokenizer problem
            report["tokens"] = {"error": f"{type(e).__name__}: {e}"}
    hv = human_vs_proposer(args, items)
    if hv is not None:
        report["proposer_vs_human"] = hv
    return report


def render_markdown(r: dict) -> str:
    t = r["totals"]
    pre = r["mode"] == "pre-review"
    out = [f"# Value-span data audit ({r['mode']})", ""]
    if pre:
        out += [
            "> **PRE-REVIEW.** These numbers describe the rule proposer's output "
            "(`gidi.annotation.value_span`), not human-verified labels. Re-run on built "
            "training data (`--mode training`) after the Quet value pass.",
            "",
        ]
    out += [
        "## Totals",
        "",
        f"- records: {t['records']} (excluding probe: {t['trainable_records_excluding_probe']}); "
        f"by split: {t['records_by_split']}",
        f"- with a value span: {t['with_value_span']}; null value span: {t['null_value_span']}",
        f"- states: {t['states']}",
        f"- auto-proposed (rule) spans: {t['auto_proposed_spans']}; human spans: "
        f"{t['human_labelled_spans']}; needing review / masked: {t['needing_review_or_masked']}",
        f"- multi-number notes (2+ numeric expressions): {t['multi_number_notes']}",
        f"- notes accented / unaccented: {t['accented_notes']} / {t['unaccented_notes']}",
    ]
    if pre:
        out += [
            f"- review queue: {r['review_queue']} records {r['review_queue_by_group']}; "
            f"flagged by the proposer, primary reasons: {r['review_reasons']}"
        ]
    categories = ", ".join(f"{k} {v}" for k, v in t["notes_by_category"].items())
    out += ["", f"Notes by category: {categories}"]
    out += [
        "",
        "## Coverage of required amount forms",
        "",
        "| form | req | total | train | val | test | probe | accented notes | unaccented notes "
        "| span has diacritics | status |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for c in r["coverage"]:
        out.append(
            f"| `{c['form']}` | {'yes' if c['required'] else 'extra'} | {c['total']} | "
            f"{c['n_train']} | {c['n_validation']} | {c['n_test']} | {c['n_probe']} | "
            f"{c['note_accented']} | {c['note_unaccented']} | {c['span_with_diacritics']} | "
            f"{c['status']} |"
        )
    out += [
        "",
        "## Surface patterns (digits masked)",
        "",
        "| pattern | all | accented notes | unaccented notes | train | val | test | probe |",
        "|---|---|---|---|---|---|---|---|",
    ]
    acc, una = r["patterns_by_note_accent"]["accented"], r["patterns_by_note_accent"]["unaccented"]
    for sig, n in r["patterns"].items():
        per = [r["patterns_by_split"][s].get(sig, 0) for s in SPLITS]
        cells = [n, acc.get(sig, 0), una.get(sig, 0), *per]
        out.append(f"| `{sig}` | " + " | ".join(map(str, cells)) + " |")
    out += ["", "## Seen / unseen patterns versus train", ""]
    for split, s in r["seen_unseen_vs_train"].items():
        out.append(
            f"- **{split}**: {s['spans']} spans; pattern seen in train "
            f"{s['pattern_seen_in_train']}, unseen {s['pattern_unseen_in_train']}; exact span "
            f"seen in train "
            f"{s['exact_span_seen_in_train']}, unseen {s['exact_span_unseen_in_train']}"
        )
        for sig, examples in s["unseen_patterns"].items():
            out.append(f"  - unseen pattern `{sig}`: {examples}")
    sl = r["span_length_chars"]
    out += ["", "## Span length", "", f"- characters: {json.dumps(sl, ensure_ascii=False)}"]
    tk = r["tokens"]
    if "tokens_per_span" in tk:
        out += [
            f"- v1 tokens per span: {json.dumps(tk['tokens_per_span'])}",
            f"- mean tokens by pattern: {tk['mean_tokens_by_pattern']}",
            f"- notes truncated at 32 tokens: {tk['notes_truncated_at_32']}; spans losing tokens "
            f"to truncation: {tk['spans_losing_tokens_to_truncation']}; spans whose tokens "
            f"straddle an edge: {tk['spans_with_token_boundary_mismatch']}",
        ]
        for ex in tk["boundary_mismatch_examples"]:
            out.append(f"  - misaligned: {ex['text']!r} span {ex['span']!r}")
    else:
        out.append(f"- tokens: {tk}")
    hv = r.get("proposer_vs_human")
    if hv:
        dump = json.dumps(hv, ensure_ascii=False, indent=2)
        out += ["", "## Proposer versus human labels", "", f"```json\n{dump}\n```"]
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--mode", choices=("pre-review", "training"), default="pre-review")
    parser.add_argument("--training-dir", type=Path, default=V2 / "training-v1")
    parser.add_argument("--out-dir", type=Path, default=Path("experiments/value-span-v1"))
    parser.add_argument("--tokenizer", type=Path, default=Path("models/gidi-finance-v1"))
    parser.add_argument("--no-tokens", action="store_true", help="skip tokenizer statistics")
    args = parser.parse_args(argv)

    report = build_report(args)
    stem = "data-audit" if args.mode == "pre-review" else "data-audit-training"
    out_dir = args.root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{stem}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (out_dir / f"{stem}.md").write_text(render_markdown(report), encoding="utf-8")
    print(f"wrote {out_dir / (stem + '.json')} and {stem}.md ({report['status']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
