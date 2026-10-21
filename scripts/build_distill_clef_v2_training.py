#!/usr/bin/env python
"""Build ``training-v7-d1`` / ``-d2`` / ``-d3`` for experiments/clef-distill-v2.

* d1 = training-v6a train + augment-01 (idea 1).
* d2 = training-v5 train + augment-01 + soft silver of the old 1,757-note pool (ideas 1 + 2).
* d3 = d2 + soft silver of the 638 new rule notes (shard ``rules``; ideas 1 + 2 + 3).

Soft silver = every rule-aware Clef-labelled note of ``distill-clef-02/labels-all.jsonl`` (no
threshold on notes): ``type`` = argmax of ``soft.type_probs``, ``target`` = Clef's chosen span (or
null), ``soft.target_trusted`` = target_prob >= 0.6, ``provenance.annotator`` = ``clef``.

``amendment_1`` of the protocol (deterministic lexical overrides, applied to every labels-all
row, old pool included): over the NFC, lowercased, diacritic-stripped text (``đ`` -> ``d``)

* R2: ``R2_RE`` -> lend; * R4-ung: ``R4_UNG_RE`` -> borrow.

The type distribution becomes 0.95 on the rule type and 0.05 spread uniformly over the other
seven; ``provenance.override`` records ``R2`` / ``R4-ung``; the target stays Clef's. R2 wins when
both match.

Leakage gate (as ``build_distill_clef_training.py`` plus probe-rules-v1): augment and silver notes
that hit it, or whose text equals a training-v5 train text (or an earlier augment/silver note),
are dropped and listed (``augment_dropped`` / ``silver_dropped``); the full gate then re-runs on
the result and raises, writing nothing, on any hit. ``--check`` rebuilds in memory and compares.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

SCRIPTS = Path(__file__).resolve().parent


def _load(name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


v1 = _load("build_annotation_v3_training")
v2 = _load("build_annotation_v3_training_v2")
v4 = _load("build_annotation_v3_training_v4")
v5 = _load("build_annotation_v3_training_v5")
hv = v2.hv

DS = Path("datasets/annotation-v3")
PREV = DS / "training-v5"
V6A = DS / "training-v6a"
AUGMENT = DS / "augment-01"
SILVER = DS / "distill-clef-02"
PROBE_RULES = Path("datasets/probe-rules-v1")
NAMES = {
    "d1": "training-v7-d1",
    "d2": "training-v7-d2",
    "d3": "training-v7-d3",
    "d4": "training-v7-d4",
}
SILVER_A_MIN = 0.6
SOURCE_BATCH = "distill-clef-02"
PREV_RECORDS, V6A_RECORDS, AUGMENT_RECORDS, SILVER_NOTES = 1123, 2023, 1106, 2395
TARGET_TRUST = 0.6
OVERRIDE_PROB = 0.95
BASE_KEYS = {"id", "text", "type", "target", "source_batch", "accented", "provenance", "group"}

# amendment_1 overrides, over fold(text) = NFC, lowercase, diacritics stripped, d-bar -> d.
R2_RE = re.compile(r"\b(?:tra|ung|mua|dong|nop)\s+(?:\S+\s+)?(?:ho|gium|dum|giup)\b")
R4_UNG_RE = re.compile(r"\b(?:ung luong|tam ung)\b")
# Exactly one of the two accented words of "dong ho" = a watch (đồng hồ) is not a "pay on behalf".
WATCH_RE = re.compile(r"đồng hồ")


def fold(text: str) -> str:
    text = unicodedata.normalize("NFC", text).lower().replace("đ", "d")
    return "".join(c for c in unicodedata.normalize("NFD", text) if not unicodedata.combining(c))


def override_rule(text: str) -> str | None:
    if R2_RE.search(fold(text)) and not WATCH_RE.search(unicodedata.normalize("NFC", text).lower()):
        return "R2"
    if R4_UNG_RE.search(fold(text)):
        return "R4-ung"
    return None


RULE_TYPE = {"R2": "lend", "R4-ung": "borrow"}


def soft_silver(row: dict[str, Any]) -> tuple[dict[str, Any], str | None, bool]:
    """``(record, override rule, argmax type changed by the override)`` for a labels-all row."""
    probs = {t: float(row["type_probs"].get(t, 0.0)) for t in v1.TYPES}
    clef_type = max(v1.TYPES, key=lambda t: probs[t])
    rule = override_rule(row["text"])
    if rule:
        rest = round((1 - OVERRIDE_PROB) / (len(v1.TYPES) - 1), 6)
        probs = {t: (OVERRIDE_PROB if t == RULE_TYPE[rule] else rest) for t in v1.TYPES}
    typ = max(v1.TYPES, key=lambda t: probs[t])
    prov: dict[str, Any] = {
        "annotator": "clef",
        "type_prob": row.get("type_probs", {}).get(typ) if rule is None else OVERRIDE_PROB,
        "target_prob": row["target_prob"],
        "source_batch": SOURCE_BATCH,
    }
    if rule:
        prov["override"] = rule
    target = row["target"]
    record = {
        "id": row["id"],
        "text": row["text"],
        "type": typ,
        "target": target,
        "source_batch": SOURCE_BATCH,
        "accented": v1.accented(row["text"]),
        "provenance": prov,
        "group": f"{SOURCE_BATCH}:{row['id']}",
        "soft": {
            "type_probs": probs,
            "target_trusted": bool(target) and float(row["target_prob"]) >= TARGET_TRUST,
        },
    }
    return record, rule, bool(rule) and typ != clef_type


def check_record(record: dict[str, Any], keys: set[str], annotator: str) -> None:
    rid = record["id"]
    if set(record) != keys:
        raise ValueError(f"{rid}: unexpected keys {sorted(record)}")
    if record["type"] not in v1.TYPES:
        raise ValueError(f"{rid}: unknown type {record['type']!r}")
    if record["provenance"].get("annotator") != annotator:
        raise ValueError(f"{rid}: provenance.annotator is not {annotator!r}")
    target = record["target"]
    if target is not None and record["text"][target["start"] : target["end"]] != target["text"]:
        raise ValueError(f"{rid}: target span does not slice the note text")


def _gate(sets: dict[str, list[dict[str, Any]]]) -> dict[str, dict[str, Any]]:
    gate = {}
    for name, rows in sets.items():
        gate[name] = {
            "ids": {r["id"] for r in rows},
            "texts": {r["text"] for r in rows} | {v1.exact_key(r["text"]) for r in rows},
            "near": hv.ReferenceSet([r["text"] for r in rows]),
            "refs": (
                [hv.Reference(r["text"]) for r in rows]
                if name.startswith(("human", "probe-rules"))
                else []
            ),
        }
    return gate


def _filter(
    records: list[dict[str, Any]],
    gate: dict[str, dict[str, Any]],
    prev_ids: set[str],
    prev_keys: set[str],
    seen: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    kept, dropped = [], []
    for record in records:
        key = v1.exact_key(record["text"])
        reasons = []
        if record["id"] in prev_ids or key in prev_keys:
            reasons.append("duplicate of a training-v5 train record")
        if record["id"] in seen or key in seen:
            reasons.append("duplicate of an earlier augment/silver record")
        for name, g in gate.items():
            if record["id"] in g["ids"]:
                reasons.append(f"id in {name}")
            if record["text"] in g["texts"] or key in g["texts"]:
                reasons.append(f"verbatim text in {name}")
            rule = g["near"].match(hv.Reference(record["text"]))
            if rule:
                reasons.append(f"near-duplicate ({rule}) of {name}")
            if g["refs"] and v4.max_char3(record, g["refs"]) >= v5.HV_CHAR3_LIMIT:
                reasons.append(f"char3 >= {v5.HV_CHAR3_LIMIT} to {name}")
        if reasons:
            dropped.append({"id": record["id"], "text": record["text"], "reasons": reasons})
            continue
        kept.append(record)
        seen.update((record["id"], key))
    return kept, dropped


def _jsonl(records: list[dict[str, Any]]) -> bytes:
    return b"".join((json.dumps(r, ensure_ascii=False) + "\n").encode("utf-8") for r in records)


def build(root: Path) -> dict[str, tuple[bytes, dict[str, Any]]]:
    """Verify every precondition and return ``{arm: (train bytes, manifest)}``; writes nothing."""
    for rel in (*v5.HV02_QUEUES, PROBE_RULES / "queue.jsonl"):
        if not (root / rel).is_file():
            raise ValueError(f"{rel} is missing: the leakage gate needs it")
    prev_bytes = (root / PREV / "train.jsonl").read_bytes()
    prev = [json.loads(line) for line in prev_bytes.decode("utf-8").splitlines()]
    v6a_bytes = (root / V6A / "train.jsonl").read_bytes()
    v6a = [json.loads(line) for line in v6a_bytes.decode("utf-8").splitlines()]
    augment = v1.read_jsonl(root / AUGMENT / "train.jsonl")
    labels = v1.read_jsonl(root / SILVER / "labels-all.jsonl")
    for what, got, want in (
        ("training-v5 train", len(prev), PREV_RECORDS),
        ("training-v6a train", len(v6a), V6A_RECORDS),
        ("augment-01", len(augment), AUGMENT_RECORDS),
        ("labels-all", len(labels), SILVER_NOTES),
    ):
        if got != want:
            raise ValueError(f"{what}: expected {want} records, got {got}")
    if v6a[: len(prev)] != prev:
        raise ValueError("training-v6a does not start with the training-v5 train records")
    for record in augment:
        check_record(record, BASE_KEYS, "augment")

    sets = v1.held_out(root)
    sets["human-value-02"] = [r for rel in v5.HV02_QUEUES for r in v1.read_jsonl(root / rel)]
    if len(sets["human-value-02"]) < 190:
        raise ValueError("human-value-02 queues look truncated")
    sets["probe-rules-v1"] = v1.read_jsonl(root / PROBE_RULES / "queue.jsonl")
    if len(sets["probe-rules-v1"]) != 64:
        raise ValueError("probe-rules-v1 queue should hold 64 notes")
    gate = _gate(sets)
    prev_ids = {r["id"] for r in prev}
    prev_keys = {v1.exact_key(r["text"]) for r in prev}

    # training-v6a is frozen; its silver-A tail was gated before probe-rules-v1 existed, so the
    # gate is re-run on it here and hits are left out of D1 (reported as ``base_dropped``).
    v6a_kept, base_dropped = _filter(v6a[len(prev) :], gate, prev_ids, prev_keys, set())
    d1_base = prev + v6a_kept
    seen: set[str] = set()
    aug_kept, aug_dropped = _filter(augment, gate, prev_ids, prev_keys, seen)

    silver_all, overrides, changed = [], Counter(), Counter()
    for row in labels:
        record, rule, flipped = soft_silver(row)
        record["shard"] = row["shard"]  # popped below; keeps the pool / rules split
        silver_all.append(record)
        if rule:
            overrides[rule] += 1
            changed[rule] += flipped
    for record in silver_all:
        check_record(
            {k: v for k, v in record.items() if k not in ("soft", "shard")}, BASE_KEYS, "clef"
        )
    kept_all, silver_dropped = _filter(silver_all, gate, prev_ids, prev_keys, seen)
    pool = [{k: v for k, v in r.items() if k != "shard"} for r in kept_all if r["shard"] != "rules"]
    rules = [
        {k: v for k, v in r.items() if k != "shard"} for r in kept_all if r["shard"] == "rules"
    ]

    # D4 (hard labels, arm A style): the rule-shard notes only, kept where Clef is confident
    # (type prob >= SILVER_A_MIN and target prob >= SILVER_A_MIN, the silver-A rule of
    # distill-clef-01; overridden rows carry type prob 0.95), without the ``soft`` field.
    rules_hard = [
        {k: v for k, v in r.items() if k != "soft"}
        for r in rules
        if r["provenance"]["type_prob"] >= SILVER_A_MIN
        and r["provenance"]["target_prob"] >= SILVER_A_MIN
    ]
    rules_label_ids = {r["id"] for r in labels if r["shard"] == "rules"}
    records = {
        "d1": d1_base + aug_kept,
        "d2": prev + aug_kept + pool,
        "d3": prev + aug_kept + pool + rules,
        "d4": d1_base + aug_kept + rules_hard,
    }
    excluded = json.loads((root / v1.DEBT / "manifest.json").read_text("utf-8"))[
        "excluded_human_value_01_ids"
    ]
    new_records = v6a_kept + aug_kept + pool + rules
    out: dict[str, tuple[bytes, dict[str, Any]]] = {}
    for recs in records.values():
        v1.assert_no_leakage(recs, sets, excluded)
    sizes = v2.assert_no_near_duplicates(new_records, sets)
    char3 = {}
    for name in ("human-value-01", "human-value-02", "probe-rules-v1"):
        refs = [hv.Reference(r["text"]) for r in sets[name]]
        scores = {r["id"]: v4.max_char3(r, refs) for r in new_records}
        over = [f"{i}: {s:.2f}" for i, s in scores.items() if s >= v5.HV_CHAR3_LIMIT]
        if over:
            raise ValueError(f"{name} char3 bound: " + "; ".join(over))
        char3[name] = round(max(scores.values()), 4)

    inputs = {
        str(p): v1.sha256_bytes((root / p).read_bytes())
        for p in (
            PREV / "train.jsonl",
            PREV / "manifest.json",
            V6A / "train.jsonl",
            AUGMENT / "train.jsonl",
            AUGMENT / "manifest.json",
            SILVER / "labels-all.jsonl",
            SILVER / "manifest.json",
            PROBE_RULES / "queue.jsonl",
            *v5.HV02_QUEUES,
        )
    }
    purposes = {
        "d1": "clef-distill-v2 D1: training-v6a train + augment-01 (idea 1)",
        "d2": "clef-distill-v2 D2: training-v5 train + augment-01 + soft silver of the old pool",
        "d3": "clef-distill-v2 D3: D2 + soft silver of the new rule notes",
        "d4": "clef-distill-v2 D4: D1 + hard-label rule-shard silver (type and target prob "
        ">= 0.6), no soft field",
    }

    def silver_notes_of(arm: str) -> int:
        return {"d1": 0, "d4": len(rules_label_ids)}.get(arm, len(labels))

    def silver_drop_of(arm: str) -> list[dict[str, Any]]:
        if arm == "d1":
            return []
        if arm == "d4":
            return [d for d in silver_dropped if d["id"] in rules_label_ids]
        return silver_dropped

    for arm, recs in records.items():
        body = _jsonl(recs)
        n_silver = sum(r["provenance"]["annotator"] == "clef" for r in recs)
        in_arm_silver = {"d1": [], "d2": pool, "d3": pool + rules, "d4": rules_hard}[arm]
        manifest = {
            "name": NAMES[arm],
            "annotation_version": "annotation-v3",
            "purpose": (
                f"{purposes[arm]} (model labels, provenance.annotator = clef / augment, NOT human "
                "labels); validation is in-sample, the frozen test split, probe-v1, "
                "human-value-01/02 and probe-rules-v1 stay held out"
            ),
            "counts": {
                "base_records": len(d1_base) if arm in ("d1", "d4") else len(prev),
                "base_dropped": len(base_dropped) if arm in ("d1", "d4") else 0,
                "augment_records": len(augment),
                "augment_dropped": len(aug_dropped),
                "augment_added": len(aug_kept),
                "silver_notes": silver_notes_of(arm),
                "silver_dropped": len(silver_drop_of(arm)),
                "silver_added": len(in_arm_silver),
                "train": len(recs),
            },
            "silver_share": round(n_silver / len(recs), 4),
            "type_counts": v1.type_counts(recs),
            "type_counts_by_source": {
                "base": v1.type_counts(d1_base if arm in ("d1", "d4") else prev),
                "augment": v1.type_counts(aug_kept),
                "silver": v1.type_counts(in_arm_silver),
            },
            "source_counts": dict(Counter(r["source_batch"] for r in recs)),
            "annotator_counts": dict(Counter(r["provenance"]["annotator"] for r in recs)),
            "soft_records": sum("soft" in r for r in recs),
            "soft_target_trusted": sum(r["soft"]["target_trusted"] for r in recs if "soft" in r),
            "overrides": {
                "rule": {"R2": R2_RE.pattern, "R4-ung": R4_UNG_RE.pattern},
                "applied_over": "NFC, lowercase, diacritics stripped, d-bar -> d "
                "(R2 skips accented 'đồng hồ' = watch)",
                "rows_matched_in_labels_all": dict(overrides),
                "rows_whose_argmax_type_changed": dict(changed),
                "rows_with_override_in_this_train": dict(
                    Counter(
                        r["provenance"]["override"]
                        for r in in_arm_silver
                        if "override" in r["provenance"]
                    )
                ),
            },
            "base_dropped": base_dropped if arm in ("d1", "d4") else [],
            "augment_dropped": aug_dropped,
            "silver_dropped": silver_drop_of(arm),
            "leakage_gate": {
                "sets": sizes,
                "rule": "no id, no verbatim or NFC+lowercase text of any train record in a set; "
                "no new (augment/silver) note a near duplicate (scripts/build_human_value_queue.py "
                "rules) of any note of a set; no new note at folded char-3 Jaccard >= "
                f"{v5.HV_CHAR3_LIMIT} to a human-value-01/02 or probe-rules-v1 note; no new note "
                "equal to a training-v5 train text. Failing notes are dropped, see "
                "augment_dropped / silver_dropped",
                "max_char3_new_notes": char3,
                "result": "pass",
            },
            "inputs_sha256": inputs,
            "outputs_sha256": {"train.jsonl": v1.sha256_bytes(body)},
            "evaluation_sets": (
                "validation.jsonl (in-sample, logging only) and test.jsonl are symlinks to the "
                "frozen annotation-v1 splits"
            ),
        }
        if arm == "d4":
            manifest["silver_filter"] = {
                "rule": f"rule-shard notes with type prob >= {SILVER_A_MIN} and target prob >= "
                f"{SILVER_A_MIN} (silver-A rule); hard labels, no soft field; amendment_1 "
                "overrides applied before the filter",
                "rules_shard_notes_after_gate": len(rules),
                "below_threshold_not_used": len(rules) - len(rules_hard),
            }
        out[arm] = (body, manifest)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    root = args.root
    try:
        built = build(root)
    except (ValueError, OSError, KeyError) as exc:
        print(f"distill-clef-v2: {exc}", file=sys.stderr)
        return 1
    links = {
        n: Path("..") / ".." / "annotation-v1" / "splits" / f"{n}.jsonl"
        for n in ("validation", "test")
    }
    problems = []
    for arm, (body, manifest) in built.items():
        out = root / DS / NAMES[arm]
        text = json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
        if args.check:
            if not (out / "train.jsonl").is_file() or (out / "train.jsonl").read_bytes() != body:
                problems.append(f"{NAMES[arm]}: train.jsonl differs from the rebuild")
            mf = out / "manifest.json"
            if not mf.is_file() or mf.read_text("utf-8") != text:
                problems.append(f"{NAMES[arm]}: manifest.json differs from the rebuild")
            for name in links:
                link = out / f"{name}.jsonl"
                if link.resolve() != (root / v1.SPLITS / f"{name}.jsonl").resolve():
                    problems.append(f"{link} is not a link to the frozen {name} split")
            continue
        out.mkdir(parents=True, exist_ok=True)
        (out / "train.jsonl").write_bytes(body)
        for name, target in links.items():
            link = out / f"{name}.jsonl"
            if link.is_symlink() or link.exists():
                link.unlink()
            link.symlink_to(target)
        (out / "manifest.json").write_text(text, encoding="utf-8")
        print(f"wrote {out}: {manifest['counts']}")
        print(f"  types: {manifest['type_counts']}")
    ov = next(iter(built.values()))[1]["overrides"]
    print(
        f"overrides: matched {ov['rows_matched_in_labels_all']}, "
        f"argmax type changed {ov['rows_whose_argmax_type_changed']}"
    )
    if problems:
        print("\n".join(problems), file=sys.stderr)
        return 1
    if args.check:
        print(
            "distill-clef-v2 check OK: "
            + str({NAMES[a]: m["counts"] for a, (_, m) in built.items()})
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
