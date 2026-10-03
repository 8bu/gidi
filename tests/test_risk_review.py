"""Tests for the risk-based review plan: grouping, auto-accept gates, scoring and escalation."""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path

import pytest

from gidi.annotation import review_gate as gate
from gidi.annotation import risk_review as rr
from gidi.annotation import value_review as vr
from gidi.annotation.schema import load_config
from gidi.annotation.value_span import load_value_config, propose_value

ROOT = Path(__file__).resolve().parents[1]
TYPE_CONFIG = load_config(ROOT / "configs" / "annotation-v1.yaml")
VALUE_CONFIG = load_value_config(ROOT / "configs" / "annotation-v2.yaml")


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses resolve string annotations through sys.modules
    spec.loader.exec_module(module)
    return module


scorer = load_script("score_value_review")
trainer = load_script("build_training_v2")

TYPES = ("expense", "income", "borrow", "lend", "refund", "transfer")


def fake_facts(n: int = 120) -> list[rr.Facts]:
    facts = []
    for i in range(n):
        stratum = (None, "multi_number", "bare_number", "slang_cu", "slang_ty")[i % 5]
        facts.append(
            rr.Facts(
                id=f"r{i:03d}",
                reasons=("no_candidate",) if i % 17 == 0 else (),
                hard_stratum=stratum,
                type=TYPES[i % len(TYPES)],
                accented=i % 3 == 0,
                form=("k", "tr", "dotted")[i % 3],
                target_present=i % 2 == 0,
                length=10 + i % 25,
                group=f"g{i // 2}",
            )
        )
    return facts


# ---------------------------------------------------------------------------------------------
# Grouping
# ---------------------------------------------------------------------------------------------


def test_plan_is_deterministic_and_independent_of_input_order() -> None:
    facts = fake_facts()
    plan = rr.plan_review(facts)
    assert plan == rr.plan_review(list(reversed(facts)))
    assert plan != rr.plan_review(facts, seed="another-seed")
    assert rr.queue_order(plan) == rr.queue_order(plan)
    assert rr.queue_order(plan) != rr.queue_order(plan, seed="another-seed")


def test_plan_group_invariants() -> None:
    facts = fake_facts()
    by_id = {f.id: f for f in facts}
    plan = rr.plan_review(facts)
    assert set(plan) == set(by_id)
    for rid, d in plan.items():
        f = by_id[rid]
        assert (d.group == rr.MUST) == bool(f.reasons)
        if d.group == rr.HARD:
            assert d.stratum == f.hard_stratum and not f.reasons
        if d.group == rr.AUDIT:
            assert f.hard_stratum is None and not f.reasons
    for stratum in ("multi_number", "bare_number", "slang_cu", "slang_ty"):
        members = [
            d for d in plan.values() if d.stratum == stratum and d.group in (rr.HARD, rr.AUTO)
        ]
        assert sum(d.group == rr.HARD for d in members) == min(3, len(members))
    assert sum(d.group == rr.AUDIT for d in plan.values()) == rr.CLEAN_AUDIT_SIZE


def test_queue_lists_must_review_first_and_excludes_auto() -> None:
    plan = rr.plan_review(fake_facts())
    order = rr.queue_order(plan)
    groups = [plan[i].group for i in order]
    assert groups == sorted(groups, key=rr.QUEUED_GROUPS.index)
    assert not any(plan[i].group == rr.AUTO for i in order)
    assert len(order) == len({i for i, d in plan.items() if d.queued})


def test_small_stratum_is_sampled_in_full() -> None:
    facts = [
        rr.Facts(f"s{i}", (), "slang_ty", "expense", True, "slang_ty", False, 15) for i in range(2)
    ]
    plan = rr.plan_review(facts)
    assert {d.group for d in plan.values()} == {rr.HARD}


def test_pick_diverse_spreads_over_features() -> None:
    items = [(f"a{i}", "x") for i in range(10)] + [("b0", "y"), ("c0", "z")]
    picked = rr.pick_diverse(items, lambda t: t[0], lambda t: (t[1],), 3, "seed", "salt")
    assert {kind for _, kind in picked} == {"x", "y", "z"}


@pytest.mark.parametrize(
    ("span", "unit"),
    [
        ("5 xị", "slang_xi"),
        ("4 xi", "slang_xi"),
        ("2 chai", "slang_chai"),
        ("1 chai rưỡi", "slang_chai"),
        ("3 lít", "slang_lit"),
        ("nửa củ", "slang_cu"),
        ("4 cu ruoi", "slang_cu"),
        ("1 tỷ 2", "slang_ty"),
        ("1 ty", "slang_ty"),
        ("50k", None),
        ("2 triệu", None),
        (None, None),
    ],
)
def test_slang_unit(span: str | None, unit: str | None) -> None:
    assert rr.slang_unit(span) == unit


def test_hard_stratum_uses_the_first_matching_stratum() -> None:
    assert rr.hard_stratum(["multi_number", "bare_number", "slang"], "5 xị") == "multi_number"
    assert rr.hard_stratum(["bare_number"], "100") == "bare_number"
    assert rr.hard_stratum(["slang"], "2 chai") == "slang_chai"
    assert rr.hard_stratum(["explicit_unit"], "1 tỷ 2") == "slang_ty"
    assert rr.hard_stratum(["slang", "number_words"], "hai trăm nghìn") == "slang_other"
    assert rr.hard_stratum(["explicit_unit"], "50k") is None


# ---------------------------------------------------------------------------------------------
# Auto-accept gates
# ---------------------------------------------------------------------------------------------


def make_note(text: str, value: str | None, *, value_status: str = "complete") -> dict:
    start = text.index(value) if value else None
    return {
        "id": "n1",
        "text": text,
        "intended_type": "expense",
        "intended_target": None,
        "intended_value": (
            {"text": value, "start": start, "end": start + len(value)} if value else None
        ),
        "value_status": value_status,
    }


def assess(note: dict, *, proposal: dict | None = None, dup=None, tokens: int = 8) -> list[str]:
    proposal = proposal or {
        "id": "n1",
        "annotation_status": "complete",
        "type": "expense",
        "target": None,
    }
    dup = dup or rr.DuplicateInfo(False, 0.5, False, 0.5)
    value = propose_value(note["text"])
    return rr.assess_record(note, proposal, value, dup, tokens, TYPE_CONFIG, VALUE_CONFIG)


def test_clean_record_passes_every_gate() -> None:
    assert assess(make_note("mua tai nghe 150k", "150k")) == []


def test_proposer_categories_force_review() -> None:
    assert "multiple_money_candidates" in assess(
        make_note("grab 89k ship 15k", None, value_status="uncertain")
    )
    assert "no_candidate" in assess(make_note("nạp momo", None, value_status="no_amount"))
    assert "compound_amount" in assess(make_note("vay 1 triệu 5", "1 triệu 5"))


def test_type_proposal_status_and_intent_disagreement_force_review() -> None:
    uncertain = {"id": "n1", "annotation_status": "uncertain", "type": None, "target": None}
    uncertain["note"] = "unclear"
    reasons = assess(make_note("mua tai nghe 150k", "150k"), proposal=uncertain)
    assert {"type_status_not_complete", "type_intent_mismatch"} <= set(reasons)
    other = {"id": "n1", "annotation_status": "complete", "type": "income", "target": None}
    assert "type_intent_mismatch" in assess(make_note("mua tai nghe 150k", "150k"), proposal=other)


def test_value_disagreeing_with_intent_forces_review() -> None:
    note = make_note("cơm tấm 100", "100")
    note["intended_value"] = {"text": "cơm", "start": 0, "end": 3}
    assert "value_intent_mismatch" in assess(note)
    assert "value_intent_mismatch" in assess(
        make_note("mua tai nghe 150k", None, value_status="no_amount")
    )


def test_invalid_suggestions_force_review() -> None:
    bad_target = {
        "id": "n1",
        "annotation_status": "complete",
        "type": "expense",
        "target": {"text": "xx", "start": 0, "end": 2},
    }
    assert "invalid_type_proposal" in assess(
        make_note("mua tai nghe 150k", "150k"), proposal=bad_target
    )
    note = make_note("mua tai nghe 150k", "150k")
    note["intended_value"] = {"text": "150k", "start": 0, "end": 4}
    assert "invalid_intent_span" in assess(note)


def test_duplicate_and_length_gates() -> None:
    note = make_note("mua tai nghe 150k", "150k")
    assert "duplicate_corpus" in assess(note, dup=rr.DuplicateInfo(True, 1.0, False, 0.2))
    assert "near_duplicate_corpus" in assess(note, dup=rr.DuplicateInfo(False, 0.9, False, 0.2))
    assert "duplicate_batch" in assess(note, dup=rr.DuplicateInfo(False, 0.2, True, 0.2))
    assert "near_duplicate_batch" in assess(note, dup=rr.DuplicateInfo(False, 0.2, False, 0.93))
    assert assess(note, dup=rr.DuplicateInfo(False, 0.899, False, 0.89)) == []
    assert "length_out_of_bounds" in assess(note, tokens=rr.MAX_TOKENS + 1)
    assert "length_out_of_bounds" in assess(make_note("5k", "5k"))
    assert "length_out_of_bounds" in assess(make_note("mua " * 40 + "150k", "150k"))


def test_scan_duplicates_flags_normalised_matches_and_ignores_same_group() -> None:
    batch = [
        {"id": "a", "text": "mua tai nghe 150k", "group": "g1"},
        {"id": "b", "text": "mua tai nghe 200k", "group": "g2"},
        {"id": "c", "text": "mua tai nghe 300k", "group": "g2"},
        {"id": "d", "text": "đổ xăng bình", "group": "g3"},
    ]
    found = rr.scan_duplicates(batch, ["Mua tai nghe 1tr", "ăn phở"])
    assert found["a"].corpus_norm_match and found["a"].batch_norm_match
    assert not found["d"].corpus_norm_match and not found["d"].batch_norm_match
    assert found["d"].corpus_sim < 0.9


# ---------------------------------------------------------------------------------------------
# Scoring and escalation
# ---------------------------------------------------------------------------------------------

SPAN = {"text": "50k", "start": 3, "end": 6}


def human(status: str = "complete", type_: str | None = "expense", target=None) -> dict:
    return {"id": "x", "annotation_status": status, "type": type_, "target": target}


def test_error_definition_covers_type_target_value_and_status() -> None:
    proposal = human()
    assert rr.pass_errors(human(), proposal) == []
    assert rr.pass_errors(human(type_="income"), proposal) == ["type"]
    assert rr.pass_errors(human(target=SPAN), proposal) == ["target"]
    assert rr.pass_errors(human(target={**SPAN, "end": 7}), human(target=SPAN)) == ["target"]
    assert rr.pass_errors(human("uncertain", None), proposal) == ["status"]
    value = human(type_="amount", target=SPAN)
    both = {"type": (proposal, proposal), "value": (value, value)}
    assert rr.record_errors(both) == []
    assert rr.record_errors({**both, "value": (human("skipped", None), value)}) == ["value:status"]
    assert rr.record_errors({**both, "type": (human(type_="income"), proposal)}) == ["type:type"]
    assert rr.record_errors({**both, "value": (human(type_="no_amount"), value)}) == [
        "value:type",
        "value:target",
    ]


AUTO_IDS = {
    "multi_number": ["m1", "m2"],
    "slang_cu": ["c1"],
    "clean": ["k1", "k2", "k3"],
}


def tallies(**errors: int) -> dict[str, rr.Tally]:
    return {s: rr.Tally(3, 3, errors.get(s, 0)) for s in ("multi_number", "slang_cu")}


def test_no_errors_accepts_the_auto_remainder() -> None:
    e = rr.decide_escalation(tallies(), rr.Tally(20, 20, 0), AUTO_IDS)
    assert e.status == "accept_auto_remainder" and not e.ids


def test_audit_tolerates_one_error_but_not_two() -> None:
    one = rr.decide_escalation(tallies(), rr.Tally(20, 20, 1), AUTO_IDS)
    assert one.status == "accept_auto_remainder" and one.clean_audit == "accept"
    two = rr.decide_escalation(tallies(), rr.Tally(20, 20, 2), AUTO_IDS)
    assert two.status == "escalated" and two.clean_audit == "expand_all"
    assert set(two.ids) == {"m1", "m2", "c1", "k1", "k2", "k3"}


def test_a_hard_stratum_error_reviews_that_whole_stratum_only() -> None:
    e = rr.decide_escalation(tallies(slang_cu=1), rr.Tally(20, 20, 0), AUTO_IDS)
    assert e.status == "escalated"
    assert e.hard_strata["slang_cu"] == "review_entire_stratum"
    assert e.hard_strata["multi_number"] == "ok"
    assert e.ids == ("c1",)


def test_incomplete_labelling_is_pending_unless_an_error_already_decides() -> None:
    partial = {"multi_number": rr.Tally(3, 1, 0), "slang_cu": rr.Tally(3, 3, 0)}
    assert rr.decide_escalation(partial, rr.Tally(20, 20, 0), AUTO_IDS).status == "pending"
    assert rr.decide_escalation(tallies(), rr.Tally(20, 5, 1), AUTO_IDS).status == "pending"
    early = rr.decide_escalation(tallies(), rr.Tally(20, 5, 2), AUTO_IDS)
    assert early.status == "escalated" and early.clean_audit == "expand_all"


def test_score_ignores_must_review_errors_for_escalation() -> None:
    targeted = scorer.DATASETS[rr.NAME]
    queue = [
        {
            "id": "u1",
            "review_group": rr.MUST,
            "stratum": "no_candidate",
            "reasons": ["no_candidate"],
        },
        {"id": "h1", "review_group": rr.HARD, "stratum": "slang_cu", "reasons": []},
        {"id": "a1", "review_group": rr.AUDIT, "stratum": "clean", "reasons": []},
    ]
    provenance = {
        "u1": {"review_group": rr.MUST, "stratum": "no_candidate"},
        "h1": {"review_group": rr.HARD, "stratum": "slang_cu"},
        "a1": {"review_group": rr.AUDIT, "stratum": "clean"},
        "x1": {"review_group": rr.AUTO, "stratum": "slang_cu"},
    }
    value_prop = human(type_="amount", target=SPAN)
    props = {
        "type": {i: human() for i in ("u1", "h1", "a1")},
        "value": {i: value_prop for i in ("u1", "h1", "a1")},
    }
    wrong_value = human(type_="no_amount")
    labels = {
        "type": {i: human() for i in ("u1", "h1", "a1")},
        "value": {"u1": wrong_value, "h1": value_prop, "a1": value_prop},
    }
    detail, esc, verdicts = scorer.score(targeted, queue, provenance, props, labels)
    assert detail["groups"][rr.MUST]["errors"] == 1
    assert detail["groups"][rr.HARD]["errors"] == 0
    assert esc.status == "accept_auto_remainder"
    assert verdicts["slang_cu"] == "passed" and verdicts["clean"] == "passed"

    labels["value"] = {"u1": value_prop, "h1": wrong_value, "a1": value_prop}
    detail, esc, verdicts = scorer.score(targeted, queue, provenance, props, labels)
    assert esc.status == "escalated" and esc.ids == ("x1",)
    assert verdicts["slang_cu"] == "failed" and verdicts["clean"] == "passed"


def test_scorer_refuses_without_human_labels(tmp_path: Path, capsys) -> None:
    v2 = tmp_path / "datasets" / "annotation-v2"
    v2.mkdir(parents=True)
    shutil.copytree(ROOT / "configs", tmp_path / "configs")
    for name in (
        "value-review-queue.jsonl",
        "value-review-provenance.jsonl",
        "value-queue.jsonl",
        "value-proposals.jsonl",
    ):
        (v2 / name).write_text("", encoding="utf-8")
    code = scorer.main(["--root", str(tmp_path), "--dataset", "existing-value-review"])
    assert code == 2
    assert "human labels absent" in capsys.readouterr().err
    assert not (tmp_path / scorer.SCORE_PATH).exists()
    (v2 / "value-labels.jsonl").write_text("", encoding="utf-8")  # present but labels nothing
    assert scorer.main(["--root", str(tmp_path), "--dataset", "existing-value-review"]) == 2
    assert not (tmp_path / scorer.SCORE_PATH).exists()


# ---------------------------------------------------------------------------------------------
# build_training_v2.py --extra layout
# ---------------------------------------------------------------------------------------------

TEXTS = {"a": "cf 50k", "b": "bún 45k", "c": "cơm tấm 100", "d": "grab 30k"}
SPANS = {"a": (3, 6), "b": (4, 7), "c": (8, 11), "d": (5, 8)}


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), "utf-8")


def value_label(rid: str, extra: dict | None = None) -> dict:
    start, end = SPANS[rid]
    span = {"text": TEXTS[rid][start:end], "start": start, "end": end}
    return {
        "id": rid,
        "annotation_status": "complete",
        "type": "amount",
        "target": span,
        **(extra or {}),
    }


def type_row(rid: str) -> dict:
    return {"id": rid, "annotation_status": "complete", "type": "expense", "target": None}


@pytest.fixture
def batch(tmp_path: Path) -> Path:
    """a, b auto-accepted; c, d queued for a human."""
    notes = [{"id": r, "text": t, "group": f"g-{r}"} for r, t in TEXTS.items()]
    write_jsonl(tmp_path / "generation-notes.jsonl", notes)
    write_jsonl(
        tmp_path / "review-queue.jsonl",
        [{"id": "c", "text": TEXTS["c"]}, {"id": "d", "text": TEXTS["d"]}],
    )
    write_jsonl(
        tmp_path / "review-provenance.jsonl",
        [
            {
                "id": r,
                "provenance": rr.PROVENANCE_AUTO if r in "ab" else rr.PROVENANCE_QUEUED,
                "stratum": {"a": "clean", "b": "multi_number"}.get(r, "clean"),
            }
            for r in TEXTS
        ],
    )
    write_jsonl(tmp_path / "auto-labels.jsonl", [type_row("a"), type_row("b")])
    rule = {"provenance": "rule", "confidence": 0.9, "rule_version": "v"}
    write_jsonl(tmp_path / "value-auto-labels.jsonl", [value_label(r, rule) for r in "ab"])
    return tmp_path


def extra(directory: Path, trusted: tuple[str, ...] = ("clean", "multi_number")):
    return trainer.extra_records(ROOT, directory, "batch", VALUE_CONFIG, set(), trusted)


def test_extra_trains_auto_accepted_and_excludes_unlabelled_queue(batch: Path) -> None:
    records, groups, review = extra(batch)
    assert sorted(r["id"] for r in records) == ["a", "b"]
    assert {r["provenance"]["annotator"] for r in records} == {"synthetic-auto"}
    assert {r["value_provenance"] for r in records} == {"rule"}
    assert sorted(groups) == ["g-a", "g-b", "g-c", "g-d"]
    assert review["queued"] == 2 and review["awaiting_review_excluded"] == 2
    assert review["auto_accepted"] == 2 and review["trained"] == 2


def test_extra_human_labels_override_auto_and_unlock_queued_records(batch: Path) -> None:
    write_jsonl(batch / "labels.jsonl", [type_row("a"), type_row("c")])
    uncertain = {
        "id": "a",
        "annotation_status": "uncertain",
        "type": None,
        "target": None,
        "note": "two amounts",
    }
    write_jsonl(batch / "value-labels.jsonl", [uncertain, value_label("c")])
    records, _, review = extra(batch)
    by_id = {r["id"]: r for r in records}
    assert sorted(by_id) == ["a", "b", "c"]
    assert by_id["a"]["provenance"]["annotator"] == "human"
    assert by_id["a"]["value"] is None and by_id["a"]["value_status"] == "uncertain"
    assert by_id["c"]["value_provenance"] == "human" and by_id["c"]["value"]["text"] == "100"
    assert by_id["b"]["provenance"]["annotator"] == "synthetic-auto"
    assert review["human"] == 2 and review["human_overrides_auto"] == 1
    assert review["awaiting_review_excluded"] == 1  # d


def test_extra_excludes_queued_records_with_only_one_human_pass(batch: Path) -> None:
    write_jsonl(batch / "labels.jsonl", [type_row("c")])
    write_jsonl(batch / "value-labels.jsonl", [])
    records, _, review = extra(batch)
    assert sorted(r["id"] for r in records) == ["a", "b"]
    assert review["partial_human_labels_excluded"] == 1


def test_extra_withholds_auto_records_of_strata_that_did_not_pass(batch: Path) -> None:
    records, _, review = extra(batch, trusted=("clean",))
    assert [r["id"] for r in records] == ["a"]
    assert review["stratum_unverified_excluded"] == 1 and review["auto_accepted"] == 1
    none, _, review = extra(batch, trusted=())
    assert none == [] and review["stratum_unverified_excluded"] == 2


def test_extra_human_label_beats_a_failed_stratum(batch: Path) -> None:
    write_jsonl(batch / "labels.jsonl", [type_row("b")])
    write_jsonl(batch / "value-labels.jsonl", [value_label("b")])
    records, _, review = extra(batch, trusted=())
    assert [r["id"] for r in records] == ["b"]
    assert records[0]["provenance"]["annotator"] == "human"
    assert review["stratum_unverified_excluded"] == 1  # a


def test_extra_refuses_auto_labels_without_synthetic_auto_provenance(batch: Path) -> None:
    write_jsonl(
        batch / "review-provenance.jsonl",
        [{"id": r, "provenance": "human"} for r in TEXTS],
    )
    with pytest.raises(SystemExit):
        extra(batch)


# ---------------------------------------------------------------------------------------------
# Existing-data review (gidi.annotation.value_review): grouping, detection
# ---------------------------------------------------------------------------------------------

STRATUM_SIZES = {"multi_number_period_or_ordinal": 20, "multi_number_quantity": 6}
STRATUM_SIZES |= {"multi_number_date_or_time": 4, "slang_cu": 2, "multi_number_percent": 1}


def existing_facts() -> list[vr.Facts]:
    facts: list[vr.Facts] = []
    n = 0
    for stratum, size in STRATUM_SIZES.items():
        for _ in range(size):
            facts.append(
                vr.Facts(f"f{n:03d}", (), stratum, TYPES[n % 4], n % 2 == 0, "k", True, "train")
            )
            n += 1
    for i in range(3):
        reasons = ("split_share",) if i == 0 else ("no_candidate", "bare_number")[:i]
        facts.append(vr.Facts(f"m{i}", reasons, None, "expense", True, "none", False, "test"))
    for i in range(60):
        form = ("<num>k", "<num>tr", "<num> triệu")[i % 3]
        split = ("train", "validation", "test", "probe")[i % 4]
        facts.append(
            vr.Facts(f"c{i:03d}", (), None, TYPES[i % 6], i % 3 == 0, form, i % 2 == 0, split)
        )
    return facts


def test_existing_plan_is_deterministic_and_independent_of_input_order() -> None:
    facts = existing_facts()
    plan = vr.plan_review(facts)
    assert plan == vr.plan_review(list(reversed(facts)))
    assert plan != vr.plan_review(facts, seed="another-seed")
    assert rr.queue_order(plan, vr.SEED) == rr.queue_order(plan, vr.SEED)


def test_existing_plan_samples_by_stratum_size_and_audits_the_clean_records() -> None:
    plan = vr.plan_review(existing_facts())
    by_stratum: dict[str, list[rr.Decision]] = {}
    for d in plan.values():
        by_stratum.setdefault(d.stratum, []).append(d)
    for stratum, size in STRATUM_SIZES.items():
        members = by_stratum[stratum]
        assert len(members) == size
        assert sum(d.group == rr.HARD for d in members) == vr.sample_size(size)
        assert {d.group for d in members} <= {rr.HARD, rr.AUTO}
    assert [d.group for d in by_stratum["slang_cu"]] == [rr.HARD, rr.HARD]  # tiny: all
    assert all(plan[f"m{i}"].group == rr.MUST for i in range(3))
    assert plan["m1"].stratum == "no_candidate"  # a must record's stratum is its first reason
    clean = by_stratum[rr.CLEAN_STRATUM]
    assert sum(d.group == rr.AUDIT for d in clean) == vr.AUDIT_SIZE
    assert sum(d.group == rr.AUTO for d in clean) == 60 - vr.AUDIT_SIZE


def test_clean_audit_spreads_over_forms_splits_and_types() -> None:
    facts = existing_facts()
    by_id = {f.id: f for f in facts}
    plan = vr.plan_review(facts)
    audit = [by_id[i] for i, d in plan.items() if d.group == rr.AUDIT]
    assert {f.split for f in audit} == {"train", "validation", "test", "probe"}
    assert {f.form for f in audit} == {"<num>k", "<num>tr", "<num> triệu"}
    assert {f.accented for f in audit} == {True, False}
    assert {f.target_present for f in audit} == {True, False}
    assert len({f.type for f in audit}) >= 5


@pytest.mark.parametrize(
    ("population", "sample"), [(0, 0), (1, 1), (2, 2), (3, 2), (5, 2), (6, 3), (58, 3)]
)
def test_stratum_sample_size(population: int, sample: int) -> None:
    assert vr.sample_size(population) == sample


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("lau vs team 1tr2 chia 5", True),
        ("lẩu vs team 4 người chia 180k", True),
        ("lau vs team 20/10 chia moi nguoi 320k", True),
        ("ăn chiều 50k", False),
        ("trich 20% luong de danh", False),
    ],
)
def test_split_share_is_detected_by_the_chia_token(text: str, expected: bool) -> None:
    assert vr.split_share(text) is expected


@pytest.mark.parametrize(
    ("text", "collision"),
    [
        ("ban dt cu dc 3 cu", True),
        ("nhan tien ban dt cu 3 cu", True),
        ("dì Xuân cho vay 8 củ mua xe máy cũ", True),
        ("thưởng dự án 3 củ", False),
        ("chu Ba muon tam cua t 5 cu, cuoi thang gui lai", False),
        ("ban xe dap cu dc 1tr3", False),
    ],
)
def test_cu_collision_needs_a_slang_span_and_another_cu_word(text: str, collision: bool) -> None:
    assert vr.cu_collision(text, propose_value(text)) is collision


@pytest.mark.parametrize(
    ("text", "reasons", "stratum"),
    [
        ("grabfood tối 89k ship 15k", ["multiple_money_candidates"], "multi_number_other"),
        ("thuong tet 2 thang luong", ["no_candidate"], "flagged_other"),
        ("rut 500 tieu tuan nay", ["bare_number"], "flagged_other"),
        ("hai tram nghin mua hoa tang me", ["number_words"], "slang_other"),
        ("tiền nhà tháng 10 bốn triệu", ["number_words"], "slang_other"),
        ("lẩu vs team 4 người chia 180k", ["split_share"], "multi_number_quantity"),
        ("ban dt cu dc 3 cu", ["cu_collision"], "slang_cu"),
        ("vay 1 triệu 2", ["compound_amount"], "flagged_other"),
        ("tien dien thang 10 620k", [], "multi_number_period_or_ordinal"),
        ("mua 1 chỉ vàng 6tr8", [], "multi_number_quantity"),
        ("quà 20/10 cho mẹ 500k", [], "multi_number_date_or_time"),
        ("mua gạo 10kg 190k", [], "multi_number_attached_unit"),
        ("mình mượn Huy 300k nạp game, thứ 6 trả", [], "multi_number_weekday"),
        ("cho vay Nghia 3tr lai 2%", [], "multi_number_percent"),
        ("thưởng dự án 3 củ", [], "slang_cu"),
        ("Thắng vay 5 củ, hẹn t10 trả", [], "slang_cu"),
        ("ăn phở 50k", [], None),
        ("ăn lẩu vs mn chia 185k", [], None),  # clean: one amount, nothing to resolve
    ],
)
def test_flagged_notes_get_must_reasons_or_a_pattern_stratum(
    text: str, reasons: list[str], stratum: str | None
) -> None:
    p = propose_value(text)
    assert vr.must_reasons(text, p) == reasons
    assert vr.pattern_stratum(p) == stratum


# ---------------------------------------------------------------------------------------------
# Existing-data review: escalation per stratum, staging, the gate and the scorer
# ---------------------------------------------------------------------------------------------

EXISTING = scorer.DATASETS[vr.NAME]
AUTOS = {
    "multi_number_quantity": ["q1", "q2"],
    "slang_cu": ["u1"],
    rr.CLEAN_STRATUM: ["k1", "k2", "k3"],
}


def existing_queue() -> tuple[list[dict], dict[str, dict]]:
    rows = [
        ("m1", rr.MUST, "split_share", ["split_share"]),
        ("s1", rr.HARD, "multi_number_quantity", []),
        ("s2", rr.HARD, "multi_number_quantity", []),
        ("h1", rr.HARD, "slang_cu", []),
        ("a1", rr.AUDIT, rr.CLEAN_STRATUM, []),
        ("a2", rr.AUDIT, rr.CLEAN_STRATUM, []),
        ("a3", rr.AUDIT, rr.CLEAN_STRATUM, []),
    ]
    queue = [{"id": i, "review_group": g, "stratum": s, "reasons": r} for i, g, s, r in rows]
    provenance = {q["id"]: {**q, "provenance": "queued-for-human"} for q in queue}
    for stratum, ids in AUTOS.items():
        provenance |= {i: {"review_group": rr.AUTO, "stratum": stratum} for i in ids}
    return queue, provenance


def score_existing(wrong: tuple[str, ...] = (), skip: tuple[str, ...] = ()):
    """Score the queue with a human who disagrees with the proposal on ``wrong`` ids."""
    queue, provenance = existing_queue()
    good = human(type_="amount", target=SPAN)
    props = {"value": {q["id"]: good for q in queue}}
    labels = {
        "value": {
            q["id"]: human(type_="no_amount") if q["id"] in wrong else good
            for q in queue
            if q["id"] not in skip
        }
    }
    return scorer.score(EXISTING, queue, provenance, props, labels), provenance


def test_a_clean_review_passes_every_stratum() -> None:
    (detail, esc, verdicts), _ = score_existing()
    assert esc.status == "accept_auto_remainder" and esc.ids == ()
    assert set(verdicts.values()) == {"passed"}
    assert set(verdicts) == {"multi_number_quantity", "slang_cu", rr.CLEAN_STRATUM}
    assert detail["groups"][rr.AUDIT]["queued"] == 3


def test_an_error_in_one_stratum_expands_only_that_stratum() -> None:
    (detail, esc, verdicts), _ = score_existing(wrong=("s1",))
    assert esc.status == "escalated"
    assert esc.ids == ("q1", "q2")
    assert verdicts == {
        "multi_number_quantity": "failed",
        "slang_cu": "passed",
        rr.CLEAN_STRATUM: "passed",
    }
    assert detail["strata"]["multi_number_quantity"]["error_ids"] == ["s1"]
    assert detail["strata"]["slang_cu"]["errors"] == 0


def test_a_must_review_error_never_escalates() -> None:
    (detail, esc, verdicts), _ = score_existing(wrong=("m1",))
    assert detail["groups"][rr.MUST]["errors"] == 1
    assert esc.status == "accept_auto_remainder" and set(verdicts.values()) == {"passed"}


def test_the_clean_audit_tolerates_one_error_and_two_stop_the_clean_set_only() -> None:
    (_, esc, verdicts), _ = score_existing(wrong=("a1",))
    assert esc.status == "accept_auto_remainder" and verdicts[rr.CLEAN_STRATUM] == "passed"
    (_, esc, verdicts), _ = score_existing(wrong=("a1", "a2"))
    assert esc.status == "escalated" and esc.clean_audit == "expand_all"
    assert set(esc.ids) == {"k1", "k2", "k3"}  # the audit scope is the clean population
    assert verdicts[rr.CLEAN_STRATUM] == "failed"
    assert verdicts["slang_cu"] == "passed" and verdicts["multi_number_quantity"] == "passed"


def test_unlabelled_samples_leave_their_stratum_pending() -> None:
    (_, esc, verdicts), _ = score_existing(skip=("h1", "a3"))
    assert esc.status == "pending"
    assert verdicts["slang_cu"] == "pending" and verdicts[rr.CLEAN_STRATUM] == "pending"
    assert verdicts["multi_number_quantity"] == "passed"


def test_clean_expansion_is_released_in_stages() -> None:
    clean = [f"k{i:03d}" for i in range(vr.CLEAN_EXPANSION_STAGE * 2 + 7)]
    provenance = {i: {"review_group": rr.AUTO, "stratum": rr.CLEAN_STRATUM} for i in clean}
    provenance["q1"] = {"review_group": rr.AUTO, "stratum": "multi_number_quantity"}
    esc = rr.Escalation("escalated", {}, "expand_all", ("q1", *clean), ())
    first = scorer.escalation_ids(EXISTING, esc, provenance, set())
    assert "q1" in first  # a failed stratum is released at once
    stage1 = [i for i in first if i != "q1"]
    assert len(stage1) == vr.CLEAN_EXPANSION_STAGE
    assert scorer.escalation_ids(EXISTING, esc, provenance, set()) == first  # idempotent
    second = scorer.escalation_ids(EXISTING, esc, provenance, set(stage1) | {"q1"})
    assert len(second) == vr.CLEAN_EXPANSION_STAGE and not set(second) & set(stage1)
    targeted = scorer.DATASETS[rr.NAME]
    assert len(scorer.escalation_ids(targeted, esc, provenance, set())) == len(clean) + 1


def write_score(root: Path, dataset: str, verdicts: dict[str, str], inputs: list[Path]) -> None:
    entry = {
        "strata_status": verdicts,
        "inputs": {str(p.relative_to(root)): gate.sha256_file(p) for p in inputs},
    }
    path = root / gate.SCORE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"datasets": {dataset: entry}}), encoding="utf-8")


def test_gate_returns_the_passed_strata_of_a_fresh_score(tmp_path: Path) -> None:
    labels = tmp_path / "labels.jsonl"
    labels.write_text("{}\n", encoding="utf-8")
    verdicts = {"clean": "passed", "slang_cu": "failed", "multi_number_quantity": "pending"}
    write_score(tmp_path, "ds", verdicts, [labels])
    assert gate.load_gate(tmp_path, "ds") == {"clean"}


def test_gate_refuses_a_missing_foreign_or_stale_score(tmp_path: Path) -> None:
    with pytest.raises(gate.GateError, match="is missing"):
        gate.load_gate(tmp_path, "ds")
    labels = tmp_path / "labels.jsonl"
    labels.write_text("{}\n", encoding="utf-8")
    write_score(tmp_path, "ds", {"clean": "passed"}, [labels])
    with pytest.raises(gate.GateError, match="no score for 'other'"):
        gate.load_gate(tmp_path, "other")
    labels.write_text("{}\n{}\n", encoding="utf-8")  # a label arrived after scoring
    with pytest.raises(gate.GateError, match="stale.*labels.jsonl"):
        gate.load_gate(tmp_path, "ds")
    labels.unlink()
    with pytest.raises(gate.GateError, match="stale"):
        gate.load_gate(tmp_path, "ds")


def existing_tree(tmp_path: Path, label_wrong: tuple[str, ...] = ()) -> Path:
    """A tiny annotation-v2 tree: m1 must; s1, s2 sampled of 4; h1; a1, a2 audit; 5 autos."""
    v2 = tmp_path / "datasets" / "annotation-v2"
    v2.mkdir(parents=True)
    shutil.copytree(ROOT / "configs", tmp_path / "configs")
    ids = ["m1", "s1", "s2", "h1", "a1", "a2", "q1", "q2", "u1", "k1", "k2", "k3"]
    texts = {i: f"cf {n}0k" for n, i in enumerate(ids, start=1)}

    def span(i: str) -> dict:
        text = f"{ids.index(i) + 1}0k"
        return {"text": text, "start": 3, "end": 3 + len(text)}

    def proposal(i: str) -> dict:
        row = {"id": i, "annotation_status": "complete", "type": "amount", "target": span(i)}
        return row | {"confidence": 0.9, "reason": "r"}

    queue_rows = [
        ("m1", rr.MUST, "split_share", ["split_share"]),
        ("s1", rr.HARD, "multi_number_quantity", []),
        ("s2", rr.HARD, "multi_number_quantity", []),
        ("h1", rr.HARD, "slang_cu", []),
        ("a1", rr.AUDIT, rr.CLEAN_STRATUM, []),
        ("a2", rr.AUDIT, rr.CLEAN_STRATUM, []),
    ]
    queue = [
        {"id": i, "text": texts[i], "review_group": g, "stratum": s, "reasons": r, "split": "x"}
        for i, g, s, r in queue_rows
    ]
    prov = [{"id": q["id"], **{k: q[k] for k in ("review_group", "stratum")}} for q in queue]
    auto = {"q1": "multi_number_quantity", "q2": "multi_number_quantity", "u1": "slang_cu"}
    auto |= {"k1": rr.CLEAN_STRATUM, "k2": rr.CLEAN_STRATUM, "k3": rr.CLEAN_STRATUM}
    prov += [{"id": i, "review_group": rr.AUTO, "stratum": s} for i, s in auto.items()]
    write_jsonl(
        v2 / "value-queue.jsonl", [{"id": i, "text": texts[i], "split": "train"} for i in ids]
    )
    write_jsonl(v2 / "value-proposals.jsonl", [proposal(i) for i in ids])
    write_jsonl(v2 / "value-review-queue.jsonl", queue)
    write_jsonl(v2 / "value-review-provenance.jsonl", prov)
    labels = []
    for q in queue:
        row = {k: proposal(q["id"])[k] for k in ("id", "annotation_status", "type", "target")}
        if q["id"] in label_wrong:
            row = {"id": q["id"], "annotation_status": "complete", "type": "no_amount"}
            row["target"] = None
        labels.append(row)
    write_jsonl(v2 / "value-labels.jsonl", labels)
    return tmp_path


def test_scorer_writes_verdicts_and_a_stratum_escalation_queue(tmp_path: Path) -> None:
    root = existing_tree(tmp_path, label_wrong=("s1",))
    assert scorer.main(["--root", str(root), "--dataset", vr.NAME]) == 0
    entry = json.loads((root / gate.SCORE_PATH).read_text())["datasets"][vr.NAME]
    assert entry["strata_status"] == {
        "multi_number_quantity": "failed",
        "slang_cu": "passed",
        "clean": "passed",
    }
    assert entry["escalation"]["auto_accepted_ids_to_review"] == ["q1", "q2"]
    v2 = root / "datasets" / "annotation-v2"
    queued = [json.loads(line)["id"] for line in (v2 / "value-escalation-queue.jsonl").open()]
    proposed = (v2 / "value-escalation-proposals.jsonl").open()
    proposals = [json.loads(line)["id"] for line in proposed]
    assert queued == proposals == ["q1", "q2"]
    assert "--labels datasets/annotation-v2/value-labels.jsonl" in " ".join(
        entry["escalation"]["queue"]["quet_commands"]
    )
    assert gate.load_gate(root, vr.NAME) == {"slang_cu", "clean"}


def test_scorer_rescoring_a_corrected_review_removes_the_escalation_queue(tmp_path: Path) -> None:
    root = existing_tree(tmp_path, label_wrong=("s1",))
    assert scorer.main(["--root", str(root), "--dataset", vr.NAME]) == 0
    v2 = root / "datasets" / "annotation-v2"
    assert (v2 / "value-escalation-queue.jsonl").exists()
    fixed = existing_tree(tmp_path / "again")
    (v2 / "value-labels.jsonl").write_bytes(
        (fixed / "datasets/annotation-v2/value-labels.jsonl").read_bytes()
    )
    assert scorer.main(["--root", str(root), "--dataset", vr.NAME]) == 0
    assert not (v2 / "value-escalation-queue.jsonl").exists()
    assert gate.load_gate(root, vr.NAME) == {"multi_number_quantity", "slang_cu", "clean"}
