from __future__ import annotations

import random
from collections import Counter

import pytest

from gidi.annotation.split import (
    SPLITS,
    accented,
    assign_splits,
    leakage_groups,
    normalize,
    template_key,
)


def rec(id_: str, text: str, type_: str = "expense") -> dict:
    return {"id": id_, "text": text, "type": type_, "accented": accented(text)}


def words(n: int, seed: int) -> list[str]:
    """``n`` pseudo-random, mutually dissimilar 3-word phrases (never near-duplicates)."""
    rng = random.Random(seed)
    letters = "bcdfghklmnpqrstvxz"
    return [" ".join("".join(rng.choices(letters, k=7)) for _ in range(3)) for _ in range(n)]


def synthetic(n_groups: int = 120) -> list[dict]:
    """``n_groups`` distinct utterances (2 types x 2 accent modes), each with a 1-3 record group."""
    records = []
    phrases = words(n_groups, seed=1)
    for g in range(n_groups):
        type_ = ("expense", "income")[g % 2]
        base = f"{phrases[g]} {'tiền' if g % 4 < 2 else 'tien'}"
        for copy in range(1 + g % 3):
            records.append(rec(f"g{g:03d}-{copy}", f"{base} {100 + copy}k", type_))
    return records


def test_normalize_strips_accents_masks_amounts_and_punctuation():
    assert normalize("Đã trả nợ Hùng 1.500k!") == "da tra no hung N !"
    assert normalize("ăn  trưa 80k") == "an trua N"
    assert normalize("mua 4P bánh") == "mua 4p banh"


def test_direction_mirrors_share_a_group():
    records = [
        rec("a", "Hùng trả nợ 500k", "repayment_out"),
        rec("b", "trả nợ Hùng 500k", "repayment_out"),
        rec("c", "tra gop cho Vinh ky 3 1tr", "repayment_out"),
        rec("d", "Vinh tra gop ky 3 cho minh 1tr", "repayment_in"),
        rec("e", "cà phê với Nam 45k"),
    ]
    assert leakage_groups(records) == [["a", "b"], ["c", "d"], ["e"]]


def test_near_duplicates_group_but_length_gap_and_different_text_do_not():
    records = [
        rec("a", "tien dien thang 10 620k"),
        rec("b", "tiền điện tháng 10 hết 612k"),
        rec("c", "tien dien thang 10 620k cho nha chu tro o quan 7"),
        rec("d", "grab den san bay 250k"),
    ]
    groups = leakage_groups(records)
    assert ["a", "b"] in groups
    assert ["c"] in groups and ["d"] in groups


def test_groups_are_sorted_and_independent_of_input_order():
    records = [rec("b", "mua sach 50k"), rec("a", "mua sach 70k"), rec("z", "đi chợ 200k")]
    assert leakage_groups(records) == [["a", "b"], ["z"]]
    assert leakage_groups(list(reversed(records))) == [["a", "b"], ["z"]]


def test_mirrored_and_near_duplicate_records_never_cross_splits():
    records = synthetic()
    records += [
        rec("m1", "Hùng trả nợ 500k", "repayment_out"),
        rec("m2", "trả nợ Hùng 500k", "repayment_out"),
        rec("n1", "tien dien thang 10 620k"),
        rec("n2", "tiền điện tháng 10 hết 612k"),
    ]
    for seed in ("s1", "s2", "s3", "s4"):
        split = assign_splits(records, seed=seed)
        assert split["m1"] == split["m2"]
        assert split["n1"] == split["n2"]


def test_every_group_lands_wholly_in_one_split():
    records = synthetic()
    split = assign_splits(records, seed="whole")
    assert set(split) == {r["id"] for r in records}
    for group in leakage_groups(records):
        assert len({split[i] for i in group}) == 1
    assert any(len(g) > 1 for g in leakage_groups(records))


def test_same_seed_is_identical_and_seeds_can_differ():
    records = synthetic()
    first = assign_splits(records, seed="same")
    assert assign_splits(records, seed="same") == first
    assert assign_splits(list(reversed(records)), seed="same") == first
    assert any(assign_splits(records, seed=f"other-{i}") != first for i in range(5))


def test_proportions_follow_ratios_within_each_stratum():
    records = synthetic(200)
    split = assign_splits(records, seed="ratios")
    strata = Counter((r["type"], r["accented"]) for r in records)
    assert len(strata) == 4
    for key, size in strata.items():
        counts = Counter(split[r["id"]] for r in records if (r["type"], r["accented"]) == key)
        for name, ratio in zip(SPLITS, (0.70, 0.15, 0.15), strict=True):
            assert abs(counts[name] - ratio * size) <= 4, (key, name, counts)


def test_held_out_splits_cover_every_type_with_at_least_three_groups():
    # rare types have 3 groups: the deficit rule alone would leave validation/test empty of them
    records = synthetic(100)
    rare = words(8, seed=2)
    for g in range(3):
        for k, t in enumerate(("refund", "transfer")):
            records.append(rec(f"{t}{g}", f"{rare[g * 2 + k]} tiền 10k", t))
    records.append(rec("lend-a", f"{rare[6]} 1tr", "lend"))
    records.append(rec("lend-b", f"{rare[7]} 2tr", "lend"))
    for seed in ("c1", "c2", "c3", "c4", "c5"):
        split = assign_splits(records, seed=seed)
        for held_out in ("validation", "test"):
            types = {r["type"] for r in records if split[r["id"]] == held_out}
            assert {"expense", "income", "refund", "transfer"} <= types, (seed, held_out)


def targeted(id_: str, text: str, target: str | None, type_: str = "expense") -> dict:
    span = None
    if target is not None:
        start = text.index(target)
        span = {"text": target, "start": start, "end": start + len(target)}
    return {**rec(id_, text, type_), "target": span}


def test_template_key_masks_target_span_and_amounts():
    a = targeted("a", "cho Long mượn 1tr", "Long", "lend")
    b = targeted("b", "cho Linh muon 200k", "Linh", "lend")
    assert template_key(a) == template_key(b) == "cho § muon N"
    assert template_key(targeted("c", "tiền nước 120k", None)) == "tien nuoc N"


def test_template_key_rejects_offsets_that_do_not_match_the_text():
    bad = {**rec("x", "cho Long mượn 1tr"), "target": {"text": "Long", "start": 0, "end": 4}}
    with pytest.raises(ValueError, match="does not match"):
        template_key(bad)


def test_target_substitution_templates_share_a_group_even_when_not_similar():
    long_name = "Nguyễn Thị Hồng Nhung"
    records = [
        targeted("a", "chuyển khoản cho Long tiền học 1tr", "Long", "lend"),
        targeted("b", f"chuyển khoản cho {long_name} tiền học 2tr", long_name, "lend"),
        targeted("c", "đi grab ra sân bay 250k", None),
    ]
    assert leakage_groups(records) == [["a", "b"], ["c"]]
    # without the target the same texts are neither mirrors nor near-duplicates
    unmasked = [{**r, "target": None} for r in records]
    assert leakage_groups(unmasked) == [["a"], ["b"], ["c"]]


def test_amount_only_substitution_and_exact_duplicates_share_a_group():
    records = [
        targeted("a", "khám răng 400k", None),
        targeted("b", "khám răng 1tr2", None),
        targeted("c", "khám răng 400k", None),
        targeted("d", "mua vé xem phim 90k", None),
    ]
    assert leakage_groups(records) == [["a", "b", "c"], ["d"]]


def test_substitution_templates_never_cross_splits():
    records = synthetic()
    lend = ("Long", "Nguyễn Thị Hồng Nhung", "Hùng", "Trần Văn Bảo Long", "Bảo")
    repay = ("Thảo", "Hoa", "Lê Thị Thanh Mai", "Lan", "Phạm Ngọc Vy Anh")
    for k, name in enumerate(lend):
        records.append(targeted(f"t{k}", f"cho {name} mượn {k + 1}tr", name, "lend"))
    for k, name in enumerate(repay):
        records.append(targeted(f"u{k}", f"trả nợ chị {name} {k + 2}00k", name, "repayment_out"))
    for seed in ("s1", "s2", "s3", "s4", "s5"):
        split = assign_splits(records, seed=seed)
        for prefix in ("t", "u"):
            assert len({split[f"{prefix}{k}"] for k in range(5)}) == 1, (seed, prefix)
