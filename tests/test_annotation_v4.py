"""annotation-v4 contract and the rule-aware Clef teacher prompt (no model calls)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
V3 = yaml.safe_load((ROOT / "configs" / "annotation-v3.yaml").read_text(encoding="utf-8"))
V4 = yaml.safe_load((ROOT / "configs" / "annotation-v4.yaml").read_text(encoding="utf-8"))


def load_teacher():
    spec = importlib.util.spec_from_file_location("clef_teacher", ROOT / "scripts/clef_teacher.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_v4_extends_v3_and_restates_only_known_types() -> None:
    assert V4["version"] == "annotation-v4"
    assert V4["extends"] == V3["version"] == "annotation-v3"
    assert set(V4["rules"]) == {"R1", "R2", "R3", "R4"}
    assert set(V4["types"]) <= set(V3["types"])


def test_teacher_default_prompt_is_unchanged_and_rules_prompt_adds_r1_to_r4() -> None:
    teacher = load_teacher()
    text = "long tra 5 lit"
    plain = teacher.questions(text)
    ruled = teacher.questions(text, rules=True)
    assert plain["type"]["instructions"] == teacher.TYPE_INSTRUCTIONS
    assert plain["target"]["instructions"] == teacher.TARGET_INSTRUCTIONS
    assert ruled["type"]["instructions"].startswith(teacher.TYPE_INSTRUCTIONS)
    for rule in ("R1", "R2", "R3", "R4"):
        assert rule in ruled["type"]["instructions"]
    assert ruled["target"]["instructions"].startswith(teacher.TARGET_INSTRUCTIONS)
    assert "'cong ty'" in ruled["target"]["instructions"]
    assert ruled["type"]["criteria"] == plain["type"]["criteria"]
    assert ruled["target"]["criteria"] == plain["target"]["criteria"]
