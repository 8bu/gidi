"""annotation-en-v1 loads in the existing validator and mirrors the annotation-v4 taxonomy."""

from __future__ import annotations

from pathlib import Path

import yaml

from gidi.annotation.schema import load_config, validate_annotation

ROOT = Path(__file__).resolve().parents[1]
EN = ROOT / "configs" / "annotation-en-v1.yaml"


def test_en_config_loads_with_v4_taxonomy() -> None:
    config = load_config(EN)
    v3 = load_config(ROOT / "configs" / "annotation-v3.yaml")
    assert config.version == "annotation-en-v1"
    assert config.types == v3.types
    assert config.null_target_types == v3.null_target_types
    assert config.statuses == v3.statuses
    assert config.queue["out"].startswith("datasets/annotation-en-v1/")
    raw = yaml.safe_load(EN.read_text(encoding="utf-8"))
    assert set(raw["rules"]) == {"R1", "R2", "R3", "R4"}
    assert raw["label_provenance"] == "silver-claude-clef"


def test_en_config_validates_a_record() -> None:
    config = load_config(EN)
    text = "venmo'd mom 50"
    record = {
        "id": "x",
        "annotation_status": "complete",
        "type": "expense",
        "target": {"text": "mom", "start": 8, "end": 11},
    }
    assert validate_annotation(record, text, config) == []
