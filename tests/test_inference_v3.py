"""A bundle whose config declares ``value_source: rule-parser`` / ``target_snap: words`` (v3).

The throwaway bundle is the frozen v1 bundle with those config keys added, so the switch is tested
without the v3 model; v1/v2 behaviour must not change.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from gidi import value_parser
from gidi.inference import BundleError, GidiPredictor, verify_bundle
from gidi.inference.bundle import BUNDLE_FILES, CONFIG_FILE, MANIFEST_FILE, load_config, sha256_file
from gidi.value_parser import parse_value

ROOT = Path(__file__).resolve().parents[1]
V1_BUNDLE = ROOT / "models" / "gidi-finance-v1"

V3_RULES = {
    "value_source": "rule-parser",
    "value_parser": {"name": "gidi.value_parser", "version": value_parser.VERSION},
    "target_snap": "words",
}
V3_KEYS = [
    "type",
    "type_confidence",
    "target",
    "target_span",
    "target_confidence",
    "value_text",
    "value_span",
    "value_confidence",
    "truncated",
    "model_version",
]
# Notes whose raw target span is a word fragment (snap changes it) and notes with / without value.
NOTES = [
    "rut tien vpbank 1tr",
    "thanh toán thẻ hsbc 7,2tr",
    "pizza 4p tối t7 389.000đ",
    "cho ban Quang muon 2tr",
    "😀 trả nợ chị Mai 500k",
    "mượn chú hai 5 xị",
    "ăn phở",
]

pytestmark = pytest.mark.skipif(not V1_BUNDLE.is_dir(), reason="deployment bundle not built")


def build_v3_bundle(dest: Path, **config_changes) -> Path:
    """Copy the v1 bundle to ``dest`` with the v3 rules (plus ``config_changes``) in its config."""
    dest.mkdir(parents=True)
    for name in BUNDLE_FILES:
        shutil.copyfile(V1_BUNDLE / name, dest / name)
    config = json.loads((V1_BUNDLE / CONFIG_FILE).read_text(encoding="utf-8"))
    config.update(V3_RULES)
    config.update(config_changes)
    config = {key: value for key, value in config.items() if value is not None}
    (dest / CONFIG_FILE).write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n")
    manifest = json.loads((V1_BUNDLE / MANIFEST_FILE).read_text(encoding="utf-8"))
    manifest["files"][CONFIG_FILE] = {
        "sha256": sha256_file(dest / CONFIG_FILE),
        "bytes": (dest / CONFIG_FILE).stat().st_size,
    }
    (dest / MANIFEST_FILE).write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return dest


@pytest.fixture(scope="module")
def v3_bundle(tmp_path_factory) -> Path:
    return build_v3_bundle(tmp_path_factory.mktemp("v3") / "bundle")


@pytest.fixture(scope="module")
def v1() -> GidiPredictor:
    return GidiPredictor.from_bundle(V1_BUNDLE)


@pytest.fixture(scope="module")
def v3(v3_bundle) -> GidiPredictor:
    return GidiPredictor.from_bundle(v3_bundle)


def test_config_declares_the_rules_and_the_bundle_still_verifies(v3_bundle):
    verify_bundle(v3_bundle)
    config = load_config(v3_bundle)
    assert config.value_source == "rule-parser"
    assert config.value_parser_version == value_parser.VERSION
    assert config.target_snap == "words"
    assert config.value_labels is None and config.value_decoding is None
    assert config.output_names == ("type_logits", "tag_logits")


def test_v1_bundle_config_has_no_rules():
    config = load_config(V1_BUNDLE)
    assert config.value_source is None and config.target_snap is None


def test_value_comes_from_the_rule_parser_on_the_callers_string(v3):
    assert v3.has_value_head
    found = missing = False
    for text in NOTES:
        out = v3.predict(text).to_dict()
        assert list(out) == V3_KEYS
        expected = parse_value(text)
        assert out["value_confidence"] is None
        if expected is None:
            missing = True
            assert out["value_text"] is None and out["value_span"] is None
            continue
        found = True
        assert out["value_span"] == [expected.start, expected.end]
        assert out["value_text"] == text[expected.start : expected.end]
    assert found and missing


def test_target_snap_follows_the_bundle_and_never_changes_type_or_value(v1, v3):
    snapped = GidiPredictor.from_bundle(V1_BUNDLE, snap_words=True)
    changed = False
    for text in NOTES:
        got = v3.predict(text)
        assert got.target_span == snapped.predict(text).target_span
        assert got.type == v1.predict(text).type
        plain = v1.predict(text).target_span
        changed |= plain != got.target_span
        if got.target_span is not None:
            assert got.target == text[got.target_span[0] : got.target_span[1]]
            assert got.target_span[0] == 0 or text[got.target_span[0] - 1].isspace()
            assert got.target_span[1] == len(text) or text[got.target_span[1]].isspace()
    assert changed, "the notes must include a fragment the snap extends"


def test_explicit_snap_flag_overrides_the_bundle(v1, v3_bundle):
    off = GidiPredictor.from_bundle(v3_bundle, snap_words=False)
    for text in NOTES:
        assert off.predict(text).target_span == v1.predict(text).target_span


def test_v1_bundle_keeps_its_behaviour(v1):
    """No value, no snap: the default of a bundle without the v3 keys."""
    no_snap = GidiPredictor.from_bundle(V1_BUNDLE, snap_words=False)
    assert not v1.has_value_head
    for text in NOTES:
        out = v1.predict(text).to_dict()
        assert list(out) == [key for key in V3_KEYS if not key.startswith("value_")]
        assert out == no_snap.predict(text).to_dict()


def test_a_bundle_frozen_with_another_parser_version_refuses_to_load(tmp_path):
    bundle = build_v3_bundle(
        tmp_path / "old-parser", value_parser={"name": "gidi.value_parser", "version": "0"}
    )
    with pytest.raises(BundleError, match="value parser version"):
        GidiPredictor.from_bundle(bundle)


@pytest.mark.parametrize(
    "changes",
    [
        {"value_source": "crf"},
        {"target_snap": "chars"},
        {"value_parser": None},
        {"value_parser": {"name": "other.parser", "version": "1"}},
        {"value_parser": {"name": "gidi.value_parser", "version": ""}},
        {"value_source": None},  # a value_parser block without value_source
        {"value_labels": ["O", "B-VALUE", "I-VALUE"]},  # rule parser excludes a value head
    ],
)
def test_inconsistent_rule_declarations_are_rejected(tmp_path, changes):
    bundle = build_v3_bundle(tmp_path / "bad", **changes)
    with pytest.raises(BundleError):
        load_config(bundle)
