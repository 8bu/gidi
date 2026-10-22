"""Word-boundary snapping in the English evaluation decode, and the v2 slang silver loader."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from evaluate_onnx import decode, snap_span

from gidi.modeling.preprocessing import TYPES
from gidi.training.english import load_slang_silver


def test_snap_span_expands_fragments_to_whole_words() -> None:
    text = "lunch w ader 15"
    start = text.index("der")
    span = {"text": "der", "start": start, "end": start + 3}
    snapped = snap_span(span, text)
    assert snapped == {"text": "ader", "start": 8, "end": 12}
    assert snap_span(None, text) is None
    whole = {"text": "lunch", "start": 0, "end": 5}
    assert snap_span(whole, text) == whole


def test_decode_snap_flag_only_moves_span_outward() -> None:
    text = "paid osa back 20"
    # tokens: "paid", "os" (fragment tagged B), "a", "back"; offsets are character spans
    offsets = [(0, 4), (5, 7), (7, 8), (9, 13)]
    data = SimpleNamespace(records=[{"text": text}], encoded=SimpleNamespace(offsets=[offsets]))
    tags = np.zeros((4, 3), dtype=np.float32)
    tags[:, 0] = 1.0
    tags[1] = [0.0, 1.0, 0.0]  # B on the fragment only
    type_logits = np.zeros((1, len(TYPES)), dtype=np.float32)
    _, raw = decode(data, type_logits, [tags])
    _, snapped = decode(data, type_logits, [tags], snap_words=True)
    assert raw[0]["text"] == "os"
    assert snapped[0] == {"text": "osa", "start": 5, "end": 8}


def test_load_slang_silver_keeps_only_consistent(tmp_path: Path) -> None:
    rows = [
        {
            "id": "a",
            "text": "gym 30",
            "type": "expense",
            "target": None,
            "consistent": True,
            "annotation_status": "complete",
        },
        {
            "id": "b",
            "text": "lent bro 30",
            "type": "lend",
            "target": {"text": "bro", "start": 5, "end": 8},
            "consistent": False,
            "annotation_status": "complete",
        },
    ]
    (tmp_path / "silver-slang-train.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8"
    )
    out = load_slang_silver(tmp_path)
    assert [r["id"] for r in out] == ["a"]
