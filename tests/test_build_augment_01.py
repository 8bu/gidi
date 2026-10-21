import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("ba", ROOT / "scripts/build_augment_01.py")
ba = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ba)


def test_augment_valid_and_reproducible():
    out, manifest = ba.build()
    out2, _ = ba.build()
    assert out == out2 and out
    src = {
        json.loads(line)["id"]: json.loads(line)
        for line in ba.SOURCE.read_text().splitlines()
        if line.strip()
    }
    assert len({r["text"] for r in out}) == len(out)
    assert manifest["records"] == len(out) <= ba.MAX_TOTAL
    per_note: dict[str, int] = {}
    for r in out:
        o = src[r["provenance"]["derived_from"]]
        per_note[o["id"]] = per_note.get(o["id"], 0) + 1
        assert r["type"] == o["type"] and r["group"] == o["group"]
        assert r["text"] != o["text"] and r["text"] not in {s["text"] for s in src.values()}
        assert (r["target"] is None) == (o["target"] is None)
        if r["target"]:
            t = r["target"]
            assert r["text"][t["start"] : t["end"]] == t["text"]
            assert ba.strip_dia(t["text"]).lower() == ba.strip_dia(o["target"]["text"]).lower()
    assert max(per_note.values()) <= ba.MAX_PER_NOTE
