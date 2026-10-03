import json

import pytest

from gidi.training.train import load_split


def test_load_split_refuses_probe_records(tmp_path):
    path = tmp_path / "train.jsonl"
    record = {"id": "probe-v1-0123456789ab", "text": "vay cô Lan 3tr", "type": "borrow"}
    path.write_text(json.dumps({**record, "target": None}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="never train"):
        load_split(path)
