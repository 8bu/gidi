"""Ad hoc check of the v3 field-issue notes: v3 vs clef-distill-v1 arm A (run from repo root)."""

import sys

sys.path.insert(0, "src")
from gidi.inference import GidiPredictor

B = "models/gidi-finance-v3"
cases = [  # (note, expected type, expected target)  -- from experiments/v3-field-issues.md
    ("Chuyển cho họ hàng 2 củ", "expense", None),
    ("trả giùm tiền cơm Khoa 50k", "lend", "Khoa"),
    ("moi nhan luong 26cu", "income", None),
    ("moi lanh luong 26tr", "income", None),
    ("long tra 5 lit", "repayment_in", "long"),
    ("Long tra 5 lit", "repayment_in", "Long"),
    ("chị Linh trả 2tr", "repayment_in", "Linh"),
    ("cong ty tra tien t10 5 lit", "income", None),
    ("khach tra tien 500k", "income", "khach"),
    ("chuyển sang ví momo 1tr", "transfer", None),
    ("chuyển cho mẹ 2 củ", "expense", "mẹ"),
    ("mới nhận lương 26cu", "income", None),
    ("lãnh lương 18tr", "income", None),
    ("ứng lương 3tr", "borrow", None),
    ("ứng giùm Lan tiền điện 400k", "lend", "Lan"),
    ("mua hộ Tú 200k", "lend", "Tú"),
    ("Khoa trả lại tiền mình trả giùm 50k", "repayment_in", "Khoa"),
]
systems = {"v3": GidiPredictor.from_bundle(B)}
for s in (1, 2, 3):
    systems[f"A{s}"] = GidiPredictor.from_bundle(
        B, model_path=f"experiments/clef-distill-v1/armA/onnx/seed{s}/model.int8.onnx"
    )
score = {k: 0 for k in systems}
for note, et, eg in cases:
    row = []
    for k, p in systems.items():
        o = p.predict(note)
        ok = o.type == et and (o.target or None) == (eg if eg else None)
        score[k] += ok
        row.append(f"{k}:{o.type}/{o.target}{'' if ok else ' ✗'}")
    print(f"{note!r:42} exp {et}/{eg} | " + " | ".join(row))
print({k: f"{v}/{len(cases)}" for k, v in score.items()})
