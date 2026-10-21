# Web playground deployment: gidi-finance-v4 4.0.0 at https://gidi.8bu.dev

Parity: `web-parity.md` / `web-parity.json` (no semantic mismatch). Same process as
`experiments/deployment-v3/web-deploy.md`.

| | |
|---|---|
| Worker | `gidi`, version id `2a74b9b6-a486-43cb-8e14-b8907266498a` |
| R2 key | `gidi-finance-v4/4.0.0/model.int8.onnx` (28,658,533 B, sha256 `a702d35ec4501f617e7536f97013c0e512cdd201fd5043e62b9185b354d9d8e7`, uploaded with `--remote`, downloaded back and hashed; the live URL hashes the same) |
| Kept for rollback | `gidi-finance-v3/3.0.0/model.int8.onnx`, `gidi-finance-v2/2.0.2/model.int8.onnx` |

Live check (BrowserOS, deployed site): `/lab` shows `gidi-finance-v4`, "SHA-256 a702d35ec450… verified";
`mới nhận lương` -> income, no target; `chuyển cho mẹ` -> expense, target `mẹ` [11, 13); the notes app
files them under Thu nhập / Chi tiêu; no console errors.
