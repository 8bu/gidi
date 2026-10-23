#!/usr/bin/env bash
# gidi-en student v3, seed 1: v2 recipe unchanged + the value-span head (scripts/train_en_student.py
# --value). Run detached. Protocol: experiments/gidi-en-student-v3/protocol.json (written before
# training). Scoring: scripts/evaluate_en_value.py (type/target as v2, value, scanner comparison).
set -euo pipefail
cd "$(dirname "$0")/../.."
E=experiments/gidi-en-student-v3
W=models/gidi-en-v3
uv run python scripts/train_en_student.py --init models/gidi-en-v1-init --out-dir $E/runs/seed1 \
  --weights-dir $W/seed1 --seed 1 --epochs 40 --batch-size 8 --lr 5e-5 --head-lr 1e-3 \
  --warmup-ratio 0.1 --device mps --slang-silver --value 2>&1 | tee $E/train-seed1.log
uv run python scripts/export_value_onnx.py --checkpoint $W/seed1 --out ${W}-onnx/seed1 \
  --report $E/export-seed1.json --notes datasets/annotation-en-v1/splits/dev.jsonl \
  --v1-int8 models/gidi-en-v2-onnx/seed1/model.int8.onnx \
  --v1-fp32 models/gidi-en-v2-onnx/seed1/model.onnx --overwrite
uv run python scripts/evaluate_en_value.py --onnx-dir ${W}-onnx/seed1 --checkpoint $W/seed1 \
  --out $E/results.json
echo ALL DONE
