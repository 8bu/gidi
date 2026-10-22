#!/usr/bin/env bash
# gidi-en student v1, seed 1. Run detached (a long job): see the bash `name` service used.
# Training data: silver-train + labels-seed. Vocabulary: pruned from TRAIN notes only.
# Test is read only by evaluate_en_student.py, after training and export.
set -euo pipefail
cd "$(dirname "$0")/../.."
E=experiments/gidi-en-student-v1
W=models/gidi-en-v1
[ -d models/gidi-en-v1-init ] || uv run python scripts/build_en_student.py \
  --out models/gidi-en-v1-init --train-notes datasets/annotation-en-v1/splits/train.jsonl \
  --max-rows 9000 --ffn-k 2048 2>&1 | tee $E/build.log
uv run python scripts/train_en_student.py --init models/gidi-en-v1-init --out-dir $E/runs/seed1 \
  --weights-dir $W/seed1 --seed 1 --epochs 40 --batch-size 8 --lr 5e-5 --head-lr 1e-3 \
  --warmup-ratio 0.1 --device mps 2>&1 | tee $E/train-seed1.log
uv run python scripts/export_onnx.py --checkpoint $W/seed1 --out ${W}-onnx/seed1 \
  --report $E/export-seed1.json --notes datasets/annotation-en-v1/splits/dev.jsonl --overwrite
uv run python scripts/evaluate_en_student.py --onnx-dir ${W}-onnx/seed1 --checkpoint $W/seed1 \
  --out $E/results.json
echo ALL DONE
