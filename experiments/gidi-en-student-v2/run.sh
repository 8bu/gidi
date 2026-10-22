#!/usr/bin/env bash
# gidi-en student v2, seed 1: v1 recipe unchanged + silver-slang-train. Run detached.
# Scoring: scripts/evaluate_en_student.py (protocol: experiments/gidi-en-student-v2/protocol.json,
# thresholds from the v1 protocol + the v2 slang success criteria), probes via evaluate_en_probe.py.
set -euo pipefail
cd "$(dirname "$0")/../.."
E=experiments/gidi-en-student-v2
W=models/gidi-en-v2
uv run python scripts/train_en_student.py --init models/gidi-en-v1-init --out-dir $E/runs/seed1 \
  --weights-dir $W/seed1 --seed 1 --epochs 40 --batch-size 8 --lr 5e-5 --head-lr 1e-3 \
  --warmup-ratio 0.1 --device mps --slang-silver 2>&1 | tee $E/train-seed1.log
uv run python scripts/export_onnx.py --checkpoint $W/seed1 --out ${W}-onnx/seed1 \
  --report $E/export-seed1.json --notes datasets/annotation-en-v1/splits/dev.jsonl --overwrite
uv run python scripts/evaluate_en_student.py --onnx-dir ${W}-onnx/seed1 --checkpoint $W/seed1 \
  --out $E/results.json
uv run python scripts/evaluate_en_probe.py --probe datasets/probe-en-slang \
  --onnx-dir ${W}-onnx/seed1 --checkpoint $W/seed1 --out $E/slang-probe.json
uv run python scripts/evaluate_en_probe.py --probe datasets/probe-en-slang \
  --onnx-dir ${W}-onnx/seed1 --checkpoint $W/seed1 --out $E/slang-probe-snap.json --snap-words
echo ALL DONE
