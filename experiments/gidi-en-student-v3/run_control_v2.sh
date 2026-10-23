#!/usr/bin/env bash
# Noise control (not part of the v3 verdict): retrain v2 seed 1 with the unchanged v2 command to
# see how far the same recipe moves between two MPS runs. Weights/ONNX go to /tmp.
set -euo pipefail
cd "$(dirname "$0")/../.."
E=experiments/gidi-en-student-v3/control-v2
W=/tmp/gidi-en-v2-control
mkdir -p $E
uv run python scripts/train_en_student.py --init models/gidi-en-v1-init --out-dir $E/runs \
  --weights-dir $W/w --seed 1 --epochs 40 --batch-size 8 --lr 5e-5 --head-lr 1e-3 \
  --warmup-ratio 0.1 --device mps --slang-silver 2>&1 | tee $E/train.log
uv run python scripts/export_onnx.py --checkpoint $W/w --out $W/onnx \
  --report $E/export.json --notes datasets/annotation-en-v1/splits/dev.jsonl --overwrite
uv run python scripts/evaluate_en_student.py --onnx-dir $W/onnx --checkpoint $W/w --out $E/results.json
uv run python scripts/evaluate_en_probe.py --probe datasets/probe-en-slang --onnx-dir $W/onnx \
  --checkpoint $W/w --out $E/slang-probe-snap.json --snap-words
echo ALL DONE
