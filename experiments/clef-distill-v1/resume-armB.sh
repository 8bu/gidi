#!/usr/bin/env bash
# resumes arm B after the first launch was killed during seed 2 (seed 1 had finished)
set -euo pipefail
cd "$(dirname "$0")/../.."
E=experiments/clef-distill-v1
d=datasets/annotation-v3/training-v6b
uv run python scripts/train_student.py --arm supervised --student student-4x768 --init pretrained --truncate-positions --vocab-spec models/compression-v1/vocab/B-rank-8000 --ffn-map models/compression-v3/ffn/combined-K2048 --retokenize --lr 5e-5 --head-lr 1e-3 --epochs 40 --targets $E/armB/teacher-targets --splits-dir $d --out-dir $E/armB/runs --weights-dir $E/armB/weights --seed 2 --seed 3 > $E/armB/train-seeds23.log 2>&1
for s in 1 2 3; do
  uv run python scripts/export_onnx.py --checkpoint $E/armB/weights/supervised/student-4x768-pretrained-pos32-vocab-B-rank-8000-ffn-combined-K2048/seed$s --out $E/armB/onnx/seed$s --report $E/armB/export-seed$s.json
done
