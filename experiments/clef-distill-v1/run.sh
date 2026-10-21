#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
E=experiments/clef-distill-v1
for arm in A B; do
  l=$(echo $arm | tr A-Z a-z)
  d=datasets/annotation-v3/training-v6$l
  uv run python scripts/build_teacher_targets.py --teacher models/distillation-v1/teacher/bamibert/lr5e-05-seed1 --data $d/train.jsonl --out $E/arm$arm/teacher-targets --device cpu
  uv run python scripts/train_student.py --arm supervised --student student-4x768 --init pretrained --truncate-positions --vocab-spec models/compression-v1/vocab/B-rank-8000 --ffn-map models/compression-v3/ffn/combined-K2048 --retokenize --lr 5e-5 --head-lr 1e-3 --epochs 40 --targets $E/arm$arm/teacher-targets --splits-dir $d --out-dir $E/arm$arm/runs --weights-dir $E/arm$arm/weights --seed 1 --seed 2 --seed 3 2>&1 | tee $E/arm$arm/train-seeds123.log
  for s in 1 2 3; do
    uv run python scripts/export_onnx.py --checkpoint $E/arm$arm/weights/supervised/student-4x768-pretrained-pos32-vocab-B-rank-8000-ffn-combined-K2048/seed$s --out $E/arm$arm/onnx/seed$s --report $E/arm$arm/export-seed$s.json
  done
done
