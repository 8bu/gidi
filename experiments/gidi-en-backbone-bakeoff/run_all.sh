#!/usr/bin/env bash
# gidi-en backbone bake-off: every run strictly one after another (train -> export -> evaluate),
# each step its own process so memory is released in between. Protocol: protocol.json (written
# before any build or training). Idempotent: a run whose results.json exists is skipped.
# Run detached:  bash experiments/gidi-en-backbone-bakeoff/run_all.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
B=experiments/gidi-en-backbone-bakeoff
INIT=models/gidi-en-bakeoff/init

build() { # candidate
  [ -f $INIT/$1/build_report.json ] || uv run python $B/build_candidate.py --candidate "$1" --out $INIT/$1
}

run() { # candidate init seed
  local c=$1 init=$2 s=$3
  local E=$B/runs/$c/seed$s W=models/gidi-en-bakeoff/$c/seed$s O=models/gidi-en-bakeoff-onnx/$c/seed$s
  [ -f $E/results.json ] && { echo "skip $c seed $s"; return; }
  echo "START $c seed $s $(date)"
  mkdir -p $E
  uv run python scripts/train_en_student.py --init "$init" --out-dir $E --weights-dir $W --seed "$s" \
    --epochs 40 --batch-size 8 --lr 5e-5 --head-lr 1e-3 --warmup-ratio 0.1 --device mps \
    --slang-silver --value 2>&1 | tee $E/train.log
  uv run python scripts/export_value_onnx.py --checkpoint $W --out $O --report $E/export.json \
    --notes datasets/annotation-en-v1/splits/dev.jsonl --v1-int8 /nonexistent \
    --v1-fp32 /nonexistent --overwrite
  uv run python scripts/evaluate_en_value.py --onnx-dir $O --checkpoint $W --out $E/results.json
  echo "DONE $c seed $s $(date)"
}

build distilroberta; build electra; build minilm
run roberta-base models/gidi-en-v1-init 2
run roberta-base models/gidi-en-v1-init 3
for c in distilroberta electra minilm; do
  for s in 1 2; do run $c $INIT/$c $s; done
done
echo ALL DONE
