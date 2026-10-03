# Models

Local model artifacts live here: downloaded pretrained checkpoints, tokenizer files, exported
ONNX graphs, and any quantized variants produced during export work.

Everything in this directory is gitignored except this README. Model files are large and
reproducible from their source and the commands recorded in `experiments/`; they do not belong
in version control.

Pretrained encoders are read from the Hugging Face cache, not copied here. Current contents:

- `baseline-v1/<model>/<run>/`: the best-epoch weights from the 10-epoch baseline sweep;
- `baseline-v1-e20/<model>/<run>/`: the final baseline runs (see
  `experiments/baseline-v1/report.md`), written by `scripts/train_baseline.py`;
- `baseline-v1-onnx/<model>/`: `model.onnx` and `model.int8.onnx` from `scripts/export_onnx.py`.
- `targeted-v2/bamibert/<run>/`: BamiBERT trained with the baseline recipe on training-v2
  (frozen train + targeted-02); see `experiments/targeted-v2/report.md`.
- `targeted-v3/bamibert/<run>/`: the same recipe on training-v3 (+ targeted-03); not chosen as
  the teacher, see `experiments/targeted-v3/report.md`.
- `distillation-v1/teacher/bamibert/lr5e-05-seed1/`: the final distillation teacher (targeted-v2
  recipe on training-v2 + frozen validation, fixed 13 epochs); `distillation-v1/{supervised,
  distilled}/student-4x256/seed<N>/`: the 4×256 students; `distillation-v1-onnx/`: ONNX FP32 +
  INT8 of the distilled seed-1 student. See `experiments/distillation-v1/report.md`.
- `distillation-v2/{supervised,distilled}/student-4x768[-pretrained]/seed<N>/`: the 4×768
  diagnostic students (random init, or BamiBERT layers 3/6/9/12 copied); `distillation-v2-onnx/`:
  ONNX FP32 + INT8 of the pretrained supervised seed-1 student. See
  `experiments/distillation-v2/report.md`.
- `compression-v1/positions/…`: the distillation-v2 pretrained students with the position table
  cut to 34 rows (bitwise lossless); `compression-v1/vocab/<policy>/`: pruned tokenizers +
  `vocab_map.json`; `compression-v1/supervised/student-4x768-pretrained-pos32-vocab-B-rank-8000/
  seed<N>/`: the retrained pruned students; `compression-v1-onnx/`: ONNX FP32 + INT8 exports. See
  `experiments/compression-v1/report.md`.
- `compression-v2/supervised/student-3x768-pretrained-pos32-vocab-B-rank-8000/seed<N>/`: the
  3×768 students (BamiBERT layers 4/8/12, pos34, B-rank-8000; not adopted);
  `compression-v2-onnx/`: ONNX FP32 + INT8 of seed 1. See `experiments/compression-v2/report.md`.
- `compression-v3/ffn/<criterion>-K<k>/ffn_map.json`: FFN neuron maps;
  `compression-v3/supervised/student-4x768-pretrained-pos32-vocab-B-rank-8000-ffn-combined-K2048/
  seed<N>/`: the FFN-pruned 4×768 students; `compression-v3-onnx/`: ONNX FP32 + INT8 of seed 1.
  See `experiments/compression-v3/report.md`.
- `compression-v4/ffn/combined-K2304/ffn_map.json`,
  `compression-v4/supervised/student-4x768-pretrained-pos32-vocab-B-rank-8000-ffn-combined-K2304/
  seed<N>/`, `compression-v4-onnx/`: the FFN-2304 check (not adopted; the deployment student is
  compression-v3's FFN 2048). See `experiments/compression-v4/report.md`.
- `gidi-finance-v1/`: the immutable deployment bundle (INT8 ONNX, tokenizer, vocab map, config,
  manifest with checksums) built by `scripts/freeze_deployment.py` from compression-v3 seed 1.
  Read-only; never edit in place. See `docs/deployment.md`.

Everything is reproducible from the commands in `experiments/baseline-v1/`.
