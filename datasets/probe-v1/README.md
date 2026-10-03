# probe-v1: diagnostic probe set (EVALUATION ONLY, NEVER TRAIN)

This set holds 81 fresh notes that probe the nine training-coverage gaps found in
`experiments/baseline-v1/error-audit.md`. It is labelled under frozen annotation-v1 semantics.

**It must never become training data**, and no augmentation may copy or paraphrase it.
`gidi.training.train.load_split` refuses any record whose id starts with `probe-`.

| file | content |
|---|---|
| `notes.jsonl` | `{id, text, pattern}`; `pattern` is the gap the note probes (evaluation slice only) |
| `queue.jsonl` | `{id, text, corpus, position}`, the annotation queue (no pattern, so annotation stays blind) |
| `labels.jsonl` | annotation-v1 labels |
| `provenance.jsonl` | `annotator` `ai` (two blind passes agreed) or `human` (Quet review) |
| `recheck.jsonl`, `proposals.jsonl` | the escalated exceptions sent to Quet |
| `manifest.json` | counts and sha256 hashes |

The notes were written from pattern descriptions, not from existing records. They were checked
against every annotation-v1 text (train, validation, test, uncertain and skipped) for exact,
token-bag, and near-duplicate overlap.

Score existing checkpoints with `uv run python scripts/evaluate_probe.py`.
