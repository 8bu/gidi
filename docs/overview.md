# Gidi Overview

## Pipeline

```text
synthetic/raw Vietnamese finance corpus
        ↓
review with Quet
        ↓
tokenizer audit
        ↓
choose pretrained encoder/tokenizer baseline
        ↓
optional domain adaptation / continued pretraining
        ↓
build labeled dataset
        ↓
train tiny classifier
        ↓
error analysis
        ↓
iterate dataset/model
        ↓
quantize + ONNX export
        ↓
on-device mobile inference
```

Steps after the tokenizer audit are planned, not implemented. Nothing in this project trains
a tokenizer or an encoder from scratch; pretrained candidates are benchmarked and only then
adapted if the measurements justify it.

## Tokenizer, encoder, heads

```text
Tokenizer
  converts raw text → tokens/IDs

Encoder
  learns contextual language representations

Classifier heads
  turn encoder representations → Gidi-specific predictions
```

These are three separate concerns and are kept separate in the code and in experiments. The
tokenizer decides how text becomes IDs and sets the vocabulary. The encoder turns those IDs
into contextual representations, where the meaning of a token depends on its neighbors. The
heads are small task-specific layers on top of the shared encoder representation; Gidi expects
at least a transaction-type head, a target-span head, and a tag head. A tokenizer change
invalidates encoder and head comparisons, so tokenizer audits come first.

## Intended architecture (not yet implemented)

```text
raw text
   |
   v
tokenizer  (Vietnamese-aware; pretrained, possibly adapted)
   |
   v
tiny encoder  (shared contextual representation)
   |
   +--> head: transaction type
   +--> head: target span
   +--> head: tags
```

Amount extraction and normalization may remain deterministic code outside the model. The
encoder is shared across heads; heads are cheap and replaceable. None of this is written yet.

## Data conventions

All data is JSONL.

```text
corpus/raw/         generated raw corpus, never overwritten
corpus/reviewed/    Quet-reviewed clean corpus
corpus/rejected/    exported rejects
datasets/           derived labeled datasets: train/ validation/ test/
prompts/            generation prompts
```

- `corpus/raw/` holds generated raw text as produced. The writer
  (`gidi.corpus.jsonl.write_jsonl`) refuses to overwrite existing files by default, so raw
  data is append-by-new-file, never edited in place.
- `corpus/reviewed/` holds what survived Quet review; `corpus/rejected/` holds the rejects
  exported from that review. Both are review outputs, not model inputs.
- `datasets/` holds derived labeled datasets, split into `train/`, `validation/`, `test/`.
  These are built from the reviewed corpus, not from raw.
- Raw corpus and labeled datasets are separate concepts at separate stages. A labeled dataset
  is a derived artifact and can be rebuilt; the raw corpus is the record of what was generated.

## Current milestone

Workspace -> raw Vietnamese finance corpus -> Quet review -> tokenizer audit.

Concretely: produce and store the raw corpus without overwriting it, review it in the external
Quet TUI, export the rejects, and then audit tokenizer candidates against the reviewed text.
No classifier training in this milestone.

## Device policy

Device selection is centralized in `src/gidi/device.py` via `resolve_device()`. Callers never
pick a device themselves. Detection order is Apple MPS first, then CUDA, then CPU, so the same
code runs on the primary development machine (Apple silicon), on CUDA hardware, and on CPU
without changes. Any Apple-MPS gaps must be handled inside the centralized path rather than by
special-casing call sites.

## Raw corpus format

A raw corpus file is JSONL: one JSON object per non-blank line. `gidi.corpus.schema` is the
single definition of the contract and the validator for it.

Required keys:

- `id`: string, non-empty, unique within the file
- `text`: string, non-empty, Unicode NFC, no leading/trailing whitespace, no newlines
- `source`: string, non-empty, e.g. `synthetic:<generator>` or `manual`

Optional keys: `prompt_id` (string), plus any other keys, which are preserved and ignored by
validation. Only `text` is consumed by the tokenizer audit. No labels live in the raw corpus;
they belong to `datasets/`, which is derived later from the reviewed text.

Validate files with:

```bash
uv run python scripts/validate_corpus.py corpus/raw/*.jsonl
```

The script prints a per-file summary (records, errors, warnings) and exits 1 if any file has
errors. Duplicate `id`s are errors; exact duplicate `text`s are warnings, since a repeated note
may be legitimate. Findings carry the physical line number of the file, and malformed JSON is
reported as an error rather than aborting the run.

Generation happens with the prompt in `prompts/raw-finance-notes.md`, which fixes the JSONL
output shape only: one `{"text": "..."}` object per line. The model supplies no metadata;
`id`, `source`, and `prompt_id` are attached locally by `scripts/ingest_generated_corpus.py`
(see [Corpus workflow](#corpus-workflow) below).

## Corpus workflow

```text
LLM generation
   ↓
generated temporary output
   ↓
ingest_generated_corpus.py
   ↓
canonical corpus/raw/*.jsonl
   ↓
validate_corpus.py
   ↓
Quet review
```

The model produces unlabeled note text and nothing else: one `{"text": "..."}` object per line,
no ids, no provenance, no labels. `id`, `source`, and `prompt_id` are attached locally by
`scripts/ingest_generated_corpus.py` from its command line, so a generation run is identified by
the command that ingested it rather than by anything the model wrote. Ids are content-derived —
the first 12 hex characters of the sha256 over `source`, `prompt_id`, and the final
NFC-normalized `text` (NUL-separated) — so re-ingesting the same generation yields identical
ids. Ingestion is strict and all-or-nothing: text is NFC-normalized, surrounding whitespace and
multiline text are reported as errors rather than repaired, and duplicate texts are reported as
warnings. Raw corpus files go through the no-overwrite writer, so an existing file is never
replaced without an explicit flag. The temporary model output is never the corpus: only the
ingestion output lands in `corpus/raw/`, and validation and Quet review run on that file.
