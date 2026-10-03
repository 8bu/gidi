# Gidi — GiaoDịch

[Tiếng Việt](README.md) · **English**

This is my experiment to build a tiny classifier.

Gidi is an experimental machine learning project building a tiny on-device model for short
Vietnamese personal-finance notes, such as `ăn phở 45k` or `nhận lương 15tr`. For one note it
predicts the transaction type, the counterparty (target) span and the amount (value) span.

The target is a model small enough to run inference locally on modern phones. There is no
server component and no inference API.

## Scope

Vietnamese-first. Inputs are short, informal notes a person writes for themselves; they are
not formal bookkeeping records. The model returns the amount as written (a span of the note);
converting it to a number stays deterministic code.

## Status

- **`gidi-finance-v2`** is the current testable candidate. It is the value-span-v7 dual
  encoder: the frozen v1 path gives type and target, a second fine-tuned path with a CRF gives
  the value span. 58.1 MB INT8 ONNX, about 3.5 ms p50 on one CPU thread. Release 2.0.3 is
  public on Hugging Face: [x8bu/gidi-finance](https://huggingface.co/x8bu/gidi-finance)
  (INT8). The GitHub release 2.0.3 has the INT8 and FP32 archives. In-browser demo:
  <https://gidi.8bu.dev>.
- **`gidi-finance-v1`** stays frozen and unchanged: type and target only, 28.66 MB INT8 ONNX.
- Research on a smaller value path is in progress; see [Roadmap](#roadmap).

Usage, output schema and known limitations: [docs/deployment.md](docs/deployment.md). The
research history is in [docs/investigation-journal.md](docs/investigation-journal.md).

Human review and annotation happen in [Quet](https://github.com/8bu/quet) (Quick Utility for
Evaluating Text), a separate open-source TUI. It is not a dependency of this project. Gidi
produces the queues and schemas that Quet consumes, and validates what it writes back. Agent
details are in `AGENTS.md`.

## Roadmap

Internal experiment names (`V7`, `V8`) are not release versions. Public models are named
`gidi-finance-vN`, with release versions such as `2.0.3`. The value path of `gidi-finance-v2`
comes from internal experiment V7.

**Current: `gidi-finance-v2`**

- Dual-encoder architecture: one encoder for type and target, a second encoder for the value span.
- INT8 ONNX distribution on Hugging Face and GitHub.
- Browser runtime (onnxruntime-web, WASM); public demo at <https://gidi.8bu.dev>.

**Research: internal experiment V8 (under evaluation)**

- Benchmarks a deterministic value parser against the V7 neural value encoder.
- Keeps the existing type and target model unchanged.
- Goal: smaller download, lower memory and lower inference cost without meaningful quality loss.
- Experimental only; it does not replace v2.

**Candidate next major release: `gidi-finance-v3` / `3.0.0` (planned only if V8 is validated)**

- Hybrid neural and deterministic architecture: one encoder for type and target, deterministic
  value extraction, no second value encoder.
- Materially smaller browser and runtime footprint.
- Not confirmed. If the V8 evidence does not support replacement, v2 stays the current release.

## Stack

- Python 3.12 managed with `uv`
- PyTorch, with Apple MPS as the primary device (cuda and cpu also supported)
- Hugging Face Transformers, Tokenizers, Datasets
- scikit-learn for evaluation and baselines
- pytest and ruff for tests, linting and formatting
- Deployment path: PyTorch -> ONNX (INT8) -> ONNX Runtime. The runtime (`src/gidi/inference/`)
  needs only numpy, onnxruntime and tokenizers.
- Playground: Vite, React and shadcn/ui in `playground/`, managed with `pnpm`

Explicitly excluded: Poetry, Conda, Lightning, Hydra, W&B, MLflow, Docker, notebook-first
workflow, cloud training.

## Development

```bash
uv sync                  # install dependencies
uv run pytest -q         # run tests
uv run ruff check .      # lint
uv run ruff format .     # format

# local playground on http://127.0.0.1:8765 (v2 bundle)
pnpm -C playground install && pnpm -C playground build
uv run python scripts/demo_ui.py

# release and publish (see docs/releasing.md)
uv run python scripts/release.py build --model gidi-finance-v2 --version 2.0.1
uv run python scripts/release.py verify --release dist/releases/gidi-finance-v2/2.0.1
uv run python scripts/publish.py all --release dist/releases/gidi-finance-v2/2.0.1 --dry-run
```

## Layout

```text
corpus/raw/         generated raw corpus (never overwritten)
corpus/reviewed/    Quet-reviewed clean corpus
corpus/rejected/    exported rejects
datasets/           derived labeled datasets (annotation-v1, annotation-v2) and splits
configs/            versioned contracts (annotation label taxonomy and queue)
prompts/            generation prompts
scripts/            runnable scripts
experiments/        experiment and deployment records
models/             local model artifacts (gitignored except README)
releases/           release specs, model card templates, license, provider config
playground/         local web playground (shadcn/ui)
.github/workflows/  release and publish CI (manual dispatch)
dist/               built releases and publication receipts (gitignored)
docs/               project documentation
src/gidi/           library code
```

All corpora and datasets are JSONL.

## Philosophy

- small > large
- measured > assumed
- deterministic > generative where possible
- mobile constraints > benchmark vanity
- simple experiments > framework complexity

## Documentation

- [docs/overview.md](docs/overview.md): pipeline, architecture, data conventions, device policy
- [docs/deployment.md](docs/deployment.md): runtime usage, output schema, limitations
- [docs/releasing.md](docs/releasing.md): release build, verification, publishing, CI
- [docs/annotation-v1.md](docs/annotation-v1.md),
  [docs/annotation-v2.md](docs/annotation-v2.md): annotation contracts
- [experiments/README.md](experiments/README.md): how experiments are recorded
- [models/README.md](models/README.md): local artifacts
