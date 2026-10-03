# Gidi

Gidi is an experimental machine learning project building a tiny on-device classifier for
short Vietnamese personal-finance notes, such as `ăn phở 45k` or `nhận lương 15tr`.

The target is a model small enough to run inference locally on modern phones. There is no
server component and no inference API.

## Scope

Vietnamese-first. Inputs are short, informal, Vietnamese-language notes a person writes for
themselves; they are not formal bookkeeping records. Amount extraction and normalization may
stay deterministic code rather than model output.

## Current phase

Deployment hardening is complete. `gidi-finance-v1` (`models/gidi-finance-v1/`, 28.66 MB INT8
ONNX) is the packaged model, and `src/gidi/inference/` is its runtime. Usage, output schema and
known limitations: [docs/deployment.md](docs/deployment.md). The research history is in
`docs/investigation-journal.md`.

Human review and annotation happen in [Quet](https://github.com/8bu/quet) (Quick Utility for
Evaluating Text), a separate open-source TUI. It is not a dependency of this project. Gidi
produces the queues and schemas that Quet consumes, and validates what it writes back. Agent
details are in `AGENTS.md`.

## Intended stack

- Python 3.12 managed with `uv`
- PyTorch, with Apple MPS as the primary device (cuda and cpu also supported)
- Hugging Face Transformers, Tokenizers, Datasets
- scikit-learn for evaluation and baselines
- pytest and ruff for tests and linting
- Deployment path: PyTorch -> ONNX (INT8) -> ONNX Runtime. The runtime (`src/gidi/inference/`)
  needs only numpy, onnxruntime and tokenizers.

Explicitly excluded: Poetry, Conda, Lightning, Hydra, W&B, MLflow, Docker, notebook-first
workflow, cloud training.

## Development

```bash
uv sync              # install dependencies
uv run pytest        # run tests
uv run ruff check .  # lint
```

Only these three commands are established so far.

## Layout

```text
corpus/raw/         generated raw corpus (never overwritten)
corpus/reviewed/    Quet-reviewed clean corpus
corpus/rejected/    exported rejects
datasets/           derived labeled datasets: train/ validation/ test/, annotation-v1/ queue
configs/            versioned contracts (annotation-v1.yaml: label taxonomy and queue)
prompts/            generation prompts
scripts/            runnable scripts
experiments/        experiment records
models/             local model artifacts (gitignored except README)
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

See [docs/overview.md](docs/overview.md) for the pipeline, intended architecture, data
conventions, and device policy. See [experiments/README.md](experiments/README.md) for how
experiments are recorded and [models/README.md](models/README.md) for local artifacts.
Release and publish workflow (local build, verification, dry-run publishing, CI):
[docs/releasing.md](docs/releasing.md).
