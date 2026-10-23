# Gidi (GiaoDịch)

**English** | [Tiếng Việt](README.vi.md)

This is my experiment to build a tiny classifier.

Gidi is an experimental machine learning project. It builds tiny on-device models for short
personal-finance notes, such as `ăn phở 45k`, `nhận lương 15tr` or `covered lunch for Sam 20`.
For one note a model predicts the transaction type and the counterparty (target) span. The
Vietnamese model also predicts the amount (value) span.

The goal is a model small enough to run inference locally on a modern phone. There is no server
and no inference API.

## Models

| Model | Language | Version | Status | Size | Link |
|---|---|---|---|---|---|
| `gidi-finance` | Vietnamese | `gidi-finance-v4`, release `4.0.0` | Current. Runs in the web app. Public Hugging Face and GitHub releases are still v3. | 28.66 MB INT8 encoder (v3 bundle: 29.3 MB) | [App](https://gidi.8bu.dev), [x8bu/gidi-finance](https://huggingface.co/x8bu/gidi-finance) |
| `gidi-en` | English | `gidi-en-v4`, release `4.0.0` | Experimental. Type, target and amount span (MiniLM). Runs in the web app. | 29.5 MB INT8 | [App](https://gidi.8bu.dev), [x8bu/gidi-en](https://huggingface.co/x8bu/gidi-en) |

Older Vietnamese releases (`gidi-finance-v3`, `-v2`, `-v1`) are described under [Status](#status).

## Scope

Both models read short, informal notes that a person writes for themselves. The notes are not
formal bookkeeping records.

- `gidi-finance` is Vietnamese. It returns the amount as written (a span of the note). Deterministic
  code converts that span to a number.
- `gidi-en` is English. Since `gidi-en-v4` it also returns the amount as written (a span of the
  note, `5k`, `$20`, `twenty bucks`), read by a learned value head. The web app turns that span
  into US dollars.

## Status

### gidi-finance (Vietnamese)

- **`gidi-finance-v4`** (release `4.0.0`) is the current model. It has the same architecture and
  runtime as v3, and a new encoder trained with Clef-teacher data and the `annotation-v4` rules
  (see `experiments/clef-distill-v2/report.md`). <https://gidi.8bu.dev> runs v4. The public
  Hugging Face and GitHub releases are still v3. The demo is a notes app that works on phones and
  tablets, with developer tools at `/lab`.
- **`gidi-finance-v3`** (release `3.0.0`) is the previous release. The system is one INT8 encoder
  for type and target, a word snap on the target span and a rule-based parser for the value span.
  The bundle is 29.3 MB INT8 ONNX, about 1.4 ms p50 on one CPU thread. It is public on Hugging
  Face ([x8bu/gidi-finance](https://huggingface.co/x8bu/gidi-finance), INT8) and on GitHub (INT8
  and FP32 archives).
- **`gidi-finance-v2`** (release `2.0.3`) is a dual encoder with a CRF value head, 58.1 MB INT8
  ONNX, about 3.5 ms p50. It stays frozen.
- **`gidi-finance-v1`** stays frozen and unchanged: type and target only, 28.66 MB INT8 ONNX.

### gidi-en (English)

- **`gidi-en-v4`** (release `4.0.0`) is the current English model. It is experimental. It predicts
  the type, the target and the value span (the amount as written). The student is MiniLM-L12-H384
  (12 layers, 384 wide, WordPiece, MIT) and won a backbone bake-off whose rule was written before
  any run (`experiments/gidi-en-backbone-bakeoff/report.md`). It is published on Hugging Face as
  [x8bu/gidi-en](https://huggingface.co/x8bu/gidi-en) (tag `gidi-en-v4-4.0.0`), INT8 29.5 MB, and
  runs in the web app. `gidi-en-v2` (`2.0.1`, roberta student, no amount) stays on Hugging Face
  under its own tag and in the app's manifest as the rollback.
- **The labels are silver.** No human annotated English. Claude, Gemini, Clef and a deterministic
  amount scan produced every label, so every score is agreement with those labels, not accuracy
  against ground truth. The contract is [docs/annotation-en-v1.md](docs/annotation-en-v1.md).
- **Two seeds, one shipped.** The seed was picked by a rule on the dev split declared before the
  release (`experiments/gidi-en-student-v4/report.md`). On the held-out test split (primary set,
  214 notes) it scores type F1 0.958, target 0.939, joint 0.902, value 0.995 against those silver
  labels. On the 357-note slang probe: type 0.919, target 0.863, joint 0.815, value 0.961.
- **The ONNX file does not include the word-boundary snapping step.** A runtime must apply it to
  the target and to the value after decoding (`gidi.inference.decode.snap_to_words`). The scores
  above use it.

### Both

Usage, output schema and known limitations are in [docs/deployment.md](docs/deployment.md). The
research history is in [docs/investigation-journal.md](docs/investigation-journal.md).

Human review and annotation happen in [Quet](https://github.com/8bu/quet) (Quick Utility for
Evaluating Text), a separate open-source TUI. It is not a dependency of this project. Gidi
produces the queues and schemas that Quet consumes, and validates what it writes back. Agent
details are in `AGENTS.md`.

## Roadmap

Internal experiment names are not release versions. Public models are named `gidi-finance-vN` or
`gidi-en-vN`, with release versions such as `3.0.0`.

**Current Vietnamese model: `gidi-finance-v4` / `4.0.0` (done); previous: `gidi-finance-v3` / `3.0.0`**

- Hybrid neural and deterministic architecture: one INT8 encoder for type and target, value
  extracted by a rule-based parser (`gidi.value_parser`), no second value encoder.
- Word snap is on by default: the target is extended to whole-word boundaries.
- `annotation-v3` taxonomy: a debt-only note is `borrow` or `lend`. The gift receiver is the
  target (the rule is in the test labels; the training data does not apply it yet).
- Half the size of v2: 29.3 MB against 58.1 MB, about 1.4 ms p50.
- On the test set `human-value-02` (190 notes, LLM-labelled): type 92.1%, target 85.3%, value
  100%, end-to-end 78.9%. Details and limitations are in the model card.

**Current English model: `gidi-en-v4` / `4.0.0` (experimental)**

- Type, target and value span, with the `annotation-en-v1` contract and silver labels.
- MiniLM-L12-H384 student, uncased WordPiece tokenizer (ported to TypeScript for the web app).
- Published on Hugging Face and live in the web app.

**Released before: `gidi-finance-v2`**

- Dual-encoder architecture: one encoder for type and target, a second encoder for the value span.
- INT8 ONNX distribution on Hugging Face and GitHub.
- Browser runtime (onnxruntime-web, WASM); public demo at <https://gidi.8bu.dev>.

**Next**

- Raise Vietnamese target accuracy (now about 85%): multi-word brand and shop names, gift
  receivers.
- Calibrate confidences and add an abstention policy.

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
uv run python scripts/release.py build --model gidi-finance-v4 --version 4.0.0
uv run python scripts/release.py verify --release dist/releases/gidi-finance-v4/4.0.0
uv run python scripts/publish.py all --release dist/releases/gidi-finance-v4/4.0.0 --dry-run
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
  [docs/annotation-v2.md](docs/annotation-v2.md): Vietnamese annotation contracts
- [docs/annotation-en-v1.md](docs/annotation-en-v1.md): English annotation contract
- [experiments/README.md](experiments/README.md): how experiments are recorded
- [experiments/gidi-en-student-v2/report.md](experiments/gidi-en-student-v2/report.md): `gidi-en-v2` report
- [experiments/gidi-en-backbone-bakeoff/report.md](experiments/gidi-en-backbone-bakeoff/report.md),
  [experiments/gidi-en-student-v4/report.md](experiments/gidi-en-student-v4/report.md): why MiniLM,
  and the `gidi-en-v4` release
- [models/README.md](models/README.md): local artifacts
