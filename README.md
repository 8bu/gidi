# Gidi — GiaoDịch

**Tiếng Việt** · [English](README.en.md)

Đây là thử nghiệm của mình để xây dựng một classifier nhỏ.

Gidi là dự án machine learning (học máy) thử nghiệm.
Dự án xây dựng model on-device nhỏ cho ghi chú tài chính cá nhân tiếng Việt ngắn,
như `ăn phở 45k` hoặc `nhận lương 15tr`.
Với mỗi ghi chú, model dự đoán loại giao dịch, span đối tác (`target`)
và span số tiền (`value`).

Mục tiêu là model đủ nhỏ để inference cục bộ trên điện thoại hiện đại.
Dự án không có server và không có inference API.

## Phạm vi

Ưu tiên tiếng Việt. Input là ghi chú ngắn, không trang trọng,
do một người viết cho chính họ.
Input không phải bản ghi kế toán chính thức.
Model trả số tiền theo cách người dùng viết, tức là một span của ghi chú.
Code deterministic chuyển span này thành số.

## Trạng thái

- **`gidi-finance-v2`** là candidate hiện tại có thể kiểm thử.
  Đây là `dual encoder` value-span-v7: path v1 đóng băng cho type và target,
  path thứ hai đã fine-tune, kèm CRF, cho value span.
  Bản này là 58.1 MB INT8 ONNX, khoảng 3.5 ms p50 trên một CPU thread.
  Release 2.0.1 được publish ở chế độ private: Hugging Face `x8bu/gidi-finance` (INT8)
  và GitHub release nháp (archive INT8 và FP32).
- **`gidi-finance-v1`** vẫn đóng băng và không đổi: chỉ có type và target,
  28.66 MB INT8 ONNX.
- Research model đã dừng. Hiện không có experiment, compression hoặc distillation mới.

Usage, output schema và known limitations nằm tại [docs/deployment.md](docs/deployment.md).
Lịch sử research nằm trong [docs/investigation-journal.md](docs/investigation-journal.md).

Human review và annotation diễn ra trong [Quet](https://github.com/8bu/quet) (Quick Utility for
Evaluating Text), một TUI (text user interface) open-source riêng. Quet không phải dependency
của dự án này. Gidi tạo queue và schema cho Quet.
Gidi validate dữ liệu Quet ghi lại.
Chi tiết về agent nằm trong `AGENTS.md`.

## Stack

- Python 3.12 được quản lý bằng `uv`
- PyTorch, với Apple MPS là device chính (cuda và cpu cũng được hỗ trợ)
- Hugging Face Transformers, Tokenizers, Datasets
- scikit-learn cho evaluation và baseline
- pytest và ruff cho test, lint và format
- Deployment path: PyTorch -> ONNX (INT8) -> ONNX Runtime. Runtime (`src/gidi/inference/`)
  chỉ cần numpy, onnxruntime và tokenizers.
- Playground: Vite, React và shadcn/ui trong `playground/`, được quản lý bằng `pnpm`

Loại trừ rõ ràng: Poetry, Conda, Lightning, Hydra, W&B, MLflow, Docker, notebook-first
workflow và cloud training.

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

## Cấu trúc

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

Tất cả corpus và dataset đều là JSONL.

## Triết lý

- nhỏ > lớn
- đo lường > giả định
- deterministic > generative khi có thể
- ràng buộc mobile > benchmark vanity
- experiment đơn giản > framework phức tạp

## Tài liệu

- [docs/overview.md](docs/overview.md): pipeline, architecture, quy ước data, policy cho device
- [docs/deployment.md](docs/deployment.md): runtime usage, output schema, limitations
- [docs/releasing.md](docs/releasing.md): release build, verification, publishing, CI
- [docs/annotation-v1.md](docs/annotation-v1.md),
  [docs/annotation-v2.md](docs/annotation-v2.md): annotation contracts
- [experiments/README.md](experiments/README.md): cách ghi experiment
- [models/README.md](models/README.md): local artifact
