# Gidi

**Tiếng Việt** · [English](README.en.md)

Gidi là dự án machine learning (học máy) thử nghiệm.
Dự án xây dựng classifier on-device nhỏ cho ghi chú tài chính cá nhân
tiếng Việt ngắn, như `ăn phở 45k` hoặc `nhận lương 15tr`.

Mục tiêu là model đủ nhỏ để inference cục bộ trên điện thoại hiện đại.
Dự án không có server và không có inference API.

## Phạm vi

Ưu tiên tiếng Việt. Input là ghi chú ngắn, không trang trọng,
do một người viết cho chính họ.
Input không phải bản ghi kế toán chính thức.
Code deterministic có thể trích xuất và normalize số tiền thay cho model.

## Giai đoạn hiện tại

Deployment hardening đã hoàn tất. `gidi-finance-v1` (`models/gidi-finance-v1/`, 28.66 MB INT8
ONNX) là model được đóng gói. `src/gidi/inference/` là runtime của model.
Xem usage, output schema và known limitations tại [docs/deployment.md](docs/deployment.md).
Lịch sử research nằm trong `docs/investigation-journal.md`.

Human review và annotation diễn ra trong [Quet](https://github.com/8bu/quet) (Quick Utility for
Evaluating Text), một TUI (text user interface) open-source riêng. Quet không phải dependency
của dự án này. Gidi tạo queue và schema cho Quet.
Gidi validate dữ liệu Quet ghi lại. Chi tiết về agent nằm trong `AGENTS.md`.

## Stack dự kiến

- Python 3.12 được quản lý bằng `uv`
- PyTorch, với Apple MPS là device chính (cuda và cpu cũng được hỗ trợ)
- Hugging Face Transformers, Tokenizers, Datasets
- scikit-learn cho evaluation và baseline
- pytest và ruff cho test và lint
- Deployment path: PyTorch -> ONNX (INT8) -> ONNX Runtime. Runtime (`src/gidi/inference/`)
  chỉ cần numpy, onnxruntime và tokenizers.

Loại trừ rõ ràng: Poetry, Conda, Lightning, Hydra, W&B, MLflow, Docker, notebook-first
workflow và cloud training.

## Development

```bash
uv sync              # install dependencies
uv run pytest        # run tests
uv run ruff check .  # lint
```

Hiện chỉ có ba command này được thiết lập.

## Cấu trúc

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

Tất cả corpus và dataset đều là JSONL.

## Triết lý

- nhỏ > lớn
- đo lường > giả định
- deterministic > generative khi có thể
- ràng buộc mobile > benchmark vanity
- experiment đơn giản > framework phức tạp

## Tài liệu

Xem [docs/overview.md](docs/overview.md) để đọc về pipeline, architecture dự kiến,
quy ước data và policy cho device.
Xem [experiments/README.md](experiments/README.md) để biết cách ghi experiment.
Xem [models/README.md](models/README.md) để biết về local artifact.

Workflow release và publish (local build, verification, dry-run publishing, CI) nằm tại
[docs/releasing.md](docs/releasing.md).
