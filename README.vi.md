# Gidi (GiaoDịch)

[English](README.md) | **Tiếng Việt**

Đây là thử nghiệm của mình để xây dựng một classifier nhỏ.

Gidi là dự án machine learning (học máy) thử nghiệm.
Dự án xây dựng các model on-device nhỏ cho ghi chú tài chính cá nhân ngắn,
như `ăn phở 45k`, `nhận lương 15tr` hoặc `covered lunch for Sam 20`.
Với mỗi ghi chú, model dự đoán loại giao dịch và span đối tác (`target`).
Model tiếng Việt còn dự đoán span số tiền (`value`).

Mục tiêu là model đủ nhỏ để inference cục bộ trên điện thoại hiện đại.
Dự án không có server và không có inference API.

## Model

| Model | Ngôn ngữ | Phiên bản | Trạng thái | Kích thước | Liên kết |
|---|---|---|---|---|---|
| `gidi-finance` | Tiếng Việt | `gidi-finance-v4`, release `4.0.0` | Hiện tại. Chạy trong ứng dụng web. Bản công khai trên Hugging Face và GitHub vẫn là v3. | Encoder INT8 28.66 MB (bundle v3: 29.3 MB) | [Ứng dụng](https://gidi.8bu.dev), [x8bu/gidi-finance](https://huggingface.co/x8bu/gidi-finance) |
| `gidi-en` | Tiếng Anh | `gidi-en-v2`, release `2.0.1` | Thử nghiệm. Chỉ có trên Hugging Face, chưa có trong ứng dụng. | INT8 29.2 MB | [x8bu/gidi-en](https://huggingface.co/x8bu/gidi-en) |

Các release tiếng Việt cũ (`gidi-finance-v3`, `-v2`, `-v1`) nằm trong mục [Trạng thái](#trạng-thái).

## Phạm vi

Cả hai model đọc ghi chú ngắn, không trang trọng, do một người viết cho chính họ.
Input không phải bản ghi kế toán chính thức.

- `gidi-finance` là model tiếng Việt. Model trả số tiền theo cách người dùng viết, tức là một span
  của ghi chú. Code deterministic chuyển span này thành số.
- `gidi-en` là model tiếng Anh. Model chỉ dự đoán type và target. Chưa có value span và chưa có
  parser số tiền.

## Trạng thái

### gidi-finance (tiếng Việt)

- **`gidi-finance-v4`** (release `4.0.0`) là model hiện tại. Kiến trúc và runtime giống v3,
  encoder mới huấn luyện thêm dữ liệu Clef và quy tắc `annotation-v4`
  (xem `experiments/clef-distill-v2/report.md`). Demo <https://gidi.8bu.dev> đang chạy v4.
  Bản công khai trên Hugging Face và GitHub vẫn là v3. Demo là ứng dụng ghi chú, dùng được trên
  điện thoại và máy tính bảng; công cụ cho nhà phát triển ở `/lab`.
- **`gidi-finance-v3`** (release `3.0.0`) là release trước.
  Hệ thống gồm một encoder INT8 cho type và target, word snap cho span `target`, và parser
  theo luật cho value span. Bundle là 29.3 MB INT8 ONNX, khoảng 1.4 ms p50 trên một CPU thread.
  Release công khai trên Hugging Face
  [x8bu/gidi-finance](https://huggingface.co/x8bu/gidi-finance) (INT8) và GitHub
  (archive INT8 và FP32).
- **`gidi-finance-v2`** (release `2.0.3`) là `dual encoder` với value head CRF,
  58.1 MB INT8 ONNX, khoảng 3.5 ms p50. Vẫn đóng băng.
- **`gidi-finance-v1`** vẫn đóng băng và không đổi: chỉ có type và target,
  28.66 MB INT8 ONNX.

### gidi-en (tiếng Anh)

- **`gidi-en-v2`** (release `2.0.1`) là model tiếng Anh đầu tiên. Model còn thử nghiệm.
  Model dự đoán type và target. Model có trên Hugging Face tại
  [x8bu/gidi-en](https://huggingface.co/x8bu/gidi-en). Bản INT8 là 29.2 MB.
  Model chưa có trong ứng dụng web.
- **Nhãn là silver.** Không có người gán nhãn tiếng Anh. Claude, Gemini và Clef tạo mọi nhãn,
  nên mọi điểm số là mức khớp với các nhãn đó, không phải độ chính xác so với ground truth.
  Hợp đồng annotation là [docs/annotation-en-v1.md](docs/annotation-en-v1.md).
- **Một seed, một lần chạy.** Báo cáo không ước lượng được phương sai và đánh giá biên so với
  ngưỡng là khiêm tốn (`experiments/gidi-en-student-v2/report.md`). Trên test split giữ lại
  (tập primary, 214 ghi chú), model đạt type F1 0.956, target 0.888, joint 0.850 so với các nhãn
  silver này. Trên slang probe 357 ghi chú, model đạt type 0.890, target 0.835, joint 0.754.
- **File ONNX không chứa bước word-boundary snapping.** Runtime phải áp dụng bước này sau khi giải
  mã BIO (`gidi.inference.decode.snap_to_words`). Các điểm số ở trên đã dùng bước này.

### Chung

Usage, output schema và known limitations nằm tại [docs/deployment.md](docs/deployment.md).
Lịch sử research nằm trong [docs/investigation-journal.md](docs/investigation-journal.md).

Human review và annotation diễn ra trong [Quet](https://github.com/8bu/quet) (Quick Utility for
Evaluating Text), một TUI (text user interface) open-source riêng. Quet không phải dependency
của dự án này. Gidi tạo queue và schema cho Quet.
Gidi validate dữ liệu Quet ghi lại.
Chi tiết về agent nằm trong `AGENTS.md`.

## Lộ trình

Tên experiment nội bộ không phải là version release.
Model public có tên `gidi-finance-vN` hoặc `gidi-en-vN` và version release như `3.0.0`.

**Model tiếng Việt hiện tại: `gidi-finance-v4` / `4.0.0` (đã xong); trước đó `gidi-finance-v3` / `3.0.0`**

- Kiến trúc lai neural và tất định: một encoder INT8 cho type và target, value được trích xuất
  bằng parser theo luật (`gidi.value_parser`), không còn value encoder thứ hai.
- Word snap bật mặc định: `target` được mở rộng ra ranh giới của cả từ.
- Taxonomy `annotation-v3`: ghi chú chỉ nói về nợ là `borrow` hoặc `lend`.
  Người nhận quà là `target` (quy tắc có trong nhãn test; dữ liệu huấn luyện chưa áp dụng).
- Nhỏ một nửa so với v2: 29.3 MB so với 58.1 MB, khoảng 1.4 ms p50.
- Trên bộ test `human-value-02` (190 ghi chú, nhãn LLM): `type` 92.1%, `target` 85.3%,
  `value` 100%, end-to-end 78.9%. Chi tiết và giới hạn nằm trong model card.

**Model tiếng Anh hiện tại: `gidi-en-v2` / `2.0.1` (thử nghiệm)**

- Chỉ có type và target, theo hợp đồng `annotation-en-v1` và nhãn silver.
- Đã có trên Hugging Face. Chưa có trong ứng dụng web.

**Đã phát hành trước đó: `gidi-finance-v2`**

- Kiến trúc `dual encoder`: một encoder cho type và target, encoder thứ hai cho value span.
- Bản phân phối INT8 ONNX trên Hugging Face và GitHub.
- Runtime trong trình duyệt (onnxruntime-web, WASM). Demo public: <https://gidi.8bu.dev>.

**Tiếp theo**

- Nâng độ chính xác của `target` tiếng Việt (hiện khoảng 85%): tên thương hiệu và cửa hàng nhiều từ,
  người nhận quà.
- Hiệu chỉnh điểm tin cậy và cơ chế từ chối.

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
uv run python scripts/release.py build --model gidi-finance-v4 --version 4.0.0
uv run python scripts/release.py verify --release dist/releases/gidi-finance-v4/4.0.0
uv run python scripts/publish.py all --release dist/releases/gidi-finance-v4/4.0.0 --dry-run
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
  [docs/annotation-v2.md](docs/annotation-v2.md): hợp đồng annotation tiếng Việt
- [docs/annotation-en-v1.md](docs/annotation-en-v1.md): hợp đồng annotation tiếng Anh
- [experiments/README.md](experiments/README.md): cách ghi experiment
- [experiments/gidi-en-student-v2/report.md](experiments/gidi-en-student-v2/report.md): báo cáo `gidi-en-v2`
- [models/README.md](models/README.md): local artifact
