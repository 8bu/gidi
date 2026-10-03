---
language: vi
library_name: onnx
license: apache-2.0
pipeline_tag: text-classification
tags:
  - onnx
  - vietnamese
  - personal-finance
  - token-classification
  - on-device
---

# {model_version} {release_version}

**Tiếng Việt** · [English](README.en.md)

Đây là model on-device nhỏ cho ghi chú tài chính cá nhân tiếng Việt ngắn như
`mượn chú hai 5 xị`.
Với mỗi ghi chú, model dự đoán:

- `type`: loại giao dịch, thuộc một trong {types};
- `target`: span của đối tác, chẳng hạn `chú hai`, hoặc `null`;
- `value`: span chứa số tiền theo cách người dùng viết,
  chẳng hạn `5 xị`, hoặc `null`.
  Span là chuỗi con của ghi chú. Model không suy ra số từ span này.

Model là `{experiment}` (seed {seed}). Model là một `dual encoder` trong một đồ thị ONNX.
Một nhánh dùng encoder `gidi-finance-v1` đã đóng băng.
Nhánh này dự đoán `type` và `target`.
Nhánh thứ hai là bản sao encoder đã fine-tune, kèm CRF. Nhánh này dự đoán `value span`.
Model chuẩn hóa ghi chú về NFC và cắt ghi chú tại {max_length} token.

## Biến thể

| biến thể | archive | tệp model | kích thước | sha256 |
|---|---|---|---|---|
| int8 | `{int8_archive}` | `model.int8.onnx` | {int8_size_mb} | `{int8_sha256}` |
| fp32 | `{fp32_archive}` | `model.onnx` | {fp32_size_mb} | `{fp32_sha256}` |

Khuyến nghị dùng `int8` để deploy. `fp32` là bản export tham chiếu
để kiểm tra parity.
Bản `fp32` chỉ có trên GitHub release, trong archive `-fp32`.
Cả hai archive đều là runtime bundle đầy đủ.
Mỗi bundle gồm model, `config.json` (label, giao diện ONNX và điểm CRF),
`tokenizer.json`, `tokenizer_config.json`, `vocab_map.json`, `LICENSE`, README này và
`release-manifest.json`. Release liệt kê sha256 của mỗi tệp trong
`manifest.json` và `checksums.txt`.
Nếu repository chỉ lưu tệp INT8, có thể kiểm tra chúng bằng lệnh
`sha256sum -c --ignore-missing checksums.txt`.

## Cách dùng

Yêu cầu: Python {python_requirement}, onnxruntime {onnxruntime_requirement}, tokenizers
{tokenizers_requirement}, numpy {numpy_requirement}, và package `gidi` (`gidi.inference`).

Giải nén archive. Sau đó nạp thư mục bằng `GidiPredictor`.

```python
from pathlib import Path

from gidi.inference.predictor import GidiPredictor

# INT8 archive: the bundle contains model.int8.onnx, which is the default model file.
predictor = GidiPredictor.from_bundle("{model_version}-{release_version}-int8")
print(predictor.predict("mượn chú hai 5 xị").to_dict())

# FP32 archive: point to its model.onnx.
root = Path("{model_version}-{release_version}-fp32")
predictor = GidiPredictor.from_bundle(root, model_path=root / "model.onnx")
```

`predict(...).to_dict()` trả về các key sau:

| key | ý nghĩa |
|---|---|
| `type` | loại giao dịch |
| `type_confidence` | xác suất softmax của `type` |
| `target` | văn bản đối tác, hoặc `null` |
| `target_span` | offset `[start, end]` theo code point trong input, hoặc `null` |
| `target_confidence` | độ tin cậy của target span |
| `value_text` | văn bản số tiền (`text[start:end]` của input), hoặc `null` |
| `value_span` | offset `[start, end]` của value, hoặc `null` |
| `value_confidence` | trung bình hình học của xác suất tag của các token trong span |
| `truncated` | `true` nếu model cắt ghi chú tại giới hạn token |
| `model_version` | `{model_version}` |

Khi tích hợp ONNX Runtime trực tiếp, graph nhận `input_ids` và `attention_mask`.
Graph trả về `type_logits`, `tag_logits` và `value_logits`.
Cần dùng CRF Viterbi decoding cho tag value bằng các điểm trong `config.json`.
Argmax không tương đương với cách này.

## Giới hạn đã biết

- `cho a Nam vay 1 triệu 20/10` cho `value` là `1 triệu 20/10`.
  Model gộp ngày vào `value span`. `value` đúng là `1 triệu`.
- `type` và `target` của INT8 khác FP32 ở một vài ghi chú gần hòa điểm,
  giống hệt model INT8 v1.
- Model không chuyển đổi NBSP và chữ số full-width. Chúng đi thẳng vào model.
- Nếu `value` nằm sau điểm cắt {max_length} token,
  model trả về `null` (`truncated` là `true`).
- Model không extract hoặc normalize số tiền hay ngày tháng.
  Model không có class "out of scope".
  Mỗi ghi chú nhận một `type` trong các `type` trên.

## Giấy phép

Apache License 2.0. Toàn văn nằm trong `LICENSE`.
