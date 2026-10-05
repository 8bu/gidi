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

# Gidi — GiaoDịch

**Tiếng Việt** · [English](README.en.md)

Đây là thử nghiệm của mình để xây dựng một classifier nhỏ.

Demo: <https://gidi.8bu.dev>. Model chạy hoàn toàn trong trình duyệt (onnxruntime-web, INT8).

Model: `{model_version}`, release `{release_version}`.

Đây là model on-device nhỏ cho ghi chú tài chính cá nhân tiếng Việt ngắn như
`mượn chú hai 5 xị`.
Với mỗi ghi chú, hệ thống dự đoán:

- `type`: loại giao dịch, thuộc một trong {types};
- `target`: span của đối tác, chẳng hạn `chú hai`, hoặc `null`;
- `value`: span chứa số tiền theo cách người dùng viết,
  chẳng hạn `5 xị`, hoặc `null`.
  Span là chuỗi con của ghi chú. Hệ thống không suy ra số từ span này.

`{model_version}` gồm ba phần:

1. Một encoder INT8 (4 lớp, 768 chiều, từ vựng đã cắt còn 8.338 token) trong một đồ thị ONNX.
   Encoder dự đoán `type` và `target`.
2. Bước *word snap*: span `target` được mở rộng ra ranh giới của cả từ.
   Ví dụ `pbank` trong `rut tien vpbank 1tr` thành `vpbank`.
3. Một parser theo luật (`gidi.value_parser`, phiên bản `{value_parser_version}`) lấy `value` từ
   ghi chú gốc. Không có value head nơ-ron. `config.json` ghi phiên bản parser; runtime từ chối
   nạp bundle nếu phiên bản parser không khớp.

Hệ thống chuẩn hóa ghi chú về NFC và cắt phần đưa vào encoder tại {max_length} token.
Parser đọc toàn bộ ghi chú, kể cả phần sau điểm cắt.

## Thay đổi so với `gidi-finance-v2`

- **Quy tắc nợ.** Ghi chú chỉ nói về một khoản nợ đang có (`còn nợ Hùng 300k`,
  `Hoa còn nợ mình 350k`) nhận `borrow` (mình nợ) hoặc `lend` (người kia nợ mình).
  Trước đây loại này bị bỏ qua. Xem `docs/annotation-v3.md`.
- **Quy tắc quà.** Người nhận quà hoặc tiền mừng là đối tác, nên là `target`
  (`mừng cưới Hoa 1 triệu` có `target` là `Hoa`). Bộ test dùng quy tắc này.
  Dữ liệu huấn luyện của model này chưa áp dụng quy tắc quà, nên ghi chú về quà vẫn là điểm
  yếu (xem Giới hạn).
- **Số tiền theo luật.** `value` đến từ parser theo luật, không còn từ value head dùng CRF.
  Parser có kết quả xác định và không có điểm tin cậy, nên `value_confidence` là `null`.
- **Word snap** bật mặc định (`target_snap: words` trong `config.json`).
- **Nhỏ một nửa.** Chỉ còn một encoder: bundle INT8 là 29,3 MB, v2 là 58,1 MB.
  Độ trễ p50 khoảng 1,4 ms (v2: 3,5 ms), CPU một luồng, Apple Silicon, batch 1.

## Kết quả

Bộ test mới `human-value-02`: 200 ghi chú, 190 ghi chú được chấm. Nhãn do hai bộ gán nhãn LLM
độc lập tạo ra, rồi đối chiếu và phân xử. Đây là nhãn LLM, không phải nhãn người.
`end-to-end` là đúng cả `type`, `target` và `value`.
Cột v2 là bundle `gidi-finance-v2` chạy nguyên trạng trên cùng 190 ghi chú.

| `human-value-02` (n = 190) | `gidi-finance-v3` | `gidi-finance-v2` |
|---|---|---|
| `type` đúng | 92,11 % | 75,79 % |
| `target` khớp chính xác | 85,26 % | 74,74 % |
| `value` khớp chính xác | 100,00 % | 98,42 % |
| end-to-end | 78,95 % | 59,47 % |

Hai bộ cũ hơn, chỉ chấm `type` và `target`. `type` và `target` của v2 giống hệt encoder
`gidi-finance-v1` INT8, nên cột v2 lấy từ encoder đó.

| bộ | `type` v3 | `type` v2 | `target` v3 | `target` v2 |
|---|---|---|---|---|
| test đóng băng (n = 105) | 94,29 % | 94,29 % | 92,38 % | 88,57 % |
| probe-v1 (n = 81) | 82,72 % | 77,78 % | 85,19 % | 81,48 % |

Các số này đến từ một lần chấm, với seed 1 cố định trước khi huấn luyện, không chọn theo điểm.
Với bộ nhỏ như vậy, chênh lệch vài điểm phần trăm chưa chắc có ý nghĩa.

## Biến thể

| biến thể | archive | tệp model | kích thước | sha256 |
|---|---|---|---|---|
| int8 | `{int8_archive}` | `model.int8.onnx` | {int8_size_mb} | `{int8_sha256}` |
| fp32 | `{fp32_archive}` | `model.onnx` | {fp32_size_mb} | `{fp32_sha256}` |

Khuyến nghị dùng `int8` để deploy. `fp32` là bản export tham chiếu
để kiểm tra parity.
Bản `fp32` chỉ có trên GitHub release, trong archive `-fp32`.
Cả hai archive đều là runtime bundle đầy đủ.
Mỗi bundle gồm model, `config.json` (label, giao diện ONNX, quy tắc `value` và `target_snap`),
`tokenizer.json`, `tokenizer_config.json`, `vocab_map.json`, `LICENSE`, README này và
`release-manifest.json`. Release liệt kê sha256 của mỗi tệp trong
`manifest.json` và `checksums.txt`.
Nếu repository chỉ lưu tệp INT8, có thể kiểm tra chúng bằng lệnh
`sha256sum -c --ignore-missing checksums.txt`.

## Cách dùng

Yêu cầu: Python {python_requirement}, onnxruntime {onnxruntime_requirement}, tokenizers
{tokenizers_requirement}, numpy {numpy_requirement}, và package `gidi`
(`gidi.inference` và `gidi.value_parser`). Parser nằm trong package `gidi`, không nằm trong
tệp ONNX.

Giải nén archive. Sau đó nạp thư mục bằng `GidiPredictor`.
`GidiPredictor` đọc `config.json` và tự bật word snap và parser.

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
| `target` | văn bản đối tác (sau word snap), hoặc `null` |
| `target_span` | offset `[start, end]` theo code point trong input, hoặc `null` |
| `target_confidence` | độ tin cậy của target span |
| `value_text` | văn bản số tiền (`text[start:end]` của input), hoặc `null` |
| `value_span` | offset `[start, end]` của value, hoặc `null` |
| `value_confidence` | luôn là `null`: parser theo luật không có điểm tin cậy |
| `truncated` | `true` nếu encoder cắt ghi chú tại giới hạn token |
| `model_version` | `{model_version}` |

Khi tích hợp ONNX Runtime trực tiếp, graph nhận `input_ids` và `attention_mask`.
Graph trả về `type_logits` và `tag_logits`. Lấy argmax của `tag_logits`, giải mã span `target`,
rồi mở rộng ra ranh giới từ như `target_snap` trong `config.json`.
Graph không có `value_logits`: phần `value` cần parser `gidi.value_parser` (hoặc bản port
tương đương đúng phiên bản `{value_parser_version}`).

## Giới hạn đã biết

- **Đối tác (`target`) đúng khoảng 85 %.** Trên `human-value-02`, 25 ghi chú sai `target`,
  12 sai `type` và 3 sai cả hai (40 ghi chú sai trong 190). Lỗi `target` thường gặp là tên
  thương hiệu hoặc cửa hàng nhiều từ bị cắt (`đi xem phim lotte cinema 130k` cho `lotte`,
  `trả góp thế giới di động 1tr8` cho `động`) và đối tác bị bỏ sót (`cf trung nguyen 45k`).
- **Ghi chú về quà và lì xì.** Model chưa học quy tắc quà. Ví dụ đã biết là sai:
  `lì xì bà ngoại 500k` (cho `income`, đúng là `expense`) và `quà tốt nghiệp em Khải 500k`
  (cho `tốt`, đúng là `Khải`).
- **Một số ghi chú nợ vẫn sai hướng**, chẳng hạn `Hung no minh 2tr chua tra` (cho
  `repayment_in`, đúng là `lend`) và `nhớ đòi Trang 450k` (cho `borrow`, đúng là `lend`).
- **Bộ test dùng nhãn LLM.** `human-value-02` có 190 ghi chú chấm được, nhãn do hai bộ gán nhãn
  LLM độc lập tạo ra, rồi phân xử, không phải nhãn người. Nhãn `value` khớp với parser trên cả
  190 ghi chú, nên con số 100 % của `value` chưa chứng minh parser đúng trên ghi chú khác.
  Các số trên chỉ là ước lượng sơ bộ.
- Điểm tin cậy của `type` và `target` chưa hiệu chỉnh. Hệ thống chưa có ngưỡng hay cơ chế từ chối.
- Hệ thống không chuyển đổi NBSP và chữ số full-width trước encoder.
- Nếu `target` nằm sau điểm cắt {max_length} token, encoder không tìm thấy nó
  (`truncated` là `true`). Parser vẫn đọc toàn bộ ghi chú.
- Hệ thống không extract hoặc normalize số tiền hay ngày tháng.
  Không có class "out of scope".
  Mỗi ghi chú nhận một `type` trong các `type` trên.

## Giấy phép

Apache License 2.0. Toàn văn nằm trong `LICENSE`.
