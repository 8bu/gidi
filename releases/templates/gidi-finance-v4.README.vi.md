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

## Thay đổi so với `gidi-finance-v3`

Kiến trúc, giao diện ONNX, tokenizer, parser giá trị và word snap giống `gidi-finance-v3`;
chỉ trọng số encoder là mới. Dữ liệu huấn luyện thêm phần tăng cường xác định và ghi chú do
một model giáo viên mạnh hơn gán nhãn theo bốn quy tắc đã làm rõ (`docs/annotation-v4.md`):
R1 ai trả (`Long trả 5 lít`), R2 trả hộ người khác (`ứng giùm Lan tiền điện 400k`),
R3 gửi tiền cho người khác (`chuyển cho mẹ 2 củ`) và R4 cách nói về lương
(`mới nhận lương 26cu`). File INT8 có cùng kích thước với v3 (28,66 MB).

## Kết quả

Chấm một lần trên cùng các bộ với `gidi-finance-v3`, dùng encoder seed 1 (cố định trước khi
huấn luyện). Nhãn `human-value-02` do hai bộ gán nhãn LLM độc lập tạo ra rồi phân xử; đây là nhãn
LLM, không phải nhãn người. `end-to-end` là đúng cả `type`, `target` và `value`.

| `human-value-02` (n = 190) | `gidi-finance-v4` | `gidi-finance-v3` |
|---|---|---|
| `type` đúng | 93,68 % | 92,11 % |
| `target` khớp chính xác | 88,95 % | 85,26 % |
| end-to-end | 84,21 % | 78,95 % |

| bộ | `type` v4 | `type` v3 | `target` v4 | `target` v3 |
|---|---|---|---|---|
| test đóng băng (n = 105) | 98,10 % | 94,29 % | 93,33 % | 92,38 % |
| probe-v1 (n = 81) | 90,12 % | 82,72 % | 85,19 % | 85,19 % |

Trên bộ probe quy tắc mới `probe-rules-v1` (64 ghi chú), đúng cả `type` lẫn `target` ở 76,6 %
số ghi chú (v3: 50,0 %). Số liệu đầy đủ và hồ sơ thí nghiệm: `experiments/clef-distill-v2/report.md`.
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

- **Trả hộ người khác (R2)** vẫn yếu: trên `probe-rules-v1`, đúng cả `type` và `target` ở khoảng
  một nửa số ghi chú `trả/ứng/mua hộ X`.
- Trên bộ kiểm tra thực tế 17 ghi chú, đúng 13. Vẫn sai: `Chuyển cho họ hàng 2 củ`,
  `long tra 5 lit` (tên viết thường), `cong ty tra tien t10 5 lit` và
  `Khoa trả lại tiền mình trả giùm 50k`.
- Tên thương hiệu hoặc cửa hàng nhiều từ có thể bị cắt, và một số đối tác bị bỏ sót.
- **Bộ test dùng nhãn LLM.** Nhãn `human-value-02` do hai bộ gán nhãn LLM tạo ra rồi phân xử,
  không phải nhãn người. Nhãn `value` khớp với parser, nên con số 100 % của `value` chưa chứng
  minh parser đúng trên ghi chú khác. Các số trên chỉ là ước lượng sơ bộ.
- Điểm tin cậy của `type` và `target` chưa hiệu chỉnh. Hệ thống chưa có ngưỡng hay cơ chế từ chối.
- Hệ thống không chuyển đổi NBSP và chữ số full-width trước encoder.
- Nếu `target` nằm sau điểm cắt {max_length} token, encoder không tìm thấy nó
  (`truncated` là `true`). Parser vẫn đọc toàn bộ ghi chú.
- Hệ thống không extract hoặc normalize số tiền hay ngày tháng.
  Không có class "out of scope".
  Mỗi ghi chú nhận một `type` trong các `type` trên.

## Giấy phép

Apache License 2.0. Toàn văn nằm trong `LICENSE`.
