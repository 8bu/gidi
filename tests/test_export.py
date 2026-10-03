"""ONNX export, INT8 quantization and latency on a tiny randomly initialised BERT."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from transformers import BertConfig

from gidi.export import (
    benchmark_latency,
    benchmark_torch_latency,
    export_onnx,
    file_size_mb,
    make_session,
    quantize_int8,
)
from gidi.export.onnx_export import SAMPLE_NOTES, _tokenize
from gidi.modeling.model import GidiMultiTaskModel

VOCAB = 64
NOTES = list(SAMPLE_NOTES)


class CharTokenizer:
    """Deterministic fast-tokenizer stand-in: one id per character, 1 = CLS, 2 = SEP, 0 = pad."""

    def __call__(self, texts, padding=True, truncation=True, max_length=32, return_tensors="np"):
        rows = [[1, *(3 + ord(c) % (VOCAB - 3) for c in t), 2][:max_length] for t in texts]
        width = max_length if padding == "max_length" else max(map(len, rows))
        ids = np.zeros((len(rows), width), dtype=np.int64)
        mask = np.zeros_like(ids)
        for i, row in enumerate(rows):
            ids[i, : len(row)] = row
            mask[i, : len(row)] = 1
        return {"input_ids": ids, "attention_mask": mask}


def _build_model() -> GidiMultiTaskModel:
    cfg = BertConfig(
        vocab_size=VOCAB,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=64,
    )
    return GidiMultiTaskModel.from_config(cfg)


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    torch.manual_seed(0)
    model = _build_model().eval()
    tokenizer = CharTokenizer()
    path = tmp_path_factory.mktemp("onnx") / "model.onnx"
    report = export_onnx(model, tokenizer, path, max_length=32, opset=17, sample_texts=NOTES)
    return model, tokenizer, path, report


def test_export_matches_pytorch_with_dynamic_axes(exported):
    model, tokenizer, path, report = exported
    assert report["exporter"] in {"dynamo", "torchscript"}
    assert report["parity"]["passed"]
    assert report["parity"]["max_abs_diff"] <= 1e-4

    session = make_session(path)
    assert [i.name for i in session.get_inputs()] == ["input_ids", "attention_mask"]
    assert {i.type for i in session.get_inputs()} == {"tensor(int64)"}
    assert [o.name for o in session.get_outputs()] == ["type_logits", "tag_logits"]

    # Batch and sequence lengths unseen at export time.
    for texts in (NOTES[:1], NOTES[:7], ["x" * 40]):
        ids, mask = _tokenize(tokenizer, texts, 32)
        got_type, got_tag = session.run(None, {"input_ids": ids, "attention_mask": mask})
        with torch.inference_mode():
            ref_type, ref_tag = model(torch.from_numpy(ids), torch.from_numpy(mask))
        assert got_type.shape == (len(texts), 8)
        assert got_tag.shape == (len(texts), ids.shape[1], 3)
        np.testing.assert_allclose(got_type, ref_type.numpy(), atol=1e-4)
        np.testing.assert_allclose(got_tag[mask == 1], ref_tag.numpy()[mask == 1], atol=1e-4)


def test_quantize_int8_shrinks_and_reports_agreement(exported, tmp_path):
    _, tokenizer, fp32_path, _ = exported
    report = quantize_int8(
        fp32_path, tmp_path / "model.int8.onnx", tokenizer=tokenizer, texts=NOTES
    )
    assert report["int8_size_mb"] == pytest.approx(file_size_mb(tmp_path / "model.int8.onnx"))
    assert report["int8_size_mb"] < report["fp32_size_mb"]
    agreement = report["agreement"]
    assert 0.0 <= agreement["type_agreement"] <= 1.0
    assert 0.0 <= agreement["tag_agreement"] <= 1.0
    assert agreement["tag_tokens"] > 0
    # Quantized graph must still run end to end with the same interface.
    session = make_session(tmp_path / "model.int8.onnx")
    ids, mask = _tokenize(tokenizer, NOTES[:3], 32)
    type_logits, tag_logits = session.run(None, {"input_ids": ids, "attention_mask": mask})
    assert type_logits.shape == (3, 8) and tag_logits.shape[-1] == 3


def test_latency_reports_separate_tokenization_and_inference(exported):
    model, tokenizer, path, _ = exported
    for result in (
        benchmark_latency(path, tokenizer, NOTES, runs=10, warmup=2, threads=1),
        benchmark_torch_latency(model, tokenizer, NOTES, runs=10, warmup=2, threads=1),
    ):
        assert result["runs"] == 10
        for part in ("tokenization", "inference", "total"):
            assert 0 < result[part]["p50_ms"] <= result[part]["p95_ms"]
        assert result["total"]["mean_ms"] == pytest.approx(
            result["tokenization"]["mean_ms"] + result["inference"]["mean_ms"]
        )
