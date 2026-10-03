"""PyTorch -> ONNX export, INT8 dynamic quantization, parity checks and CPU latency.

The model contract is ``forward(input_ids, attention_mask) -> (type_logits, tag_logits)``; the
exported graph has int64 inputs ``input_ids``/``attention_mask`` with dynamic batch and sequence
axes and float outputs ``type_logits`` (B, num_types) / ``tag_logits`` (B, L, num_tags).
"""

from __future__ import annotations

import statistics
import time
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import torch

INPUT_NAMES = ("input_ids", "attention_mask")
OUTPUT_NAMES = ("type_logits", "tag_logits")
DEFAULT_ATOL = 1e-4

SAMPLE_NOTES = (
    "Chị Thảo trả lại 1tr",
    "an sang 35k",
    "luong thang 9 ve 15tr",
    "mượn anh Nam 2 triệu",
    "cho Lan vay 500k",
    "chuyen khoan cho me 3tr",
    "hoàn tiền đơn shopee 120k",
    "trả nợ chú Hùng 1tr5",
    "Minh trả tiền cơm 80k",
    "đổ xăng 50k",
)


def _tokenize(tokenizer: Any, texts: Sequence[str], max_length: int, padding: bool | str = True):
    enc = tokenizer(
        list(texts),
        padding=padding,
        truncation=True,
        max_length=max_length,
        return_tensors="np",
    )
    return (
        np.ascontiguousarray(enc["input_ids"], dtype=np.int64),
        np.ascontiguousarray(enc["attention_mask"], dtype=np.int64),
    )


def _select_sample(texts: Sequence[str] | None) -> list[str]:
    return list(texts) if texts else list(SAMPLE_NOTES)


def file_size_mb(path: str | Path) -> float:
    """Size in MB (1e6 bytes) of an ONNX file including its external-data sidecars."""
    path = Path(path)
    total = path.stat().st_size
    for sidecar in path.parent.glob(f"{path.name}.data"):
        total += sidecar.stat().st_size
    for sidecar in path.parent.glob(f"{path.stem}*.onnx_data"):
        total += sidecar.stat().st_size
    return total / 1e6


def make_session(path: str | Path, threads: int | None = None) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    if threads is not None:
        opts.intra_op_num_threads = threads
        opts.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])


def _run(session: ort.InferenceSession, ids: np.ndarray, mask: np.ndarray):
    type_logits, tag_logits = session.run(
        list(OUTPUT_NAMES), {"input_ids": ids, "attention_mask": mask}
    )
    return type_logits, tag_logits


def _torch_logits(model: torch.nn.Module, ids: np.ndarray, mask: np.ndarray):
    with torch.inference_mode():
        type_logits, tag_logits = model(torch.from_numpy(ids), torch.from_numpy(mask))
    return type_logits.numpy(), tag_logits.numpy()


def _dynamo_export(
    model, args, out_path: Path, opset: int, output_names: Sequence[str] = OUTPUT_NAMES
) -> None:
    from torch.export import Dim

    batch, seq = Dim("batch", min=1, max=4096), Dim("seq", min=2, max=512)
    dynamic_shapes = {name: {0: batch, 1: seq} for name in INPUT_NAMES}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        program = torch.onnx.export(
            model,
            args,
            input_names=list(INPUT_NAMES),
            output_names=list(output_names),
            dynamic_shapes=dynamic_shapes,
            opset_version=opset,
            dynamo=True,
        )
    if program is None:
        raise RuntimeError("dynamo exporter returned no program")
    program.save(str(out_path))


def _legacy_export(
    model, args, out_path: Path, opset: int, output_names: Sequence[str] = OUTPUT_NAMES
) -> None:
    # type logits are per note; every further output is per token
    dynamic_axes = {
        **{name: {0: "batch", 1: "seq"} for name in INPUT_NAMES},
        output_names[0]: {0: "batch"},
        **{name: {0: "batch", 1: "seq"} for name in output_names[1:]},
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.onnx.export(
            model,
            args,
            str(out_path),
            input_names=list(INPUT_NAMES),
            output_names=list(output_names),
            dynamic_axes=dynamic_axes,
            opset_version=opset,
            dynamo=False,
        )


def _remove_onnx(path: Path) -> None:
    path.unlink(missing_ok=True)
    for sidecar in (Path(f"{path}.data"), path.with_suffix(".onnx_data")):
        sidecar.unlink(missing_ok=True)


def check_parity(
    model: torch.nn.Module,
    session: ort.InferenceSession,
    tokenizer: Any,
    texts: Sequence[str] | None = None,
    max_length: int = 32,
    atol: float = DEFAULT_ATOL,
) -> dict:
    """Compare onnxruntime CPU outputs with PyTorch on a padded batch and on each note alone."""
    texts = _select_sample(texts)
    model.eval()
    batch_ids, batch_mask = _tokenize(tokenizer, texts, max_length)
    ref_type, ref_tag = _torch_logits(model, batch_ids, batch_mask)
    got_type, got_tag = _run(session, batch_ids, batch_mask)
    valid = batch_mask.astype(bool)
    max_type = float(np.abs(ref_type - got_type).max())
    max_tag = float(np.abs(ref_tag - got_tag)[valid].max())

    # Different shapes than the export trace: every note at batch 1 with its natural length.
    single_max = 0.0
    for text in texts:
        ids, mask = _tokenize(tokenizer, [text], max_length)
        r_type, r_tag = _torch_logits(model, ids, mask)
        g_type, g_tag = _run(session, ids, mask)
        single_max = max(
            single_max,
            float(np.abs(r_type - g_type).max()),
            float(np.abs(r_tag - g_tag).max()),
        )
    max_abs = max(max_type, max_tag, single_max)
    return {
        "samples": len(texts),
        "atol": atol,
        "max_abs_diff_type_logits": max_type,
        "max_abs_diff_tag_logits": max_tag,
        "max_abs_diff_single_note": single_max,
        "max_abs_diff": max_abs,
        "passed": bool(max_abs <= atol),
    }


def export_onnx(
    model: torch.nn.Module,
    tokenizer: Any,
    out_path: str | Path,
    max_length: int = 32,
    opset: int = 17,
    sample_texts: Sequence[str] | None = None,
    atol: float = DEFAULT_ATOL,
) -> dict:
    """Export ``model`` to ``out_path`` and validate it against PyTorch.

    Tries the torch dynamo exporter first, then the legacy TorchScript exporter. Returns a dict
    with the exporter that succeeded, the error text of any exporter that failed, and the parity
    report. Raises if neither exporter produces a graph that matches PyTorch within ``atol``.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model = model.to("cpu").eval()
    texts = _select_sample(sample_texts)
    ids, mask = _tokenize(tokenizer, texts[:2], max_length, padding="max_length")
    args = (torch.from_numpy(ids), torch.from_numpy(mask))

    errors: dict[str, str] = {}
    for name, fn in (("dynamo", _dynamo_export), ("torchscript", _legacy_export)):
        _remove_onnx(out_path)
        try:
            fn(model, args, out_path, opset)
            session = make_session(out_path)
            parity = check_parity(model, session, tokenizer, texts, max_length, atol)
        except Exception as exc:
            errors[name] = f"{type(exc).__name__}: {exc}"[:2000]
            continue
        if not parity["passed"]:
            errors[name] = f"parity failed: max_abs_diff={parity['max_abs_diff']:.3e}"
            continue
        return {
            "path": str(out_path),
            "exporter": name,
            "opset_requested": opset,
            "errors": errors,
            "size_mb": file_size_mb(out_path),
            "parity": parity,
        }
    _remove_onnx(out_path)
    raise RuntimeError(f"ONNX export failed with every exporter: {errors}")


def quantize_int8(
    fp32_path: str | Path,
    int8_path: str | Path,
    tokenizer: Any = None,
    texts: Sequence[str] | None = None,
    max_length: int = 32,
) -> dict:
    """Dynamic INT8 weight quantization; report sizes and INT8-vs-FP32 prediction agreement.

    Agreement is over the sample ``texts`` (default: built-in notes): ``type_agreement`` is the
    fraction of notes with the same type argmax, ``tag_agreement`` the fraction of real (non-pad)
    tokens with the same tag argmax. Skipped when ``tokenizer`` is None.

    The dynamo exporter can leave stale intermediate ``value_info`` that makes onnxruntime's
    shape inference reject the graph, so the quantizer is fed a copy with it stripped (onnxruntime
    re-infers shapes itself; graph inputs/outputs keep their dynamic shapes).
    """
    import onnx
    from onnxruntime.quantization import QuantType, quantize_dynamic

    fp32_path, int8_path = Path(fp32_path), Path(int8_path)
    int8_path.parent.mkdir(parents=True, exist_ok=True)
    _remove_onnx(int8_path)

    stripped = int8_path.with_name(f"{int8_path.stem}.stripped.onnx")
    proto = onnx.load(str(fp32_path))
    del proto.graph.value_info[:]
    onnx.save_model(proto, str(stripped))
    del proto
    try:
        quantize_dynamic(str(stripped), str(int8_path), weight_type=QuantType.QInt8)
    finally:
        _remove_onnx(stripped)

    report: dict[str, Any] = {
        "path": str(int8_path),
        "fp32_size_mb": file_size_mb(fp32_path),
        "int8_size_mb": file_size_mb(int8_path),
    }
    report["size_ratio"] = report["int8_size_mb"] / report["fp32_size_mb"]
    if tokenizer is not None:
        report["agreement"] = compare_predictions(
            fp32_path, int8_path, tokenizer, texts, max_length
        )
    return report


def compare_predictions(
    fp32_path: str | Path,
    int8_path: str | Path,
    tokenizer: Any,
    texts: Sequence[str] | None = None,
    max_length: int = 32,
) -> dict:
    texts = _select_sample(texts)
    ids, mask = _tokenize(tokenizer, texts, max_length)
    fp_type, fp_tag = _run(make_session(fp32_path), ids, mask)
    q_type, q_tag = _run(make_session(int8_path), ids, mask)
    valid = mask.astype(bool)
    return {
        "samples": len(texts),
        "type_agreement": float((fp_type.argmax(-1) == q_type.argmax(-1)).mean()),
        "tag_agreement": float((fp_tag.argmax(-1) == q_tag.argmax(-1))[valid].mean()),
        "tag_tokens": int(valid.sum()),
        "max_abs_diff_type_logits": float(np.abs(fp_type - q_type).max()),
    }


def _percentiles(values_ms: list[float]) -> dict:
    ordered = sorted(values_ms)
    p95 = ordered[min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))]
    return {
        "p50_ms": statistics.median(ordered),
        "p95_ms": p95,
        "mean_ms": statistics.fmean(ordered),
    }


def benchmark_latency(
    session_or_path: ort.InferenceSession | str | Path,
    tokenizer: Any,
    texts: Sequence[str],
    runs: int = 200,
    warmup: int = 20,
    max_length: int = 32,
    threads: int | None = None,
) -> dict:
    """CPU single-note (batch 1) latency in ms; tokenization and inference timed separately."""
    session = (
        session_or_path
        if isinstance(session_or_path, ort.InferenceSession)
        else make_session(session_or_path, threads)
    )

    def step(text: str):
        t0 = time.perf_counter()
        ids, mask = _tokenize(tokenizer, [text], max_length)
        t1 = time.perf_counter()
        _run(session, ids, mask)
        t2 = time.perf_counter()
        return (t1 - t0) * 1e3, (t2 - t1) * 1e3

    return _timed(step, texts, runs, warmup)


def benchmark_torch_latency(
    model: torch.nn.Module,
    tokenizer: Any,
    texts: Sequence[str],
    runs: int = 200,
    warmup: int = 20,
    max_length: int = 32,
    threads: int | None = None,
) -> dict:
    """PyTorch CPU FP32 single-note latency for reference (same protocol as ONNX)."""
    model = model.to("cpu").eval()
    previous = torch.get_num_threads()
    if threads is not None:
        torch.set_num_threads(threads)

    def step(text: str):
        t0 = time.perf_counter()
        ids, mask = _tokenize(tokenizer, [text], max_length)
        t1 = time.perf_counter()
        _torch_logits(model, ids, mask)
        t2 = time.perf_counter()
        return (t1 - t0) * 1e3, (t2 - t1) * 1e3

    try:
        return _timed(step, texts, runs, warmup)
    finally:
        torch.set_num_threads(previous)


def _timed(step, texts: Sequence[str], runs: int, warmup: int) -> dict:
    texts = list(texts)
    if not texts:
        raise ValueError("benchmark needs at least one text")
    for i in range(warmup):
        step(texts[i % len(texts)])
    tok, inf = [], []
    for i in range(runs):
        t, f = step(texts[i % len(texts)])
        tok.append(t)
        inf.append(f)
    total = [a + b for a, b in zip(tok, inf, strict=True)]
    return {
        "runs": runs,
        "warmup": warmup,
        "texts": len(texts),
        "tokenization": _percentiles(tok),
        "inference": _percentiles(inf),
        "total": _percentiles(total),
    }
