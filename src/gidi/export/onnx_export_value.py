"""ONNX export of the 3-head (type, target, value) model; INT8 exactly as v1.

The graph is the v1 graph plus one output: int64 inputs ``input_ids`` / ``attention_mask`` with
dynamic batch and sequence axes (unchanged), outputs ``type_logits`` (B, num_types),
``tag_logits`` (B, L, 3) and ``value_logits`` (B, L, 3) in the label order
``["O", "B-VALUE", "I-VALUE"]``. The v1 output names and their order are unchanged. Export uses
the v1 exporters (dynamo first, TorchScript fallback, opset 17) and INT8 uses the v1
``quantize_int8`` (dynamic, ``QuantType.QInt8`` weights) unchanged; only the parity and agreement
checks are widened to three outputs.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnx import numpy_helper

from gidi.export.onnx_export import (
    DEFAULT_ATOL,
    OUTPUT_NAMES,
    _dynamo_export,
    _legacy_export,
    _remove_onnx,
    _select_sample,
    _tokenize,
    file_size_mb,
    make_session,
    quantize_int8,
)

VALUE_OUTPUT_NAMES = (*OUTPUT_NAMES, "value_logits")


def _run(session: ort.InferenceSession, ids: np.ndarray, mask: np.ndarray):
    return session.run(list(VALUE_OUTPUT_NAMES), {"input_ids": ids, "attention_mask": mask})


def _torch_logits(model: torch.nn.Module, ids: np.ndarray, mask: np.ndarray):
    with torch.inference_mode():
        outputs = model(torch.from_numpy(ids), torch.from_numpy(mask))
    return tuple(o.numpy() for o in outputs)


def check_parity(
    model: torch.nn.Module,
    session: ort.InferenceSession,
    tokenizer: Any,
    texts: Sequence[str] | None = None,
    max_length: int = 32,
    atol: float = DEFAULT_ATOL,
) -> dict:
    """ONNX Runtime CPU vs PyTorch on a padded batch and on each note alone, per output."""
    texts = _select_sample(texts)
    model.eval()
    ids, mask = _tokenize(tokenizer, texts, max_length)
    ref, got = _torch_logits(model, ids, mask), _run(session, ids, mask)
    valid = mask.astype(bool)
    batch = {
        "type_logits": float(np.abs(ref[0] - got[0]).max()),
        "tag_logits": float(np.abs(ref[1] - got[1])[valid].max()),
        "value_logits": float(np.abs(ref[2] - got[2])[valid].max()),
    }
    single = dict.fromkeys(batch, 0.0)
    for text in texts:  # shapes different from the export trace: batch 1, natural length
        ids, mask = _tokenize(tokenizer, [text], max_length)
        r, g = _torch_logits(model, ids, mask), _run(session, ids, mask)
        for name, a, b in zip(single, r, g, strict=True):
            single[name] = max(single[name], float(np.abs(a - b).max()))
    max_abs = max(*batch.values(), *single.values())
    return {
        "samples": len(texts),
        "atol": atol,
        "max_abs_diff_type_logits": batch["type_logits"],
        "max_abs_diff_tag_logits": batch["tag_logits"],
        "max_abs_diff_value_logits": batch["value_logits"],
        "max_abs_diff_single_note": max(single.values()),
        "max_abs_diff_single_note_type_logits": single["type_logits"],
        "max_abs_diff_single_note_tag_logits": single["tag_logits"],
        "max_abs_diff_single_note_value_logits": single["value_logits"],
        "max_abs_diff": max_abs,
        "passed": bool(max_abs <= atol),
    }


def export_value_onnx(
    model: torch.nn.Module,
    tokenizer: Any,
    out_path: str | Path,
    max_length: int = 32,
    opset: int = 17,
    sample_texts: Sequence[str] | None = None,
    atol: float = DEFAULT_ATOL,
) -> dict:
    """Export ``model`` to ``out_path`` and validate it against PyTorch (see ``export_onnx``)."""
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
            fn(model, args, out_path, opset, VALUE_OUTPUT_NAMES)
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


def compare_predictions(
    fp32_path: str | Path,
    int8_path: str | Path,
    tokenizer: Any,
    texts: Sequence[str] | None = None,
    max_length: int = 32,
) -> dict:
    """INT8 vs FP32 argmax agreement for type, target tags and value tags (real tokens)."""
    texts = _select_sample(texts)
    ids, mask = _tokenize(tokenizer, texts, max_length)
    fp = _run(make_session(fp32_path), ids, mask)
    q = _run(make_session(int8_path), ids, mask)
    valid = mask.astype(bool)
    return {
        "samples": len(texts),
        "type_agreement": float((fp[0].argmax(-1) == q[0].argmax(-1)).mean()),
        "tag_agreement": float((fp[1].argmax(-1) == q[1].argmax(-1))[valid].mean()),
        "value_agreement": float((fp[2].argmax(-1) == q[2].argmax(-1))[valid].mean()),
        "tag_tokens": int(valid.sum()),
        "max_abs_diff_type_logits": float(np.abs(fp[0] - q[0]).max()),
        "max_abs_diff_value_logits": float(np.abs(fp[2] - q[2])[valid].max()),
    }


def quantize_value_int8(
    fp32_path: str | Path,
    int8_path: str | Path,
    tokenizer: Any = None,
    texts: Sequence[str] | None = None,
    max_length: int = 32,
) -> dict:
    """The v1 ``quantize_int8`` (size report) plus the 3-output INT8-vs-FP32 agreement."""
    report = quantize_int8(fp32_path, int8_path, tokenizer=None)
    if tokenizer is not None:
        report["agreement"] = compare_predictions(
            fp32_path, int8_path, tokenizer, texts, max_length
        )
    return report


def onnx_interface(path: str | Path) -> dict[str, Any]:
    """Input/output names and symbolic shapes of an ONNX file (for reports and tests)."""
    session = make_session(path)
    return {
        "inputs": [i.name for i in session.get_inputs()],
        "outputs": [o.name for o in session.get_outputs()],
        "output_shapes": {o.name: list(o.shape) for o in session.get_outputs()},
    }


def _initializer_digests(path: str | Path, min_elements: int) -> dict[str, tuple]:
    """``{name: (dtype, shape, sha256 of the raw bytes)}`` for every non-trivial initializer."""
    model = onnx.load(str(path), load_external_data=False)
    digests = {}
    for init in model.graph.initializer:
        array = numpy_helper.to_array(init)
        if array.size >= min_elements:
            digest = hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()
            digests[init.name] = (str(array.dtype), tuple(array.shape), digest)
    return digests


def compare_initializers(
    reference: str | Path, candidate: str | Path, min_elements: int = 2, listed: int = 20
) -> dict:
    """Does every weight initializer of ``reference`` also exist, bit for bit, in ``candidate``?

    Matching is by content (dtype, shape, bytes), one candidate tensor per reference tensor,
    because exporters and the quantizer rename anonymous initializers (``val_337``); by-name
    equality is reported as well. Scalars and one-element constants are ignored. The candidate
    may hold more tensors (the value branch of a dual-encoder graph).
    """
    ref = _initializer_digests(reference, min_elements)
    cand = _initializer_digests(candidate, min_elements)
    available = Counter(cand.values())
    unmatched = []
    for name, digest in ref.items():
        if available[digest] > 0:
            available[digest] -= 1
        else:
            unmatched.append({"name": name, "dtype": digest[0], "shape": list(digest[1])})
    shared = [name for name in ref if name in cand]
    return {
        "reference": str(reference),
        "candidate": str(candidate),
        "reference_tensors": len(ref),
        "candidate_tensors": len(cand),
        "matched_by_content": len(ref) - len(unmatched),
        "unmatched": len(unmatched),
        "unmatched_listed": unmatched[:listed],
        "same_name_tensors": len(shared),
        "same_name_equal": sum(ref[n] == cand[n] for n in shared),
        "all_reference_tensors_present": not unmatched,
    }


def onnx_opset(path: str | Path) -> int:
    """The default-domain opset version an ONNX file declares."""
    model = onnx.load(str(path), load_external_data=False)
    return max(i.version for i in model.opset_import if i.domain in ("", "ai.onnx"))
