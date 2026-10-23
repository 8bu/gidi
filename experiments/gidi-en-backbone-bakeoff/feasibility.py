#!/usr/bin/env python
"""Feasibility of the bake-off candidates: width rule, int8 size and ONNX op set (random weights).

No quality is measured. For each candidate the *fixed width rule* of protocol.json picks the
student shape from the parameter budget only; the student is then built with random weights,
exported with the repo's dynamo exporter (opset 17), dynamically quantised to int8 like the
shipped model, and its op-type set compared with the shipped gidi-en v3 int8 graph.

    uv run python experiments/gidi-en-backbone-bakeoff/feasibility.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import onnx
import torch
from transformers import AutoConfig

from gidi.export.onnx_export import _dynamo_export, make_session, quantize_int8
from gidi.modeling.model import GidiMultiTaskModel

OUT = Path("experiments/gidi-en-backbone-bakeoff/feasibility.json")
SHIPPED = Path("models/gidi-en-v3-onnx/seed1/model.int8.onnx")
BUDGET_PARAMS = 29_200_000
MAX_LENGTH = 32


def ops(path: Path) -> list[str]:
    return sorted({n.op_type for n in onnx.load(str(path)).graph.node})


def meta_params(config, hub_class_kwargs=None) -> int:
    with torch.device("meta"):
        model = GidiMultiTaskModel.from_config(config)
    return model.param_count()


def hub_config(name: str, **over):
    config = AutoConfig.from_pretrained(name)
    for key, value in over.items():
        setattr(config, key, value)
    return config


def solve(name: str, vary: str, step: int, lo: int, hi: int, **fixed) -> tuple[int, int]:
    """Largest multiple of ``step`` in [lo, hi] of attribute ``vary`` with params <= budget."""
    best = None
    for value in range(lo, hi + 1, step):
        params = meta_params(hub_config(name, **fixed, **{vary: value}))
        if params <= BUDGET_PARAMS:
            best = (value, params)
    if best is None:
        raise SystemExit(f"{name}: no {vary} fits the budget")
    return best


def measure(label: str, config, opset: int = 17) -> dict:
    torch.manual_seed(0)
    model = GidiMultiTaskModel.from_config(config).eval()
    ids = torch.randint(5, int(config.vocab_size), (2, MAX_LENGTH))
    ids[:, 0] = 0
    mask = torch.ones_like(ids)
    mask[1, 20:] = 0
    with tempfile.TemporaryDirectory() as tmp:
        fp32, int8 = Path(tmp, "m.onnx"), Path(tmp, "m.int8.onnx")
        _dynamo_export(model, (ids, mask), fp32, opset)
        q = quantize_int8(fp32, int8)
        sess = make_session(fp32)
        with torch.inference_mode():
            t_ref, g_ref = model(ids, mask)
        t, g = sess.run(
            ["type_logits", "tag_logits"],
            {"input_ids": ids.numpy(), "attention_mask": mask.numpy()},
        )
        real = mask.bool().numpy()
        diff = max(
            float(np.abs(t - t_ref.numpy()).max()), float(np.abs((g - g_ref.numpy())[real]).max())
        )
        # shorter, unpadded input must also run (dynamic axes)
        sess.run(None, {"input_ids": ids.numpy()[:1, :9], "attention_mask": mask.numpy()[:1, :9]})
        graph_ops = ops(int8)
        return {
            "label": label,
            "params": model.param_count(),
            "fp32_bytes": fp32.stat().st_size,
            "int8_bytes": int8.stat().st_size,
            "fp32_parity_max_abs_diff": diff,
            "ops": graph_ops,
            "ops_not_in_shipped": sorted(set(graph_ops) - set(ops(SHIPPED))),
            "size_report": q,
        }


def main() -> int:
    results: dict = {"budget_params": BUDGET_PARAMS, "shipped_ops": ops(SHIPPED), "candidates": {}}

    def add(key: str, **kw) -> None:
        print("==", key, flush=True)
        try:
            results["candidates"][key] = measure(key, **kw)
            r = results["candidates"][key]
            print(
                key,
                r["params"],
                r["int8_bytes"],
                r["ops_not_in_shipped"],
                r["fp32_parity_max_abs_diff"],
                flush=True,
            )
        except Exception as exc:  # recorded, not hidden
            results["candidates"][key] = {
                "label": key,
                "error": f"{type(exc).__name__}: {exc}"[:3000],
            }
            print(key, "ERROR", str(exc)[:500], flush=True)
        OUT.write_text(json.dumps(results, indent=1) + "\n")

    rob = dict(
        num_hidden_layers=4, intermediate_size=2048, vocab_size=9000, max_position_embeddings=34
    )
    add("roberta-base (control)", config=hub_config("roberta-base", **rob))

    k, _ = solve(
        "distilbert/distilroberta-base",
        "intermediate_size",
        64,
        64,
        3072,
        vocab_size=9000,
        max_position_embeddings=34,
    )
    results["distilroberta_ffn_k"] = k
    add(
        f"distilroberta-base 6L K={k}",
        config=hub_config(
            "distilbert/distilroberta-base",
            intermediate_size=k,
            vocab_size=9000,
            max_position_embeddings=34,
        ),
    )

    v, _ = solve(
        "microsoft/MiniLM-L12-H384-uncased",
        "vocab_size",
        500,
        9000,
        30522,
        max_position_embeddings=33,
    )
    results["minilm_vocab_rows"] = v
    add(
        f"MiniLM-L12-H384 12L vocab={v}",
        config=hub_config(
            "microsoft/MiniLM-L12-H384-uncased", vocab_size=v, max_position_embeddings=33
        ),
    )

    add(
        "electra-base-discriminator 4L K=2048 vocab=9000",
        config=hub_config(
            "google/electra-base-discriminator",
            num_hidden_layers=4,
            intermediate_size=2048,
            vocab_size=9000,
            max_position_embeddings=33,
        ),
    )

    mb = hub_config("answerdotai/ModernBERT-base", num_hidden_layers=4, vocab_size=9000)
    if getattr(mb, "layer_types", None):
        mb.layer_types = mb.layer_types[:4]
    # the hub ids (pad 50283, ...) fall outside a 9,000-row table; real ids come after pruning
    for attr, value in (("pad_token_id", 1), ("bos_token_id", 0), ("eos_token_id", 2)):
        setattr(mb, attr, value)
    mb.cls_token_id, mb.sep_token_id = 0, 2
    mb.reference_compile = False
    add("ModernBERT-base 4L vocab=9000 (opset 17)", config=mb)
    add("ModernBERT-base 4L vocab=9000 (opset 18)", config=mb, opset=18)

    OUT.write_text(json.dumps(results, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
