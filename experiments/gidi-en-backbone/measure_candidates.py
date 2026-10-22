#!/usr/bin/env python
"""Measure ONNX/int8 feasibility of English backbone candidates with RANDOM weights.

Per candidate (architecture from the hub config only, no pretrained weights):
  * ``full``   stock config, int8 ONNX size and ONNX op set (ORT Web compatibility evidence);
  * ``pruned`` vocab 8000, 34 position rows, all layers, stock FFN;
  * ``cut``    like ``pruned`` but 4 layers, so (pruned - cut) / (layers - 4) = int8 bytes/layer;
and the hub licence tag and parameter count. Output: candidates.json next to this file.

Usage: uv run python experiments/gidi-en-backbone/measure_candidates.py
"""

from __future__ import annotations

import copy
import json
import tempfile
from pathlib import Path

import onnx
import torch
from huggingface_hub import HfApi
from transformers import AutoConfig, AutoTokenizer

from gidi.export.onnx_export import export_onnx, file_size_mb, quantize_int8
from gidi.modeling.model import GidiMultiTaskModel

OUT = Path(__file__).parent / "candidates.json"
NOTES = [json.loads(line)["text"] for line in (Path(__file__).parent / "audit-notes.jsonl").open()]
CANDIDATES = (
    "roberta-base",
    "distilroberta-base",
    "bert-base-uncased",
    "distilbert-base-uncased",
    "google/electra-small-discriminator",
    "microsoft/MiniLM-L12-H384-uncased",
    "google/bert_uncased_L-4_H-512_A-8",
    "microsoft/deberta-v3-small",
)
PRUNED_VOCAB = 8000
POS_ROWS = 34


class ModTokenizer:
    """Fold ids into the pruned vocabulary so a random-weight model can run (size test only)."""

    def __init__(self, tok, vocab: int) -> None:
        self.tok, self.vocab = tok, vocab

    def __call__(self, *a, **k):
        enc = self.tok(*a, **k)
        enc["input_ids"] = enc["input_ids"] % self.vocab
        return enc


def layers_key(cfg) -> str:
    return "n_layers" if hasattr(cfg, "n_layers") else "num_hidden_layers"


def set_layers(cfg, n: int) -> None:
    setattr(cfg, layers_key(cfg), n)


def n_layers(cfg) -> int:
    return int(getattr(cfg, layers_key(cfg)))


def variant(name: str, cfg, tok, tmp: Path, tag: str) -> dict:
    torch.manual_seed(0)
    model = GidiMultiTaskModel.from_config(cfg, encoder_name=name).eval()
    vocab = int(cfg.vocab_size)
    out: dict = {"params": model.param_count()}
    try:
        fp = tmp / f"{tag}.onnx"
        exp = export_onnx(model, ModTokenizer(tok, vocab), fp, 32, 17, NOTES[:16], atol=1e-3)
        q = quantize_int8(fp, tmp / f"{tag}.int8.onnx")
        ops = sorted({n.op_type for n in onnx.load(str(fp)).graph.node})
        out.update(
            exporter=exp["exporter"],
            parity=exp["parity"]["max_abs_diff"],
            fp32_mb=round(q["fp32_size_mb"], 2),
            int8_mb=round(file_size_mb(tmp / f"{tag}.int8.onnx"), 2),
            fp32_ops=ops,
        )
        int8_ops = sorted({n.op_type for n in onnx.load(str(tmp / f"{tag}.int8.onnx")).graph.node})
        out["int8_ops"] = int8_ops
    except Exception as exc:  # recorded as evidence, not hidden
        out["error"] = f"{type(exc).__name__}: {exc}"[:600]
    return out


def main() -> None:
    api = HfApi()
    results = {}
    for name in CANDIDATES:
        print("==", name, flush=True)
        cfg = AutoConfig.from_pretrained(name)
        tok = AutoTokenizer.from_pretrained(name)
        info = api.model_info(name)
        lic = next((t.split(":", 1)[1] for t in info.tags if t.startswith("license:")), None)
        row = {
            "license": lic,
            "arch": cfg.model_type,
            "layers": n_layers(cfg),
            "hidden": getattr(cfg, "hidden_size", getattr(cfg, "dim", None)),
            "vocab": cfg.vocab_size,
            "hub_safetensors_mb": round(
                (info.safetensors.total * 4 / 1e6) if info.safetensors else 0, 1
            ),
        }
        with tempfile.TemporaryDirectory() as t:
            tmp = Path(t)
            row["full"] = variant(name, cfg, tok, tmp, "full")
            pr = copy.deepcopy(cfg)
            pr.vocab_size = PRUNED_VOCAB
            pr.max_position_embeddings = POS_ROWS
            row["pruned"] = variant(name, pr, tok, tmp, "pruned")
            if n_layers(cfg) > 4:
                cut = copy.deepcopy(pr)
                set_layers(cut, 4)
                row["cut4"] = variant(name, cut, tok, tmp, "cut4")
                a, b = row["pruned"].get("int8_mb"), row["cut4"].get("int8_mb")
                if a and b:
                    row["int8_mb_per_layer"] = round((a - b) / (n_layers(cfg) - 4), 3)
        results[name] = row
        print(json.dumps({k: v for k, v in row.items()}, default=str)[:900], flush=True)
    OUT.write_text(json.dumps(results, indent=1) + "\n")


if __name__ == "__main__":
    main()
