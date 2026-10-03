"""Save/load a trained Gidi model as a self-contained directory.

Layout of ``out_dir``::

    model.safetensors       encoder + both heads
    config.json             encoder architecture (HF config) so the encoder can be rebuilt
    tokenizer files         whatever ``tokenizer.save_pretrained`` writes
    gidi_checkpoint.json    {"encoder", "types", "tags", "max_length", ...caller meta}
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from safetensors.torch import load_model, save_model
from transformers import AutoConfig, PreTrainedTokenizerBase

from gidi.modeling.model import GidiMultiTaskModel
from gidi.modeling.preprocessing import DEFAULT_MAX_LENGTH, TAGS, TYPES
from gidi.modeling.tokenization import load_tokenizer

WEIGHTS_FILE = "model.safetensors"
META_FILE = "gidi_checkpoint.json"


def save_checkpoint(
    model: GidiMultiTaskModel,
    tokenizer: PreTrainedTokenizerBase,
    out_dir: str | Path,
    meta: dict[str, Any],
) -> Path:
    """Write ``model``, ``tokenizer`` and metadata to ``out_dir``; returns the directory.

    ``meta`` extends (and may override) the standard keys ``encoder``, ``types``, ``tags``,
    ``max_length``. Weights are written from CPU copies so MPS/CUDA models save unchanged.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_model(model, str(out / WEIGHTS_FILE))
    model.encoder.config.save_pretrained(out)
    tokenizer.save_pretrained(out)
    record = {
        "encoder": model.encoder_name,
        "types": list(TYPES),
        "tags": list(TAGS),
        "max_length": DEFAULT_MAX_LENGTH,
        "num_types": model.num_types,
        "num_tags": model.num_tags,
        "dropout": model.dropout_p,
        **meta,
    }
    (out / META_FILE).write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return out


def load_checkpoint(
    out_dir: str | Path,
) -> tuple[GidiMultiTaskModel, PreTrainedTokenizerBase, dict[str, Any]]:
    """Rebuild ``(model, tokenizer, meta)`` from a ``save_checkpoint`` directory (CPU, eval)."""
    out = Path(out_dir)
    meta = json.loads((out / META_FILE).read_text(encoding="utf-8"))
    model = GidiMultiTaskModel.from_config(
        AutoConfig.from_pretrained(out),
        num_types=meta["num_types"],
        num_tags=meta["num_tags"],
        dropout=meta["dropout"],
        encoder_name=meta["encoder"],
    )
    load_model(model, str(out / WEIGHTS_FILE), strict=True)
    model.eval()
    return model, load_tokenizer(str(out)), meta
