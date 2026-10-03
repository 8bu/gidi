"""Fixed-epoch mode (``TrainConfig.stop_epoch``) on a tiny random encoder."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from tokenizers import Tokenizer, models, pre_tokenizers, processors
from transformers import BertConfig, BertModel, PreTrainedTokenizerFast

from gidi.corpus.jsonl import read_jsonl, write_jsonl
from gidi.modeling.checkpoint import load_checkpoint
from gidi.training.train import TrainConfig, train_run

VOCAB = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "chị", "thảo", "trả", "lại", "1tr", "nam", "ăn", "phở"]


def record(id_: str, text: str, type_: str, target: str | None) -> dict:
    span = None
    if target:
        start = text.index(target)
        span = {"text": target, "start": start, "end": start + len(target)}
    return {"id": id_, "text": text, "type": type_, "target": span}


@pytest.fixture
def encoder_dir(tmp_path: Path) -> Path:
    path = tmp_path / "encoder"
    config = BertConfig(
        vocab_size=len(VOCAB),
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
        max_position_embeddings=32,
    )
    BertModel(config).save_pretrained(path)
    tok = Tokenizer(models.WordLevel({t: i for i, t in enumerate(VOCAB)}, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tok.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 2), ("[SEP]", 3)]
    )
    PreTrainedTokenizerFast(
        tokenizer_object=tok,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
    ).save_pretrained(path)
    return path


@pytest.fixture
def splits_dir(tmp_path: Path) -> Path:
    rows = [
        record("a", "ăn phở 1tr", "expense", None),
        record("b", "chị thảo trả lại 1tr", "repayment_in", "thảo"),
        record("c", "trả lại chị nam 1tr", "repayment_out", "nam"),
        record("d", "ăn phở nam", "expense", None),
    ]
    path = tmp_path / "splits"
    for name in ("train", "validation", "test"):
        write_jsonl(path / f"{name}.jsonl", rows)
    return path


def config(tmp_path: Path, encoder: Path, splits: Path, **kwargs) -> TrainConfig:
    return TrainConfig(
        model=str(encoder),
        lr=1e-3,
        seed=1,
        batch_size=2,
        epochs=6,
        patience=1,
        splits_dir=str(splits),
        out_dir=str(tmp_path / "runs"),
        weights_dir=str(tmp_path / "weights"),
        device="cpu",
        **kwargs,
    )


def test_stop_epoch_trains_exactly_that_many_epochs_and_keeps_last_weights(
    tmp_path: Path, encoder_dir: Path, splits_dir: Path
) -> None:
    cfg = config(tmp_path, encoder_dir, splits_dir, stop_epoch=3)
    summary = train_run(cfg)
    run = tmp_path / "runs" / "encoder" / "lr0.001-seed1"

    # patience=1 would stop a selection run early; fixed-epoch mode ignores it
    assert summary["epochs_run"] == 3
    assert summary["best_epoch"] == 3
    assert len(read_jsonl(run / "train_log.jsonl")) == 3
    # the schedule horizon stays epochs * steps_per_epoch, so the lr has not decayed to zero
    assert read_jsonl(run / "train_log.jsonl")[-1]["lr"] > 0
    assert json.loads((run / "config.json").read_text())["stop_epoch"] == 3

    _, _, meta = load_checkpoint(tmp_path / "weights" / "encoder" / "lr0.001-seed1")
    assert meta["stop_epoch"] == 3
    assert meta["selection"] == "none (fixed epoch)"
    assert meta["best_epoch"] == 3


def test_stop_epoch_outside_epochs_is_rejected(
    tmp_path: Path, encoder_dir: Path, splits_dir: Path
) -> None:
    with pytest.raises(ValueError, match="stop_epoch"):
        train_run(config(tmp_path, encoder_dir, splits_dir, stop_epoch=7))
    assert not (tmp_path / "runs").exists()
