import pytest
import torch
from transformers import RobertaConfig

from gidi.compression.positions import truncate_positions
from gidi.modeling.model import GidiMultiTaskModel

PAD = 1
MAX_LENGTH = 12


def tiny_model(rows: int = 64, position_embedding_type: str = "absolute") -> GidiMultiTaskModel:
    torch.manual_seed(0)
    config = RobertaConfig(
        vocab_size=40,
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=32,
        max_position_embeddings=rows,
        type_vocab_size=1,
        pad_token_id=PAD,
        bos_token_id=0,
        eos_token_id=2,
        position_embedding_type=position_embedding_type,
    )
    model = GidiMultiTaskModel.from_config(config)
    # distinguish every row so a wrong slice cannot go unnoticed
    with torch.no_grad():
        model.encoder.embeddings.position_embeddings.weight.normal_()
    return model.eval()


def padded_batch(lengths: list[int], width: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(1)
    ids = torch.full((len(lengths), width), PAD, dtype=torch.long)
    mask = torch.zeros_like(ids)
    for row, n in enumerate(lengths):
        ids[row, :n] = torch.randint(3, 40, (n,), generator=generator)
        mask[row, :n] = 1
    return ids, mask


def test_truncated_model_is_bitwise_identical_on_full_length_and_padded_inputs():
    model = tiny_model()
    short = truncate_positions(model, MAX_LENGTH)
    # one row hits max_length exactly (largest position index = max_length + pad id)
    ids, mask = padded_batch([MAX_LENGTH, 5, 1, 9], width=MAX_LENGTH)
    with torch.inference_mode():
        for batch_ids, batch_mask in [(ids, mask), (ids[:, :9], mask[:, :9])]:
            expected = model(batch_ids, batch_mask)
            actual = short(batch_ids, batch_mask)
            assert torch.equal(actual[0], expected[0])
            assert torch.equal(actual[1], expected[1])


def test_truncation_keeps_exactly_the_needed_rows_and_all_other_tensors():
    model = tiny_model(rows=64)
    short = truncate_positions(model, MAX_LENGTH)
    rows = MAX_LENGTH + PAD + 1
    assert short.encoder.config.max_position_embeddings == rows
    assert model.encoder.config.max_position_embeddings == 64  # input untouched
    old, new = model.state_dict(), short.state_dict()
    assert old.keys() == new.keys()
    for key, tensor in new.items():
        if key.endswith("position_embeddings.weight"):
            assert tensor.shape[0] == rows
            assert torch.equal(tensor, old[key][:rows])
        else:
            assert torch.equal(tensor, old[key]), key
    assert short.param_count() == model.param_count() - (64 - rows) * 16


def test_exactly_enough_rows_is_a_valid_noop_and_one_fewer_is_refused():
    needed = MAX_LENGTH + PAD + 1
    same = truncate_positions(tiny_model(rows=needed), MAX_LENGTH)
    assert same.encoder.config.max_position_embeddings == needed
    with pytest.raises(ValueError, match="position rows"):
        truncate_positions(tiny_model(rows=needed - 1), MAX_LENGTH)


def test_non_absolute_position_embeddings_are_refused():
    with pytest.raises(ValueError, match="absolute"):
        truncate_positions(tiny_model(position_embedding_type="relative_key"), MAX_LENGTH)
