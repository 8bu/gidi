import pytest
import torch
from transformers import RobertaConfig

from gidi.compression.ffn import neuron_scores, prune_ffn, select_neurons
from gidi.modeling.model import GidiMultiTaskModel

PAD = 1
SIZE = 32


def tiny_model() -> GidiMultiTaskModel:
    torch.manual_seed(0)
    config = RobertaConfig(
        vocab_size=40,
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=SIZE,
        max_position_embeddings=20,
        type_vocab_size=1,
        pad_token_id=PAD,
        bos_token_id=0,
        eos_token_id=2,
    )
    model = GidiMultiTaskModel.from_config(config)
    with torch.no_grad():
        for layer in model.encoder.encoder.layer:
            layer.intermediate.dense.bias.normal_()
    return model.eval()


def batch() -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(1)
    ids = torch.full((3, 10), PAD, dtype=torch.long)
    mask = torch.zeros_like(ids)
    for row, n in enumerate([10, 4, 7]):
        ids[row, :n] = torch.randint(3, 40, (n,), generator=generator)
        mask[row, :n] = 1
    return ids, mask


def test_pruned_model_equals_original_with_dropped_neurons_silenced():
    # gelu(0) = 0: zeroing a neuron's input row and bias removes exactly its contribution, so the
    # pruned model must compute the same function as the original with those neurons silenced.
    model = tiny_model()
    keep = [[0, 3, 5, 8, 13, 21, 30, 31], [1, 2, 4, 9, 16, 17, 25, 29]]
    pruned = prune_ffn(model, keep)
    silenced = prune_ffn(model, [list(range(SIZE))] * 2)
    with torch.no_grad():
        for layer, idx in zip(silenced.encoder.encoder.layer, keep, strict=True):
            drop = sorted(set(range(SIZE)) - set(idx))
            layer.intermediate.dense.weight[drop] = 0
            layer.intermediate.dense.bias[drop] = 0
    ids, mask = batch()
    with torch.inference_mode():
        for got, want in zip(pruned(ids, mask), silenced(ids, mask), strict=True):
            torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-6)
    assert pruned.encoder.config.intermediate_size == 8
    assert model.encoder.config.intermediate_size == SIZE  # input untouched
    assert pruned.param_count() == model.param_count() - 2 * (SIZE - 8) * (2 * 16 + 1)


def test_keeping_every_neuron_is_bitwise_identical():
    model = tiny_model()
    same = prune_ffn(model, [list(range(SIZE))] * 2)
    ids, mask = batch()
    with torch.inference_mode():
        for got, want in zip(same(ids, mask), model(ids, mask), strict=True):
            assert torch.equal(got, want)


@pytest.mark.parametrize(
    "keep",
    [
        [[0, 1], [0, 1, 2]],  # unequal widths
        [[1, 0], [0, 1]],  # not ascending
        [[0, 0], [0, 1]],  # duplicate
        [[0, SIZE], [0, 1]],  # out of range
        [[0, 1]],  # wrong layer count
    ],
)
def test_invalid_neuron_maps_are_rejected(keep):
    with pytest.raises(ValueError):
        prune_ffn(tiny_model(), keep)


def test_select_neurons_keeps_top_scores_and_breaks_ties_to_lower_index():
    scores = torch.tensor([0.5, 2.0, 0.5, 3.0, 0.5])
    assert select_neurons(scores, 3) == [0, 1, 3]
    with pytest.raises(ValueError):
        select_neurons(scores, 0)


def test_activation_scores_ignore_padding():
    model = tiny_model()
    ids, mask = batch()
    garbage = ids.clone()
    garbage[mask == 0] = 7  # different pad-slot ids must not change any score
    a = neuron_scores(model, ids, mask)
    b = neuron_scores(model, garbage, mask)
    for la, lb in zip(a, b, strict=True):
        torch.testing.assert_close(la["activation"], lb["activation"])
