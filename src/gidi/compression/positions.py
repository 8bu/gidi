"""Truncate the absolute position-embedding table to the rows a short-note model can index.

RoBERTa numbers real tokens ``pad_id + 1, pad_id + 2, ...`` and gives pad slots ``pad_id``
(``RobertaEmbeddings.create_position_ids_from_input_ids``). A sequence of at most ``max_length``
tokens therefore indexes rows ``0..max_length + pad_id`` (row 0 is never read but keeps the
alignment), i.e. ``max_length + pad_id + 1`` rows. Dropping the rest changes no output.
"""

from __future__ import annotations

import copy

from gidi.modeling.model import GidiMultiTaskModel


def required_rows(max_length: int, pad_token_id: int) -> int:
    """Rows needed so every position index of a ``max_length``-token sequence is valid."""
    return max_length + pad_token_id + 1


def truncate_positions(model: GidiMultiTaskModel, max_length: int) -> GidiMultiTaskModel:
    """Return a new model whose position table keeps only the first ``required_rows`` rows.

    Every other tensor (and both heads) is copied unchanged; ``model`` is not mutated.
    """
    config = model.encoder.config
    if getattr(config, "position_embedding_type", "absolute") != "absolute":
        raise ValueError(
            f"only absolute position embeddings can be truncated, got "
            f"{config.position_embedding_type!r}"
        )
    if max_length < 1:
        raise ValueError(f"max_length must be positive, got {max_length}")
    rows = required_rows(max_length, int(config.pad_token_id))
    have = int(config.max_position_embeddings)
    if have < rows:
        raise ValueError(
            f"model has {have} position rows but max_length={max_length} needs {rows} "
            f"(max_length + pad_token_id + 1)"
        )

    new_config = copy.deepcopy(config)
    new_config.max_position_embeddings = rows
    new = GidiMultiTaskModel.from_config(
        new_config,
        num_types=model.num_types,
        num_tags=model.num_tags,
        dropout=model.dropout_p,
        encoder_name=model.encoder_name,
    )
    table = model.encoder.embeddings.position_embeddings.weight
    table_key = next(name for name, p in model.named_parameters() if p is table)
    state = dict(model.state_dict())
    state[table_key] = state[table_key][:rows].clone()
    new.load_state_dict(state, strict=True)
    new.train(model.training)
    return new
