"""Value-span model: BIO alignment/masking, 3-head model, decoding, metrics, ONNX export."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers, processors
from transformers import BertConfig, PreTrainedTokenizerFast

from gidi.evaluation.value_eval import evaluate_records
from gidi.evaluation.value_metrics import (
    evaluate_value,
    overlap_metrics,
    surface_pattern,
    token_indices,
)
from gidi.evaluation.value_predict import decode_predictions, onnx_logits, torch_logits
from gidi.export.onnx_export import make_session
from gidi.export.onnx_export_value import (
    VALUE_OUTPUT_NAMES,
    export_value_onnx,
    onnx_interface,
    quantize_value_int8,
)
from gidi.inference.decode import decode_first_span
from gidi.modeling.model import GidiMultiTaskModel
from gidi.modeling.preprocessing import IGNORE_INDEX, TYPES
from gidi.modeling.value import (
    VALUE_TAGS,
    GidiValueModel,
    encode_value,
    load_value_checkpoint,
    prepare_value,
    save_value_checkpoint,
    validate_value_record,
)
from gidi.modeling.value_decode import decode_all_spans, decode_value
from gidi.training.train_value import eval_losses, multitask_loss

WORDS = [
    "cơm", "tấm", "100", "mượn", "chú", "hai", "5", "xị", "ăn", "2", "tô", "phở", "70",
    "mua", "3", "vé", "150k", "trả", "góp", "kỳ", "1tr5", "an", "sang", "35k", "tiền", "điện",
]  # fmt: skip
VOCAB = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", *WORDS]


def tiny_tokenizer() -> PreTrainedTokenizerFast:
    tok = Tokenizer(models.WordLevel({t: i for i, t in enumerate(VOCAB)}, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tok.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 2), ("[SEP]", 3)]
    )
    return PreTrainedTokenizerFast(
        tokenizer_object=tok,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
    )


def tiny_config() -> BertConfig:
    return BertConfig(
        vocab_size=len(VOCAB),
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=32,
        max_position_embeddings=64,
    )


def span(text: str, piece: str) -> dict:
    start = text.index(piece)
    return {"text": piece, "start": start, "end": start + len(piece)}


def record(
    text: str,
    value: str | None,
    *,
    status: str = "complete",
    target: str | None = None,
    type_: str = "expense",
    rid: str = "r",
    provenance: str = "human",
    **extra,
) -> dict:
    return {
        "id": rid,
        "text": text,
        "type": type_,
        "target": span(text, target) if target else None,
        "value": span(text, value) if value else None,
        "value_status": status,
        "value_provenance": provenance,
        **extra,
    }


def tags(enc, i: int) -> list[int]:
    return [t for t in enc.value_labels[i].tolist()]


# --------------------------------------------------------------------------- alignment


def test_value_tags_follow_target_alignment():
    recs = [
        record("ăn 2 tô phở 70", "70"),
        record("mượn chú hai 5 xị", "5 xị", target="chú hai", type_="borrow"),
    ]
    enc = encode_value(tiny_tokenizer(), recs, 32)
    # [CLS] ăn 2 tô phở 70 [SEP]
    assert tags(enc, 0)[:7] == [IGNORE_INDEX, 0, 0, 0, 0, 1, IGNORE_INDEX]
    # [CLS] mượn chú hai 5 xị [SEP]: B-VALUE then I-VALUE
    assert tags(enc, 1)[:7] == [IGNORE_INDEX, 0, 0, 0, 1, 2, IGNORE_INDEX]
    # the target head alignment of the same batch is the v1 one
    assert enc.encoded.tag_labels[1].tolist()[:7] == [IGNORE_INDEX, 0, 1, 2, 0, 0, IGNORE_INDEX]
    assert VALUE_TAGS == ("O", "B-VALUE", "I-VALUE")


def test_padding_positions_are_ignored():
    recs = [record("ăn 2 tô phở 70", "70"), record("cơm tấm 100", "100")]
    enc = encode_value(tiny_tokenizer(), recs, 32)
    assert tags(enc, 1)[5:] == [IGNORE_INDEX] * (len(tags(enc, 1)) - 5)
    assert tags(enc, 1)[:5] == [IGNORE_INDEX, 0, 0, 1, IGNORE_INDEX]


def test_non_complete_records_are_fully_masked_but_keep_target_tags():
    recs = [
        record("mượn chú hai 5 xị", "5 xị", status="uncertain", target="chú hai", type_="borrow"),
        record("tiền điện", None),
    ]
    enc = encode_value(tiny_tokenizer(), recs, 32)
    assert set(tags(enc, 0)) == {IGNORE_INDEX}
    assert enc.supervised == [False, True]
    assert enc.encoded.tag_labels[0].tolist()[:7] == [IGNORE_INDEX, 0, 1, 2, 0, 0, IGNORE_INDEX]
    # complete + no amount: every real token is O, specials stay ignored
    assert tags(enc, 1)[:4] == [IGNORE_INDEX, 0, 0, IGNORE_INDEX]


def test_masked_record_with_unalignable_value_does_not_raise():
    # the (unsupervised) value is not looked at, so a bad boundary cannot abort training
    text = "cơm tấm 100"
    rec = record(text, "100", status="uncertain")
    rec["value"] = {"text": "ấm 1", "start": 5, "end": 9}
    enc = encode_value(tiny_tokenizer(), [rec], 32)
    assert set(tags(enc, 0)) == {IGNORE_INDEX}


def test_straddled_value_boundary_is_strict_or_counted():
    text = "cơm tấm 100"
    rec = record(text, "100")
    rec["value"] = {"text": text[9:11], "start": 9, "end": 11}  # "00": inside the token "100"
    with pytest.raises(ValueError):
        encode_value(tiny_tokenizer(), [rec], 32)
    # a token overlapping the span is tagged, so the rebuilt span differs from the gold span
    lenient = encode_value(tiny_tokenizer(), [rec], 32, strict=False)
    assert lenient.value_boundary_mismatch == [True]
    assert lenient.n_value_boundary_mismatch == 1


def test_value_beyond_max_length_is_all_o_and_flagged():
    rec = record("cơm tấm ăn phở tô vé 100", "100")
    enc = encode_value(tiny_tokenizer(), [rec], 4)  # [CLS] cơm tấm [SEP]
    assert tags(enc, 0) == [IGNORE_INDEX, 0, 0, IGNORE_INDEX]
    assert enc.value_span_truncated == [True]


def test_partially_truncated_value_keeps_surviving_tags():
    rec = record("mượn chú hai 5 xị", "5 xị")
    enc = encode_value(tiny_tokenizer(), [rec], 6)  # [CLS] mượn chú hai 5 [SEP]
    assert tags(enc, 0) == [IGNORE_INDEX, 0, 0, 0, 1, IGNORE_INDEX]
    assert enc.value_span_truncated == [True]


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda r: r.pop("value_status"), "lacks 'value_status'"),
        (lambda r: r.update(value_status="skipped"), "value_status"),
        (lambda r: r.update(value_provenance="ai"), "value_provenance"),
        (lambda r: r["value"].update(text="x"), "reproduce"),
        (lambda r: r["value"].update(start=9, end=9), "offsets"),
    ],
)
def test_validate_value_record_rejects_bad_blocks(mutate, message):
    rec = record("cơm tấm 100", "100")
    mutate(rec)
    with pytest.raises(ValueError, match=message):
        validate_value_record(rec)


def test_validate_rejects_value_with_surrounding_whitespace():
    rec = record("cơm tấm 100", "100")
    rec["value"] = {"text": " 100", "start": 7, "end": 11}
    with pytest.raises(ValueError, match="whitespace"):
        validate_value_record(rec)


# --------------------------------------------------------------------------- model


def v1_model(seed: int) -> GidiMultiTaskModel:
    torch.manual_seed(seed)
    return GidiMultiTaskModel.from_config(tiny_config())


def test_from_v1_keeps_encoder_and_heads_and_global_rng():
    base = v1_model(1)
    before = {k: v.clone() for k, v in base.state_dict().items()}
    torch.manual_seed(123)
    expected_next = torch.rand(3)
    torch.manual_seed(123)
    model = GidiValueModel.from_v1(base, seed=1)
    assert torch.equal(torch.rand(3), expected_next)  # the global RNG stream is untouched
    for key, tensor in model.state_dict().items():
        if key.startswith("value_head"):
            continue
        assert torch.equal(tensor, before[key]), key
    assert set(model.state_dict()) - set(before) == {"value_head.weight", "value_head.bias"}
    assert model.value_head.weight.shape == (3, 16)


def test_value_head_is_seeded_and_not_the_target_head():
    a = GidiValueModel.from_v1(v1_model(1), seed=1)
    b = GidiValueModel.from_v1(v1_model(1), seed=1)
    c = GidiValueModel.from_v1(v1_model(1), seed=2)
    assert torch.equal(a.value_head.weight, b.value_head.weight)
    assert not torch.equal(a.value_head.weight, c.value_head.weight)
    assert not torch.equal(a.value_head.weight, a.tag_head.weight)
    bound = 1 / 16**0.5
    assert a.value_head.weight.abs().max() <= bound and a.value_head.bias.abs().max() <= bound


def test_forward_returns_three_heads():
    model = GidiValueModel.from_v1(v1_model(0), seed=0)
    ids = torch.randint(4, len(VOCAB), (3, 7))
    out = model(ids, torch.ones_like(ids))
    assert isinstance(out, tuple) and len(out) == 3
    assert [tuple(o.shape) for o in out] == [(3, len(TYPES)), (3, 7, 3), (3, 7, 3)]
    assert model.param_count() == v1_model(0).param_count() + 16 * 3 + 3


def test_multitask_loss_is_equal_weight_sum_and_masks_value():
    torch.manual_seed(0)
    type_l, tag_l = torch.randn(2, len(TYPES)), torch.randn(2, 4, 3)
    value_l = torch.randn(2, 4, 3, requires_grad=True)
    types = torch.tensor([0, 3])
    tag_y = torch.tensor([[IGNORE_INDEX, 0, 1, IGNORE_INDEX]] * 2)
    value_y = torch.full((2, 4), IGNORE_INDEX)
    losses = multitask_loss(type_l, tag_l, value_l, types, tag_y, value_y)
    assert losses["value"] == 0.0
    assert torch.isclose(losses["total"], losses["type"] + losses["target"])
    losses["total"].backward()
    assert value_l.grad is None or float(value_l.grad.abs().sum()) == 0.0

    value_y = torch.tensor([[IGNORE_INDEX, 0, 1, IGNORE_INDEX], [IGNORE_INDEX] * 4])
    losses = multitask_loss(type_l, tag_l, value_l, types, tag_y, value_y)
    expected = torch.nn.functional.cross_entropy(value_l[0, 1:3], torch.tensor([0, 1]))
    assert torch.isclose(losses["value"], expected)
    assert torch.isclose(losses["total"], losses["type"] + losses["target"] + losses["value"])
    losses["total"].backward()
    # positions without a label (special/pad positions and the masked record) get no gradient
    assert float(value_l.grad[1].abs().sum()) == 0.0
    assert float(value_l.grad[0, 0].abs().sum()) == 0.0


def test_eval_losses_report_each_component():
    tok = tiny_tokenizer()
    data = prepare_value(
        tok,
        [
            record("cơm tấm 100", "100"),
            record("trả góp kỳ 3 1tr5", "1tr5", status="uncertain"),
        ],
        32,
    )
    model = GidiValueModel.from_v1(v1_model(0), seed=0)
    losses = eval_losses(model, data, torch.device("cpu"))
    assert set(losses) == {"type", "target", "value", "total"}
    assert losses["total"] == pytest.approx(losses["type"] + losses["target"] + losses["value"])


# --------------------------------------------------------------------------- decoding


def test_decode_all_spans_matches_first_span_rule():
    text = "ăn 2 tô phở 70"
    offsets = [(0, 0), (0, 2), (2, 4), (4, 7), (7, 11), (11, 14), (0, 0)]
    tag_ids = [0, 0, 1, 2, 0, 1, 0]
    spans = decode_all_spans(offsets, tag_ids, text)
    assert [(s.start, s.end, s.members) for s in spans] == [(3, 7, (2, 3)), (12, 14, (5,))]
    first = decode_first_span(offsets, tag_ids, text)
    assert first is not None and (first.start, first.end) == (spans[0].start, spans[0].end)
    assert decode_all_spans(offsets, [0] * 7, text) == []
    # an I after O starts a span, a later B closes the previous one
    assert [s.members for s in decode_all_spans(offsets, [0, 2, 2, 1, 1, 0, 0], text)] == [
        (1, 2),
        (3,),
        (4,),
    ]


def test_decode_value_picks_most_confident_span_and_scores_null():
    text = "ăn 2 tô phở 70"
    offsets = [(0, 0), (0, 2), (2, 4), (4, 7), (7, 11), (11, 14), (0, 0)]
    real = [False, True, True, True, True, True, False]
    logits = np.full((7, 3), -4.0)
    logits[:, 0] = 4.0
    logits[2] = [0.0, 1.0, -4.0]  # weak B on "2"
    logits[5] = [-4.0, 6.0, -4.0]  # strong B on "70"
    span, confidence = decode_value(logits, offsets, text, real)
    assert span is not None and text[span.start : span.end] == "70"
    assert confidence > 0.99
    # no span: the weakest P(O) over the real tokens
    none_span, none_conf = decode_value(np.tile([3.0, 0.0, 0.0], (7, 1)), offsets, text, real)
    assert none_span is None
    assert none_conf == pytest.approx(np.exp(3.0) / (np.exp(3.0) + 2.0))


def _tie_logits() -> np.ndarray:
    logits = np.full((7, 3), -4.0)
    logits[:, 0] = 4.0
    logits[2] = logits[5] = [0.0, 3.0, -4.0]
    return logits


def test_decode_value_tie_goes_to_earlier_span():
    text = "ăn 2 tô phở 70"
    offsets = [(0, 0), (0, 2), (2, 4), (4, 7), (7, 11), (11, 14), (0, 0)]
    real = [False, True, True, True, True, True, False]
    span, _ = decode_value(_tie_logits(), offsets, text, real)
    assert span is not None and (span.start, span.end) == (3, 4)


# --------------------------------------------------------------------------- metrics


def pred(type_="expense", target=None, value=None, text=None):
    return {
        "type": type_,
        "target": span(text, target) if target else None,
        "value": span(text, value) if value else None,
    }


def test_surface_pattern_masks_digits_only():
    assert surface_pattern("5 xị") == "n xị"
    assert surface_pattern("1tr5") == "ntrn"
    assert surface_pattern("1.500.000") == "n.n.n"
    assert surface_pattern("50  K") == "n k"


def test_token_indices_and_overlap():
    text = "mượn chú hai 5 xị"
    offsets = [(0, 0), (0, 4), (4, 8), (8, 12), (12, 14), (14, 17), (0, 0)]
    assert token_indices(text, offsets, span(text, "5 xị")) == {4, 5}
    assert token_indices(text, offsets, None) == frozenset()
    result = overlap_metrics([frozenset({4, 5}), frozenset()], [frozenset({5}), frozenset()])
    assert result == {
        "precision": 1.0,
        "recall": 0.5,
        "f1": pytest.approx(2 / 3),
        "tp": 1,
        "n_pred": 1,
        "n_gold": 2,
    }
    assert overlap_metrics([frozenset()], [frozenset()]) is None


def test_evaluate_value_counts_only_complete_records_for_value_metrics():
    texts = ["cơm tấm 100", "mượn chú hai 5 xị", "tiền điện", "ăn 2 tô phở 70", "mua 3 vé 150k"]
    records = [
        record(texts[0], "100", rid="a", source_batch="x"),
        record(texts[1], "5 xị", rid="b", target="chú hai", type_="borrow", source_batch="x"),
        record(texts[2], None, rid="c", source_batch="y"),
        record(texts[3], "70", rid="d", status="uncertain", source_batch="y"),
        record(texts[4], "150k", rid="e", source_batch="y"),
    ]
    preds = [
        pred(text=texts[0], value="100"),  # exact
        pred("borrow", "chú hai", "xị", text=texts[1]),  # value partially right
        pred(text=texts[2], value=None),  # correct null
        pred(text=texts[3], value="2"),  # masked: wrong but excluded
        pred("income", None, "150k", text=texts[4]),  # type wrong
    ]
    data = prepare_value(tiny_tokenizer(), records, 32)
    out = evaluate_value(records, preds, offsets=data.encoded.offsets)
    assert (out["n"], out["n_value_complete"], out["n_value_masked"]) == (5, 4, 1)
    value = out["value"]
    assert value["n"] == 4
    assert value["exact_match"] == pytest.approx(3 / 4)  # a, c, e exact; null==null counts
    assert value["present_accuracy"] == 1.0
    span_counts = {k: value["span"][k] for k in ("tp", "n_gold", "n_pred")}
    assert span_counts == {"tp": 2, "n_gold": 3, "n_pred": 3}
    assert value["outcomes"]["partial_overlap"] == 1 and value["outcomes"]["correct_null"] == 1
    # token overlap: gold tokens {100},{5,xị},{150k}; predicted {100},{xị},{150k}
    assert value["token"]["tp"] == 3 and value["token"]["n_gold"] == 4
    assert value["token"]["precision"] == 1.0
    # full joint over complete records: only a and c have type, target and value all right
    assert out["full_joint"] == {"n": 4, "accuracy": pytest.approx(2 / 4)}
    # type/target metrics use every record
    assert out["overall"]["type"]["n"] == 5
    assert out["overall"]["type"]["accuracy"] == pytest.approx(4 / 5)
    assert out["overall"]["type_target_joint"] == pytest.approx(4 / 5)
    assert out["slices"]["source_batch"]["x"]["n"] == 2
    assert out["slices"]["source_batch"]["y"]["n"] == 2  # c and e: d is masked


def test_evaluate_value_slices_unseen_pattern_length_and_categories():
    texts = ["cơm tấm 100", "mượn chú hai 5 xị", "mua 3 vé 150k", "trả góp kỳ 3 1tr5"]
    records = [
        record(texts[0], "100", rid="a"),
        record(texts[1], "5 xị", rid="b", target="chú hai", type_="borrow"),
        record(texts[2], "150k", rid="c"),
        record(texts[3], "1tr5", rid="d"),
    ]
    train = [
        record("an sang 35k", "35k", rid="t1"),  # pattern "nk"  -> 150k is seen
        record("tiền điện 100", "100", rid="t2"),  # pattern "n"   -> 100 is seen
        record("mua 5 xị", "5 xị", rid="t3", status="uncertain"),  # not complete: ignored
    ]
    values = ["100", "5 xị", "150k", "1"]
    preds = [pred(text=t, value=v) for t, v in zip(texts, values, strict=True)]
    preds[1]["target"] = span(texts[1], "chú hai")
    preds[1]["type"] = "borrow"
    data = prepare_value(tiny_tokenizer(), records, 32)
    cats = {"cơm tấm 100": ["bare_number"], "mượn chú hai 5 xị": ["slang"]}
    out = evaluate_value(
        records,
        preds,
        offsets=data.encoded.offsets,
        train_records=train,
        categories=lambda text: cats.get(text, []),
    )
    pattern = out["slices"]["surface_pattern"]
    assert {k: v["n"] for k, v in pattern.items()} == {"seen": 2, "unseen": 2}
    assert pattern["unseen"]["value"]["exact_match"] == pytest.approx(0.5)  # "5 xị" ok, "1tr5" no
    assert pattern["seen"]["value"]["exact_match"] == 1.0
    length = out["slices"]["amount_length"]
    assert length["multi_token"]["n"] == 1 and length["single_token"]["n"] == 3
    assert length["multi_token"]["full_joint"] == 1.0
    space = out["slices"]["amount_space"]
    assert space["space"]["n"] == 1 and space["no_space"]["n"] == 3
    category = out["slices"]["category"]
    assert category["bare_number"]["n"] == 1 and category["slang"]["n"] == 1
    assert category["none"]["n"] == 2


def test_evaluate_value_rejects_misaligned_inputs():
    with pytest.raises(ValueError):
        evaluate_value([record("cơm tấm 100", "100")], [])


# --------------------------------------------------------------------------- checkpoint/export


def test_checkpoint_roundtrip_and_v1_checkpoint_rejected(tmp_path):
    tok = tiny_tokenizer()
    model = GidiValueModel.from_v1(v1_model(3), seed=3).eval()
    save_value_checkpoint(model, tok, tmp_path / "ckpt", {"seed": 3, "max_length": 32})
    loaded, _, meta = load_value_checkpoint(tmp_path / "ckpt")
    assert meta["value_tags"] == list(VALUE_TAGS) and meta["annotation_version"] == "annotation-v2"
    ids = torch.tensor([[2, 4, 5, 3]])
    mask = torch.ones_like(ids)
    for a, b in zip(model(ids, mask), loaded(ids, mask), strict=True):
        assert torch.equal(a, b)

    from gidi.modeling.checkpoint import save_checkpoint

    save_checkpoint(v1_model(0), tok, tmp_path / "v1", {})
    with pytest.raises(ValueError, match="no value head"):
        load_value_checkpoint(tmp_path / "v1")


def test_onnx_export_adds_value_logits_and_matches_torch(tmp_path):
    tok = tiny_tokenizer()
    model = GidiValueModel.from_v1(v1_model(0), seed=0).eval()
    notes = ["cơm tấm 100", "mượn chú hai 5 xị", "mua 3 vé 150k"]
    fp32 = tmp_path / "model.onnx"
    report = export_value_onnx(model, tok, fp32, 32, 17, notes)
    assert report["parity"]["passed"]
    interface = onnx_interface(fp32)
    assert tuple(interface["outputs"]) == VALUE_OUTPUT_NAMES
    assert VALUE_OUTPUT_NAMES[:2] == ("type_logits", "tag_logits")
    assert interface["inputs"] == ["input_ids", "attention_mask"]
    assert len(interface["output_shapes"]["value_logits"]) == 3
    assert interface["output_shapes"]["value_logits"][2] == 3

    # decoded predictions of the PyTorch model and its FP32 export agree
    records = [record(t, None) for t in notes]
    data = prepare_value(tok, records, 32)
    session = make_session(fp32, threads=1)
    ref = decode_predictions(data, torch_logits(model, data, torch.device("cpu")))
    got = decode_predictions(data, onnx_logits(session, data))
    assert [p["value"] for p in ref] == [p["value"] for p in got]
    assert [p["type"] for p in ref] == [p["type"] for p in got]

    int8 = tmp_path / "model.int8.onnx"
    q = quantize_value_int8(fp32, int8, tok, notes, 32)
    assert q["int8_size_mb"] < q["fp32_size_mb"]
    assert set(q["agreement"]) >= {"type_agreement", "tag_agreement", "value_agreement"}
    assert tuple(onnx_interface(int8)["outputs"]) == VALUE_OUTPUT_NAMES


def test_compare_initializers_finds_v1_weights_inside_the_value_graph_but_not_the_reverse(tmp_path):
    from gidi.export.onnx_export import export_onnx
    from gidi.export.onnx_export_value import compare_initializers

    tok = tiny_tokenizer()
    base = v1_model(0).eval()
    notes = ["cơm tấm 100", "mượn chú hai 5 xị"]
    v1_onnx, value_onnx = tmp_path / "v1.onnx", tmp_path / "value.onnx"
    export_onnx(base, tok, v1_onnx, 32, 17, notes)
    export_value_onnx(GidiValueModel.from_v1(base, seed=0).eval(), tok, value_onnx, 32, 17, notes)

    inside = compare_initializers(v1_onnx, value_onnx)
    assert inside["reference_tensors"] > 0 and inside["all_reference_tensors_present"]
    reverse = compare_initializers(value_onnx, v1_onnx)  # the value head's weights are not in v1
    assert not reverse["all_reference_tensors_present"] and reverse["unmatched"] > 0


def test_evaluate_records_end_to_end_on_tiny_model():
    tok = tiny_tokenizer()
    model = GidiValueModel.from_v1(v1_model(0), seed=0).eval()
    records = [record("cơm tấm 100", "100", rid="a"), record("tiền điện", None, rid="b")]
    metrics, rows, preds = evaluate_records(
        records,
        lambda d: torch_logits(model, d, torch.device("cpu")),
        tok,
        32,
        train_records=records,
        categories=lambda text: ["bare_number"],
    )
    assert len(rows) == len(preds) == 2
    assert {"gold_value", "pred_value", "value_ok", "full_joint_ok", "categories"} <= set(rows[0])
    assert metrics["alignment"]["value_boundary_mismatch"] == 0
    assert metrics["n_value_complete"] == 2
