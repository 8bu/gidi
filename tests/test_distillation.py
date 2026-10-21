import json
import math
from pathlib import Path

import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from torch.nn import functional as F
from transformers import PreTrainedTokenizerFast, RobertaConfig

from gidi.distillation.guard import assert_distillation_trainable
from gidi.distillation.loss import KDConfig, kd_loss
from gidi.distillation.student import (
    build_student,
    copy_pretrained_encoder,
    layer_map,
    student_config,
)
from gidi.distillation.targets import (
    build_targets,
    check_targets,
    load_targets,
    soft_tensors,
    verify_encoding,
    write_targets,
)
from gidi.modeling.checkpoint import save_checkpoint
from gidi.modeling.model import GidiMultiTaskModel
from gidi.modeling.preprocessing import IGNORE_INDEX, TYPES
from gidi.training.train import load_split

REPO = Path(__file__).resolve().parents[1]


def _batch(seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    type_logits = torch.randn(4, 8, generator=g)
    tag_logits = torch.randn(4, 6, 3, generator=g)
    type_ids = torch.tensor([0, 3, 7, 2])
    tags = torch.tensor(
        [
            [IGNORE_INDEX, 0, 1, 2, 0, IGNORE_INDEX],
            [IGNORE_INDEX, 0, 0, IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX],
            [IGNORE_INDEX, 1, 2, 2, 0, IGNORE_INDEX],
            [IGNORE_INDEX, 0, 1, 0, 0, 0],
        ]
    )
    teacher_type = torch.randn(4, 8, generator=g)
    teacher_tag = torch.randn(4, 6, 3, generator=g)
    return type_logits, tag_logits, type_ids, tags, teacher_type, teacher_tag


def test_supervised_loss_is_the_train_run_hard_loss_and_needs_no_teacher():
    type_logits, tag_logits, type_ids, tags, _, _ = _batch()
    out = kd_loss(type_logits, tag_logits, type_ids, tags, KDConfig(alpha_hard=1.0))
    expected = F.cross_entropy(type_logits, type_ids) + F.cross_entropy(
        tag_logits.reshape(-1, 3), tags.reshape(-1), ignore_index=IGNORE_INDEX
    )
    assert torch.allclose(out["total"], expected)
    assert out["type_kl"] == 0 and out["tag_kl"] == 0


def test_kl_vanishes_when_student_matches_teacher():
    _, _, type_ids, tags, teacher_type, teacher_tag = _batch()
    cfg = KDConfig(temperature=3.0, alpha_hard=0.25)
    out = kd_loss(teacher_type, teacher_tag, type_ids, tags, cfg, teacher_type, teacher_tag)
    assert out["type_kl"].abs() < 1e-6 and out["tag_kl"].abs() < 1e-6
    hard = out["type_hard"] + out["tag_hard"]
    assert torch.allclose(out["total"], cfg.alpha_hard * hard, atol=1e-5)


def test_distilled_loss_requires_teacher_logits():
    type_logits, tag_logits, type_ids, tags, _, _ = _batch()
    with pytest.raises(ValueError, match="teacher"):
        kd_loss(type_logits, tag_logits, type_ids, tags, KDConfig(alpha_hard=0.5))


def test_ignored_positions_do_not_affect_loss_or_gradient():
    type_logits, tag_logits, type_ids, tags, teacher_type, teacher_tag = _batch()
    cfg = KDConfig()
    tag_logits = tag_logits.clone().requires_grad_()
    base = kd_loss(type_logits, tag_logits, type_ids, tags, cfg, teacher_type, teacher_tag)
    base["total"].backward()
    ignored = (tags == IGNORE_INDEX).unsqueeze(-1)
    assert torch.all(tag_logits.grad.masked_select(ignored.expand_as(tag_logits)) == 0)

    noisy_student = torch.where(ignored, tag_logits.detach() + 100.0, tag_logits.detach())
    noisy_teacher = torch.where(ignored, -teacher_tag * 7, teacher_tag)
    other = kd_loss(type_logits, noisy_student, type_ids, tags, cfg, teacher_type, noisy_teacher)
    assert torch.allclose(base["total"], other["total"])
    assert torch.allclose(base["tag_kl"], other["tag_kl"])


@pytest.mark.parametrize("temperature", [1.0, 4.0])
def test_soft_term_is_t_squared_times_the_temperature_kl(temperature):
    # Two classes: teacher logits (2, 0), uniform student. KL = ln2 + p ln p + (1-p) ln(1-p)
    # with p = sigmoid(2 / T). Alpha 0 keeps only the soft terms; one real tag position.
    p = 1.0 / (1.0 + math.exp(-2.0 / temperature))
    kl = math.log(2) + p * math.log(p) + (1 - p) * math.log(1 - p)
    zeros = torch.zeros(1, 2)
    teacher = torch.tensor([[2.0, 0.0]])
    cfg = KDConfig(temperature=temperature, alpha_hard=0.0, type_weight=1.0, span_weight=2.0)
    out = kd_loss(
        zeros,
        zeros.unsqueeze(1),
        torch.tensor([0]),
        torch.tensor([[0]]),
        cfg,
        teacher,
        teacher.unsqueeze(1),
    )
    assert out["type_kl"].item() == pytest.approx(kl, rel=1e-5)
    assert out["tag_kl"].item() == pytest.approx(kl, rel=1e-5)
    assert out["type_loss"].item() == pytest.approx(temperature**2 * kl, rel=1e-5)
    assert out["total"].item() == pytest.approx((1 + 2) * temperature**2 * kl, rel=1e-5)


@pytest.mark.parametrize("alpha", [1.0, 0.5])
def test_hard_only_batch_with_soft_arguments_equals_the_old_loss_bit_for_bit(alpha):
    type_logits, tag_logits, type_ids, tags, teacher_type, teacher_tag = _batch()
    cfg = KDConfig(alpha_hard=alpha)
    teacher = (teacher_type, teacher_tag) if cfg.uses_teacher else (None, None)
    old = kd_loss(type_logits, tag_logits, type_ids, tags, cfg, *teacher)
    assert torch.equal(old["type_hard"], F.cross_entropy(type_logits, type_ids))
    off = torch.zeros(4, dtype=torch.bool)
    new = kd_loss(
        type_logits,
        tag_logits,
        type_ids,
        tags,
        cfg,
        *teacher,
        soft_type_probs=torch.zeros(4, len(TYPES)),
        soft_rows=off,
        span_rows=~off,
    )
    for key in old:
        assert torch.equal(old[key], new[key]), key


def test_soft_type_loss_is_cross_entropy_against_the_renormalised_distribution():
    # Row 0 is soft with unnormalised probs (1, 3, 0, ...) -> (0.25, 0.75); row 1 is hard (class 2).
    type_logits = torch.zeros(2, 8)
    type_logits[0, :2] = torch.tensor([1.0, 0.0])
    type_logits[1, 2] = 3.0
    probs = torch.zeros(2, 8)
    probs[0, :2] = torch.tensor([1.0, 3.0])
    soft_rows = torch.tensor([True, False])
    log_z0 = math.log(math.exp(1.0) + 1.0 + 6.0)
    soft_ce = -(0.25 * (1.0 - log_z0) + 0.75 * (0.0 - log_z0))
    hard_ce = math.log(math.exp(3.0) + 7.0) - 3.0
    tags = torch.full((2, 1), IGNORE_INDEX)
    out = kd_loss(
        type_logits,
        torch.zeros(2, 1, 3),
        torch.tensor([0, 2]),
        tags,
        KDConfig(alpha_hard=1.0, type_weight=2.0),
        soft_type_probs=probs,
        soft_rows=soft_rows,
    )
    assert out["type_hard"].item() == pytest.approx((soft_ce + hard_ce) / 2, rel=1e-5)
    assert out["total"].item() == pytest.approx(2.0 * (soft_ce + hard_ce) / 2, rel=1e-5)
    # A one-hot soft row is the hard loss.
    onehot = torch.zeros(2, 8)
    onehot[:, 0] = 1.0
    same = kd_loss(
        type_logits,
        torch.zeros(2, 1, 3),
        torch.tensor([0, 0]),
        tags,
        KDConfig(alpha_hard=1.0),
        soft_type_probs=onehot,
        soft_rows=torch.tensor([True, True]),
    )
    plain = kd_loss(
        type_logits, torch.zeros(2, 1, 3), torch.tensor([0, 0]), tags, KDConfig(alpha_hard=1.0)
    )
    assert same["type_hard"].item() == pytest.approx(plain["type_hard"].item(), rel=1e-6)


def test_untrusted_target_rows_are_masked_from_the_span_loss():
    type_logits, tag_logits, type_ids, tags, _, _ = _batch()
    tag_logits = tag_logits.clone().requires_grad_()
    span_rows = torch.tensor([True, False, True, True])
    cfg = KDConfig(alpha_hard=1.0)
    masked = kd_loss(type_logits, tag_logits, type_ids, tags, cfg, span_rows=span_rows)
    masked["total"].backward()
    assert torch.all(tag_logits.grad[1] == 0) and torch.any(tag_logits.grad[0] != 0)
    kept = [0, 2, 3]
    expected = F.cross_entropy(
        tag_logits.detach()[kept].reshape(-1, 3),
        tags[kept].reshape(-1),
        ignore_index=IGNORE_INDEX,
    )
    assert masked["tag_hard"].item() == pytest.approx(expected.item(), rel=1e-6)
    # Masking every row leaves a clean zero span loss, not NaN.
    none = kd_loss(type_logits, tag_logits, type_ids, tags, cfg, span_rows=torch.zeros(4).bool())
    assert none["tag_hard"].item() == 0.0


def _base_config() -> RobertaConfig:
    return RobertaConfig(
        vocab_size=20481, pad_token_id=1, bos_token_id=0, eos_token_id=2, hidden_size=768
    )


def test_student_architecture_and_seeded_init():
    cfg = student_config(base_config=_base_config())
    assert (cfg.num_hidden_layers, cfg.hidden_size, cfg.num_attention_heads) == (4, 256, 4)
    assert (cfg.vocab_size, cfg.pad_token_id, cfg.max_position_embeddings) == (20481, 1, 34)

    a = build_student(7, base_config=_base_config())
    b = build_student(7, base_config=_base_config())
    c = build_student(8, base_config=_base_config())
    assert 8_000_000 < a.param_count() < 9_000_000
    same = zip(a.state_dict().values(), b.state_dict().values(), strict=True)
    assert all(torch.equal(x, y) for x, y in same)
    assert not torch.equal(a.type_head.weight, c.type_head.weight)


def test_4x768_student_keeps_bamibert_layer_and_position_shapes():
    base = _base_config()
    cfg = student_config(student="student-4x768", base_config=base)
    assert (cfg.num_hidden_layers, cfg.hidden_size, cfg.num_attention_heads) == (4, 768, 12)
    assert cfg.intermediate_size == base.intermediate_size == 3072
    assert cfg.max_position_embeddings == base.max_position_embeddings


def test_layer_map_takes_evenly_spaced_blocks_ending_at_the_top():
    assert layer_map(12, 4) == [2, 5, 8, 11]
    assert layer_map(12, 12) == list(range(12))
    with pytest.raises(ValueError):
        layer_map(4, 12)


def _tiny_encoder(layers: int, hidden: int = 16):
    cfg = RobertaConfig(
        vocab_size=40,
        hidden_size=hidden,
        num_hidden_layers=layers,
        num_attention_heads=2,
        intermediate_size=32,
        max_position_embeddings=20,
        type_vocab_size=1,
    )
    return GidiMultiTaskModel.from_config(cfg).encoder


def test_pretrained_copy_is_exact_and_covers_every_student_tensor():
    torch.manual_seed(0)
    source, student = _tiny_encoder(12), _tiny_encoder(4)
    report = copy_pretrained_encoder(student, source)
    assert report["n_tensors"] == len(student.state_dict())
    for k, src_layer in enumerate([2, 5, 8, 11]):
        pairs = zip(
            student.encoder.layer[k].state_dict().values(),
            source.encoder.layer[src_layer].state_dict().values(),
            strict=True,
        )
        assert all(torch.equal(a, b) for a, b in pairs)
    for name in ("word_embeddings", "position_embeddings", "LayerNorm"):
        a = getattr(student.embeddings, name).weight
        assert torch.equal(a, getattr(source.embeddings, name).weight)


def test_pretrained_copy_refuses_shape_mismatch():
    with pytest.raises(ValueError, match="shape mismatch"):
        copy_pretrained_encoder(_tiny_encoder(4, hidden=8), _tiny_encoder(12, hidden=16))


def test_guard_refuses_probe_and_targeted_03_records():
    ok = {"id": "baseline-01-1e11850c97d6", "text": "x", "source_batch": "baseline-01"}
    assert_distillation_trainable([ok], REPO)
    probe = {"id": "probe-v1-0123456789ab", "text": "x"}
    by_prefix = {"id": "targeted-annotation-v1-03-abc", "text": "x"}
    by_batch = {
        "id": "other-1",
        "text": "x",
        "provenance": {"source_batch": "targeted-annotation-v1-03"},
    }
    for bad in (probe, by_prefix, by_batch):
        with pytest.raises(ValueError, match="not allowed"):
            assert_distillation_trainable([ok, bad], REPO)


# --- target cache, end to end on a tiny random teacher ---------------------------------------

WORDS = ["ship", "baemin", "67k", "an", "com", "grab", "lan", "vay", "tra", "5tr"]
RECORDS = [
    {"id": "t-1", "text": "ship baemin 67k", "type": "expense", "target": ("baemin", 5, 11)},
    {"id": "t-2", "text": "vay lan 5tr", "type": "borrow", "target": ("lan", 4, 7)},
    {"id": "t-3", "text": "an com grab", "type": "expense", "target": ("grab", 7, 11)},
]


def _tokenizer(words: list[str]) -> PreTrainedTokenizerFast:
    vocab = {"<pad>": 0, "<unk>": 1, **{w: i + 2 for i, w in enumerate(words)}}
    tok = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    return PreTrainedTokenizerFast(tokenizer_object=tok, pad_token="<pad>", unk_token="<unk>")


def _write_records(path: Path) -> list[dict]:
    records = [
        {**r, "target": dict(zip(("text", "start", "end"), r["target"], strict=True))}
        for r in RECORDS
    ]
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    return records


@pytest.fixture
def teacher_dir(tmp_path):
    config = RobertaConfig(
        vocab_size=len(WORDS) + 2,
        pad_token_id=0,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
        max_position_embeddings=40,
    )
    torch.manual_seed(0)
    model = GidiMultiTaskModel.from_config(config, encoder_name="tiny")
    return save_checkpoint(model, _tokenizer(WORDS), tmp_path / "teacher", {})


def test_target_cache_roundtrip_check_and_tamper_detection(tmp_path, teacher_dir):
    data = tmp_path / "train.jsonl"
    records = _write_records(data)
    targets = build_targets(teacher_dir, data, max_length=12, root=REPO)
    assert targets.teacher_tag_logits.shape == (3, 12, 3)
    assert targets.teacher_type_logits.shape == (3, len(TYPES))
    # Raw logits, not argmax, and gold hard labels are kept alongside.
    assert targets.teacher_tag_logits.dtype == torch.float32
    assert targets.type_ids.tolist() == [TYPES.index(r["type"]) for r in records]
    assert (targets.tag_labels[:, -1] == IGNORE_INDEX).all()

    cache = tmp_path / "cache"
    write_targets(targets, cache)
    with pytest.raises(FileExistsError):
        write_targets(targets, cache)
    assert check_targets(cache, build_targets(teacher_dir, data, max_length=12, root=REPO)) == []

    loaded = load_targets(cache)
    tokenizer = _tokenizer(WORDS)
    verify_encoding(loaded, load_split(data), tokenizer, 12)
    with pytest.raises(ValueError, match="does not reproduce"):
        verify_encoding(loaded, load_split(data), _tokenizer(WORDS[::-1]), 12)

    (cache / "index.jsonl").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="sha256"):
        load_targets(cache)


def test_build_targets_refuses_forbidden_records(tmp_path, teacher_dir):
    data = tmp_path / "train.jsonl"
    records = _write_records(data)
    records[1]["id"] = "probe-v1-0123456789ab"
    data.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="probe"):
        build_targets(teacher_dir, data, max_length=12, root=REPO)


def test_soft_field_is_carried_into_the_cache_and_hard_caches_stay_unchanged(tmp_path, teacher_dir):
    data = tmp_path / "train.jsonl"
    records = _write_records(data)
    plain = build_targets(teacher_dir, data, max_length=12, root=REPO)
    assert all("soft" not in row for row in plain.index) and "soft" not in plain.manifest
    assert soft_tensors(plain.index) is None

    soft = {"type_probs": {"expense": 0.6, "lend": 0.4}, "target_trusted": False}
    records[1]["soft"] = soft
    data.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    targets = build_targets(teacher_dir, data, max_length=12, root=REPO)
    assert targets.index[1]["soft"] == soft and "soft" not in targets.index[0]
    assert targets.manifest["soft"] == {"n_soft": 1, "n_target_untrusted": 1}
    # Hard fields are identical to the cache without soft.
    assert torch.equal(targets.input_ids, plain.input_ids)
    assert torch.equal(targets.type_ids, plain.type_ids)

    cache = tmp_path / "cache"
    write_targets(targets, cache)
    loaded = load_targets(cache)
    verify_encoding(loaded, load_split(data), _tokenizer(WORDS), 12)
    probs, soft_rows, span_rows = soft_tensors(loaded.index)
    assert soft_rows.tolist() == [False, True, False] and span_rows.tolist() == [True, False, True]
    assert probs[1, TYPES.index("expense")].item() == pytest.approx(0.6)
    assert probs[1, TYPES.index("lend")].item() == pytest.approx(0.4)
    assert probs[0].sum().item() == 0.0
    # A train file whose soft field differs from the cache is refused.
    other = [dict(r) for r in load_split(data)]
    other[1]["soft"] = {**soft, "target_trusted": True}
    with pytest.raises(ValueError, match="soft"):
        verify_encoding(loaded, other, _tokenizer(WORDS), 12)


def test_malformed_soft_field_is_refused(tmp_path, teacher_dir):
    data = tmp_path / "train.jsonl"
    records = _write_records(data)
    for bad in (
        {"type_probs": {"nonsense": 1.0}, "target_trusted": True},
        {"type_probs": {"expense": 0.0}, "target_trusted": True},
        {"type_probs": {"expense": 1.0}, "target_trusted": "yes"},
    ):
        records[0]["soft"] = bad
        data.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
        with pytest.raises(ValueError, match="soft"):
            build_targets(teacher_dir, data, max_length=12, root=REPO)
