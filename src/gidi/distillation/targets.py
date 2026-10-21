"""Teacher target cache: the frozen teacher's raw logits over the distillation training data.

``build_targets`` encodes the records with the teacher's saved tokenizer (padded to the fixed
``max_length`` width), runs the teacher once in eval mode (dropout off), and keeps everything the
student needs, so no student run ever touches the teacher. Layout of the cache directory::

    targets.safetensors   input_ids, attention_mask, tag_labels [N, W] int64;
                          type_ids [N] int64 (gold); teacher_type_logits [N, 8] float32;
                          teacher_tag_logits [N, W, 3] float32 (raw: not argmax, not
                          temperature-scaled, full padded width; padded positions are
                          meaningless and are never read because their tag label is -100)
    index.jsonl           row, id, text, type, target, offsets (W (start, end) pairs, (0, 0) off
                          real tokens)
    manifest.json         provenance (teacher + data hashes), shapes, dtypes, TYPES/TAGS order,
                          teacher-vs-gold agreement, hashes of the two files above

Every record is passed through ``assert_distillation_trainable`` first.

Records may carry the optional ``soft`` field ``{"type_probs": {type: p}, "target_trusted": bool}``
(annotation-v4 Clef soft labels). It is copied verbatim into the matching ``index.jsonl`` row (rows
of records without it are unchanged, so existing caches stay valid) and the manifest gains a
``soft`` summary; ``soft_tensors`` turns the rows into the tensors the student loss reads.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file
from transformers import PreTrainedTokenizerBase

from gidi.corpus.jsonl import read_jsonl, write_jsonl
from gidi.device import resolve_device
from gidi.distillation.guard import assert_distillation_trainable
from gidi.modeling.checkpoint import WEIGHTS_FILE, load_checkpoint
from gidi.modeling.preprocessing import IGNORE_INDEX, TAGS, TYPES
from gidi.training.train import Prepared, _write_json, load_split, prepare

TENSORS_FILE = "targets.safetensors"
INDEX_FILE = "index.jsonl"
MANIFEST_FILE = "manifest.json"
TOKENIZER_FILE = "tokenizer.json"
LOGIT_ATOL = 1e-4  # --check tolerance for the float logits (MPS is not bit-reproducible)
INT_TENSORS = ("input_ids", "attention_mask", "tag_labels", "type_ids")
FLOAT_TENSORS = ("teacher_type_logits", "teacher_tag_logits")


@dataclass
class TeacherTargets:
    input_ids: torch.Tensor  # [N, W] int64
    attention_mask: torch.Tensor  # [N, W] int64
    tag_labels: torch.Tensor  # [N, W] int64, -100 off real tokens
    type_ids: torch.Tensor  # [N] int64, gold
    teacher_type_logits: torch.Tensor  # [N, 8] float32
    teacher_tag_logits: torch.Tensor  # [N, W, 3] float32
    index: list[dict[str, Any]]
    manifest: dict[str, Any]

    def __len__(self) -> int:
        return int(self.input_ids.shape[0])

    def tensors(self) -> dict[str, torch.Tensor]:
        return {k: getattr(self, k).contiguous() for k in (*INT_TENSORS, *FLOAT_TENSORS)}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def pad_to_width(x: torch.Tensor, width: int, value: int) -> torch.Tensor:
    """Right-pad the second dimension of ``x`` to ``width`` with ``value``."""
    n, length = x.shape[:2]
    if length > width:
        raise ValueError(f"cannot pad width {length} down to {width}")
    out = x.new_full((n, width, *x.shape[2:]), value)
    out[:, :length] = x
    return out


def padded_encoding(
    data: Prepared, width: int, pad_id: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[list[list[int]]]]:
    """``input_ids, attention_mask, tag_labels, offsets`` of ``data`` padded to ``width``."""
    enc = data.encoded
    offsets = [[[s, e] for s, e in row] + [[0, 0]] * (width - len(row)) for row in enc.offsets]
    return (
        pad_to_width(enc.input_ids, width, pad_id),
        pad_to_width(enc.attention_mask, width, 0),
        pad_to_width(enc.tag_labels, width, IGNORE_INDEX),
        offsets,
    )


def validate_soft(record: dict[str, Any]) -> dict[str, Any] | None:
    """The record's ``soft`` field if it has one (validated), else ``None``."""
    soft = record.get("soft")
    if soft is None:
        return None
    rid = record.get("id")
    probs, trusted = soft.get("type_probs"), soft.get("target_trusted")
    if not isinstance(probs, dict) or not isinstance(trusted, bool):
        raise ValueError(f"record {rid!r}: soft needs type_probs (dict) and target_trusted (bool)")
    if unknown := sorted(set(probs) - set(TYPES)):
        raise ValueError(f"record {rid!r}: soft.type_probs has unknown types {unknown}")
    if (
        any(not isinstance(p, int | float) or p < 0 for p in probs.values())
        or sum(probs.values()) <= 0
    ):
        raise ValueError(
            f"record {rid!r}: soft.type_probs must be non-negative with a positive sum"
        )
    return soft


def index_rows(
    records: list[dict[str, Any]], offsets: list[list[list[int]]]
) -> list[dict[str, Any]]:
    rows = []
    for i, (r, offs) in enumerate(zip(records, offsets, strict=True)):
        row = {
            "row": i,
            "id": r["id"],
            "text": r["text"],
            "type": r["type"],
            "target": r["target"],
            "offsets": offs,
        }
        if (soft := validate_soft(r)) is not None:
            row["soft"] = soft
        rows.append(row)
    return rows


def soft_tensors(
    index: list[dict[str, Any]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """``(type_probs [N, C] float32, soft_rows [N] bool, span_rows [N] bool)``, or ``None``.

    ``None`` when no index row has ``soft`` (the training path is then exactly the hard one).
    ``span_rows`` is False for rows whose ``soft.target_trusted`` is false. ``type_probs`` is
    zero on hard rows (they are never read).
    """
    if not any("soft" in row for row in index):
        return None
    probs = torch.zeros(len(index), len(TYPES), dtype=torch.float32)
    soft_rows = torch.zeros(len(index), dtype=torch.bool)
    span_rows = torch.ones(len(index), dtype=torch.bool)
    for i, row in enumerate(index):
        if (soft := validate_soft(row)) is None:
            continue
        soft_rows[i] = True
        span_rows[i] = soft["target_trusted"]
        for name, p in soft["type_probs"].items():
            probs[i, TYPES.index(name)] = float(p)
    return probs, soft_rows, span_rows


def soft_summary(index: list[dict[str, Any]]) -> dict[str, int] | None:
    """Counts of soft rows for the manifest, or ``None`` without any."""
    soft = [row["soft"] for row in index if "soft" in row]
    if not soft:
        return None
    return {"n_soft": len(soft), "n_target_untrusted": sum(not s["target_trusted"] for s in soft)}


@torch.no_grad()
def build_targets(
    teacher_dir: str | Path,
    data_path: str | Path,
    *,
    limit: int | None = None,
    max_length: int = 32,
    device: str = "cpu",
    batch_size: int = 64,
    root: Path = Path("."),
) -> TeacherTargets:
    """Run the teacher over the first ``limit`` (default all) records of ``data_path``."""
    teacher_dir, data_path = Path(teacher_dir), Path(data_path)
    records = load_split(data_path)
    if limit is not None:
        records = records[:limit]
    assert_distillation_trainable(records, root)

    model, tokenizer, meta = load_checkpoint(teacher_dir)
    if tuple(meta["types"]) != TYPES or tuple(meta["tags"]) != TAGS:
        raise ValueError(f"{teacher_dir}: checkpoint class order differs from TYPES/TAGS")
    dev = resolve_device(device)
    model.to(dev).eval()

    data = prepare(tokenizer, records, max_length)
    pad_id = int(tokenizer.pad_token_id)
    input_ids, attention_mask, tag_labels, offsets = padded_encoding(data, max_length, pad_id)
    type_chunks, tag_chunks = [], []
    for i in range(0, len(records), batch_size):
        type_logits, tag_logits = model(
            input_ids[i : i + batch_size].to(dev), attention_mask[i : i + batch_size].to(dev)
        )
        type_chunks.append(type_logits.float().cpu())
        tag_chunks.append(tag_logits.float().cpu())
    type_logits = torch.cat(type_chunks)
    tag_logits = torch.cat(tag_chunks)

    valid = tag_labels != IGNORE_INDEX
    type_accuracy = float((type_logits.argmax(-1) == data.type_ids).float().mean())
    tag_accuracy = float((tag_logits.argmax(-1)[valid] == tag_labels[valid]).float().mean())
    dist_manifest = data_path.parent / MANIFEST_FILE
    targets = TeacherTargets(
        input_ids=input_ids,
        attention_mask=attention_mask,
        tag_labels=tag_labels,
        type_ids=data.type_ids,
        teacher_type_logits=type_logits,
        teacher_tag_logits=tag_logits,
        index=index_rows(records, offsets),
        manifest={},
    )
    targets.manifest = {
        "teacher": {
            "checkpoint": str(teacher_dir),
            "model_sha256": sha256_file(teacher_dir / WEIGHTS_FILE),
            "tokenizer_sha256": sha256_file(teacher_dir / TOKENIZER_FILE),
            "checkpoint_meta": meta,
            "device": str(dev),
        },
        "data": {
            "path": str(data_path),
            "sha256": sha256_file(data_path),
            "limit": limit,
            "manifest_sha256": sha256_file(dist_manifest) if dist_manifest.exists() else None,
        },
        "n": len(records),
        "max_length": max_length,
        "types": list(TYPES),
        "tags": list(TAGS),
        "shapes": {k: list(t.shape) for k, t in targets.tensors().items()},
        "dtypes": {k: str(t.dtype).removeprefix("torch.") for k, t in targets.tensors().items()},
        "teacher_vs_gold": {
            "type_accuracy": type_accuracy,
            "tag_token_accuracy": tag_accuracy,
            "n_valid_tag_positions": int(valid.sum()),
        },
    }
    if (soft := soft_summary(targets.index)) is not None:
        targets.manifest["soft"] = soft
    return targets


def write_targets(targets: TeacherTargets, out_dir: str | Path, *, overwrite: bool = False) -> None:
    """Write the three cache files; ``manifest.json`` gets the hashes of the other two."""
    out = Path(out_dir)
    paths = [out / name for name in (TENSORS_FILE, INDEX_FILE, MANIFEST_FILE)]
    if not overwrite and (clash := [p for p in paths if p.exists()]):
        raise FileExistsError(f"refusing to overwrite existing target cache files: {clash}")
    out.mkdir(parents=True, exist_ok=True)
    save_file(targets.tensors(), str(out / TENSORS_FILE))
    write_jsonl(out / INDEX_FILE, targets.index, overwrite=True)
    targets.manifest["files"] = {
        TENSORS_FILE: sha256_file(out / TENSORS_FILE),
        INDEX_FILE: sha256_file(out / INDEX_FILE),
    }
    _write_json(out / MANIFEST_FILE, targets.manifest, overwrite=True)


def load_targets(cache_dir: str | Path) -> TeacherTargets:
    """Load a cache, failing loudly if a file does not match the hash recorded in its manifest."""
    cache = Path(cache_dir)
    manifest = json.loads((cache / MANIFEST_FILE).read_text(encoding="utf-8"))
    for name, expected in manifest["files"].items():
        actual = sha256_file(cache / name)
        if actual != expected:
            raise ValueError(f"{cache / name}: sha256 {actual} != manifest {expected}")
    tensors = load_file(str(cache / TENSORS_FILE))
    for key, shape in manifest["shapes"].items():
        if list(tensors[key].shape) != shape:
            raise ValueError(f"{cache}: tensor {key} has shape {list(tensors[key].shape)}")
    index = read_jsonl(cache / INDEX_FILE)
    if len(index) != manifest["n"]:
        raise ValueError(f"{cache}: index has {len(index)} rows, manifest says {manifest['n']}")
    return TeacherTargets(index=index, manifest=manifest, **tensors)


def check_targets(cache_dir: str | Path, fresh: TeacherTargets) -> list[str]:
    """Problems found comparing the stored cache with a regenerated one (empty means equal).

    Integer tensors, the index and the manifest provenance must match exactly; the float logits
    within ``LOGIT_ATOL``.
    """
    stored = load_targets(cache_dir)  # also checks the stored files against their own hashes
    problems: list[str] = []
    for key in INT_TENSORS:
        if not torch.equal(getattr(stored, key), getattr(fresh, key)):
            problems.append(f"{key} differs")
    for key in FLOAT_TENSORS:
        a, b = getattr(stored, key), getattr(fresh, key)
        if a.shape != b.shape:
            problems.append(f"{key} shape {tuple(a.shape)} != {tuple(b.shape)}")
        elif (diff := float((a - b).abs().max())) > LOGIT_ATOL:
            problems.append(f"{key} max abs diff {diff:.3g} > {LOGIT_ATOL}")
    if stored.index != fresh.index:
        problems.append("index.jsonl rows differ")
    for key, value in fresh.manifest.items():
        if key == "teacher":
            value = {k: v for k, v in value.items() if k != "device"}
            old = {k: v for k, v in stored.manifest[key].items() if k != "device"}
        else:
            old = stored.manifest.get(key)
        if old != value:
            problems.append(f"manifest[{key!r}] differs: stored {old!r}, regenerated {value!r}")
    return problems


def verify_encoding(
    targets: TeacherTargets,
    records: list[dict[str, Any]],
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
) -> Prepared:
    """Re-encode ``records`` with the student tokenizer and require the cached encoding.

    Raises ``ValueError`` unless ids, texts, gold types/targets, ``input_ids``,
    ``attention_mask``, ``tag_labels`` and ``type_ids`` all match the cache exactly. Returns the
    re-encoded split (its ``offsets`` etc. are only needed for bookkeeping).
    """
    if len(records) != len(targets):
        raise ValueError(f"{len(records)} records but the target cache has {len(targets)} rows")
    for row, rec in zip(targets.index, records, strict=True):
        for key in ("id", "text", "type", "target"):
            if row[key] != rec[key]:
                raise ValueError(f"cache row {row['row']} {key} {row[key]!r} != {rec[key]!r}")
        if row.get("soft") != rec.get("soft"):
            raise ValueError(
                f"cache row {row['row']} soft {row.get('soft')!r} != {rec.get('soft')!r}"
            )
    if targets.manifest["max_length"] != max_length:
        raise ValueError(f"cache max_length {targets.manifest['max_length']} != {max_length}")
    data = prepare(tokenizer, records, max_length)
    ids, mask, tags, offsets = padded_encoding(data, max_length, int(tokenizer.pad_token_id))
    mismatched = [
        name
        for name, got, want in (
            ("input_ids", ids, targets.input_ids),
            ("attention_mask", mask, targets.attention_mask),
            ("tag_labels", tags, targets.tag_labels),
            ("type_ids", data.type_ids, targets.type_ids),
        )
        if not torch.equal(got, want)
    ]
    if offsets != [row["offsets"] for row in targets.index]:
        mismatched.append("offsets")
    if mismatched:
        raise ValueError(
            "re-encoding with the student tokenizer does not reproduce the cache: "
            + ", ".join(mismatched)
        )
    return data
