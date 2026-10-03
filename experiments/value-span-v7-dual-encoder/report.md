# value-span-v7-dual-encoder: report

**Verdict: A. v1 type/target invariance passes and all frozen value-quality criteria pass, on the
mean and on seed 1.** Stopped as the protocol requires: no export, no playground change, no
further experiment.

The protocol `experiments/value-span-v7-dual-encoder/protocol.json` (0444, sha256
`7792e088…f166`) was written before training. The following are unchanged:

- `gidi-finance-v1`;
- value-span-v1, v2-direction-repair and v3–v6, including their protocols.

Numbers come from `comparison.json`, `invariance.json`, `value-errors.jsonl`,
`multi-number-errors.jsonl`, `train.log` and `models/value-span-v7-dual-encoder/seed*/metrics.json`.

## 1. Question

Does a fully adaptable, value-specific encoder recover value-span-v1 quality while v1
type/target stays untouched?

**Yes.**

- Test value exact is 0.988 ± 0.004, the same as value-span-v1's 0.988 (shared encoder).
- Type and target outputs are bit-identical to gidi-finance-v1.
- Full representation adaptation is what the value task needed. The frozen-v1 readouts
  (v3–v6) plateaued at 0.77–0.90.

## 2. Architecture

```
tokenizer (shared) → input_ids, attention_mask
├─ PATH A (frozen, eval; exactly gidi-finance-v1)
│    original v1 encoder → shared dropout (identity in eval)
│    ├─ type head (mean pool)      → type_logits
│    └─ target BIO head            → target_logits
└─ PATH B (trainable value path; never reads path A's states)
     value encoder = deep copy of the trained v1 encoder (RoBERTa 4 × 768, 12 heads, FFN 2048,
       internal dropout 0.1) → Dropout(0.1)
     → Linear(768,256) → GELU → Dropout(0.1) → Linear(256,3)   emissions = value_logits
     → linear-chain CRF (O / B-VALUE / I-VALUE, learned transitions) → Viterbi
```

Head arch `encoder-mlp-crf`. The value encoder sits under `value_head.encoder.*`, so every
existing freezing check and the frozen-state hash cover path A unchanged.

**Initialisation:**

- Path A is `models/compression-v3/…/K2048/seed1`, loaded as trained (safetensors sha256
  `bfb2195a…0556`).
- The value encoder is `copy.deepcopy` of that trained encoder, not fresh pretrained weights.
  Each seed checks that it is bit-identical to the v1 encoder at init (69 tensors) and shares no
  storage with it.
- MLP and CRF: as in v5, from private generators `1_000_003 + seed` (+ 1_000_000, + 2_000_000).
- Seeds 1–3 vary only the MLP/CRF initialisation, the batch order (`seed + epoch`) and the
  dropout stream (`torch.manual_seed(seed)` after the model is built). Path A and the
  value-encoder init are identical across seeds.

**Parameters:**

| part | parameters | trainable |
|---|---|---|
| v1 encoder (path A) | 28,487,936 | no |
| v1 type + target heads | 8,459 | no |
| value encoder (path B) | 28,487,936 | yes |
| value MLP | 197,635 | yes |
| CRF | 15 | yes |
| **total** | **57,181,981** | **28,685,586 (50.2%)** |

## 3. Training recipe (declared before training)

The value-span-v1 settings, applied to path B only:

- **Optimizer:** AdamW; value encoder lr 5e-5; value MLP + CRF lr 1e-3.
- **Schedule:** batch 8, 40 epochs, linear warmup 0.1 then linear decay.
- **Weight decay:** 0.01 on weight matrices, including embeddings. 0 on biases, LayerNorms and
  CRF transitions (the v1 no-decay rule).
- **Dropout:** 0.1 inside the value encoder, on its output and in the MLP.
- **Clipping:** max grad norm 1.0 over the whole value branch.
- **Loss:** CRF NLL only, mean over notes with complete value labels; uncertain notes are
  excluded. No type or target loss, and no auxiliary loss.
- **Data:** `annotation-v2/training-v1` unchanged (train 895). No direction-repair rows,
  augmentation or label changes.
- Last epoch kept; no sweep.
- **Run:** MPS, 4,480 steps per seed, 626 / 457 / 372 s. Final training NLL was 0.0000 / 0.0002 /
  0.0003. Validation exact at epoch 40 was 1.000 / 0.993 / 0.993.
  - Monitoring only. The 107 frozen-v1 validation rows are inside train, as in value-span-v1.

## 4. Freezing and invariance (passes)

Checks before and during training (in each seed's `metrics.json`):

- the optimizer held exactly 76 path-B tensors: value encoder 69, MLP 4, CRF 3;
  `requires_grad` was set only on `value_head.*`;
- all 73 frozen path-A parameters had grad None after the first backward pass;
- the path-A state sha256 is `705488d3…c72c` for every seed, before training, after training
  and in the saved checkpoint, and it equals the base.

Invariance after training, on CPU, for all 3 seeds on test (143, which contains test-v1's 105 and
test-targeted's 38) and on probe (81):

- maximum absolute difference in type and target logits: 0;
- 0 mismatches in type predictions, target BIO predictions and target spans;
- 0 mismatches against the frozen v1 seed-1 prediction files.

## 5. Value metrics

Per seed, plus mean ± std (test, complete value labels):

| metric | seed 1 | seed 2 | seed 3 | mean ± std |
|---|---|---|---|---|
| exact | 0.9856 | 0.9856 | 0.9928 | 0.9880 ± 0.0042 |
| span precision | 0.9856 | 0.9856 | 0.9928 | 0.9880 |
| span recall | 0.9856 | 0.9856 | 0.9928 | 0.9880 |
| span F1 | 0.9856 | 0.9856 | 0.9928 | 0.9880 |
| token F1 | 0.9970 | 0.9970 | 0.9985 | 0.9975 ± 0.0009 |
| present/null | 1.000 | 1.000 | 1.000 | 1.000 |

| set | n | exact s1 / s2 / s3 | mean ± std |
|---|---|---|---|
| test-v1 | 105 | 0.981 / 0.981 / 0.990 | 0.984 ± 0.006 |
| test-targeted | 34 | 1.000 / 1.000 / 1.000 | 1.000 |
| probe | 81 | 1.000 / 1.000 / 1.000 | 1.000 |
| validation (monitoring) | 146 | 1.000 / 0.993 / 0.993 | 0.995 |

Gate (each criterion must hold on the mean and on seed 1):

| criterion | mean | seed 1 | threshold | result |
|---|---|---|---|---|
| test value exact | 0.9880 | 0.9856 | ≥ 0.95 | pass |
| human-only exact (n = 12) | 0.9167 | 0.9167 | ≥ 0.90 | pass |
| present/null | 1.0000 | 1.0000 | ≥ 0.97 | pass |
| multi_number exact (n = 22) | 0.9545 | 0.9545 | ≥ 0.90 | pass |
| slice floor (n ≥ 5) | 0.9545 (multi_number) | 0.9545 | ≥ 0.80 | pass |

Test slices (exact):

| slice | n | s1 | s2 | s3 | mean |
|---|---|---|---|---|---|
| explicit_unit | 113 | 0.982 | 0.982 | 0.991 | 0.985 |
| bare_number | 6 | 1.000 | 1.000 | 1.000 | 1.000 |
| multi_number | 22 | 0.955 | 0.955 | 0.955 | 0.955 |
| slang | 10 | 1.000 | 1.000 | 1.000 | 1.000 |
| unaccented | 50 | 1.000 | 1.000 | 1.000 | 1.000 |
| unseen_span | 37 | 0.973 | 0.973 | 1.000 | 0.982 |
| long_multi_token | 63 | 0.968 | 0.968 | 0.984 | 0.974 |

## 6. Comparison (means over 3 seeds)

| | v3 linear | v4 MLP | v5 MLP+CRF | v6 adapter | **v7 dual encoder** | value-span-v1 (shared FT) |
|---|---|---|---|---|---|---|
| test exact | 0.765 | 0.863 | 0.902 | 0.887 | **0.988** | 0.988 |
| test-v1 | 0.829 | 0.908 | 0.933 | 0.924 | **0.984** | 0.987 |
| test-targeted | 0.569 | 0.725 | 0.804 | 0.775 | **1.000** | 0.990 |
| probe | 0.840 | 0.889 | 0.951 | 0.975 | **1.000** | 1.000 |
| token F1 (test) | 0.892 | 0.961 | 0.968 | 0.959 | **0.998** | 0.995 |
| human-only | 0.417 | 0.750 | 0.750 | 0.750 | **0.917** | 0.944 |
| present/null | 0.971 | 1.000 | 1.000 | 0.998 | **1.000** | 1.000 |
| explicit_unit | 0.817 | 0.888 | 0.920 | 0.906 | **0.985** | 0.988 |
| bare_number | 0.667 | 0.500 | 0.500 | 0.833 | **1.000** | 0.944 |
| multi_number | 0.606 | 0.727 | 0.682 | 0.864 | **0.955** | 0.955 |
| slang | 0.100 | 0.667 | 0.833 | 0.700 | **1.000** | 1.000 |
| unaccented | 0.820 | 0.867 | 0.900 | 0.900 | **1.000** | 0.993 |
| unseen_span | 0.703 | 0.730 | 0.820 | 0.793 | **0.982** | 0.982 |
| long_multi_token | 0.640 | 0.820 | 0.894 | 0.815 | **0.974** | 0.979 |
| type/target = v1 | yes | yes | yes | yes | **yes (bit-identical)** | no (seed-1 type regression) |
| trainable params | 2.3k | 198k | 198k | 4.1M | **28.7M** | whole model |

**Key comparison: isolated value encoder (v7) vs shared encoder (value-span-v1).** Value quality
is equivalent:

- test exact 0.988 for both;
- multi-number 0.955 for both;
- probe 1.000 for both.

v7 is slightly lower on test-v1 (0.984 vs 0.987) and human-only (0.917 vs 0.944, one note of 12).
It is equal or higher on test-targeted, bare number and unaccented. Unlike value-span-v1, v7 leaves
type and target exactly v1. The isolation costs no value quality. Sharing the encoder with the
type/target tasks was neither needed for value quality nor harmless to type.

## 7. Error analysis

All value errors (test and probe, complete labels):

| category | test s1 / s2 / s3 | probe |
|---|---|---|
| wrong-number selection | 0 / 0 / 0 | 0 |
| truncation | 1 / 1 / 0 | 0 |
| left overrun | 0 / 0 / 0 | 0 |
| right overrun into date/time | 1 / 1 / 1 | 0 |
| punctuation overrun | 0 / 0 / 0 | 0 |
| other right overrun | 0 / 0 / 0 | 0 |
| missed value | 0 / 0 / 0 | 0 |
| total | 2 / 2 / 1 | 0 |

The errors:

- `cho a Nam vay 1 triệu 20/10` → `1 triệu 20/10` in all seeds. This is a date absorbed into the
  value; it is the tracked held-out case, gold provenance human.
- `cashback thẻ vib 230k` → `30k` in seeds 1 and 2: a B restart inside one amount (rule
  provenance). Seed 3 gets it right.

Every v5/v6 failure class is gone:

- leading number piece predicted O (`540k`, `170`, `612`);
- partial subword units (`600 nghin`, `5 xị`, `2tr`);
- unit dropping (`150k`, `250 ngan`, `700 ngàn`);
- wrong candidates (`300 hom 5/10`, `ky 2 900k`, `thằng`);
- punctuation (`5 củ,`, `1tr,`).

## 8. Multi-number errors (inspected by hand)

`multi-number-errors.jsonl` has 3 rows, all the same note:

| note | prediction (s1/s2/s3) | gold | kind |
|---|---|---|---|
| `cho a Nam vay 1 triệu 20/10` | `1 triệu 20/10` (all) | `1 triệu` | date/period absorption |

There is no wrong-candidate selection, truncation or punctuation error among the multi-number
notes. `tra no chi Mai 300 hom 5/10`, which value-span-v1 seed 1 got wrong (`300 hom 5/10`), is
exact in all v7 seeds.

## 9. Tracked held-out cases (reported only; never used for training, tuning or templates)

- `cho a Nam vay 1 triệu 20/10` → `1 triệu 20/10` in all seeds: a right overrun into the date.
  value-span-v1 made the same error (seeds 1 and 3). It remains the one systematic value error.
- `Thắng vay 5 củ, hẹn t10 trả` → `5 củ` in all seeds (exact).

## 10. Runtime and model-size implications (nothing exported)

| | gidi-finance-v1 | v7 dual encoder |
|---|---|---|
| parameters | 28.50M | 57.18M (2.01×) |
| PyTorch checkpoint (FP32 safetensors) | 114.0 MB | ≈ 229 MB (measured 218 MiB) |
| INT8 ONNX | 28.66 MB (measured) | ≈ 57.4 MB, estimated as 2 × the v1 encoder graph + 0.2M head params; not exported |
| bundle (model + tokenizer + maps) | ≈ 29.3 MB | ≈ 58 MB (estimate) |
| CPU latency, 1 thread, batch 1, PyTorch FP32, 143 test notes | 3.83 ms/note | 7.59 ms/note (1.98×) |

- **Inference cost:** two full encoder passes over the same input, so about 2× FLOPs and latency.
  The CRF Viterbi over at most 32 tokens × 3 tags is negligible.
- **Shared tokenization: yes.**
  - Both encoders consume the identical `input_ids` and `attention_mask` (same tokenizer, vocab
    map and max_length 32), so tokenization, NFC normalization and offsets run once.
  - The two encoder passes are independent and could run in parallel, or as one ONNX graph with
    two subgraphs.
  - [INFERENCE] The embeddings are not shared: the value encoder's embedding table (6.43M params)
    was fine-tuned too.
- [INFERENCE] Possible size reductions are not tested here: INT8 for both encoders is already
  assumed; distilling or compressing the value encoder, or sharing the frozen lower layers, would
  each need their own experiment.

## 11. Verdict

**A.** Invariance passes exactly (path A = gidi-finance-v1, bit-identical logits). All five
frozen value criteria pass on the mean and on seed 1. Stopped: no export, no playground change,
no further experiment.

## 12. Known limitations

- The small slices make individual criteria fragile:
  - human-only n = 12 (0.917 = 11/12; the miss is the tracked `cho a Nam` note);
  - multi_number n = 22; bare 6; slang 10.
- Most value labels are rule-provenance (audited by stratum).
- Validation rows from frozen v1 are in train, so the validation numbers are optimistic.
- Training ran on MPS (not bit-reproducible); evaluation ran on CPU. The training loss reached
  about 0. There is no sign of held-out degradation, but last-epoch selection on 895 notes is a
  memorisation risk for a 28.5M-parameter branch.
- The date-overrun case (`1 triệu 20/10`) is unresolved. The date-adjacent amount pattern seems
  under-represented in training (`[INFERENCE]`).
- Model size and latency double compared with v1. This was not an adoption criterion here, but it
  matters for deployment.
- value-span-v1 started from the compression-v3 student init and trained all three tasks; v7
  starts from the trained v1 encoder. The comparison isolates sharing and initialisation
  together.

## 13. Possible next directions (listed only, NOT started)

1. **Deployment experiment:**
   - export the dual model as gidi-finance-v2 (INT8 ONNX with two encoders);
   - run ONNX parity, the INT8 value-agreement gate, size and latency on target hardware;
   - update the playground after the gate passes.
2. **Size reduction:** distil or compress the value encoder (fewer layers or a smaller FFN, as in
   compression-v3) under the same frozen value gate.
3. **Shared lower layers:** keep frozen v1 layers 1–k shared and clone only the top layers for
   the value path, trading size against value quality.
4. **Date-adjacent amounts:** collect fresh human-labelled notes with amount + date patterns. They
   must not come from held-out errors.

## 14. Artifacts

| path | content |
|---|---|
| `protocol.json` | frozen protocol (0444) |
| `train.log` | training log |
| `models/value-span-v7-dual-encoder/seed{n}/` | checkpoints (PyTorch), `metrics.json` with freezing and clone checks |
| `invariance.json` | type/target invariance |
| `comparison.json`, `eval/` | metrics, slices, criteria, error counts, tracked notes, verdict |
| `value-errors.jsonl`, `multi-number-errors.jsonl` | every value error with its category; multi-number errors |

Code changes (additive; earlier head archs and their checkpoints unchanged):

- `src/gidi/modeling/value.py`:
  - head arch `encoder-mlp-crf`;
  - `CRFValueHead` gained an optional value `encoder` (deep copy of the v1 encoder) plus output
    dropout, and receives `input_ids`;
  - `build_value_head(..., encoder=)`.
- `src/gidi/training/train_value_head.py`:
  - `encoder_lr`;
  - optimizer groups by part (head / adapter / encoder) and decay;
  - `check_value_encoder_clone`, recorded in `metrics.json`;
  - `torch.manual_seed(seed)` for the value-branch dropout stream.
- `scripts/train_value_head.py`: `--encoder-lr`.
- `tests/test_train_value_head.py`: a test that the value encoder starts as a v1 clone and trains
  without touching v1.
