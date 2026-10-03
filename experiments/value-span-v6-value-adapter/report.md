# value-span-v6-value-adapter: report

**Verdict: B. Invariance passes, value quality fails. Stop.** No second adapter block, BiLSTM,
encoder-layer adapters, partial unfreezing, new augmentation or loss weighting was started.

The protocol `experiments/value-span-v6-value-adapter/protocol.json` (0444, sha256
`633bd254…32bd`) was written before training. The following are unchanged:

- `gidi-finance-v1`;
- value-span-v1, v2-direction-repair, v3, v4 and v5, including their protocols.

Numbers come from `comparison.json`, `invariance.json`, `value-errors.jsonl`,
`multi-number-errors.jsonl` and `train.log`.

## 1. Question

Can one small value-only contextual block, placed in front of the v5 MLP + CRF head, close most of
the gap to value-span-v1 (full joint fine-tune, test value exact 0.988) without changing v1
type/target behaviour?

**Answer: no.**

- Test value exact is 0.887 ± 0.011. That is below v5's 0.902, and the gap to v1 is 0.101
  (v5: 0.086).
- Probe improves to 0.975.
- Invariance is exact.

## 2. Architecture (as declared)

```
frozen v1 encoder → H (last_hidden_state; shared dropout = identity in eval)
├─ frozen type head    (mean-pooled H)              unchanged v1 path
├─ frozen target head  (H)                          unchanged v1 path
└─ value branch (trainable, reads H; returns new tensors, H never modified):
     value adapter: 1 × nn.TransformerEncoderLayer
        d_model 768, 12 heads (head dim 64), FFN 1024, GELU, dropout 0.1
        (attention weights, FFN activation, both residual branches), post-LN, eps 1e-5,
        key-padding mask from the attention mask
        x1 = LN1(H  + Drop(MHSA(H)))
        x2 = LN2(x1 + Drop(Linear2(Drop(GELU(Linear1(x1))))))
     → Linear(768,256) → GELU → Dropout(0.1) → Linear(256,3)   emissions
     → linear-chain CRF (O / B-VALUE / I-VALUE; learned transitions, no hard constraints)
     → Viterbi
```

Trainable parameters, 4,137,746 in total:

| part | parameters |
|---|---|
| adapter | 3,940,096 |
| MLP | 197,635 |
| CRF | 15 |

- **Initialisation:**
  - MLP and CRF as in v5;
  - adapter: PyTorch default init under `torch.manual_seed(1_000_003 + seed + 3_000_000)` in a
    forked RNG.
  - Seeds 1–3 vary only the value-branch initialisation and batch order.
- **Objective:** CRF NLL only, mean over notes with complete value labels; uncertain notes are
  excluded. No auxiliary loss was needed: the CRF NLL trains the adapter, emissions and
  transitions end to end.
- **Recipe** (declared before training, no sweep):
  - AdamW: adapter lr 3e-4; MLP and CRF lr 1e-3, as in v4/v5;
  - weight decay 0.01 on weight matrices only;
  - batch 8, 40 epochs, warmup 0.1 then linear decay, grad norm 1.0 over the value branch;
  - last epoch kept.
  - Data: `annotation-v2/training-v1` unchanged.
- **Run:** MPS, about 160 s per seed, 4,480 steps. Final training NLL was 0.022 / 0.017 / 0.015.
  Validation exact at epoch 40 was 0.952 / 0.959 / 0.959.
- **Runtime schema:** unchanged if this is ever exported (`value_text`, `value_span`,
  `value_confidence`, …). No numeric normalization. Not exported.

## 3. Freezing and invariance (passes)

Checks before and during training (recorded in each seed's `metrics.json`):

- the optimizer held exactly the 19 value-branch tensors: 12 adapter, 4 MLP and 3 CRF;
- `requires_grad` is set only on `value_head.*`;
- all 73 frozen parameters had grad None after the first backward pass;
- the frozen-state sha256 equals the base's before training, after training and in the saved
  checkpoint.

Invariance after training, on CPU, for all 3 seeds on test (143) and probe (81):

- maximum absolute difference in type and target logits: 0;
- 0 mismatches in type predictions, target BIO predictions and target spans;
- 0 mismatches against the frozen v1 seed-1 prediction files.

## 4. Value metrics

Per seed, plus mean ± std:

| set | metric | seed 1 | seed 2 | seed 3 | mean ± std |
|---|---|---|---|---|---|
| test (139) | exact | 0.8993 | 0.8849 | 0.8777 | 0.8873 ± 0.011 |
| | span P / R | 0.899 / 0.899 | 0.891 / 0.885 | 0.878 / 0.878 | 0.889 / 0.887 |
| | span F1 | 0.8993 | 0.8881 | 0.8777 | 0.8884 ± 0.011 |
| | token F1 | 0.9625 | 0.9594 | 0.9546 | 0.9588 ± 0.004 |
| | present/null | 1.000 | 0.993 | 1.000 | 0.998 |
| test-v1 (105) | exact | 0.9333 | 0.9238 | 0.9143 | 0.9238 ± 0.010 |
| test-targeted (34) | exact | 0.7941 | 0.7647 | 0.7647 | 0.7745 ± 0.017 |
| probe (81) | exact | 0.9753 | 0.9753 | 0.9753 | 0.9753 ± 0 |
| validation (146, monitoring) | exact | 0.9521 | 0.9589 | 0.9589 | 0.9566 ± 0.004 |

Gate (each criterion must hold on the mean and on seed 1):

| criterion | mean | seed 1 | threshold | result |
|---|---|---|---|---|
| test value exact | 0.8873 | 0.8993 | ≥ 0.95 | fail |
| human-only exact | 0.7500 | 0.7500 | ≥ 0.90 | fail |
| present/null | 0.9976 | 1.0000 | ≥ 0.97 | pass |
| multi_number exact | 0.8636 | 0.9091 | ≥ 0.90 | fail (mean) |
| slice floor (n ≥ 5) | 0.700 (slang) | 0.800 | ≥ 0.80 | fail (mean) |

Test slices (exact; per seed and mean):

| slice | n | s1 | s2 | s3 | mean |
|---|---|---|---|---|---|
| explicit_unit | 113 | 0.912 | 0.912 | 0.894 | 0.906 |
| bare_number | 6 | 0.833 | 0.833 | 0.833 | 0.833 |
| multi_number | 22 | 0.909 | 0.864 | 0.818 | 0.864 |
| slang | 10 | 0.800 | 0.600 | 0.700 | 0.700 |
| unaccented | 50 | 0.900 | 0.900 | 0.900 | 0.900 |
| unseen_span | 37 | 0.838 | 0.784 | 0.757 | 0.793 |
| long_multi_token | 63 | 0.841 | 0.794 | 0.810 | 0.815 |

## 5. Comparison (means over 3 seeds)

| | v3 linear | v4 MLP | v5 MLP+CRF | **v6 adapter** | value-span-v1 (joint FT) |
|---|---|---|---|---|---|
| test exact | 0.765 | 0.863 | 0.902 | **0.887** | 0.988 |
| test-v1 | 0.829 | 0.908 | 0.933 | **0.924** | 0.987 |
| test-targeted | 0.569 | 0.725 | 0.804 | **0.775** | 0.990 |
| probe | 0.840 | 0.889 | 0.951 | **0.975** | 1.000 |
| token F1 (test) | 0.892 | 0.961 | 0.968 | **0.959** | 0.995 |
| human-only | 0.417 | 0.750 | 0.750 | **0.750** | 0.944 |
| present/null | 0.971 | 1.000 | 1.000 | **0.998** | 1.000 |
| explicit_unit | 0.817 | 0.888 | 0.920 | **0.906** | 0.988 |
| bare_number | 0.667 | 0.500 | 0.500 | **0.833** | 0.944 |
| multi_number | 0.606 | 0.727 | 0.682 | **0.864** | 0.955 |
| slang | 0.100 | 0.667 | 0.833 | **0.700** | 1.000 |
| unaccented | 0.820 | 0.867 | 0.900 | **0.900** | 0.993 |
| unseen_span | 0.703 | 0.730 | 0.820 | **0.793** | 0.982 |
| long_multi_token | 0.640 | 0.820 | 0.894 | **0.815** | 0.979 |
| type/target = v1 | yes | yes | yes | **yes** | no (seed-1 type regression) |

## 6. Error categories (seeds 1 / 2 / 3)

| category | test v6 | test v5 | probe v6 | probe v5 |
|---|---|---|---|---|
| wrong-number selection | 3 / 3 / 4 | 1 / 1 / 1 | 1 / 0 / 0 | 0 / 0 / 0 |
| boundary truncation | 10 / 10 / 11 | 8 / 7 / 7 | 1 / 1 / 1 | 2 / 2 / 2 |
| left overrun | 0 / 0 / 1 | 1 / 1 / 2 | 0 / 1 / 1 | 1 / 1 / 1 |
| right overrun into date/time | 1 / 1 / 1 | 2 / 2 / 2 | 0 | 0 |
| punctuation overrun | 0 / 1 / 0 | 2 / 2 / 2 | 0 | 0 |
| other right overrun | 0 | 0 | 0 | 0 |
| missed value | 0 / 1 / 0 | 0 | 0 | 1 / 1 / 1 |
| total | 14 / 16 / 17 | 14 / 13 / 15 | 2 / 2 / 2 | 4 / 4 / 4 |

**v5's emission-level failures that the adapter fixes, in all seeds unless noted:**

- Leading number piece predicted O:
  - `540k` → `k` is fixed in seeds 1 and 2; seed 3 overruns to `8 540k`;
  - `170` → `0` is fixed;
  - `612` → `12` is fixed.
- Partial numeric token spans: probe `9tr5` → `tr5` and probe `5tr2` → `tr2` are fixed.
- Left and punctuation overruns: `4g 120k` → `g 120k` and `1tr,` → `1tr` are fixed.

**v5's failures that remain:**

- B restart inside one amount: `230k` is still split. Seed 1 still predicts `2` (B on `30`), and
  seeds 2 and 3 now pick `30k`.
- Unit dropped: `150k` → `150`, `250 ngan` → `250`.
- Wrong number: `1tr2` → `k`.
- Amount vs period/date ambiguity, unresolved:
  - `cho a Nam vay 1 triệu 20/10` → `1 triệu 20/10` in all seeds (`20/10` tagged I);
  - `tra no chi Mai 300 hom 5/10` → `5` in all seeds, which is worse than v5's `300 hom`.

**New errors from the adapter** (none of these failed in v5):

- Unit words truncated mid-word, the dominant new failure:
  - `600 nghin` → `600 ngh` (`in` → O), in all seeds;
  - `5 xị` → `5 x` (`ị` → O), in all seeds;
  - `45.000 vnd` → `45.000`, in all seeds;
  - `700 ngàn` → `700`, in all seeds;
  - `2tr` → `tr` in seeds 1 and 3, and `2tr6` → `tr6` in seed 2.
- Wrong span chosen: `đòi được nợ thằng Lâm 400k` → `thằng`, in all seeds. Viterbi tags
  `thằng` as a lone I (an O→I start) and `400k` as a separate span, and the confidence rule picks
  `thằng`.

Net effect: the adapter fixes the numeric-continuity errors v5 could not override, but it breaks
roughly as many easy unit spans.

[INFERENCE] Together with the training NLL of 0.015–0.022, against v5's ~0.09, this looks like a
4.1M-parameter block overfitting 895 notes. Validation (0.957) is higher than v5's (0.948), but
the held-out test-targeted set is lower (0.775 vs 0.804).

## 7. Multi-number errors (inspected by hand)

`multi-number-errors.jsonl`; seed 1 has 2 test notes and 1 probe note, and all seeds together
have 12 rows.

| note | s1 | s2 | s3 | kind |
|---|---|---|---|---|
| `cho a Nam vay 1 triệu 20/10` | `1 triệu 20/10` | same | same | date/period absorption |
| `tra no chi Mai 300 hom 5/10` | `5` | `5` | `5` | wrong candidate (the date's day) |
| `Thắng vay 5 củ, hẹn t10 trả` | exact | `5 củ,` | `10` | s2 punctuation; s3 wrong candidate (`t10` month) |
| `tiền điện nhà số 12 tháng 8 540k` | exact | exact | `8 540k` | other boundary error (absorbs the preceding month number) |
| probe `khoan vay online tamo dong ky 2 900k` | `2` | `2 900k` | `2 900k` | s1 wrong candidate; s2/s3 absorb the instalment index |

All remaining multi-number errors are amount-vs-date/period/index confusions. There are no pure
truncations of the amount.

## 8. Tracked held-out cases (reported only; never used for training, tuning or templates)

- `cho a Nam vay 1 triệu 20/10` → `1 triệu 20/10` in all seeds, a right overrun into the date
  (as in v5; v4 was exact).
- `Thắng vay 5 củ, hẹn t10 trả`:
  - seed 1: `5 củ` (exact);
  - seed 2: `5 củ,` (punctuation overrun);
  - seed 3: `10` (wrong number, the `t10` month).

## 9. Verdict

**B.** Invariance passes exactly; value quality fails:

- test exact 0.887 against ≥ 0.95;
- human-only 0.75 against ≥ 0.90;
- multi_number mean 0.864 against ≥ 0.90;
- slang mean 0.70 against ≥ 0.80.

Seed 1 alone passes multi_number (0.909) and the slice floor (0.80), but not test or human-only
exact. The value-specific contextual block does **not** close most of the gap to value-span-v1:
test exact falls 0.902 → 0.887 against v5 and the gap widens. Probe and multi-number improve.

Stopped. No export, no playground change, and no further experiment started.

## 10. Known limitations

- Test, human-only and slice sizes are small: human-only n = 12; slang 10, bare 6. A single
  note moves human-only by 0.08 and those slices by 0.10–0.17.
- Most value labels are rule-provenance (audited by stratum), so exact match measures agreement
  with audited rule labels.
- Training ran on MPS, which is not bit-reproducible. Evaluation ran on CPU.
- Last-epoch selection with a 4.1M-parameter branch on 895 notes. Overfitting is visible in the
  training NLL, but per protocol it was neither tuned nor early-stopped.
- The span-choice rule (highest emission-softmax confidence among Viterbi spans) can pick a stray
  span (`thằng`). This is decoder behaviour shared with v5, not a v6 change.
- test has 0 complete no-amount labels; null accuracy is measured on validation only.

## 11. Possible next directions (listed only, NOT started)

1. Accept the v5/v6 level as a bounded frozen-v1 result, and decide whether value extraction
   needs encoder adaptation (value-span-v1 reached 0.988 but regressed v1 type on seed 1).
2. Joint fine-tuning with an explicit type/target preservation constraint (distillation to v1
   logits), instead of exact freezing.
3. A separate small value-only encoder (a second model) running alongside frozen v1 at runtime,
   if the size budget allows.
4. Hard BIO constraints in Viterbi (forbid O→I), a decoding-only change, re-evaluated under a new
   protocol on the existing checkpoints.
5. More human-labelled value data for unit words and date/amount contexts. It must come from a
   fresh corpus, not from held-out errors.

## 12. Artifacts

| path | content |
|---|---|
| `protocol.json` | frozen protocol (0444) |
| `train.log` | training log |
| `models/value-span-v6-value-adapter/seed{n}/metrics.json` | freezing checks |
| `invariance.json` | type/target invariance |
| `comparison.json`, `eval/` | metrics, slices, criteria, error counts, tracked notes, verdict |
| `value-errors.jsonl`, `multi-number-errors.jsonl` | every value error with its category; multi-number errors |

Code changes (additive; linear, mlp and mlp-crf heads and their checkpoints unchanged):

- `src/gidi/modeling/value.py`:
  - head arch `adapter-mlp-crf`;
  - `build_value_adapter`, which builds one post-LN `TransformerEncoderLayer`, 12 heads, FFN
    1024;
  - `CRFValueHead` gained an optional `adapter` and takes the attention mask;
  - `GidiValueModel.forward` passes the mask to CRF heads, and type/target still read the
    unmodified `H`.
- `src/gidi/modeling/crf.py`: `torch.where` masking in the gold-path score, so padded
  non-finite emissions cannot leak.
- `src/gidi/training/train_value_head.py` and `scripts/train_value_head.py`:
  - `adapter_lr` / `--adapter-lr`;
  - optimizer groups by adapter vs head and by decay (LayerNorm and bias not decayed).
- `tests/test_train_value_head.py`: an adapter-mlp-crf freezing case (tiny hidden size 24, so
  the 12 heads divide it).
