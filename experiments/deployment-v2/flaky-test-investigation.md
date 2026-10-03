# Flaky full-suite signal: investigation (deployment-v2, phase 0)

## Observation

In the value-span-v7 closing gates, one of three full `uv run pytest -q` runs printed `607 passed,
20 errors` (627 tests). The other two passed. Only the summary line was captured, so the failing
test names were never recorded.

## Attempts to reproduce

All runs used `uv run pytest -q -p no:cacheprovider` from the repo root, with the v1 playground
service running on 127.0.0.1:8765, as during the original run.

| attempt | runs | result |
|---|---|---|
| sequential full suite | 6 | 627 passed each time |
| two full suites at once (second started 5–8 s after the first) | 2 × 2 | 627 passed each time |
| runtime-critical subset, sequential | 5 | 121 passed each time |

The runtime-critical subset is `test_inference`, `test_inference_value`, `test_export`,
`test_crf`, `test_freeze_deployment`, `test_verify_deployment`, `test_demo_ui`,
`test_value_model` and `test_train_value_head`.

Including the original two clean runs: 0 errors in 15 of 16 full-suite runs and in 5 of 5 subset
runs. **The failure did not reproduce.**

## Analysis

Of the suite's shared fixtures, only one is used by exactly 20 tests: the module-scoped `server`
fixture in `tests/test_demo_ui.py`, which binds the old playground HTTP server on port 0 around
the v1 predictor. A setup failure in a module fixture is reported as one error per dependent
test. The other shared fixtures don't fit the count:

| fixture | tests |
|---|---|
| `predictor` (several modules) | 36 |
| `v2_bundle` | 17 |
| `tokenizer` | 24 |

[INFERENCE] The 20 errors were most likely a one-off setup failure of the `test_demo_ui` server
fixture, the test HTTP server rather than the model. The fixture binds an ephemeral port on
127.0.0.1. Two cheap explanations were tested and did not reproduce:

- the concurrent pytest temp-dir cleanup;
- the fixed playground port.

Ruled out by the evidence:

- **Test order:** pytest uses its fixed default order and has no random-order plugin.
- **Randomness:** all subset runs were identical.
- **MPS vs CPU:** the inference and export tests run on CPU and onnxruntime.
- **Stale artifacts:** the bundles are verified byte-identical by `freeze_deployment --check`.

## Decision

- Recorded as unreproduced.
- The model, export, runtime, tokenizer/offset and bundle-integrity tests were deterministic and
  clean in every run, so the failure does not block export.
- No speculative fixes were made.
- The playground server and its tests are rewritten in phase 7.
- The final verification repeats the full suite several times and captures the full output, not
  just the summary line.
