| candidate | seeds | composite | sec. type F1 | sec. target | slang joint | slang value | GenZ joint | prim. type F1 | prim. target | int8 bytes |
|---|---|---|---|---|---|---|---|---|---|---|
| roberta-base 4L/768 K2048 v9000 (control) | 3 | 0.8471 (0.8453-0.8483) | 0.8826 (0.8765-0.8928) | 0.8379 (0.8207-0.8552) | 0.7563 (0.7311-0.7703) | 0.9823 (0.9776-0.9860) | 0.7766 (0.7766-0.7766) | 0.9389 (0.9316-0.9455) | 0.8925 (0.8879-0.8972) | 29,170,089 |
| distilroberta-base 6L/768 K832 v9000 | 2 | 0.8537 (0.8437-0.8637) | 0.8976 (0.8917-0.9035) | 0.8517 (0.8483-0.8552) | 0.7353 (0.7283-0.7423) | 0.9860 (0.9804-0.9916) | 0.7979 (0.7447-0.8511) | 0.9407 (0.9315-0.9499) | 0.9019 (0.9019-0.9019) | 29,036,406 |
| electra-base-discriminator 4L/768 K2048 v9000 | 2 | 0.8630 (0.8581-0.8678) | 0.8761 (0.8665-0.8857) | 0.8655 (0.8621-0.8690) | 0.7969 (0.7815-0.8123) | 0.9678 (0.9636-0.9720) | 0.8085 (0.8085-0.8085) | 0.9287 (0.9176-0.9398) | 0.9159 (0.9112-0.9206) | 29,169,805 |
| MiniLM-L12-H384 12L/384 K1536 v20500 | 2 | 0.8786 (0.8757-0.8814) | 0.9066 (0.8904-0.9228) | 0.8948 (0.8897-0.9000) | 0.8095 (0.8039-0.8151) | 0.9734 (0.9608-0.9860) | 0.8085 (0.8085-0.8085) | 0.9499 (0.9423-0.9575) | 0.9393 (0.9393-0.9393) | 29,499,600 |

Per run (composite / sec type / sec target / slang joint / slang value / GenZ joint):

- roberta-base seed 1: 0.8453 / 0.8784 / 0.8207 / 0.7675 / 0.9832 / 0.7766
- roberta-base seed 2: 0.8478 / 0.8765 / 0.8379 / 0.7703 / 0.9776 / 0.7766
- roberta-base seed 3: 0.8483 / 0.8928 / 0.8552 / 0.7311 / 0.9860 / 0.7766
- distilroberta seed 1: 0.8437 / 0.8917 / 0.8483 / 0.7423 / 0.9916 / 0.7447
- distilroberta seed 2: 0.8637 / 0.9035 / 0.8552 / 0.7283 / 0.9804 / 0.8511
- electra seed 1: 0.8581 / 0.8665 / 0.8621 / 0.7815 / 0.9720 / 0.8085
- electra seed 2: 0.8678 / 0.8857 / 0.8690 / 0.8123 / 0.9636 / 0.8085
- minilm seed 1: 0.8814 / 0.9228 / 0.9000 / 0.8151 / 0.9608 / 0.8085
- minilm seed 2: 0.8757 / 0.8904 / 0.8897 / 0.8039 / 0.9860 / 0.8085

distilroberta: **no** (separated=False)
  - composite_margin: FAIL {'challenger_mean': 0.8536979156021542, 'control_mean': 0.8471327864578413, 'delta': 0.006565129144312953, 'need': 0.01}
  - no_regression_primary_type: pass {'delta': 0.0017682650516978038}
  - no_regression_primary_target: pass {'delta': 0.009345794392523477}
  - int8_size: pass {'max_bytes': 29036406}
  - ops_and_opset: pass {'ops_not_in_shipped': [], 'opsets': [17]}
electra: **no** (separated=True)
  - composite_margin: pass {'challenger_mean': 0.8629710188605781, 'control_mean': 0.8471327864578413, 'delta': 0.01583823240273685, 'need': 0.01}
  - no_regression_primary_type: FAIL {'delta': -0.01023739272698676}
  - no_regression_primary_target: pass {'delta': 0.02336448598130858}
  - int8_size: pass {'max_bytes': 29169805}
  - ops_and_opset: pass {'ops_not_in_shipped': [], 'opsets': [17]}
minilm: **WIN-WITH-PORT** (separated=True)
  - composite_margin: pass {'challenger_mean': 0.8785767443028057, 'control_mean': 0.8471327864578413, 'delta': 0.031443957844964454, 'need': 0.01}
  - no_regression_primary_type: pass {'delta': 0.010966670310117999}
  - no_regression_primary_target: pass {'delta': 0.04672897196261694}
  - int8_size: pass {'max_bytes': 29499600}
  - ops_and_opset: pass {'ops_not_in_shipped': [], 'opsets': [17]}

Decision: minilm
