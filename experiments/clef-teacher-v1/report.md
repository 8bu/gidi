# Clef as a distillation teacher (feasibility probe)

Question: can Cloudflare Clef (27B decision model; omp's `judge` role) label Gidi notes well enough
to be a teacher for the on-device student?

Setup (`scripts/clef_teacher.py`): one `choice` question for the type (the 8 annotation-v3 type
definitions, read from `configs/annotation-v3.yaml`) and one `choice` for the target (every word
n-gram of the note up to 4 words, plus `none`). Clef returns a probability per option. The gold
target span is among the candidates for 100% of notes. Few-shot variant: the 16 nearest labelled
notes (char n-gram TF-IDF) go into `state`. Nothing here is written into a dataset.

| Variant | Eval set | n | type | target | type+target |
|---|---|---|---|---|---|
| zero-shot, short type rubric | validation (training-v5) | 110 | 0.773 | 0.800 | 0.664 |
| zero-shot, contract text + rules | validation | 110 | 0.709 | 0.800 | 0.582 |
| **16-shot kNN** (pool = train) | validation | 110 | 0.936 | 0.955 | 0.909 |
| **16-shot kNN** (pool = train + validation) | human-value-02 (LLM-labelled, never-train) | 190 | 0.953 | 0.874 | 0.832 |
| `gidi-finance-v3` student, for reference | human-value-02 | 190 | 0.921 | 0.853 | about 0.79 (end-to-end 0.7895 with value 1.0) |

Selective accuracy of the type on human-value-02 (max type probability): all 190 notes 0.953;
>= 0.5: 96% of notes at 0.978; >= 0.7: 85% at 0.988; >= 0.9: 38% at 1.000.

Reading:

- Zero-shot Clef is not usable as a teacher: it does not know the annotation conventions (`muon chi
  Lan` is `borrow`, gold bars are `transfer`). More prompt text made it worse.
- With retrieved labelled examples it clears the student on human-value-02, but the gap (0.832 vs
  about 0.79 type+target) is within noise at n=190, and the labels of that set come from Claude, not
  humans.
- The validation split is not held out for the v3 student (it trained on it), so no student number
  is quoted on validation.
- The prompt was tuned on validation only; human-value-02 was scored once.
