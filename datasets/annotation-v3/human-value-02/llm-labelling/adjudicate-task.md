# Task

Two independent annotators (X and Y) labelled each note below under the rules above and
disagreed on at least one field. For **each** note decide which label is better according to
the rules, or decide that the note is genuinely unclear. Judge only from the note text and the
rules; you have no files, tools or other context. Do not edit a label and do not merge fields.

## Output format

**One JSON object per line, one per note, in the input order, nothing else** (no markdown, no
commentary):

```
{"id": "<note id>", "choice": "X" | "Y" | "uncertain", "reason": "<one sentence>",
 "label": <only when choice is "uncertain": a complete label object, see below>}
```

For `"uncertain"` only (use it when the note truly does not establish the type, the target or the
value, or when both labels break the rules in a way that no label is acceptable), `label` is
`{"annotation_status": "uncertain", "type": <a type or null>, "target": <{"text": ..., "start": ...,
"end": ...} or null>, "value": <{"text": ..., "start": ..., "end": ...} or null>, "span_status":
{"value": "complete" | "uncertain"}, "note": "<what is unclear, non-empty>"}` with exact
code-point offsets into the note. Prefer `X` or `Y` whenever one of them follows the rules.
