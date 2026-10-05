# Task

You are an annotator for short Vietnamese personal-finance notes (the user's own money). Label
every note below with the **annotation-v3 combined pass** rules above (type, target span and
value span in one label). annotation-v3 overrides annotation-v1/v2 where they differ: a
debt-only note is `borrow` (the user owes) or `lend` (the other party owes the user), or
`uncertain` with a note when the direction is unclear; it is not `skipped`. A gift or ceremony
money goes to its receiver, who IS the target (`quà sinh nhật bé Na 300k` -> `Na`, `mừng cưới Hoa
1 triệu` -> `Hoa`), with a kinship/title prefix before a proper name dropped. `skipped` means the
note is unusable (not a finance note); use it only for that.

Judge each note only from its own text and the rules. You have no files, tools or other
context: do not look anything up. Do not copy or normalise anything: the spans are exact
substrings of the note.

## Output format

Output **one JSON object per line, one line per note, in the input order, and nothing else** (no
markdown fence, no commentary, no blank lines). Keys:

```
{"id": "<note id>", "annotation_status": "complete" | "uncertain" | "skipped",
 "type": "<one of the eight types>" | null,
 "target": {"text": "<exact substring>", "start": <int>, "end": <int>} | null,
 "value": {"text": "<exact substring>", "start": <int>, "end": <int>} | null,
 "span_status": {"value": "complete" | "uncertain"},
 "note": "<short reason>"}
```

* `start` / `end` are 0-based Python code-point offsets into the note exactly as given (`end`
  exclusive); `text` must equal `note[start:end]`. Count carefully, including spaces and
  combining characters (the notes are NFC).
* `type` is null only when `annotation_status` is `skipped` or `uncertain` with an unclear type;
  `transfer` has `target` null; `target` is null when no counterparty is named; `value` is null
  when the note states no amount.
* `span_status` is omitted when `annotation_status` is `skipped`, otherwise it is present with
  the status of the value span.
* `note` is required (non-empty) when `annotation_status` is `uncertain` or `span_status.value`
  is `uncertain`; otherwise omit it or keep it to a few words (for example an oddity in the
  note's spacing).
