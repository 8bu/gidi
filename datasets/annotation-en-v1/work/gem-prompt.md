# Task: label English personal-finance notes (annotation-en-v1, silver)

Read `docs/annotation-en-v1.md` and the `types`, `rules` (R1-R4) and target comments in
`configs/annotation-en-v1.yaml` completely and follow them exactly. Do NOT read any other labels
file; do not look at other datasets/annotation-* directories; label independently.

Input: a JSONL file of `{"id","text"}`. Output: a JSONL file, one line per input note, same order,
every id exactly once:

`{"id": "...", "type": "<one of the 8 types>", "target": "<exact substring of text>" | null,
  "status": "complete" | "uncertain" | "skipped", "note": "<why, only if uncertain>"}`

- `type`: expense, income, borrow, lend, repayment_in, repayment_out, transfer, refund.
- `target`: the counterparty per the target rules (minimal exact substring copied verbatim from
  `text`, case and typos included; null when none; transfer is always null; salary notes null;
  payment apps/banks/items/bills are not targets; drop titles before a proper name).
- Use `status: "uncertain"` (type null allowed only then... but still give your best `type`) only
  when the text truly does not establish type or target (e.g. `iou 15` with no direction, bare
  `Sam 20`); use `"skipped"` with type null and target null for non-finance text. Default is
  `"complete"`.
- Do not write a script that guesses labels from keywords; read and judge each note yourself,
  in chunks of ~50 notes. A script may be used only to write/validate the output file
  (valid JSON, target is exact substring of text, ids match).
- Finally verify: line count equals input, every `target` is a substring of its `text`.

## Clarifications (these resolve ambiguities in the guide; apply them)

1. **Named merchants, services, platforms, brands and businesses ARE targets** when the money goes
   to / comes from them (inherited v3/v4 rule; the guide's `uber home $18` row is the one outlier,
   treat it as target `uber`). Examples: `zomato biryani delivery 280` -> `zomato`;
   `mcdonalds 8.99` -> `mcdonalds`; `uber to station 9.20` -> `uber`; `aws 13th month 4200`
   (income) -> `aws`; `olx furniture sale 6,500` -> `olx`; `halifax credit card 85`
   (repayment_out) -> `halifax`; `weekly grocery haul at fairprice` -> `fairprice`;
   `anytime fitness membership 98` -> `anytime fitness` (minimal span naming the business: drop
   generic trailing words like `store`/`restaurant` only if a distinctive name remains:
   `omnidesk store` -> `omnidesk`, `paradise dynasty restaurant` -> `paradise dynasty`).
   Items, dishes, bill names (`rent`, `electric bill`, `council tax`), activities, payment
   channels (venmo, zelle, gpay, paynow, upi, bank names when just the channel) and companions
   are NOT targets. A generic business noun alone (`pharmacy`, `toll booth`) is a target only for
   a refund/return from it; otherwise null.
2. **Salary phrasing (R4)**: `salary came in`, `payday`, `paycheck`, `salary credited`, `monthly
   salary from <employer> credited`, `direct deposit` -> income with target null even when an
   employer is named. A target appears only in the `X paid ...` form (`company paid oct 5k` ->
   `company`, `boss paid bonus 500` -> `boss`).
3. If several people/businesses plausibly are the counterparty -> status uncertain.
4. Sale / marketplace income (`craigslist sale 150`, `sold bike on olx 120`): the platform is the
   target (`craigslist`, `olx`); a named buyer person is the target instead if present.
