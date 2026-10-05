#!/usr/bin/env python
r"""Two independent LLM labellers + adjudication for the 200-note test set ``human-value-02``.

The user approved LLM labels for this test set (an exception to "never auto-generate human
labels"). They are **LLM labels, never human labels**; the set is never trained on.

Usage (each step is idempotent; raw model output is kept under ``llm-labelling/raw``):

    uv run python scripts/label_human_value_02_llm.py prompts      # rules.md, task prompts, chunks
    uv run python scripts/label_human_value_02_llm.py run a        # labeller A: Claude CLI
    uv run python scripts/label_human_value_02_llm.py normalize    # a-labels.jsonl, b-labels.jsonl
    uv run python scripts/label_human_value_02_llm.py agree        # agreement.json
    uv run python scripts/label_human_value_02_llm.py adjudicate   # third Claude pass
    uv run python scripts/label_human_value_02_llm.py finalize     # labels.jsonl + provenance
    uv run python scripts/label_human_value_02_llm.py manifest     # manifest-labels.json

Labeller B is an independent LLM labeller (B); its raw output under ``raw/b`` is kept as
received and ``normalize`` reads it. Isolation: each labeller ran in an EMPTY working
directory (no repo, no git, no tools) and saw only ``rules.md`` (the annotation rules
copied from the docs and the Quet schema), ``task.md`` and the ``{id, text}`` of its chunk of
notes. Neither labeller sees the other's labels, the ``strata`` of the queue, training data or
any model prediction. Only the adjudicator sees both labels (anonymised as X / Y in a hashed
order) for the notes on which the labellers disagree.

``normalize`` repairs only mechanical offset errors: when ``text[start:end]`` differs from the
span text it recomputes the offsets by locating that exact text in the note (nearest occurrence
to the claimed start); the chosen words are never changed, and a span text that does not occur
in the note stays an error.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import tempfile
import unicodedata
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
DIR = ROOT / "datasets/annotation-v3/human-value-02"
WORK = DIR / "llm-labelling"
RAW = WORK / "raw"
QUEUE_ALL = DIR / "review-queue-all.jsonl"
SCHEMA = ROOT / "configs/annotation-v3.quet.yaml"
CLAUDE = "claude"
CLAUDE_MODEL = "opus"
CLAUDE_EFFORT = "high"
CHUNK = 50

FIELDS = ("annotation_status", "type", "target", "value", "span_status", "note")

TASK = """# Task

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
"""

ADJ_TASK = """# Task

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
"""


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text("utf-8").splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
    )


def _cut(text: str, start: str, end: str | None = None) -> str:
    """``text`` from the heading ``start`` up to (not including) the heading ``end``."""
    i = text.index(start)
    j = text.index(end, i) if end else len(text)
    return text[i:j].rstrip() + "\n"


def rules_text() -> str:
    docs = ROOT / "docs"
    v1 = (docs / "annotation-v1.md").read_text("utf-8")
    v2 = _cut((docs / "annotation-v2.md").read_text("utf-8"), "# Annotation v2", "## Value pass")
    v3 = _cut((docs / "annotation-v3.md").read_text("utf-8"), "# Annotation v3", "## Re-label")
    guide = _cut((DIR / "annotation-guide.md").read_text("utf-8"), "## How to label a note")
    schema = SCHEMA.read_text("utf-8")
    parts = [
        "# Annotation rules (the only source of truth)\n",
        "## Part 1: annotation-v1 (transaction types and target spans)\n\n" + v1,
        "## Part 2: annotation-v2 (value span convention, combined pass)\n\n" + v2,
        "## Part 3: annotation-v3 (debt-only notes; overrides parts 1 and 2 where they differ)\n\n"
        + v3,
        "## Part 4: combined-pass guide\n\n" + guide,
        "## Part 5: the combined schema (types, statuses, spans)\n\n```yaml\n" + schema + "```\n",
    ]
    return "\n\n".join(parts)


def notes(queue: Path = QUEUE_ALL) -> list[dict[str, str]]:
    return [{"id": r["id"], "text": r["text"]} for r in _read_jsonl(queue)]


def chunks(rows: list[dict[str, str]]) -> list[list[dict[str, str]]]:
    return [rows[i : i + CHUNK] for i in range(0, len(rows), CHUNK)]


def notes_block(rows: list[dict[str, str]]) -> str:
    return "# Notes\n\n" + "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)


def cmd_prompts(_: argparse.Namespace) -> None:
    WORK.mkdir(exist_ok=True)
    (WORK / "rules.md").write_text(rules_text(), encoding="utf-8")
    (WORK / "task.md").write_text(TASK, encoding="utf-8")
    (WORK / "adjudicate-task.md").write_text(ADJ_TASK, encoding="utf-8")
    for i, rows in enumerate(chunks(notes()), 1):
        _write_jsonl(WORK / f"notes-{i}.jsonl", rows)
    print(
        json.dumps(
            {
                n: _sha((WORK / n).read_bytes())
                for n in ("rules.md", "task.md", "adjudicate-task.md")
            }
        )
    )


def _prompt(task: str, body: str) -> str:
    return (WORK / "rules.md").read_text("utf-8") + "\n\n---\n\n" + task + "\n\n---\n\n" + body


def _run_claude(prompt: str, out: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="hv02-claude-") as cwd:
        proc = subprocess.run(
            [
                CLAUDE, "-p", "--model", CLAUDE_MODEL, "--effort", CLAUDE_EFFORT,
                "--tools", "", "--disable-slash-commands", "--setting-sources", "",
                "--strict-mcp-config", "--no-session-persistence", "--output-format", "json",
            ],  # fmt: skip
            input=prompt, capture_output=True, text=True, cwd=cwd, timeout=3000,
        )  # fmt: skip
    if proc.returncode:
        raise SystemExit(
            f"claude failed ({out.name}): {proc.stderr[-2000:] or proc.stdout[-2000:]}"
        )
    payload = json.loads(proc.stdout)
    out.with_suffix(".meta.json").write_text(
        json.dumps(
            {k: payload.get(k) for k in ("modelUsage", "usage", "total_cost_usd", "is_error")},
            ensure_ascii=False,
            indent=1,
        ),  # fmt: skip
        encoding="utf-8",
    )
    if payload.get("is_error"):
        raise SystemExit(f"claude error ({out.name}): {payload.get('result')}")
    out.write_text(payload["result"], encoding="utf-8")


def cmd_run(args: argparse.Namespace) -> None:
    runner = _run_claude
    out_dir = RAW / args.who
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    if args.missing:
        # notes of the current queue without a label from this labeller (the queue changed after
        # the first run: a note was swapped out by the leakage gate), as one extra chunk
        have = {r["id"] for r in _read_jsonl(DIR / f"{args.who}-labels.jsonl")}
        rows = [r for r in notes() if r["id"] not in have]
        if rows:
            _write_jsonl(WORK / "notes-extra.jsonl", rows)
            n = len(list(out_dir.glob("chunk-extra-*.txt"))) + 1
            body = notes_block(rows)
            jobs.append(
                (
                    _prompt((WORK / "task.md").read_text("utf-8"), body),
                    out_dir / f"chunk-extra-{n}.txt",
                )
            )
    else:
        for i, rows in enumerate(chunks(notes()), 1):
            out = out_dir / f"chunk-{i}.txt"
            if out.exists() and not args.force:
                continue
            jobs.append((_prompt((WORK / "task.md").read_text("utf-8"), notes_block(rows)), out))
    with ThreadPoolExecutor(max_workers=len(jobs) or 1) as pool:
        list(pool.map(lambda j: runner(*j), jobs))
    print(f"labeller {args.who}: ran {len(jobs)} chunk(s)")


def parse_jsonl(text: str) -> list[dict[str, Any]]:
    rows = []
    for line in text.splitlines():
        line = line.strip().rstrip(",")
        if line.startswith("{") and line.endswith("}"):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def fix_span(span: Any, text: str) -> tuple[Any, str | None]:
    """Mechanical offset repair. Returns ``(span, how)``; ``how`` is None when untouched."""
    if not isinstance(span, dict) or not isinstance(span.get("text"), str):
        return span, None
    word = _nfc(span["text"])
    start, end = span.get("start"), span.get("end")
    if (
        isinstance(start, int)
        and isinstance(end, int)
        and 0 <= start <= end <= len(text)
        and text[start:end] == word
        and word == span["text"]
    ):
        return span, None
    hits = [m.start() for m in re.finditer(re.escape(word), text)]
    if not hits:
        return span, "unfixable"
    claimed = start if isinstance(start, int) else 0
    best = min(hits, key=lambda h: (abs(h - claimed), h))
    return {"text": word, "start": best, "end": best + len(word)}, "fixed"


def normalise_label(row: dict[str, Any], text: str, stats: Counter[str]) -> dict[str, Any]:
    out = {k: row[k] for k in ("id", *FIELDS) if k in row}
    for field in ("target", "value"):
        if out.get(field) is None:
            out[field] = None
            continue
        out[field], how = fix_span(out[field], text)
        if how:
            stats[f"{field}_{how}"] += 1
    if out.get("annotation_status") == "skipped":
        out.pop("span_status", None)
    if "note" in out and not (isinstance(out["note"], str) and out["note"].strip()):
        out.pop("note")
    return out


def normalise(who: str) -> tuple[list[dict[str, Any]], Counter[str]]:
    queue = {r["id"]: r["text"] for r in _read_jsonl(QUEUE_ALL)}
    raw: dict[str, dict[str, Any]] = {}
    stats: Counter[str] = Counter()
    for path in sorted((RAW / who).glob("chunk-*.txt")):
        for row in parse_jsonl(path.read_text("utf-8")):
            rid = row.get("id")
            if rid not in queue:
                stats["unknown_id"] += 1
            elif rid in raw:
                stats["duplicate_id"] += 1
            else:
                raw[rid] = row
    stats["missing_id"] = len(set(queue) - set(raw))
    out = [normalise_label(raw[r], queue[r], stats) for r in queue if r in raw]
    return out, stats


def cmd_normalize(_: argparse.Namespace) -> None:
    report = {}
    for who in ("a", "b"):
        rows, stats = normalise(who)
        _write_jsonl(DIR / f"{who}-labels.jsonl", rows)
        report[who] = {"records": len(rows), **dict(stats)}
    print(json.dumps(report, indent=1))


def _span_text(label: dict[str, Any], field: str) -> str | None:
    span = label.get(field)
    return span["text"] if isinstance(span, dict) else None


def _value_status(label: dict[str, Any]) -> str | None:
    return (label.get("span_status") or {}).get("value")


def compare(a: dict[str, Any], b: dict[str, Any]) -> dict[str, bool]:
    return {
        "annotation_status": a.get("annotation_status") == b.get("annotation_status"),
        "type": a.get("type") == b.get("type"),
        "target_text": _span_text(a, "target") == _span_text(b, "target"),
        "value_text": _span_text(a, "value") == _span_text(b, "value"),
        "value_status": _value_status(a) == _value_status(b),
    }


def load_pair() -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], list[str]]:
    a = {r["id"]: r for r in _read_jsonl(DIR / "a-labels.jsonl")}
    b = {r["id"]: r for r in _read_jsonl(DIR / "b-labels.jsonl")}
    ids = [r["id"] for r in _read_jsonl(QUEUE_ALL)]
    missing = [i for i in ids if i not in a or i not in b]
    if missing:
        raise SystemExit(f"{len(missing)} ids lack a label from A or B")
    return a, b, ids


def cmd_agree(_: argparse.Namespace) -> None:
    a, b, ids = load_pair()
    per_field: Counter[str] = Counter()
    full = 0
    lenient = 0
    disagreements = []
    confusion: Counter[str] = Counter()
    for rid in ids:
        cmp = compare(a[rid], b[rid])
        per_field.update(k for k, ok in cmp.items() if ok)
        if all(cmp.values()):
            full += 1
        else:
            disagreements.append({"id": rid, "fields": [k for k, ok in cmp.items() if not ok]})
        both_uncertain = {a[rid].get("annotation_status"), b[rid].get("annotation_status")} == {
            "uncertain"
        }
        if all(cmp.values()) or both_uncertain:
            lenient += 1
        if not cmp["type"]:
            confusion[f"{a[rid].get('type')} | {b[rid].get('type')}"] += 1
    n = len(ids)
    status_a = Counter(a[i]["annotation_status"] for i in ids)
    status_b = Counter(b[i]["annotation_status"] for i in ids)
    report = {
        "records": n,
        "field_agreement": {
            k: {"agree": per_field[k], "rate": round(per_field[k] / n, 4)}
            for k in ("annotation_status", "type", "target_text", "value_text", "value_status")
        },  # fmt: skip
        "full_label_agreement": {"agree": full, "rate": round(full / n, 4)},
        "full_label_or_both_uncertain": {"agree": lenient, "rate": round(lenient / n, 4)},
        "full_label_rule": "annotation_status, type, target text, value text and value status "
        "all equal (notes and offsets ignored)",
        "disagreeing_records": n - full,
        "status_counts": {"a": dict(status_a), "b": dict(status_b)},
        "type_disagreements_a_vs_b": dict(confusion.most_common()),
        "disagreements": disagreements,
    }
    (DIR / "agreement.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in report.items() if k != "disagreements"}, indent=1))


def _anon(rid: str) -> bool:
    """True when A is shown as X (hashed, so neither labeller is systematically first)."""
    return int(_sha(f"hv02-adjudication:{rid}".encode())[:8], 16) % 2 == 0


def _shown(label: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in label.items() if k != "id"}


def cmd_adjudicate(args: argparse.Namespace) -> None:
    a, b, _ = load_pair()
    queue = {r["id"]: r["text"] for r in _read_jsonl(QUEUE_ALL)}
    todo = [
        d["id"] for d in json.loads((DIR / "agreement.json").read_text("utf-8"))["disagreements"]
    ]
    lines = []
    for rid in todo:
        x, y = (a[rid], b[rid]) if _anon(rid) else (b[rid], a[rid])
        lines.append(
            json.dumps(
                {"id": rid, "text": queue[rid], "X": _shown(x), "Y": _shown(y)}, ensure_ascii=False
            )
        )
    body = "# Notes with the two labels\n\n" + "\n".join(lines) + "\n"
    out = RAW / "adj" / "adjudication.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.force or not out.exists():
        _run_claude(_prompt((WORK / "adjudicate-task.md").read_text("utf-8"), body), out)
    print(f"adjudicated {len(todo)} note(s)")


def cmd_finalize(_: argparse.Namespace) -> None:
    a, b, ids = load_pair()
    queue = {r["id"]: r["text"] for r in _read_jsonl(QUEUE_ALL)}
    decisions = {
        r["id"]: r for r in parse_jsonl((RAW / "adj" / "adjudication.txt").read_text("utf-8"))
    }
    labels: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    stats: Counter[str] = Counter()
    for rid in ids:
        la, lb = a[rid], b[rid]
        if all(compare(la, lb).values()):
            label = dict(la)
            if "note" not in label and "note" in lb:
                label["note"] = lb["note"]
            provenance.append({"id": rid, "annotator": "llm", "provenance": "agreed"})
        else:
            dec = decisions.get(rid)
            if dec is None or dec.get("choice") not in {"X", "Y", "uncertain"}:
                raise SystemExit(f"{rid}: no valid adjudication")
            choice = dec["choice"]
            if choice == "uncertain":
                label = normalise_label({"id": rid, **dec["label"]}, queue[rid], stats)
                picked = "uncertain"
            else:
                x_is_a = _anon(rid)
                picked = "A" if (choice == "X") == x_is_a else "B"
                label = dict(la if picked == "A" else lb)
            provenance.append(
                {
                    "id": rid,
                    "annotator": "llm",
                    "provenance": "adjudicated",
                    "picked": picked,
                    "reason": dec.get("reason", ""),
                    "disagreed_fields": [k for k, ok in compare(la, lb).items() if not ok],
                }
            )
        labels.append({k: label[k] for k in ("id", *FIELDS) if k in label})
    _write_jsonl(DIR / "labels.jsonl", labels)
    _write_jsonl(DIR / "labels-provenance.jsonl", provenance)
    print(
        json.dumps(
            {
                "labels": len(labels),
                "provenance": dict(Counter(p["provenance"] for p in provenance)),
                "picked": dict(Counter(p.get("picked") for p in provenance if "picked" in p)),
                "status": dict(Counter(lbl["annotation_status"] for lbl in labels)),
            }
        )
    )


def cmd_manifest(_: argparse.Namespace) -> None:
    def sha(name: str) -> str:
        return _sha((DIR / name).read_bytes())

    prov = _read_jsonl(DIR / "labels-provenance.jsonl")
    labels = _read_jsonl(DIR / "labels.jsonl")
    agreement = json.loads((DIR / "agreement.json").read_text("utf-8"))
    raw = {f"{p.parent.name}/{p.name}": _sha(p.read_bytes()) for p in sorted(RAW.glob("*/*.txt"))}
    meta = {}
    for who in ("a", "adj"):
        models: Counter[str] = Counter()
        for p in sorted((RAW / who).glob("*.meta.json")):
            models.update((json.loads(p.read_text("utf-8")).get("modelUsage") or {}).keys())
        meta[who] = dict(models)
    n = len(labels)
    manifest = {
        "name": "human-value-02-labels",
        "annotation_version": "annotation-v3",
        "label_source": "LLM labels (two independent labellers + adjudication); NOT human labels",
        "user_approval_exception": (
            "The user explicitly approved LLM labels for this test set (they will not label it). "
            "This is an exception to the repo rule 'never auto-generate human labels'; these "
            "labels are never called human labels, and the exception applies to nothing else."
        ),
        "never_train": True,
        "notes": n,
        "queue": "review-queue-all.jsonl (review-queue.jsonl, 150 notes, bytes unchanged, + "
        "review-queue-b.jsonl, 50 notes)",
        "labellers": {
            "a": {
                "role": "independent labeller A",
                "tool": f"Claude Code CLI `claude -p --model {CLAUDE_MODEL} --effort "
                f"{CLAUDE_EFFORT} --tools '' --setting-sources ''`, empty working directory",
                "models": meta["a"],
                "chunks": "4 chunks of 50 notes + 1 extra chunk of 1 note (see swapped_note)",
            },
            "b": {
                "role": "independent LLM labeller (B)",
                "chunks": "4 chunks of 50 notes + 1 extra chunk of 1 note (see swapped_note)",
            },
            "adjudicator": {
                "role": "third pass over the notes where A and B disagree; sees both labels "
                "anonymised as X / Y (A is X for a note when sha256('hv02-adjudication:<id>') "
                "is even), picks X, Y or `uncertain`",
                "tool": "Claude Code CLI, same flags as A",
                "models": meta["adj"],
                "note": "same model family as labeller A; all 9 picks went to A's label",
            },
        },
        "isolation": "each labeller saw only rules.md + task.md + the {id, text} of its chunk: no "
        "strata, no training data, no model prediction, no repo, and never the other labeller's "
        "labels; the adjudicator saw both labels of the disagreeing notes only",
        "prompts_sha256": {
            "rules.md": sha("llm-labelling/rules.md"),
            "task.md": sha("llm-labelling/task.md"),
            "adjudicate-task.md": sha("llm-labelling/adjudicate-task.md"),
        },
        "prompt_template": "rules.md + '\\n\\n---\\n\\n' + task.md + '\\n\\n---\\n\\n' + "
        "'# Notes' block (one {id, text} JSON per line)",
        "rules_sources": [
            "docs/annotation-v1.md",
            "docs/annotation-v2.md (up to 'Value pass labels')",
            "docs/annotation-v3.md (up to 'Re-label batch')",
            "human-value-02/annotation-guide.md (from 'How to label a note')",
            "configs/annotation-v3.quet.yaml",
        ],
        "agreement": {
            k: agreement[k]
            for k in (
                "records",
                "field_agreement",
                "full_label_agreement",
                "full_label_or_both_uncertain",
                "full_label_rule",
                "disagreeing_records",
                "status_counts",
                "type_disagreements_a_vs_b",
            )
        },
        "final_labels": {
            "provenance": dict(Counter(p["provenance"] for p in prov)),
            "adjudicated_picks": dict(Counter(p["picked"] for p in prov if "picked" in p)),
            "status": dict(Counter(lbl["annotation_status"] for lbl in labels)),
            "type": dict(Counter(str(lbl["type"]) for lbl in labels)),
            "value_uncertain": sum(
                (lbl.get("span_status") or {}).get("value") == "uncertain" for lbl in labels
            ),
        },
        "offset_repairs": {
            "rule": "only mechanical: span text re-located in the note (nearest occurrence to the "
            "claimed start); chosen words never changed; 0 unfixable",
            "a": {"value": 2},
            "b": {"target": 5, "value": 6},
        },
        "swapped_note": {
            "withdrawn": "lì xì cháu Bin 100k",
            "replacement": "quà thôi nôi bé Gấu 350k",
            "why": "the withdrawn note is an exact text in contrast-04/source.jsonl (a candidate "
            "the contrast-04 build dropped against this set); it was labelled in the first run "
            "(raw chunk-4) and is not part of the set or of labels.jsonl",
        },
        "files_sha256": {
            f: sha(f)
            for f in (
                "review-queue.jsonl",
                "review-queue-b.jsonl",
                "review-queue-all.jsonl",
                "a-labels.jsonl",
                "b-labels.jsonl",
                "agreement.json",
                "labels.jsonl",
                "labels-provenance.jsonl",
            )
        },
        "raw_outputs_sha256": raw,
        "scripts": {
            "scripts/build_human_value_02_b.py": _sha(
                (ROOT / "scripts/build_human_value_02_b.py").read_bytes()
            ),
            "scripts/label_human_value_02_llm.py": _sha(Path(__file__).read_bytes()),
        },
        "validate_command": "uv run python scripts/validate_annotations.py "
        "datasets/annotation-v3/human-value-02/labels.jsonl --config "
        "configs/annotation-v3.quet.yaml --queue "
        "datasets/annotation-v3/human-value-02/review-queue-all.jsonl",
    }
    (DIR / "manifest-labels.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest["final_labels"]))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prompts").set_defaults(fn=cmd_prompts)
    run = sub.add_parser("run")
    run.add_argument("who", choices=["a"])
    run.add_argument("--force", action="store_true")
    run.add_argument("--missing", action="store_true", help="only notes without a label yet")
    run.set_defaults(fn=cmd_run)
    sub.add_parser("normalize").set_defaults(fn=cmd_normalize)
    sub.add_parser("agree").set_defaults(fn=cmd_agree)
    adj = sub.add_parser("adjudicate")
    adj.add_argument("--force", action="store_true")
    adj.set_defaults(fn=cmd_adjudicate)
    sub.add_parser("finalize").set_defaults(fn=cmd_finalize)
    sub.add_parser("manifest").set_defaults(fn=cmd_manifest)
    args = parser.parse_args(argv)
    args.fn(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
