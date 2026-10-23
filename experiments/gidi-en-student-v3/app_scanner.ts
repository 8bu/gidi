// Runs the playground's English amount scan (`findEnglishValue`) over a JSON array of notes read
// from stdin and prints the matching `[start, end] | null` code-point spans as JSON.
// Used by scripts/evaluate_en_value.py for the regex-vs-model comparison.
import { findEnglishValue } from "../../playground/src/app/amount.ts"

const notes: string[] = JSON.parse(await Bun.stdin.text())
console.log(JSON.stringify(notes.map((note) => findEnglishValue(note).span)))
