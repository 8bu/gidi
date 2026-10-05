/**
 * Byte-level BPE tokenizer for the bundle's `tokenizer.json`. Port of `gidi.inference.tokenizer`
 * (the Hugging Face `tokenizers` crate configured for this model):
 *
 * - no normalizer (the caller applies NFC); offsets are code points of the string given;
 * - added special tokens (`<s>`, `<pad>`, ...) are matched literally in the text first
 *   (leftmost-longest) and bypass the pre-tokenizer;
 * - pre-tokenizer: `Split` with the Oniguruma regex ` ?[^(\s|[.,!?…。，、।۔،])]+` (behavior
 *   `Isolated`) then `ByteLevel` (no prefix space, no regex);
 * - BPE with the merge table (lowest rank first, leftmost on ties);
 * - `TemplateProcessing` `<s> A </s>` and right-side truncation to `maxLength` incl. specials.
 *
 * The regex is implemented by hand: the nested class `[^(\s|[.,!?…。，、।۔،])]` means "none of
 * `( ) |`, a Unicode White_Space char, or one of `. , ! ? … 。 ， 、 । ۔ ،`".
 */

import { toCodePoints } from "./text.ts"

export interface TokenizedText {
  ids: number[]
  /** Per-token `[start, end)` code point offsets in the tokenized string; `[0, 0]` for specials. */
  offsets: [number, number][]
  specialTokensMask: number[]
  truncated: boolean
}

interface AddedToken {
  id: number
  content: string
  cps: number[]
}

interface TokenizerJson {
  added_tokens: { id: number; content: string; special: boolean }[]
  pre_tokenizer: unknown
  post_processor: {
    type: string
    single: ({ SpecialToken: { id: string } } | { Sequence: { id: string } })[]
    special_tokens: Record<string, { ids: number[] }>
  }
  model: {
    type: string
    unk_token: string | null
    continuing_subword_prefix: string | null
    end_of_word_suffix: string | null
    fuse_unk: boolean
    byte_fallback: boolean
    ignore_merges: boolean
    dropout: number | null
    vocab: Record<string, number>
    merges: (string | [string, string])[]
  }
}

/** Oniguruma `\s` (Unicode White_Space) plus the class members of the Split pattern. */
const SPLIT_EXCLUDED = new Set<number>([
  0x28, // (
  0x29, // )
  0x7c, // |
  0x2e, // .
  0x2c, // ,
  0x21, // !
  0x3f, // ?
  0x2026, // …
  0x3002, // 。
  0xff0c, // ，
  0x3001, // 、
  0x0964, // ।
  0x06d4, // ۔
  0x060c, // ،
  0x09,
  0x0a,
  0x0b,
  0x0c,
  0x0d,
  0x20,
  0x85,
  0xa0,
  0x1680,
  0x2028,
  0x2029,
  0x202f,
  0x205f,
  0x3000,
])

function isWordChar(cp: number): boolean {
  if (SPLIT_EXCLUDED.has(cp)) return false
  return !(cp >= 0x2000 && cp <= 0x200a)
}

/** Byte-level BPE `bytes_to_unicode`: the printable stand-in character of each byte. */
const BYTE_TO_CHAR: string[] = (() => {
  const keep: number[] = []
  for (let b = 0x21; b <= 0x7e; b++) keep.push(b)
  for (let b = 0xa1; b <= 0xac; b++) keep.push(b)
  for (let b = 0xae; b <= 0xff; b++) keep.push(b)
  const table: string[] = new Array<string>(256)
  let extra = 0
  for (let b = 0; b < 256; b++) {
    table[b] = String.fromCodePoint(keep.includes(b) ? b : 256 + extra++)
  }
  return table
})()

function utf8Bytes(cp: number): number[] {
  if (cp < 0x80) return [cp]
  if (cp < 0x800) return [0xc0 | (cp >> 6), 0x80 | (cp & 0x3f)]
  if (cp < 0x10000)
    return [0xe0 | (cp >> 12), 0x80 | ((cp >> 6) & 0x3f), 0x80 | (cp & 0x3f)]
  return [
    0xf0 | (cp >> 18),
    0x80 | ((cp >> 12) & 0x3f),
    0x80 | ((cp >> 6) & 0x3f),
    0x80 | (cp & 0x3f),
  ]
}

interface MergeEntry {
  rank: number
  newId: number
}

interface BpeSymbol {
  id: number
  prev: number
  next: number
  /** 0 once merged into its left neighbour. */
  length: number
  /** Offsets of the first/last original code point this symbol covers. */
  start: number
  end: number
}

interface Candidate {
  pos: number
  rank: number
  newId: number
}

/** Binary min-heap on `(rank, pos)`, like the Rust BPE merge queue. */
class MergeQueue {
  private readonly items: Candidate[] = []

  private static less(a: Candidate, b: Candidate): boolean {
    return a.rank !== b.rank ? a.rank < b.rank : a.pos < b.pos
  }

  push(item: Candidate): void {
    const items = this.items
    let i = items.length
    items.push(item)
    while (i > 0) {
      const parent = (i - 1) >> 1
      if (!MergeQueue.less(items[i], items[parent])) break
      ;[items[i], items[parent]] = [items[parent], items[i]]
      i = parent
    }
  }

  pop(): Candidate | undefined {
    const items = this.items
    if (items.length === 0) return undefined
    const top = items[0]
    const last = items.pop()!
    if (items.length > 0) {
      items[0] = last
      let i = 0
      for (;;) {
        const l = 2 * i + 1
        const r = l + 1
        let m = i
        if (l < items.length && MergeQueue.less(items[l], items[m])) m = l
        if (r < items.length && MergeQueue.less(items[r], items[m])) m = r
        if (m === i) break
        ;[items[i], items[m]] = [items[m], items[i]]
        i = m
      }
    }
    return top
  }
}

export class BundleTokenizer {
  readonly maxLength: number
  private readonly vocab: Map<string, number>
  private readonly merges = new Map<number, MergeEntry>()
  private readonly vocabSize: number
  private readonly unkId: number | null
  private readonly added: AddedToken[]
  private readonly bosId: number
  private readonly eosId: number

  constructor(tokenizerJson: unknown, maxLength: number) {
    const tj = tokenizerJson as TokenizerJson
    this.maxLength = maxLength
    BundleTokenizer.assertSupported(tj)

    this.vocab = new Map(Object.entries(tj.model.vocab))
    this.vocabSize = Math.max(...this.vocab.values()) + 1
    this.unkId =
      tj.model.unk_token === null
        ? null
        : (this.vocab.get(tj.model.unk_token) ?? null)
    tj.model.merges.forEach((merge, rank) => {
      const [a, b] = typeof merge === "string" ? merge.split(" ") : merge
      const idA = this.vocab.get(a)
      const idB = this.vocab.get(b)
      const newId = this.vocab.get(a + b)
      if (idA === undefined || idB === undefined || newId === undefined) {
        throw new Error(
          `tokenizer.json: merge ${rank} (${a} ${b}) is not in the vocabulary`
        )
      }
      this.merges.set(idA * this.vocabSize + idB, { rank, newId })
    })

    this.added = tj.added_tokens
      .map((t) => ({
        id: t.id,
        content: t.content,
        cps: toCodePoints(t.content),
      }))
      .sort((x, y) => y.cps.length - x.cps.length)

    const specials = tj.post_processor.special_tokens
    const [bos, eos] = tj.post_processor.single.flatMap((item) =>
      "SpecialToken" in item ? [specials[item.SpecialToken.id].ids[0]] : []
    )
    this.bosId = bos
    this.eosId = eos
  }

  private static assertSupported(tj: TokenizerJson): void {
    const model = tj.model
    const template = tj.post_processor?.single?.map((i) =>
      "SpecialToken" in i ? "S" : "Sequence" in i ? "A" : "?"
    )
    const unsupported =
      model.type !== "BPE" ||
      model.dropout !== null ||
      model.continuing_subword_prefix ||
      model.end_of_word_suffix ||
      model.byte_fallback ||
      model.ignore_merges ||
      template?.join("") !== "SAS" ||
      JSON.stringify(tj.pre_tokenizer) !==
        JSON.stringify(SUPPORTED_PRE_TOKENIZER)
    if (unsupported)
      throw new Error(
        "tokenizer.json: configuration not supported by this port"
      )
  }

  encode(cps: readonly number[]): TokenizedText {
    // A lone surrogate becomes U+FFFD (also one code point, so offsets stay aligned).
    const text = cps.map((cp) => (cp >= 0xd800 && cp <= 0xdfff ? 0xfffd : cp))
    const budget = this.maxLength - 2
    const ids: number[] = []
    const offsets: [number, number][] = []

    const push = (id: number, start: number, end: number): boolean => {
      if (ids.length >= budget) return false
      ids.push(id)
      offsets.push([start, end])
      return true
    }

    let truncated = false
    let segmentStart = 0
    const flushSegment = (segmentEnd: number): boolean => {
      for (const [a, b] of this.preTokenize(text, segmentStart, segmentEnd)) {
        for (const [id, start, end] of this.bpe(text, a, b)) {
          if (!push(id, start, end)) return false
        }
      }
      return true
    }

    let i = 0
    let ok = true
    while (i < text.length && ok) {
      const match = this.addedTokenAt(text, i)
      if (match === null) {
        i++
        continue
      }
      ok = flushSegment(i) && push(match.id, i, i + match.cps.length)
      i += match.cps.length
      segmentStart = i
    }
    if (ok) ok = flushSegment(text.length)
    if (!ok) truncated = true

    return {
      ids: [this.bosId, ...ids, this.eosId],
      offsets: [[0, 0], ...offsets, [0, 0]],
      specialTokensMask: [1, ...ids.map(() => 0), 1],
      truncated,
    }
  }

  private addedTokenAt(text: readonly number[], at: number): AddedToken | null {
    outer: for (const token of this.added) {
      if (at + token.cps.length > text.length) continue
      for (let k = 0; k < token.cps.length; k++) {
        if (text[at + k] !== token.cps[k]) continue outer
      }
      return token
    }
    return null
  }

  /** `Split(regex, Isolated)`: matches and the gaps between them as `[start, end)` pieces. */
  private preTokenize(
    text: readonly number[],
    from: number,
    to: number
  ): [number, number][] {
    const pieces: [number, number][] = []
    let gapStart = from
    let i = from
    while (i < to) {
      let j = i
      if (text[j] === 0x20 && j + 1 < to && isWordChar(text[j + 1])) j++
      if (!isWordChar(text[j])) {
        i++
        continue
      }
      while (j < to && isWordChar(text[j])) j++
      if (gapStart < i) pieces.push([gapStart, i])
      pieces.push([i, j])
      i = j
      gapStart = j
    }
    if (gapStart < to) pieces.push([gapStart, to])
    return pieces
  }

  /** ByteLevel + BPE of one piece: `[id, start, end)` per token, offsets in code points. */
  private bpe(
    text: readonly number[],
    from: number,
    to: number
  ): [number, number, number][] {
    const symbols: BpeSymbol[] = []
    const out: [number, number, number][] = []
    for (let cp = from; cp < to; cp++) {
      for (const byte of utf8Bytes(text[cp])) {
        // A byte character missing from the vocabulary becomes `unk` (no fusing), or is dropped.
        const id = this.vocab.get(BYTE_TO_CHAR[byte]) ?? this.unkId
        if (id === null) continue
        const index = symbols.length
        symbols.push({
          id,
          prev: index - 1,
          next: index + 1,
          length: 1,
          start: cp,
          end: cp + 1,
        })
      }
    }
    if (symbols.length === 0) return out
    symbols[symbols.length - 1].next = -1

    const queue = new MergeQueue()
    const enqueue = (left: number): void => {
      const right = symbols[left].next
      if (right < 0) return
      const entry = this.merges.get(
        symbols[left].id * this.vocabSize + symbols[right].id
      )
      if (entry !== undefined)
        queue.push({ pos: left, rank: entry.rank, newId: entry.newId })
    }
    for (let k = 0; k + 1 < symbols.length; k++) enqueue(k)

    for (let top = queue.pop(); top !== undefined; top = queue.pop()) {
      const left = symbols[top.pos]
      if (left.length === 0 || left.next < 0) continue
      const right = symbols[left.next]
      const entry = this.merges.get(left.id * this.vocabSize + right.id)
      if (
        right.length === 0 ||
        entry === undefined ||
        entry.newId !== top.newId
      )
        continue
      left.id = top.newId
      left.length += right.length
      left.end = right.end
      left.next = right.next
      right.length = 0
      if (right.next >= 0) symbols[right.next].prev = top.pos
      if (left.prev >= 0) enqueue(left.prev)
      enqueue(top.pos)
    }

    for (const symbol of symbols) {
      if (symbol.length > 0) out.push([symbol.id, symbol.start, symbol.end])
    }
    return out
  }
}

const SUPPORTED_PRE_TOKENIZER = {
  type: "Sequence",
  pretokenizers: [
    {
      type: "Split",
      pattern: { Regex: " ?[^(\\s|[.,!?…。，、।۔،])]+" },
      behavior: "Isolated",
      invert: false,
    },
    {
      type: "ByteLevel",
      add_prefix_space: false,
      trim_offsets: true,
      use_regex: false,
    },
  ],
}
