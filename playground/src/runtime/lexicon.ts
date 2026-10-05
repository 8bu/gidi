/**
 * Word lists of the deterministic value parser. Port of `gidi.value_parser.lexicon`.
 *
 * Every word is accent-folded lowercase (`tỷ` -> `ty`, `đồng` -> `dong`): the parser matches on
 * its folded shadow string, so accented and unaccented notes hit the same entries.
 */

/** A static word list; test membership with `inList`. */
export type WordList = Readonly<Record<string, true>>

const set = (...words: string[]): WordList =>
  Object.fromEntries(words.map((word) => [word, true as const]))
const union = (...lists: WordList[]): WordList => Object.assign({}, ...lists)

export const inList = (list: WordList, word: string): boolean =>
  Object.hasOwn(list, word)

// --- money units -------------------------------------------------------------------------
export const THOUSAND_LETTER = set("k")
export const THOUSAND_WORDS = set("nghin", "ngan")
export const MILLION = set("tr", "trieu")
export const BILLION = set("ty")
// Slang for 100k (xi, chai, lit) and a million (cu). Ambiguous with quantities, see SLANG_QTY.
export const SLANG = set("cu", "xi", "chai", "lit")
export const CURRENCY_WORDS = set("d", "vnd", "dong")
export const CURRENCY_SIGN = 0x20ab // ₫
// Units after which a half ("rưỡi") or a trailing digit can continue the amount (2 củ rưỡi).
export const CONTINUABLE = union(MILLION, BILLION, SLANG, THOUSAND_WORDS)
export const WORD_UNITS = set(
  "nghin",
  "ngan",
  "trieu",
  "ty",
  "cu",
  "chai",
  "lit",
  "xi"
)
export const HALF = set("ruoi")

// Slang unit followed by one of these is a quantity of goods, not money (5 lít xăng, 2 chai bia).
export const SLANG_QTY = set(
  "xang",
  "dau",
  "nhot",
  "nuoc",
  "bia",
  "ruou",
  "cafe",
  "coca",
  "pepsi",
  "gas",
  "khoai"
)
// Same, but the folded form is ambiguous with another word, so the accented spelling is required
// (`sữa` milk vs `sửa` repair, `trà` tea vs `trả` pay, `hành` onion vs `Hạnh`).
export const SLANG_QTY_ACCENTED = set(
  "sữa",
  "trà",
  "hành",
  "tỏi",
  "gừng",
  "cải"
)

// --- number words (for "bốn triệu", "hai trăm nghìn", "nửa củ") ----------------------------
const DIGIT_WORDS = set(
  "mot",
  "hai",
  "ba",
  "bon",
  "nam",
  "sau",
  "bay",
  "tam",
  "chin",
  "muoi"
)
export const NUMBER_WORDS = union(
  DIGIT_WORDS,
  set("tram", "linh", "le", "lam", "muoi")
)
export const HALF_PREFIX = set("nua")

// --- context that explains a number as something other than money --------------------------
// Word BEFORE the number: month, instalment, period, week, day, ...
export const PERIOD_BEFORE = set(
  "thang", "ky", "dot", "lan", "quy", "nam", "tuan", "ngay", "mung", "thu", "lop", "tang",
  "phong", "chia", "so", "hang"
) // prettier-ignore
// Of those, words that are also common given names: only an excluding context when lowercase.
export const NAME_LIKE = set("lan", "thu", "nam", "quy", "tang", "phong", "dot")
// Word AFTER the number: time units and counted things.
export const QUANTITY_AFTER = set(
  // time and periods
  "thang", "ngay", "tuan", "nam", "quy", "gio", "phut", "giay", "buoi", "hom", "dem",
  "lan", "dot", "ky", "tieng", "tuoi", "t", "h", "p",
  // counted things
  "nguoi", "em", "be", "to", "ly", "coc", "ve", "phan", "suat", "cuon", "quyen", "mon",
  "cai", "chiec", "con", "cay", "bo", "doi", "cap", "goi", "hop", "thung", "lon", "lang",
  "chi", "phong", "tang", "tiet", "bai", "trang", "chuyen", "cuoc", "vien", "mieng",
  "trai", "bat", "dia", "tui", "cuc", "thanh", "sp",
  // measures
  "kg", "kilo", "gam", "gram", "g", "mg", "can", "yen", "tan", "m", "km", "cm", "mm",
  "ml", "cc", "l", "m2", "m3", "inch", "gb", "tb", "mb"
) // prettier-ignore
export const WEEKDAY_WORD = "thu" // "thứ 6": a single digit 2-8 after it is a weekday
