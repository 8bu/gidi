export const PRESETS = [
  "cơm tấm 100",
  "bún bò 50k",
  "mượn chú hai 5 xị",
  "ăn 2 tô phở 70",
  "tiền điện tháng 10 hết 612",
  "trả góp kỳ 3 1tr5",
  "20/10 mua quà mẹ 500",
  "mua 3 vé 150k",
  "đổ 5 lít xăng 120k",
  "cho a Nam vay 1 triệu 20/10",
] as const

/**
 * The one preset whose value span is a documented model weakness: the right value is
 * `1 triệu`. The note is shown only when the runtime's actual value differs, so it
 * disappears if the model ever gets it right. Output is never altered.
 */
export const KNOWN_LIMITATION = {
  text: "cho a Nam vay 1 triệu 20/10",
  expectedValue: "1 triệu",
} as const
