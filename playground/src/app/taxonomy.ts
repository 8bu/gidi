import type { TxType } from "./types"

/** +1 money comes in, -1 money goes out, 0 neutral. Names and hints live in `i18n.ts`. */
export const TYPE_SIGN: Record<TxType, 1 | -1 | 0> = {
  expense: -1,
  income: 1,
  borrow: 1,
  lend: -1,
  repayment_in: 1,
  repayment_out: -1,
  transfer: 0,
  refund: 1,
}
