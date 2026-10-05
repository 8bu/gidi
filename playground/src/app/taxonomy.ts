import type { TxType } from "./types"

export interface TypeMeta {
  /** Short Vietnamese name shown on a stack. */
  label: string
  /** One line explaining when a note belongs to this type. */
  hint: string
  /** +1 money comes in, -1 money goes out, 0 neutral. */
  sign: 1 | -1 | 0
}

export const TYPE_META: Record<TxType, TypeMeta> = {
  expense: {
    label: "Chi tiêu",
    hint: "Tiền bạn bỏ ra để mua hàng hay trả phí.",
    sign: -1,
  },
  income: {
    label: "Thu nhập",
    hint: "Tiền bạn kiếm được như lương, thưởng, bán hàng.",
    sign: 1,
  },
  borrow: {
    label: "Đi vay",
    hint: "Bạn mượn tiền của người khác, sẽ phải trả lại.",
    sign: 1,
  },
  lend: {
    label: "Cho vay",
    hint: "Bạn cho người khác mượn tiền, họ sẽ trả lại.",
    sign: -1,
  },
  repayment_in: {
    label: "Được trả nợ",
    hint: "Người khác trả lại khoản tiền bạn đã cho vay.",
    sign: 1,
  },
  repayment_out: {
    label: "Trả nợ",
    hint: "Bạn trả lại khoản tiền đã vay của người khác.",
    sign: -1,
  },
  transfer: {
    label: "Chuyển tiền",
    hint: "Chuyển tiền giữa các tài khoản hay ví của bạn.",
    sign: 0,
  },
  refund: {
    label: "Hoàn tiền",
    hint: "Tiền được trả lại sau khi hủy đơn hay trả hàng.",
    sign: 1,
  },
}
