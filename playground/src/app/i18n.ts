/**
 * Every user-facing string of the notes app, Vietnamese and English. No i18n library: a typed
 * dictionary per language, picked by `useI18n()` (see `language.ts`). Strings that carry a number
 * or a name are functions so each language keeps its own word order and plural.
 */

import type { Lang } from "@/lib/language"

import type { TxType } from "./types"

export interface TypeMeta {
  /** Short name shown on a stack. */
  label: string
  /** One line explaining when a note belongs to this type. */
  hint: string
}

export interface Strings {
  /** BCP 47 tag for `Intl` and `<html lang>`. */
  locale: string
  docTitle: string
  docDescription: string
  types: Record<TxType, TypeMeta>

  // Header
  menu: string
  totalsLabel: string
  income: string
  expense: string
  remaining: string
  totalsTitle(income: string, expense: string, net: string): string
  onDevice: string
  loadingModel: string
  loadingModelProgress(loaded: string, total: string): string
  loadOnce: string
  loadFailedShort: string
  loadFailed: string
  retry: string
  appearance: string
  themeLight: string
  themeDark: string
  themeSystem: string
  exportNotes: string
  importNotes: string
  about: string
  aboutHref: string
  lab: string
  github: string
  huggingface: string
  huggingfaceHref: string
  author: string
  privacy: string

  // Language and model
  languageGroup: string
  languageHeading: string
  languageNames: Record<Lang, string>
  /** Title of the header switch button for `lang`. */
  switchTo(lang: Lang, model: string): string
  earlyRelease: string
  /** Notes in the app that the *other* model classified. */
  /** One line: notes are stored separately per language. */
  notesPerLanguage: string

  // Model download (`1,5 MB` / `1.5 MB`)
  megabytes(bytes: number): string
  loadFailedGeneric: string

  // Notes app
  examples: readonly string[]
  emptyHint: string
  exported(count: number): string
  imported(count: number): string
  importNone: string
  importFailed: string
  exportFilePrefix: string

  // Composer
  newNote: string
  placeholder: string
  emptyNote: string
  modelNotRunning: string
  modelStillLoading: string
  readFailed: string
  enterToSend: string
  tapToSend: string
  downloading(progress: string | null): string
  reload: string
  loadFailedRetry: string
  reading: string
  send: string

  // Classification
  badType(type: string): string
  saveFailed(message: string): string

  // Stacks
  stackEmptyLabel(label: string): string
  stackLabel(label: string, count: number, total: string): string
  stackTitle(label: string, total: string): string
  stackEmpty: string
  undo: string
  editNote: string
  deleteNote: string
  unknownTarget: string
  seeReading: string
  edited: string
  noAmount: string
  originalReading: string
  typeField: string
  targetField: string
  amountField: string
  none: string
  restore: string
  editSaveFailed: string
  deleted: string
  deleteFailed: string
  targetsHeading: string
  clearFilter: string
  collapse: string
  showMore(count: number): string
  stackEmptyHint: string
  noFilterMatch: string
  notesCount(count: number): string
  notesHeading: string

  // Editor
  chooseType: string
  chooseTypeHint: string
  amountPlaceholder: string
  amountUnreadable: string
  amountOptional: string
  targetPlaceholder: string
  moveToStack(label: string): string
  cancel: string
  save: string

  // Import / export file errors
  notJson: string
  notGidiExport: string
  unsupportedVersion: string
  invalidRecord(index: number): string
}

const vi: Strings = {
  locale: "vi-VN",
  docTitle: "Gidi – Ghi chú chi tiêu tự phân loại, chạy trên máy",
  docDescription:
    "Gidi đọc ghi chú chi tiêu tiếng Việt như “ăn phở 45k”, tự phân loại 8 loại giao dịch, tìm người liên quan và số tiền. Chạy ngay trên máy bạn, miễn phí.",
  types: {
    expense: {
      label: "Chi tiêu",
      hint: "Tiền bạn bỏ ra để mua hàng hay trả phí.",
    },
    income: {
      label: "Thu nhập",
      hint: "Tiền bạn kiếm được như lương, thưởng, bán hàng.",
    },
    borrow: {
      label: "Đi vay",
      hint: "Bạn mượn tiền của người khác, sẽ phải trả lại.",
    },
    lend: {
      label: "Cho vay",
      hint: "Bạn cho người khác mượn tiền, họ sẽ trả lại.",
    },
    repayment_in: {
      label: "Được trả nợ",
      hint: "Người khác trả lại khoản tiền bạn đã cho vay.",
    },
    repayment_out: {
      label: "Trả nợ",
      hint: "Bạn trả lại khoản tiền đã vay của người khác.",
    },
    transfer: {
      label: "Chuyển tiền",
      hint: "Chuyển tiền giữa các tài khoản hay ví của bạn.",
    },
    refund: {
      label: "Hoàn tiền",
      hint: "Tiền được trả lại sau khi hủy đơn hay trả hàng.",
    },
  },

  menu: "Menu",
  totalsLabel: "Tổng hợp thu chi",
  income: "Thu",
  expense: "Chi",
  remaining: "Còn lại",
  totalsTitle: (income, expense, net) =>
    `Thu ${income}  ·  Chi ${expense}  ·  Còn lại ${net}`,
  onDevice: "Chạy trên thiết bị",
  loadingModel: "Đang tải mô hình",
  loadingModelProgress: (loaded, total) =>
    `Đang tải mô hình, ${loaded} / ${total}`,
  loadOnce: "Chỉ tải một lần, sau đó chạy ngay trên thiết bị",
  loadFailedShort: "Không tải được",
  loadFailed: "Không tải được mô hình",
  retry: "Thử lại",
  appearance: "Giao diện",
  themeLight: "Sáng",
  themeDark: "Tối",
  themeSystem: "Theo hệ thống",
  exportNotes: "Xuất ghi chú (JSON)",
  importNotes: "Nhập ghi chú (JSON)",
  about: "Giới thiệu",
  aboutHref: "/gioi-thieu",
  lab: "Phòng thử nghiệm",
  github: "Mã nguồn (GitHub)",
  huggingface: "Mô hình (Hugging Face)",
  huggingfaceHref: "https://huggingface.co/x8bu/gidi-finance",
  author: "Tác giả: 8bu.dev",
  privacy:
    "Mô hình chạy trên thiết bị. Ghi chú không rời khỏi máy. Chỉ đếm lượt dùng ẩn danh, không có nội dung, tên hay số tiền.",

  languageGroup: "Ngôn ngữ và mô hình",
  languageHeading: "Ngôn ngữ · mô hình",
  languageNames: { vi: "Tiếng Việt", en: "English" },
  switchTo: (lang, model) =>
    lang === "vi"
      ? `Chuyển sang tiếng Việt (${model})`
      : `Chuyển sang English (${model})`,
  earlyRelease: "bản thử nghiệm",
  notesPerLanguage: "Ghi chú được lưu riêng cho từng ngôn ngữ.",

  megabytes: (bytes) => `${(bytes / 1e6).toFixed(1).replace(".", ",")} MB`,
  loadFailedGeneric: "Không thể tải mô hình.",

  examples: [
    "mua sữa vinamilk hết 500k",
    "còn nợ chị Mai 700k",
    "quà sinh nhật bé Na 300k",
    "lương tháng 10 về 15tr",
    "chuyển khoản cho mẹ 2tr",
  ],
  emptyHint: "Chưa có ghi chú nào. Thử một ví dụ:",
  exported: (count) => `Đã xuất ${count} ghi chú`,
  imported: (count) => `Đã nhập ${count} ghi chú`,
  importNone: "Không có ghi chú mới trong tệp này.",
  importFailed: "Không nhập được tệp này.",
  exportFilePrefix: "gidi-ghi-chu",

  newNote: "Ghi chú mới",
  placeholder: "mua sữa vinamilk hết 500k\nmỗi dòng là một ghi chú",
  emptyNote: "Hãy viết một dòng ghi chú trước khi gửi.",
  modelNotRunning: "Mô hình chưa chạy được. Hãy thử tải lại.",
  modelStillLoading: "Mô hình đang tải. Gửi được ngay khi tải xong.",
  readFailed: "Không đọc được ghi chú này. Thử lại nhé.",
  enterToSend: "Enter để gửi, Shift + Enter xuống dòng",
  tapToSend: "Chạm Gửi để cất ghi chú",
  downloading: (progress) =>
    `Đang tải mô hình về máy bạn${progress === null ? "" : ` (${progress})`}. Bạn cứ viết sẵn, mọi thứ chạy ngay trên thiết bị.`,
  reload: "Tải lại",
  loadFailedRetry: "Không tải được mô hình. Thử lại",
  reading: "Đang đọc",
  send: "Gửi",

  badType: (type) => `Mô hình trả về loại không hợp lệ: ${type}`,
  saveFailed: (message) => `Không lưu được ghi chú: ${message}`,

  stackEmptyLabel: (label) => `${label}: chưa có ghi chú`,
  stackLabel: (label, count, total) =>
    `${label}: ${count} ghi chú, tổng ${total}`,
  stackTitle: (label, total) => `${label}: tổng ${total}`,
  stackEmpty: "Trống",
  undo: "Hoàn tác",
  editNote: "Sửa ghi chú",
  deleteNote: "Xóa ghi chú",
  unknownTarget: "Chưa rõ đối tượng",
  seeReading: "Xem mô hình đã đọc gì",
  edited: "đã sửa",
  noAmount: "Chưa có số tiền",
  originalReading: "Mô hình đã đọc ban đầu",
  typeField: "Loại",
  targetField: "Đối tượng",
  amountField: "Số tiền",
  none: "không có",
  restore: "Khôi phục",
  editSaveFailed: "Không lưu được chỉnh sửa.",
  deleted: "Đã xóa ghi chú",
  deleteFailed: "Không xóa được ghi chú.",
  targetsHeading: "Đối tượng",
  clearFilter: "Bỏ lọc",
  collapse: "Thu gọn",
  showMore: (count) => `Xem thêm ${count}`,
  stackEmptyHint:
    "Chồng này đang trống. Viết một ghi chú ở giữa màn hình để bắt đầu.",
  noFilterMatch: "Không có ghi chú nào khớp bộ lọc.",
  notesCount: (count) => `${count} ghi chú`,
  notesHeading: "Ghi chú",

  chooseType: "Chọn loại",
  chooseTypeHint: "Khi lưu, ghi chú sẽ bay sang chồng của loại mới.",
  amountPlaceholder: "500k, 1,2tr hoặc 500000",
  amountUnreadable: "Chưa đọc được số tiền. Thử 500k, 1,2tr hoặc 500000.",
  amountOptional: "Để trống nếu ghi chú không có số tiền.",
  targetPlaceholder: "Để trống nếu không có",
  moveToStack: (label) => `Khi lưu, ghi chú sẽ bay sang chồng “${label}”.`,
  cancel: "Hủy",
  save: "Lưu",

  notJson: "Tệp không phải JSON hợp lệ.",
  notGidiExport: "Tệp này không phải bản xuất của Gidi.",
  unsupportedVersion: "Phiên bản tệp xuất chưa được hỗ trợ.",
  invalidRecord: (index) => `Ghi chú thứ ${index} trong tệp không hợp lệ.`,
}

const en: Strings = {
  locale: "en-US",
  docTitle: "Gidi – Self-sorting spending notes that run on your device",
  docDescription:
    "Gidi reads short money notes like “venmo'd mom 50”, sorts them into 8 transaction types and finds the person involved. It runs on your device, free.",
  types: {
    expense: {
      label: "Spending",
      hint: "Money you spend on things or fees.",
    },
    income: {
      label: "Income",
      hint: "Money you earn: pay, bonuses, sales.",
    },
    borrow: {
      label: "Borrowed",
      hint: "You borrow money from someone and will pay it back.",
    },
    lend: {
      label: "Lent",
      hint: "You lend someone money and they will pay it back.",
    },
    repayment_in: {
      label: "Paid back to you",
      hint: "Someone pays back money you lent them.",
    },
    repayment_out: {
      label: "You paid back",
      hint: "You pay back money you borrowed.",
    },
    transfer: {
      label: "Transfer",
      hint: "Money moved between your own accounts or wallets.",
    },
    refund: {
      label: "Refund",
      hint: "Money returned after a cancelled order or a return.",
    },
  },

  menu: "Menu",
  totalsLabel: "Income and spending totals",
  income: "In",
  expense: "Out",
  remaining: "Net",
  totalsTitle: (income, expense, net) =>
    `In ${income}  ·  Out ${expense}  ·  Net ${net}`,
  onDevice: "Runs on your device",
  loadingModel: "Loading the model",
  loadingModelProgress: (loaded, total) =>
    `Loading the model, ${loaded} / ${total}`,
  loadOnce: "Downloaded once, then it runs right on your device",
  loadFailedShort: "Couldn't load",
  loadFailed: "Couldn't load the model",
  retry: "Try again",
  appearance: "Appearance",
  themeLight: "Light",
  themeDark: "Dark",
  themeSystem: "Match system",
  exportNotes: "Export notes (JSON)",
  importNotes: "Import notes (JSON)",
  about: "About",
  aboutHref: "/about",
  lab: "Playground",
  github: "Source code (GitHub)",
  huggingface: "Model (Hugging Face)",
  huggingfaceHref: "https://huggingface.co/x8bu/gidi-en",
  author: "Author: 8bu.dev",
  privacy:
    "The model runs on your device. Notes never leave it. Only anonymous usage counts are collected, never content, names or amounts.",

  languageGroup: "Language and model",
  languageHeading: "Language · model",
  languageNames: { vi: "Tiếng Việt", en: "English" },
  switchTo: (lang, model) =>
    lang === "vi"
      ? `Switch to Tiếng Việt (${model})`
      : `Switch to English (${model})`,
  earlyRelease: "early release",
  notesPerLanguage: "Notes are kept separately for each language.",

  megabytes: (bytes) => `${(bytes / 1e6).toFixed(1)} MB`,
  loadFailedGeneric: "Couldn't load the model.",

  examples: [
    "venmo'd mom 50",
    "bro paid me back 20",
    "got paid",
    "lent sam 30 for pizza",
    "uber to the airport 42.50",
  ],
  emptyHint: "No notes yet. Try an example:",
  exported: (count) => `Exported ${count} ${count === 1 ? "note" : "notes"}`,
  imported: (count) => `Imported ${count} ${count === 1 ? "note" : "notes"}`,
  importNone: "No new notes in this file.",
  importFailed: "Couldn't import this file.",
  exportFilePrefix: "gidi-notes",

  newNote: "New note",
  placeholder: "venmo'd mom 50\none note per line",
  emptyNote: "Write a note before sending.",
  modelNotRunning: "The model isn't running. Try reloading it.",
  modelStillLoading:
    "The model is still loading. You can send as soon as it's done.",
  readFailed: "Couldn't read this note. Try again.",
  enterToSend: "Enter to send, Shift + Enter for a new line",
  tapToSend: "Tap Send to file the note",
  downloading: (progress) =>
    `Downloading the model to your device${progress === null ? "" : ` (${progress})`}. Keep writing, everything runs on your device.`,
  reload: "Reload",
  loadFailedRetry: "Couldn't load the model. Try again",
  reading: "Reading",
  send: "Send",

  badType: (type) => `The model returned an invalid type: ${type}`,
  saveFailed: (message) => `Couldn't save the note: ${message}`,

  stackEmptyLabel: (label) => `${label}: no notes yet`,
  stackLabel: (label, count, total) =>
    `${label}: ${count} ${count === 1 ? "note" : "notes"}, total ${total}`,
  stackTitle: (label, total) => `${label}: total ${total}`,
  stackEmpty: "Empty",
  undo: "Undo",
  editNote: "Edit note",
  deleteNote: "Delete note",
  unknownTarget: "Unknown person",
  seeReading: "See what the model read",
  edited: "edited",
  noAmount: "No amount",
  originalReading: "What the model first read",
  typeField: "Type",
  targetField: "Person",
  amountField: "Amount",
  none: "none",
  restore: "Restore",
  editSaveFailed: "Couldn't save the edit.",
  deleted: "Note deleted",
  deleteFailed: "Couldn't delete the note.",
  targetsHeading: "People",
  clearFilter: "Clear filter",
  collapse: "Show less",
  showMore: (count) => `Show ${count} more`,
  stackEmptyHint:
    "This stack is empty. Write a note in the middle of the screen to start.",
  noFilterMatch: "No notes match the filter.",
  notesCount: (count) => `${count} ${count === 1 ? "note" : "notes"}`,
  notesHeading: "Notes",

  chooseType: "Choose a type",
  chooseTypeHint: "When you save, the note flies to the stack of the new type.",
  amountPlaceholder: "50, 12.50 or 2k",
  amountUnreadable: "Couldn't read that amount. Try 50, 12.50 or 2k.",
  amountOptional: "Leave empty if the note has no amount.",
  targetPlaceholder: "Leave empty if there is none",
  moveToStack: (label) =>
    `When you save, the note flies to the “${label}” stack.`,
  cancel: "Cancel",
  save: "Save",

  notJson: "The file is not valid JSON.",
  notGidiExport: "This file is not a Gidi export.",
  unsupportedVersion: "This export file version is not supported.",
  invalidRecord: (index) => `Note ${index} in the file is not valid.`,
}

export const STRINGS: Readonly<Record<Lang, Strings>> = { vi, en }
