#!/usr/bin/env bash
# Regenerates icons, social cards and favicon.ico into ../public from the SVG sources here.
# Needs: rsvg-convert, magick (ImageMagick), and the "Be Vietnam Pro" font installed.
set -euo pipefail
cd "$(dirname "$0")"
OUT=../public
mkdir -p "$OUT"

cp favicon.svg "$OUT/favicon.svg"

png() { # <src.svg> <size> <out.png>
  rsvg-convert -w "$2" -h "$2" "$1" | magick png:- -strip -define png:compression-level=9 "$3"
}
png icon-apple.svg 180 "$OUT/apple-touch-icon.png"
png icon-any.svg 192 "$OUT/icon-192.png"
png icon-any.svg 512 "$OUT/icon-512.png"
png icon-maskable.svg 512 "$OUT/icon-maskable-512.png"

for s in 16 32 48; do rsvg-convert -w $s -h $s favicon.svg -o "/tmp/gidi-fav-$s.png"; done
magick /tmp/gidi-fav-16.png /tmp/gidi-fav-32.png /tmp/gidi-fav-48.png "$OUT/favicon.ico"

card() { # <out.png> <T1> <T2> <L1> <L2> <L3>
  sed -e "s|@T1@|$2|" -e "s|@T2@|$3|" -e "s|@L1@|$4|" -e "s|@L2@|$5|" -e "s|@L3@|$6|" og.tpl.svg > "/tmp/gidi-og-$$.svg"
  rsvg-convert -w 1200 -h 630 "/tmp/gidi-og-$$.svg" \
    | magick png:- -strip -colors 128 -define png:compression-level=9 PNG8:"$1"
  rm -f "/tmp/gidi-og-$$.svg"
}
card "$OUT/og.png"    "Ghi chú chi tiêu tự phân loại," "chạy ngay trên máy" "Chi tiêu" "Thu nhập" "Cho vay"
card "$OUT/og-en.png" "On-device classifier for"      "Vietnamese money notes" "Expense" "Income" "Lend"
