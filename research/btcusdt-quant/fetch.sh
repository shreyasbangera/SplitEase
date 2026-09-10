#!/bin/bash
# Download Binance public klines from the S3 origin of data.binance.vision
BASE="https://s3-ap-northeast-1.amazonaws.com/data.binance.vision/data"
MARKET=$1   # spot | futures/um
SYM=$2; INT=$3; START=$4; END=$5   # YYYY-MM
OUT=$6
mkdir -p "$OUT"
d=$START
list=""
while [[ "$d" < "$END" || "$d" == "$END" ]]; do
  list="$list $d"
  d=$(date -u -d "${d}-01 +1 month" +%Y-%m)
done
echo $list | tr ' ' '\n' | xargs -P 12 -I{} sh -c \
  "curl -sf --max-time 90 -o '$OUT/${SYM}-${INT}-{}.zip' '$BASE/$MARKET/monthly/klines/$SYM/$INT/${SYM}-${INT}-{}.zip' || echo MISS {}"
