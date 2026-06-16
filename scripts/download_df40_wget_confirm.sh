#!/bin/bash
set -u

OUT_DIR="/work/cvcs2026/resnet_gang/datasets/DF40/zips"
COOKIE="/work/cvcs2026/resnet_gang/cookies_drive.txt"
LOG_DIR="/work/cvcs2026/resnet_gang/logs"
STATUS="$LOG_DIR/df40_wget_status.txt"

mkdir -p "$OUT_DIR" "$LOG_DIR"
cd "$OUT_DIR"
: > "$STATUS"

FILES=(
"blendface.zip 1prCL6Rh1kvreBQJcQ409JR5XvD1Tj3xU"
"danet.zip 1CMC6GVVxjA0RlcihXCHtRC25JW22jHrv"
"e4s.zip 1lE1THBYba1iJfZBYFuHKV3bz2DDYFuwP"
"facedancer.zip 1Z69Zv02b2tTnIKL66MnVKHXWjWoZT1V_"
"fomm.zip 1lQ_BIXsHkBCu7b-QPyJMK3SV3gidF--W"
"inswap.zip 1SSzDaSqs-JcyRok-g7hlCsIEy-RxF50d"
"lia.zip 1OaW_xGmdObW9BnSfU5vOSB5Ma7UY8TgN"
"mobileswap.zip 1cwpdl5a9DxECdVDi8T91QpLUgMi_TZe8"
"pirender.zip 1ucCUQYGdgHrcvblL6z2zqi1ivpv-V9H6"
"sadtalker.zip 1noHwPXw9cX_UzKpPA_Iq-wm6D2E6nqc0"
"simswap.zip 1wqo00e58oNE-3PQFo0MVw39jQZPxFEKB"
"tpsm.zip 1LHCFNT3pElzuTp1ejs6X0nsUEH9gpFNC"
"wav2lip.zip 1vm1GnDl07BUxH15gqiBA9yIaY_-69Shr"
)

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$STATUS"
}

download_one() {
    name="$1"
    id="$2"

    if [ -s "$name" ]; then
        log "SKIP $name already exists"
        return 0
    fi

    tmp_html="${name}.html"
    rm -f "$tmp_html" "$name"

    log "FETCH HTML $name"

    wget \
      --load-cookies "$COOKIE" \
      -O "$tmp_html" \
      "https://drive.google.com/uc?export=download&id=${id}"

    confirm=$(grep -o 'name="confirm" value="[^"]*"' "$tmp_html" | sed 's/.*value="//;s/"//')
    uuid=$(grep -o 'name="uuid" value="[^"]*"' "$tmp_html" | sed 's/.*value="//;s/"//')
    at=$(grep -o 'name="at" value="[^"]*"' "$tmp_html" | sed 's/.*value="//;s/"//')

    if [ -z "$confirm" ] || [ -z "$uuid" ] || [ -z "$at" ]; then
        log "FAIL token extraction for $name"
        head -20 "$tmp_html" | tee -a "$STATUS"
        return 1
    fi

    log "DOWNLOAD $name"

    wget \
      --load-cookies "$COOKIE" \
      --continue \
      -O "$name" \
      "https://drive.usercontent.google.com/download?id=${id}&export=download&authuser=0&confirm=${confirm}&uuid=${uuid}&at=${at}"

    if [ $? -eq 0 ] && [ -s "$name" ]; then
        size=$(du -h "$name" | cut -f1)
        log "OK $name size=$size"
        rm -f "$tmp_html"
        return 0
    else
        log "FAIL download $name"
        rm -f "$name"
        return 1
    fi
}

log "START DF40 selected zip download"

for entry in "${FILES[@]}"; do
    name=$(echo "$entry" | awk '{print $1}')
    id=$(echo "$entry" | awk '{print $2}')
    download_one "$name" "$id" || true
done

log "DONE"
ls -lh "$OUT_DIR"/*.zip 2>/dev/null | tee -a "$STATUS"
