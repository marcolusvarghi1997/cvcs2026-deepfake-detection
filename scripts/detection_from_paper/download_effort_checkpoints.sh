#!/bin/bash
set -u

OUT_DIR="/work/cvcs2026/resnet_gang/external/Effort-AIGI-Detection/checkpoints"
COOKIE="/work/cvcs2026/resnet_gang/cookies_drive.txt"
LOG_DIR="/work/cvcs2026/resnet_gang/logs"
STATUS="$LOG_DIR/effort_checkpoints_wget_status.txt"

mkdir -p "$OUT_DIR" "$LOG_DIR"
cd "$OUT_DIR"
: > "$STATUS"

FILES=(
    "effort_clip_L14_trainOn_sdv14.pth 1UXf1hC9FC1yV93uKwXSkdtepsgpIAU9d"
    "effort_clip_L14_trainOn_chameleon.pth 1GlJ1y4xmTdqV0FfIcyBwNNU6cQird9DR"
)

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$STATUS"
}

download_one() {
    local name="$1"
    local id="$2"

    if [ -s "$name" ]; then
        log "SKIP $name already exists"
        return 0
    fi

    local tmp="${name}.download"
    local tmp_html="${name}.html"

    rm -f "$tmp" "$tmp_html" "$name"

    log "FETCH $name id=$id"

    wget \
        --load-cookies "$COOKIE" \
        --save-cookies "$COOKIE" \
        --keep-session-cookies \
        -O "$tmp" \
        "https://drive.google.com/uc?export=download&id=${id}"

    if [ $? -ne 0 ]; then
        log "FAIL initial request for $name"
        rm -f "$tmp"
        return 1
    fi

    # Se Google ha restituito direttamente il checkpoint anziché HTML.
    if ! grep -qiE '<html|<!doctype html|name="confirm"' "$tmp"; then
        mv "$tmp" "$name"

        if [ -s "$name" ]; then
            local size
            size=$(du -h "$name" | cut -f1)
            log "OK direct download $name size=$size"
            return 0
        fi

        log "FAIL empty direct download for $name"
        rm -f "$name"
        return 1
    fi

    mv "$tmp" "$tmp_html"

    local confirm
    local uuid
    local at

    confirm=$(
        grep -o 'name="confirm" value="[^"]*"' "$tmp_html" |
        head -1 |
        sed 's/.*value="//;s/"//'
    )

    uuid=$(
        grep -o 'name="uuid" value="[^"]*"' "$tmp_html" |
        head -1 |
        sed 's/.*value="//;s/"//'
    )

    at=$(
        grep -o 'name="at" value="[^"]*"' "$tmp_html" |
        head -1 |
        sed 's/.*value="//;s/"//'
    )

    if [ -z "$confirm" ]; then
        log "FAIL confirm token extraction for $name"
        head -30 "$tmp_html" | tee -a "$STATUS"
        return 1
    fi

    log "DOWNLOAD confirmed file $name"

    if [ -n "$uuid" ] && [ -n "$at" ]; then
        download_url="https://drive.usercontent.google.com/download?id=${id}&export=download&authuser=0&confirm=${confirm}&uuid=${uuid}&at=${at}"
    else
        download_url="https://drive.usercontent.google.com/download?id=${id}&export=download&confirm=${confirm}"
    fi

    wget \
        --load-cookies "$COOKIE" \
        --save-cookies "$COOKIE" \
        --keep-session-cookies \
        --continue \
        -O "$name" \
        "$download_url"

    if [ $? -eq 0 ] && [ -s "$name" ]; then
        if grep -qiE '<html|<!doctype html' "$name"; then
            log "FAIL Google returned HTML instead of checkpoint for $name"
            mv "$name" "${name}.error.html"
            return 1
        fi

        local size
        size=$(du -h "$name" | cut -f1)

        log "OK $name size=$size"
        rm -f "$tmp_html"
        return 0
    fi

    log "FAIL download $name"
    rm -f "$name"
    return 1
}

log "START Effort checkpoint download"

for entry in "${FILES[@]}"; do
    read -r name id <<< "$entry"
    download_one "$name" "$id" || true
done

log "DONE"

ls -lh "$OUT_DIR"/*.pth 2>/dev/null | tee -a "$STATUS"
