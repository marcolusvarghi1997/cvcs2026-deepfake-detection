#!/bin/bash
set -euo pipefail

ZIP_DIR="/work/cvcs2026/resnet_gang/datasets/DF40/zips"
OUT_DIR="/work/cvcs2026/resnet_gang/datasets/DF40/extracted"
ANALYSIS_DIR="/work/cvcs2026/resnet_gang/datasets/DF40/analysis"

mkdir -p "$OUT_DIR" "$ANALYSIS_DIR"

cd "$ZIP_DIR"

for z in *.zip; do
  name="${z%.zip}"
  target="$OUT_DIR/$name"

  if [ -d "$target" ] && [ "$(find "$target" -type f | head -1)" ]; then
    echo "[SKIP] $z already extracted"
    continue
  fi

  echo "[EXTRACT] $z -> $target"
  mkdir -p "$target"
  unzip -q "$z" -d "$target"
  echo "[OK] $z"
done

echo "[ANALYZE]"
cd /work/cvcs2026/resnet_gang/datasets/DF40

find extracted -maxdepth 4 -type d | sort > "$ANALYSIS_DIR/dirs_maxdepth4.txt"
find extracted -type f | awk -F. '{print tolower($NF)}' | sort | uniq -c > "$ANALYSIS_DIR/extensions.txt"
find extracted -type f | sort | head -200 > "$ANALYSIS_DIR/sample_files.txt"
du -sh extracted > "$ANALYSIS_DIR/size.txt"

echo "[DONE]"
cat "$ANALYSIS_DIR/size.txt"
cat "$ANALYSIS_DIR/extensions.txt"
