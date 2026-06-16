from pathlib import Path
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
from collections import Counter
import time
import os

ROOT = Path("/work/cvcs2026/resnet_gang/datasets/DF40")
REAL_ROOT = ROOT / "real"
FAKE_ROOT = ROOT / "fake"

OUT = Path("/work/cvcs2026/resnet_gang/datasets/standardized/df40.parquet")
LOG = Path("/work/cvcs2026/resnet_gang/logs/df40_parquet_progress.txt")

OUT.parent.mkdir(parents=True, exist_ok=True)
LOG.parent.mkdir(parents=True, exist_ok=True)

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
BATCH_SIZE = 1000
LOG_EVERY = 5000

schema = pa.schema([
    ("image", pa.binary()),
    ("label", pa.int64()),
    ("generator", pa.string()),
    ("dataset", pa.string()),
    ("split", pa.string()),
    ("source_id", pa.string()),
])

def log(msg):
    elapsed = time.time() - START
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] +{elapsed:.1f}s {msg}"
    print(line, flush=True)
    with LOG.open("a") as f:
        f.write(line + "\n")
        f.flush()

def image_files(root: Path):
    return sorted([
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in EXTS
    ])

def get_real_generator(parts):
    if "ff_real" in parts:
        return "real_ff"
    if "cdf_real" in parts:
        return "real_cdf"
    return "real_unknown"

def get_fake_generator(parts):
    base_generator = parts[1].lower()

    if "ff" in parts:
        domain = "ff"
    elif "cdf" in parts:
        domain = "cdf"
    else:
        domain = "unknown"

    return f"{base_generator}_{domain}"

START = time.time()

if LOG.exists():
    LOG.unlink()

log("START convert_df40_to_parquet.py")
log(f"ROOT={ROOT}")
log(f"REAL_ROOT={REAL_ROOT}")
log(f"FAKE_ROOT={FAKE_ROOT}")
log(f"OUT={OUT}")

log("Scanning real files...")
real_files = image_files(REAL_ROOT)
log(f"Real files found: {len(real_files)}")

log("Scanning fake files...")
fake_files = image_files(FAKE_ROOT)
log(f"Fake files found: {len(fake_files)}")

total_expected = len(real_files) + len(fake_files)
log(f"Total expected rows: {total_expected}")

if OUT.exists():
    log(f"Removing existing output: {OUT}")
    OUT.unlink()

writer = pq.ParquetWriter(OUT, schema=schema, compression="snappy")

batch = []
total = 0
label_counter = Counter()
generator_counter = Counter()

def parquet_size():
    if OUT.exists():
        return f"{OUT.stat().st_size / (1024**3):.2f} GB"
    return "not_created"

def flush_batch():
    global batch, total

    if not batch:
        return

    df = pd.DataFrame(batch)
    table = pa.Table.from_pandas(df, schema=schema, preserve_index=False)
    writer.write_table(table)

    total += len(batch)
    batch.clear()

try:
    log("Adding real images...")
    for i, p in enumerate(tqdm(real_files, desc="Adding DF40 real", unit="img", ncols=100), start=1):
        rel = p.relative_to(ROOT)
        parts = rel.parts
        generator = get_real_generator(parts)

        batch.append({
            "image": p.read_bytes(),
            "label": 0,
            "generator": generator,
            "dataset": "df40",
            "split": "test",
            "source_id": str(rel),
        })

        label_counter[0] += 1
        generator_counter[generator] += 1

        if len(batch) >= BATCH_SIZE:
            flush_batch()

        if i % LOG_EVERY == 0:
            log(f"REAL progress: {i}/{len(real_files)} | written={total} | parquet_size={parquet_size()}")

    flush_batch()
    log(f"REAL done | written={total} | parquet_size={parquet_size()}")

    log("Adding fake images...")
    for i, p in enumerate(tqdm(fake_files, desc="Adding DF40 fake", unit="img", ncols=100), start=1):
        rel = p.relative_to(ROOT)
        parts = rel.parts
        generator = get_fake_generator(parts)

        batch.append({
            "image": p.read_bytes(),
            "label": 1,
            "generator": generator,
            "dataset": "df40",
            "split": "test",
            "source_id": str(rel),
        })

        label_counter[1] += 1
        generator_counter[generator] += 1

        if len(batch) >= BATCH_SIZE:
            flush_batch()

        if i % LOG_EVERY == 0:
            log(f"FAKE progress: {i}/{len(fake_files)} | written={total} | parquet_size={parquet_size()}")

    flush_batch()
    log(f"FAKE done | written={total} | parquet_size={parquet_size()}")

finally:
    writer.close()
    log("ParquetWriter closed")

elapsed = time.time() - START

log("DONE")
log(f"Saved: {OUT}")
log(f"Rows written: {total}")
log(f"Rows expected: {total_expected}")
log(f"Labels: {dict(label_counter)}")
log("Generators:")
for k, v in sorted(generator_counter.items()):
    log(f"  {k}: {v}")
log(f"Elapsed seconds: {round(elapsed, 2)}")