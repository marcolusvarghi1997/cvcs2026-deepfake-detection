from pathlib import Path
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
import time

ROOT = Path("/work/cvcs2026/resnet_gang/datasets/CNNDetection/repo/dataset/test")
OUT = Path("/work/cvcs2026/resnet_gang/datasets/standardized/cnndetection.parquet")
OUT.parent.mkdir(parents=True, exist_ok=True)

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
BATCH_SIZE = 1000

schema = pa.schema([
    ("image", pa.binary()),
    ("label", pa.int64()),
    ("generator", pa.string()),
    ("dataset", pa.string()),
    ("split", pa.string()),
    ("source_id", pa.string()),
])

print("Scanning files...")
files = [
    p for p in ROOT.rglob("*")
    if p.is_file() and p.suffix.lower() in EXTS
]

print(f"Found files: {len(files)}")
print(f"Output: {OUT}")

if OUT.exists():
    OUT.unlink()

writer = pq.ParquetWriter(OUT, schema=schema, compression="snappy")

batch = []
total = 0
start = time.time()

for p in tqdm(files, desc="Converting CNNDetection", unit="img", ncols=100):
    rel = p.relative_to(ROOT)
    parts = rel.parts

    if "0_real" in parts:
        label = 0
    elif "1_fake" in parts:
        label = 1
    else:
        continue

    batch.append({
        "image": p.read_bytes(),
        "label": label,
        "generator": parts[0].lower(),
        "dataset": "cnndetection",
        "split": "test",
        "source_id": str(rel),
    })

    if len(batch) >= BATCH_SIZE:
        df = pd.DataFrame(batch)
        table = pa.Table.from_pandas(df, schema=schema, preserve_index=False)
        writer.write_table(table)
        total += len(batch)
        batch.clear()

if batch:
    df = pd.DataFrame(batch)
    table = pa.Table.from_pandas(df, schema=schema, preserve_index=False)
    writer.write_table(table)
    total += len(batch)
    batch.clear()

writer.close()

elapsed = time.time() - start

print("DONE")
print("Saved:", OUT)
print("Rows:", total)
print("Elapsed seconds:", round(elapsed, 2))