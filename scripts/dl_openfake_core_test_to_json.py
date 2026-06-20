from pathlib import Path
import json
import hashlib
import re
from tqdm import tqdm
from datasets import load_dataset


OUT_ROOT = Path("/work/cvcs2026/resnet_gang/datasets/json_standardized")

IMG_ROOT = OUT_ROOT / "images" / "openfake"
JSON_PATH = OUT_ROOT / "json" / "openfake.jsonl"

IMG_ROOT.mkdir(parents=True, exist_ok=True)
JSON_PATH.parent.mkdir(parents=True, exist_ok=True)


def safe_name(x):
    x = str(x).strip()
    x = re.sub(r"[^a-zA-Z0-9._-]+", "_", x)
    return x[:120] if x else "unknown"


ds = load_dataset(
    "ComplexDataLab/OpenFake",
    "core",
    split="test",
    streaming=True
)

n = 0

with open(JSON_PATH, "w") as jf:

    for row in tqdm(ds, desc="OpenFake core/test"):

        img = row["image"]

        raw_label = str(row["label"]).lower().strip()

        if raw_label == "real":
            label = 0
        elif raw_label == "fake":
            label = 1
        else:
            raise ValueError(f"Label non riconosciuta: {row['label']}")

        source_model = (
            row.get("generator")
            or row.get("model")
            or row.get("source")
            or "unknown"
        )

        if label == 0:
            generator = "real"
        else:
            generator = source_model

        source_id = str(
            row.get("id")
            or row.get("source_id")
            or f"openfake_core_test_{n}"
        )

        unique_key = f"openfake|core|test|{source_model}|{label}|{source_id}|{n}"
        h = hashlib.sha1(unique_key.encode()).hexdigest()[:16]

        out_img = (
            IMG_ROOT
            / "test"
            / safe_name(generator)
            / str(label)
            / f"{h}.png"
        )

        out_img.parent.mkdir(parents=True, exist_ok=True)

        if not out_img.exists() or out_img.stat().st_size == 0:
            img.convert("RGB").save(out_img)

        rec = {
            "image_path": str(out_img),
            "label": int(label),
            "generator": str(generator),
            "source_model": str(source_model),
            "dataset": "openfake",
            "subset": "core",
            "split": "test",
            "source_id": source_id
        }

        jf.write(json.dumps(rec) + "\n")

        n += 1

print(f"Done: {n}")