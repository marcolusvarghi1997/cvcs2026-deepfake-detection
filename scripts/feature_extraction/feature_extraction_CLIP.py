import argparse
import json
from pathlib import Path

import numpy as np
import open_clip
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


# ============================================================
# ARGOMENTI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--case",
        dest="case",
        choices=["case1", "case2"],
        required=True,
        help="Caso sperimentale da elaborare",
    )

    parser.add_argument(
        "--split",
        dest="split_name",
        choices=["train", "val", "test"],
        required=True,
        help="Split da elaborare",
    )

    return parser.parse_args()


ARGS = parse_args()

CASE = ARGS.case
SPLIT_NAME = ARGS.split_name


# ============================================================
# CONFIG
# ============================================================

BATCH_SIZE = 64
NUM_WORKERS = 6
OVERWRITE = False


# ============================================================
# PATH
# ============================================================

JSONL_PATH = Path(
    f"/work/cvcs2026/resnet_gang/cases_jsons/"
    f"{CASE}/{SPLIT_NAME}.jsonl"
)

OUTPUT_DIR = Path(
    f"/work/cvcs2026/resnet_gang/outputs/"
    f"CLIP/features/{CASE}"
)

METADATA_PATH = OUTPUT_DIR / f"metadata_{SPLIT_NAME}.parquet"
FEATURES_PATH = OUTPUT_DIR / f"features_{SPLIT_NAME}.npy"


# ============================================================
# DATASET
# ============================================================

class JsonlImageDataset(Dataset):
    def __init__(self, items, transform):
        self.items = items
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        image_path = item["image_path"]

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image = self.transform(image)

        return {
            "image": image,
            "image_path": image_path,
            "label": int(item["label"]),
            "generator": item.get("generator") or "unknown",
            "source_model": item.get("source_model") or "unknown",
            "dataset": item.get("dataset") or "unknown",
            "source_split": item.get("split") or "unknown",
            "source_id": str(item.get("source_id", image_path)),
        }


def load_jsonl(jsonl_path):
    items = []

    with jsonl_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                items.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"JSON non valido alla riga {line_number}: "
                    f"{jsonl_path}"
                ) from error

    return items


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_features():
    if CASE not in {"case1", "case2"}:
        raise ValueError(f"Case non valido: {CASE}")

    if SPLIT_NAME not in {"train", "val", "test"}:
        raise ValueError(f"Split non valido: {SPLIT_NAME}")

    if not JSONL_PATH.is_file():
        raise FileNotFoundError(f"File non trovato: {JSONL_PATH}")

    if not OVERWRITE and (
        FEATURES_PATH.exists() or METADATA_PATH.exists()
    ):
        raise FileExistsError(
            "Output già esistente. Imposta OVERWRITE=True "
            "per sovrascriverlo.\n"
            f"{FEATURES_PATH}\n"
            f"{METADATA_PATH}"
        )

    items = load_jsonl(JSONL_PATH)

    if not items:
        raise ValueError(f"Il JSONL è vuoto: {JSONL_PATH}")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model, _, transform = open_clip.create_model_and_transforms(
        "ViT-L-14",
        pretrained="openai",
    )

    model.eval()
    model.to(device)

    dataset = JsonlImageDataset(items, transform)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
    )

    print(f"Device: {device}")
    print(f"Case: {CASE}")
    print(f"Split: {SPLIT_NAME}")
    print(f"Immagini: {len(dataset)}")

    all_features = []
    metadata_rows = []

    with torch.inference_mode():
        for batch in tqdm(loader, desc="Feature extraction"):
            images = batch["image"].to(
                device,
                non_blocking=True,
            )

            features = model.encode_image(images)

            features = F.normalize(
                features,
                p=2,
                dim=-1,
                eps=1e-12,
            )

            features = (
                features
                .cpu()
                .numpy()
                .astype(np.float32)
            )

            all_features.append(features)

            for i in range(len(features)):
                metadata_rows.append({
                    "image_path": batch["image_path"][i],
                    "label": int(batch["label"][i]),
                    "generator": batch["generator"][i],
                    "source_model": batch["source_model"][i],
                    "dataset": batch["dataset"][i],
                    "source_split": batch["source_split"][i],
                    "case": CASE,
                    "case_split": SPLIT_NAME,
                    "source_id": batch["source_id"][i],
                })

    X = np.concatenate(all_features, axis=0)
    metadata = pd.DataFrame(metadata_rows)

    if len(X) != len(dataset):
        raise RuntimeError(
            f"Numero feature errato: {len(X)} != {len(dataset)}"
        )

    if len(metadata) != len(X):
        raise RuntimeError(
            "Metadata e feature non sono allineati"
        )

    if not np.isfinite(X).all():
        raise RuntimeError(
            "Le feature contengono NaN o valori infiniti"
        )

    if not set(metadata["label"].unique()).issubset({0, 1}):
        raise ValueError(
            "Le label devono contenere solamente 0 e 1"
        )

    norms = np.linalg.norm(X, axis=1)

    if not np.allclose(norms, 1.0, atol=1e-4):
        raise RuntimeError(
            "Le feature non risultano normalizzate. "
            f"Norma minima: {norms.min():.6f}, "
            f"massima: {norms.max():.6f}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    np.save(FEATURES_PATH, X)
    metadata.to_parquet(METADATA_PATH, index=False)

    print(f"Feature: {FEATURES_PATH}")
    print(f"Metadata: {METADATA_PATH}")
    print(f"Shape: {X.shape}")
    print(metadata["label"].value_counts().sort_index())


if __name__ == "__main__":
    extract_features()