import os
import json
import torch
from PIL import Image
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import open_clip

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#=================================================
#Da modificare 
PROTOCOL = "test2" # "test1", "test2"
SPLIT_NAME = "val" # "train", "val", "test"
#=============================================


JSONL_PATH = Path(f"/work/cvcs2026/resnet_gang/test_jsons/{PROTOCOL}/{SPLIT_NAME}.jsonl")

OUTPUT_DIR = Path(f"/work/cvcs2026/resnet_gang/results/CLIP/features/{PROTOCOL}")

METADATA_PATH = OUTPUT_DIR / f"metadata_{SPLIT_NAME}.parquet"
FEATURES_PATH = OUTPUT_DIR / f"features_{SPLIT_NAME}.npy"


class JsonlImageDataset(Dataset):
    def __init__(self, items, transform):
        self.items = items
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        image_path = item["image_path"]

        with Image.open(image_path) as img:
            img = img.convert("RGB")
            img = self.transform(img)

        return {
            "image": img,
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
    with open(jsonl_path, "r") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items


def test_on_jsonl(jsonl_path):

    if not os.path.isfile(jsonl_path):
        print(f"[Errore] File non trovato: {jsonl_path}")
        return

    model, _, transform = open_clip.create_model_and_transforms(
        "ViT-L-14",
        pretrained="openai"
    )

    model.eval()
    model.to(device)

    dataset = load_jsonl(jsonl_path)

    print(f"Il dataset contiene {len(dataset)} elementi.")
    print("Esempio prima entry:")
    print(dataset[0])

    torch_dataset = JsonlImageDataset(dataset, transform)

    loader = DataLoader(
        torch_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=6,
        pin_memory=True,
    )

    all_features = []
    rows = []

    for batch in tqdm(loader):
        images = batch["image"].to(device, non_blocking=True)

        with torch.no_grad():
            features = model.encode_image(images)

            features = features / features.norm(
                dim=-1,
                keepdim=True
            )

            features_np = features.cpu().numpy().astype(np.float32)

        all_features.append(features_np)

        bs = features_np.shape[0]

        for i in range(bs):
            rows.append({
                "image_path": batch["image_path"][i],
                "label": int(batch["label"][i]),
                "generator": batch["generator"][i],
                "source_model": batch["source_model"][i],
                "dataset": batch["dataset"][i],
                "source_split": batch["source_split"][i],
                "protocol": PROTOCOL,
                "protocol_split": SPLIT_NAME,
                "source_id": batch["source_id"][i],
            })

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    X = np.concatenate(all_features, axis=0).astype(np.float32)
    df = pd.DataFrame(rows)

    assert len(df) == len(X), "Metadata e features non allineati!"

    df.to_parquet(METADATA_PATH, index=False)
    np.save(FEATURES_PATH, X)

    print(f"Salvato metadata in: {METADATA_PATH}")
    print(f"Salvate features in: {FEATURES_PATH}")
    print(f"Shape features: {X.shape}")
    print(df.head())


if __name__ == "__main__":

    test_on_jsonl(
        jsonl_path=JSONL_PATH
    )
