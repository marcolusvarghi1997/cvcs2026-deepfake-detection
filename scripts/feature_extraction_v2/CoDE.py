import os
import json
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
import transformers
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#=================================================
#Da modificare 
PROTOCOL = "test2"
SPLIT_NAME = "test" # "train", "val", "test"
#=============================================


JSONL_PATH = Path(f"/work/cvcs2026/resnet_gang/test_jsons/{PROTOCOL}/{SPLIT_NAME}.jsonl")

OUTPUT_DIR = Path(f"/work/cvcs2026/resnet_gang/results/CoDE/features/{PROTOCOL}")

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

        img = Image.open(image_path).convert("RGB")
        img = self.transform(img)

        return {
            "image": img,
            "image_path": image_path,
            "label": item["label"],
            "generator": item.get("generator"),
            "dataset": item.get("dataset"),
            "source_split": item.get("split"),
            "protocol_split": SPLIT_NAME,
            "protocol": PROTOCOL,
            "source_id": item.get("source_id", image_path),
        }

class VITContrastiveHF(nn.Module):
    def __init__(self, repo_name):
        super().__init__()
        self.model = transformers.AutoModel.from_pretrained(
            repo_name,
            add_pooling_layer=False
        )
        self.model.pooler = nn.Identity()

    def forward(self, x):
        out = self.model(x)
        features = out.last_hidden_state[:, 0, :]
        return features


def load_jsonl(jsonl_path):
    items = []
    with open(jsonl_path, "r") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items


def test_on_jsonl(jsonl_path, repo="aimagelab/CoDE"):

    if not os.path.isfile(jsonl_path):
        print(f"[Errore] File non trovato: {jsonl_path}")
        return

    model = VITContrastiveHF(repo_name=repo)
    model.eval()
    model.to(device)

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

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

    with torch.no_grad():
        for batch in tqdm(loader):
            images = batch["image"].to(device, non_blocking=True)
            features = model(images)
            features_np = features.cpu().numpy().astype(np.float32)

            all_features.append(features_np)

            bs = features_np.shape[0]

            for i in range(bs):
                rows.append({
                    "image_path": batch["image_path"][i],
                    "label": int(batch["label"][i]),
                    "generator": batch["generator"][i],
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
        jsonl_path=JSONL_PATH,
        repo="aimagelab/CoDE"
    )
