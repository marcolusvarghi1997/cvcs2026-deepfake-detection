#!/usr/bin/env python3

import argparse
import hashlib
import json
import os
import platform
import random
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import sklearn
import torch
import yaml
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF
from tqdm import tqdm


ALLOWED_DETECTORS = {
    "effort",
    "clip",
    "cnn_aug",
    "xception",
    "efficientnetb4",
    "f3net",
    "spsl",
    "srm",
}

PATH_FIELDS = (
    "path",
    "image_path",
    "filepath",
    "file_path",
    "filename",
    "image",
)

LABEL_FIELDS = (
    "label",
    "target",
    "class",
    "is_fake",
    "y",
    "gt",
    "ground_truth",
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--detector", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)

    parser.add_argument(
        "--deepfakebench-root",
        type=Path,
        default=Path(
            "/work/cvcs2026/resnet_gang/external/DeepfakeBench"
        ),
    )

    parser.add_argument(
        "--jsonl",
        type=Path,
        default=Path(
            "/work/cvcs2026/resnet_gang/datasets/"
            "json_standardized/json/openfake.jsonl"
        ),
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "/work/cvcs2026/resnet_gang/outputs/"
            "detectors_from_deepfakebench/detection_from_paper"
        ),
    )

    parser.add_argument("--image-root", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--overwrite", action="store_true")

    return parser.parse_args()


def sha256_file(path: Path):
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def find_field(
    record: dict[str, Any],
    candidates: tuple[str, ...],
):
    for field in candidates:
        if field in record and record[field] is not None:
            return field, record[field]

    raise KeyError(
        f"Nessun campo tra {candidates}. "
        f"Campi presenti: {sorted(record)}"
    )


def normalize_label(value: Any):
    if isinstance(value, (bool, np.bool_)):
        return int(value)

    if isinstance(value, (int, np.integer)):
        value = int(value)

        if value in (0, 1):
            return value

    if isinstance(value, (float, np.floating)):
        value = float(value)

        if value in (0.0, 1.0):
            return int(value)

    text = str(value).strip().lower()

    if text in {
        "0",
        "real",
        "authentic",
        "genuine",
        "natural",
        "pristine",
    }:
        return 0

    if text in {
        "1",
        "fake",
        "synthetic",
        "generated",
        "ai",
        "ai-generated",
        "deepfake",
    }:
        return 1

    raise ValueError(f"Label non riconosciuta: {value!r}")


def resolve_path(
    value: Any,
    jsonl_path: Path,
    image_root: Path | None,
):
    path = Path(str(value))

    if path.is_absolute():
        return path

    if image_root is not None:
        return image_root / path

    return jsonl_path.parent / path


def load_records(
    jsonl_path: Path,
    image_root: Path | None,
):
    records = []
    detected_path_field = None
    detected_label_field = None

    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                raw_record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"JSON non valido alla riga {line_number}: {exc}"
                ) from exc

            path_field, raw_path = find_field(
                raw_record,
                PATH_FIELDS,
            )

            label_field, raw_label = find_field(
                raw_record,
                LABEL_FIELDS,
            )

            image_path = resolve_path(
                raw_path,
                jsonl_path,
                image_root,
            )

            records.append(
                {
                    "row_id": len(records),
                    "image_path": str(image_path),
                    "label": normalize_label(raw_label),
                    "original_json": json.dumps(
                        raw_record,
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                }
            )

            detected_path_field = (
                detected_path_field or path_field
            )

            detected_label_field = (
                detected_label_field or label_field
            )

    if not records:
        raise RuntimeError(
            f"Nessun record trovato in {jsonl_path}"
        )

    return (
        records,
        detected_path_field,
        detected_label_field,
    )


class OpenFakeDataset(Dataset):
    def __init__(
        self,
        records,
        resolution,
        mean,
        std,
    ):
        self.records = records
        self.resolution = int(resolution)
        self.mean = list(mean)
        self.std = list(std)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        image_path = record["image_path"]

        image = cv2.imread(
            image_path,
            cv2.IMREAD_COLOR,
        )

        if image is None:
            raise RuntimeError(
                f"Impossibile leggere: {image_path}"
            )

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB,
        )

        image = cv2.resize(
            image,
            (
                self.resolution,
                self.resolution,
            ),
            interpolation=cv2.INTER_CUBIC,
        )

        image = TF.to_tensor(image)

        image = TF.normalize(
            image,
            mean=self.mean,
            std=self.std,
        )

        return {
            "image": image,
            "label": int(record["label"]),
            "row_id": int(record["row_id"]),
            "image_path": image_path,
            "original_json": record["original_json"],
        }


def collate_batch(batch):
    return {
        "image": torch.stack(
            [item["image"] for item in batch]
        ),
        "label": torch.tensor(
            [item["label"] for item in batch],
            dtype=torch.long,
        ),
        "row_id": torch.tensor(
            [item["row_id"] for item in batch],
            dtype=torch.long,
        ),
        "image_path": [
            item["image_path"] for item in batch
        ],
        "original_json": [
            item["original_json"] for item in batch
        ],
    }


def import_registry(root: Path):
    root = root.resolve()
    training_path = root / "training"

    if not training_path.is_dir():
        raise FileNotFoundError(training_path)

    os.chdir(root)

    sys.path.insert(
        0,
        str(training_path),
    )

    sys.path.insert(
        0,
        str(root),
    )

    import detectors  # noqa: F401
    from detectors import DETECTOR

    return DETECTOR


def extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in (
            "state_dict",
            "model_state_dict",
            "model",
            "net",
        ):
            if (
                key in checkpoint
                and isinstance(checkpoint[key], dict)
            ):
                checkpoint = checkpoint[key]
                break

    if not isinstance(checkpoint, dict):
        raise TypeError(
            "Formato checkpoint non riconosciuto"
        )

    state_dict = {}

    for key, value in checkpoint.items():
        if not torch.is_tensor(value):
            continue

        if key.startswith("module."):
            key = key[7:]

        state_dict[key] = value

    if not state_dict:
        raise RuntimeError(
            "Checkpoint senza state_dict valido"
        )

    return state_dict


def validate_config(config):
    if config.get("video_mode", False):
        raise ValueError(
            "Detector video non supportato"
        )

    if config.get("with_mask", False):
        raise ValueError(
            "Detector con maschere non supportato"
        )

    if config.get("with_landmark", False):
        raise ValueError(
            "Detector con landmark non supportato"
        )

    required = (
        "model_name",
        "resolution",
        "mean",
        "std",
    )

    missing = [
        key
        for key in required
        if key not in config
    ]

    if missing:
        raise KeyError(
            f"Campi mancanti nel config: {missing}"
        )


def get_model_class(registry, model_name):
    if hasattr(registry, "module_dict"):
        available = registry.module_dict

        if model_name not in available:
            raise KeyError(
                f"Model name non registrato: {model_name}. "
                f"Disponibili: {sorted(available)}"
            )

        return available[model_name]

    return registry[model_name]


def get_fake_scores(output):
    if not isinstance(output, dict):
        raise TypeError(
            "Output detector non valido"
        )

    if "prob" in output:
        scores = output["prob"]
    elif "logits" in output:
        logits = output["logits"]

        if logits.ndim == 2 and logits.shape[1] == 2:
            scores = torch.softmax(
                logits,
                dim=1,
            )[:, 1]
        else:
            scores = torch.sigmoid(
                logits.reshape(-1)
            )
    elif "cls" in output:
        logits = output["cls"]

        if logits.ndim == 2 and logits.shape[1] == 2:
            scores = torch.softmax(
                logits,
                dim=1,
            )[:, 1]
        else:
            scores = torch.sigmoid(
                logits.reshape(-1)
            )
    else:
        raise KeyError(
            f"Nessuno score trovato. Chiavi: "
            f"{sorted(output)}"
        )

    return scores.reshape(-1)


def compute_metrics(
    labels,
    predictions,
    scores,
):
    tn, fp, fn, tp = confusion_matrix(
        labels,
        predictions,
        labels=[0, 1],
    ).ravel()

    metrics = {
        "n_samples": int(len(labels)),
        "n_real": int(np.sum(labels == 0)),
        "n_fake": int(np.sum(labels == 1)),
        "accuracy": float(
            accuracy_score(
                labels,
                predictions,
            )
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                labels,
                predictions,
            )
        ),
        "precision": float(
            precision_score(
                labels,
                predictions,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                labels,
                predictions,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                labels,
                predictions,
                zero_division=0,
            )
        ),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }

    if len(np.unique(labels)) == 2:
        metrics["auroc"] = float(
            roc_auc_score(
                labels,
                scores,
            )
        )

        metrics["average_precision"] = float(
            average_precision_score(
                labels,
                scores,
            )
        )
    else:
        metrics["auroc"] = None
        metrics["average_precision"] = None

    return metrics


def save_predictions(
    dataframe,
    output_dir,
):
    parquet_path = (
        output_dir / "predictions.parquet"
    )

    try:
        dataframe.to_parquet(
            parquet_path,
            index=False,
        )

        return parquet_path.name

    except Exception as exc:
        csv_path = (
            output_dir / "predictions.csv"
        )

        dataframe.to_csv(
            csv_path,
            index=False,
        )

        (
            output_dir / "parquet_error.txt"
        ).write_text(
            str(exc),
            encoding="utf-8",
        )

        return csv_path.name


def main():
    args = parse_args()

    detector = args.detector.lower()

    if detector not in ALLOWED_DETECTORS:
        raise ValueError(
            f"Detector non consentito: {detector}. "
            f"Consentiti: {sorted(ALLOWED_DETECTORS)}"
        )

    config_path = args.config.resolve()
    weights_path = args.weights.resolve()
    jsonl_path = args.jsonl.resolve()
    repository_root = (
        args.deepfakebench_root.resolve()
    )

    for path in (
        config_path,
        weights_path,
        jsonl_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    with config_path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        config = yaml.safe_load(handle)

    validate_config(config)

    output_dir = (
        args.output_root.resolve()
        / detector
    )

    if (
        output_dir.exists()
        and any(output_dir.iterdir())
        and not args.overwrite
    ):
        raise FileExistsError(
            f"Output già esistente: {output_dir}"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA non disponibile"
        )

    device = torch.device("cuda")

    records, path_field, label_field = (
        load_records(
            jsonl_path,
            args.image_root,
        )
    )

    dataset = OpenFakeDataset(
        records=records,
        resolution=config["resolution"],
        mean=config["mean"],
        std=config["std"],
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_batch,
        persistent_workers=args.workers > 0,
    )

    registry = import_registry(
        repository_root
    )

    model_name = str(
        config["model_name"]
    )

    model_class = get_model_class(
        registry,
        model_name,
    )

    model = model_class(config)

    checkpoint = torch.load(
        weights_path,
        map_location="cpu",
    )

    state_dict = extract_state_dict(
        checkpoint
    )

    missing, unexpected = model.load_state_dict(
        state_dict,
        strict=False,
    )

    if missing:
        print(
            f"[WARNING] Missing keys: {missing}"
        )

    if unexpected:
        print(
            f"[WARNING] Unexpected keys: {unexpected}"
        )

    model.eval()
    model.to(device)

    for parameter in model.parameters():
        parameter.requires_grad_(False)

    score_batches = []
    label_batches = []
    row_batches = []
    image_paths = []
    original_json_rows = []

    start_time = time.time()

    with torch.inference_mode():
        for batch in tqdm(
            loader,
            desc=f"{detector} OpenFake",
            unit="batch",
        ):
            images = batch["image"].to(
                device,
                non_blocking=True,
            )

            labels = batch["label"].to(
                device,
                non_blocking=True,
            )

            data_dict = {
                "image": images,
                "label": labels,
                "mask": None,
                "landmark": None,
            }

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=args.amp,
            ):
                output = model(
                    data_dict,
                    inference=True,
                )

            scores = get_fake_scores(output)

            score_batches.append(
                scores.detach()
                .float()
                .cpu()
                .numpy()
                .astype(np.float64)
            )

            label_batches.append(
                batch["label"]
                .numpy()
                .astype(np.int64)
            )

            row_batches.append(
                batch["row_id"]
                .numpy()
                .astype(np.int64)
            )

            image_paths.extend(
                batch["image_path"]
            )

            original_json_rows.extend(
                batch["original_json"]
            )

    elapsed = time.time() - start_time

    scores = np.concatenate(
        score_batches
    )

    labels = np.concatenate(
        label_batches
    )

    row_ids = np.concatenate(
        row_batches
    )

    predictions = (
        scores >= args.threshold
    ).astype(np.int64)

    order = np.argsort(row_ids)

    scores = scores[order]
    labels = labels[order]
    row_ids = row_ids[order]
    predictions = predictions[order]

    image_paths = [
        image_paths[index]
        for index in order
    ]

    original_json_rows = [
        original_json_rows[index]
        for index in order
    ]

    metrics = compute_metrics(
        labels,
        predictions,
        scores,
    )

    metrics.update(
        {
            "detector": detector,
            "model_name": model_name,
            "threshold": float(
                args.threshold
            ),
            "elapsed_seconds": float(
                elapsed
            ),
            "samples_per_second": float(
                len(labels) / elapsed
            ),
        }
    )

    predictions_df = pd.DataFrame(
        {
            "row_id": row_ids,
            "image_path": image_paths,
            "label": labels,
            "fake_score": scores,
            "prediction": predictions,
            "correct": (
                predictions == labels
            ),
            "original_json": (
                original_json_rows
            ),
        }
    )

    prediction_file = save_predictions(
        predictions_df,
        output_dir,
    )

    metadata = {
        "detector": detector,
        "model_name": model_name,
        "protocol": (
            "DeepfakeBench pretrained checkpoint "
            "evaluated without retraining on OpenFake"
        ),
        "repository_root": str(
            repository_root
        ),
        "config": str(config_path),
        "config_sha256": sha256_file(
            config_path
        ),
        "weights": str(weights_path),
        "weights_sha256": sha256_file(
            weights_path
        ),
        "jsonl": str(jsonl_path),
        "jsonl_sha256": sha256_file(
            jsonl_path
        ),
        "path_field": path_field,
        "label_field": label_field,
        "resolution": int(
            config["resolution"]
        ),
        "mean": list(config["mean"]),
        "std": list(config["std"]),
        "batch_size": int(
            args.batch_size
        ),
        "workers": int(args.workers),
        "threshold": float(
            args.threshold
        ),
        "amp": bool(args.amp),
        "seed": int(args.seed),
        "prediction_file": (
            prediction_file
        ),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "sklearn": sklearn.__version__,
        "gpu": torch.cuda.get_device_name(
            0
        ),
    }

    (
        output_dir / "metrics.json"
    ).write_text(
        json.dumps(
            metrics,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    (
        output_dir / "run_metadata.json"
    ).write_text(
        json.dumps(
            metadata,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    (
        output_dir
        / "resolved_config.yaml"
    ).write_text(
        yaml.safe_dump(
            config,
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            metrics,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()