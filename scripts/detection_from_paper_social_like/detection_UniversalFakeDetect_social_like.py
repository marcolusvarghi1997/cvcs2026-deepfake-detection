#!/usr/bin/env python3

import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn
import torch
import torchvision
from social_like import apply_social_like

from PIL import Image, ImageFile
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
from torchvision import transforms
from tqdm import tqdm


# ============================================================
# CONFIG
# ============================================================

# Repository ufficiale UniversalFakeDetect.
UFD_REPO = Path(
    "/work/cvcs2026/resnet_gang/external/UniversalFakeDetect"
)

# Checkpoint ufficiale del classificatore lineare.
CHECKPOINT_PATH = (
    UFD_REPO / "pretrained_weights/fc_weights.pth"
)

JSONL_PATH = Path(
    "/work/cvcs2026/resnet_gang/datasets/json_standardized/json/"
    "openfake.jsonl"
)

OUTPUT_DIR = Path(
    "/work/cvcs2026/resnet_gang/outputs/CLIP/"
    "detection_from_paper"
)

ARCHITECTURE = "CLIP:ViT-L/14"

BATCH_SIZE = 128
NUM_WORKERS = 8
PIN_MEMORY = True
REQUIRE_CUDA = True
OVERWRITE = False

# Impostare solo se i path nel JSONL sono relativi a una directory
# diversa da quella che contiene openfake.jsonl.
IMAGE_ROOT: Path | None = None

PATH_CANDIDATES = (
    "path",
    "image_path",
    "filepath",
    "file_path",
    "filename",
    "image",
)

LABEL_CANDIDATES = (
    "label",
    "target",
    "class",
    "is_fake",
    "y",
    "gt",
    "ground_truth",
)

# Normalizzazione ufficiale CLIP usata da UniversalFakeDetect.
CLIP_MEAN = [
    0.48145466,
    0.4578275,
    0.40821073,
]

CLIP_STD = [
    0.26862954,
    0.26130258,
    0.27577711,
]

# Preprocessing ufficiale di test di UniversalFakeDetect.
OFFICIAL_TRANSFORM = transforms.Compose(
    [
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=CLIP_MEAN,
            std=CLIP_STD,
        ),
    ]
)

ImageFile.LOAD_TRUNCATED_IMAGES = True


# ============================================================
# UTILITIES
# ============================================================

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def first_present(
    record: dict[str, Any],
    candidates: tuple[str, ...],
) -> tuple[str, Any]:
    for key in candidates:
        if key in record and record[key] is not None:
            return key, record[key]

    raise KeyError(
        f"Nessun campo trovato tra {candidates}. "
        f"Campi disponibili: {sorted(record.keys())}"
    )


def normalize_label(value: Any) -> int:
    """
    Ground truth comune:
    0 = Real
    1 = Fake
    """
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

    real_values = {
        "0",
        "real",
        "authentic",
        "genuine",
        "natural",
        "pristine",
    }

    fake_values = {
        "1",
        "fake",
        "synthetic",
        "generated",
        "ai",
        "ai-generated",
        "deepfake",
    }

    if text in real_values:
        return 0

    if text in fake_values:
        return 1

    raise ValueError(
        f"Label non riconosciuta o ambigua: {value!r}"
    )


def resolve_image_path(raw_path: Any) -> Path:
    path = Path(str(raw_path))

    if path.is_absolute():
        return path

    if IMAGE_ROOT is not None:
        return IMAGE_ROOT / path

    return JSONL_PATH.parent / path


def load_jsonl_records():
    records: list[dict[str, Any]] = []

    detected_path_field: str | None = None
    detected_label_field: str | None = None

    with JSONL_PATH.open(
        "r",
        encoding="utf-8",
    ) as handle:
        for line_number, line in enumerate(
            handle,
            start=1,
        ):
            line = line.strip()

            if not line:
                continue

            try:
                raw_record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"JSON non valido alla riga "
                    f"{line_number}: {exc}"
                ) from exc

            path_field, raw_path = first_present(
                raw_record,
                PATH_CANDIDATES,
            )

            label_field, raw_label = first_present(
                raw_record,
                LABEL_CANDIDATES,
            )

            if detected_path_field is None:
                detected_path_field = path_field

            if detected_label_field is None:
                detected_label_field = label_field

            image_path = resolve_image_path(raw_path)
            label = normalize_label(raw_label)

            records.append(
                {
                    "row_id": len(records),
                    "image_path": str(image_path),
                    "label": label,
                    "original_json": json.dumps(
                        raw_record,
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                }
            )

    if not records:
        raise RuntimeError(
            f"Nessun record trovato in {JSONL_PATH}"
        )

    assert detected_path_field is not None
    assert detected_label_field is not None

    return (
        records,
        detected_path_field,
        detected_label_field,
    )


# ============================================================
# DATASET
# ============================================================

class OpenFakeDataset(Dataset):
    def __init__(
        self,
        records: list[dict[str, Any]],
    ) -> None:
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        image_path = Path(record["image_path"])

        try:
            with Image.open(image_path) as image:
                image_tensor = OFFICIAL_TRANSFORM(
                    image.convert("RGB")
                )
        except Exception as exc:
            raise RuntimeError(
                f"Errore durante la lettura di "
                f"{image_path}: {exc}"
            ) from exc

        return (
            image_tensor,
            int(record["label"]),
            int(record["row_id"]),
            record["image_path"],
            record["original_json"],
        )


# ============================================================
# MODEL
# ============================================================

def import_official_model_factory():
    if not UFD_REPO.is_dir():
        raise FileNotFoundError(
            "Repository UniversalFakeDetect non trovato: "
            f"{UFD_REPO}"
        )

    repo_string = str(UFD_REPO)

    if repo_string not in sys.path:
        sys.path.insert(
            0,
            repo_string,
        )

    try:
        from models import get_model
    except Exception as exc:
        raise ImportError(
            "Impossibile importare models.get_model dal "
            "repository ufficiale UniversalFakeDetect."
        ) from exc

    return get_model


def load_official_model(
    device: torch.device,
):
    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(
            "Checkpoint ufficiale non trovato: "
            f"{CHECKPOINT_PATH}"
        )

    get_model = import_official_model_factory()

    model = get_model(
        ARCHITECTURE
    )

    # Il checkpoint ufficiale contiene i pesi del solo
    # classificatore lineare finale.
    state_dict = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
    )

    model.fc.load_state_dict(
        state_dict
    )

    model.eval()
    model.to(device)

    for parameter in model.parameters():
        parameter.requires_grad_(False)

    return model


def run_inference(
    model,
    loader: DataLoader,
    device: torch.device,
):
    feature_batches: list[np.ndarray] = []
    logit_batches: list[np.ndarray] = []
    score_batches: list[np.ndarray] = []
    label_batches: list[np.ndarray] = []
    row_id_batches: list[np.ndarray] = []

    image_paths: list[str] = []
    original_json_rows: list[str] = []

    with torch.inference_mode():
        for (
            images,
            labels,
            row_ids,
            batch_paths,
            batch_original_json,
        ) in tqdm(
            loader,
            desc="UniversalFakeDetect official inference",
            unit="batch",
        ):
            images = images.to(
                device,
                non_blocking=True,
            )

            # Feature ufficiali:
            # output del visual encoder CLIP ViT-L/14.
            features = model(
                images,
                return_feature=True,
            )

            # Classificatore lineare ufficiale.
            logits = model.fc(
                features
            ).flatten()

            # Score continuo crescente verso Fake.
            fake_scores = torch.sigmoid(
                logits
            )

            feature_batches.append(
                features.float().cpu().numpy()
            )

            logit_batches.append(
                logits.float().cpu().numpy()
            )

            score_batches.append(
                fake_scores.float().cpu().numpy()
            )

            label_batches.append(
                labels.numpy().astype(
                    np.int64
                )
            )

            row_id_batches.append(
                row_ids.numpy().astype(
                    np.int64
                )
            )

            image_paths.extend(
                batch_paths
            )

            original_json_rows.extend(
                batch_original_json
            )

    return (
        np.concatenate(
            feature_batches,
            axis=0,
        ).astype(np.float32),
        np.concatenate(
            logit_batches,
            axis=0,
        ).astype(np.float64),
        np.concatenate(
            score_batches,
            axis=0,
        ).astype(np.float64),
        np.concatenate(
            label_batches,
            axis=0,
        ),
        np.concatenate(
            row_id_batches,
            axis=0,
        ),
        image_paths,
        original_json_rows,
    )


# ============================================================
# METRICS
# ============================================================

def compute_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    scores: np.ndarray,
) -> dict[str, Any]:
    tn, fp, fn, tp = confusion_matrix(
        labels,
        predictions,
        labels=[0, 1],
    ).ravel()

    metrics: dict[str, Any] = {
        "classifier": "linear",
        "n_samples": int(
            len(labels)
        ),
        "n_real": int(
            np.sum(labels == 0)
        ),
        "n_fake": int(
            np.sum(labels == 1)
        ),
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
        "precision_fake": float(
            precision_score(
                labels,
                predictions,
                zero_division=0,
            )
        ),
        "recall_fake": float(
            recall_score(
                labels,
                predictions,
                zero_division=0,
            )
        ),
        "f1_fake": float(
            f1_score(
                labels,
                predictions,
                zero_division=0,
            )
        ),
        "auroc": float(
            roc_auc_score(
                labels,
                scores,
            )
        ),
        "average_precision": float(
            average_precision_score(
                labels,
                scores,
            )
        ),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }

    return metrics


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    if not JSONL_PATH.is_file():
        raise FileNotFoundError(
            f"JSONL non trovato: {JSONL_PATH}"
        )

    if REQUIRE_CUDA and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA non disponibile. "
            "Il job deve richiedere una GPU."
        )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    features_path = (
        OUTPUT_DIR / "features_test.npy"
    )

    metadata_path = (
        OUTPUT_DIR / "metadata_test.parquet"
    )

    predictions_parquet = (
        OUTPUT_DIR / "predictions_test.parquet"
    )

    predictions_csv = (
        OUTPUT_DIR / "predictions_test.csv"
    )

    metrics_csv = (
        OUTPUT_DIR / "metrics_test.csv"
    )

    metrics_json = (
        OUTPUT_DIR / "metrics_test.json"
    )

    run_info_json = (
        OUTPUT_DIR / "run_info.json"
    )

    outputs = (
        features_path,
        metadata_path,
        predictions_parquet,
        predictions_csv,
        metrics_csv,
        metrics_json,
        run_info_json,
    )

    if not OVERWRITE:
        existing = [
            str(path)
            for path in outputs
            if path.exists()
        ]

        if existing:
            raise FileExistsError(
                "Output già presenti. "
                "Imposta OVERWRITE=True per sovrascrivere:\n"
                + "\n".join(existing)
            )

    (
        records,
        path_field,
        label_field,
    ) = load_jsonl_records()

    missing_paths = [
        record["image_path"]
        for record in records
        if not Path(
            record["image_path"]
        ).is_file()
    ]

    if missing_paths:
        preview = "\n".join(
            missing_paths[:10]
        )

        raise FileNotFoundError(
            f"{len(missing_paths)} immagini non trovate. "
            f"Prime occorrenze:\n{preview}"
        )

    dataset = OpenFakeDataset(
        records
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(
            PIN_MEMORY
            and device.type == "cuda"
        ),
        persistent_workers=(
            NUM_WORKERS > 0
        ),
    )

    print(
        f"[INFO] repository: {UFD_REPO}"
    )
    print(
        f"[INFO] checkpoint: {CHECKPOINT_PATH}"
    )
    print(
        f"[INFO] JSONL: {JSONL_PATH}"
    )
    print(
        f"[INFO] samples: {len(records)}"
    )
    print(
        f"[INFO] path field: {path_field}"
    )
    print(
        f"[INFO] label field: {label_field}"
    )
    print(
        f"[INFO] architecture: {ARCHITECTURE}"
    )
    print(
        f"[INFO] classifier: linear"
    )
    print(
        f"[INFO] device: {device}"
    )

    model = load_official_model(
        device
    )

    (
        features,
        logits,
        scores,
        labels,
        row_ids,
        image_paths,
        original_json_rows,
    ) = run_inference(
        model=model,
        loader=loader,
        device=device,
    )

    if features.ndim != 2:
        raise ValueError(
            f"Feature non 2D: {features.shape}"
        )

    if features.shape[1] != 768:
        raise ValueError(
            "Dimensione feature inattesa per "
            f"CLIP ViT-L/14: {features.shape}"
        )

    if not np.isfinite(
        features
    ).all():
        raise ValueError(
            "Le feature contengono NaN o infiniti."
        )

    if not np.isfinite(
        logits
    ).all():
        raise ValueError(
            "I logit contengono NaN o infiniti."
        )

    if not np.isfinite(
        scores
    ).all():
        raise ValueError(
            "Gli score contengono NaN o infiniti."
        )

    # Regola ufficiale:
    # sigmoid(logit) > 0.5 => Fake.
    predictions = (
        scores > 0.5
    ).astype(np.int64)

    metrics = compute_metrics(
        labels=labels,
        predictions=predictions,
        scores=scores,
    )

    np.save(
        features_path,
        features,
        allow_pickle=False,
    )

    metadata = pd.DataFrame(
        {
            "row_id": row_ids,
            "image_path": image_paths,
            "label": labels,
            "label_name": np.where(
                labels == 1,
                "Fake",
                "Real",
            ),
            "original_json": original_json_rows,
        }
    )

    metadata.to_parquet(
        metadata_path,
        index=False,
    )

    predictions_frame = metadata.copy()

    predictions_frame["raw_logit"] = (
        logits
    )

    predictions_frame["fake_score"] = (
        scores
    )

    predictions_frame["prediction"] = (
        predictions
    )

    predictions_frame["prediction_name"] = (
        np.where(
            predictions == 1,
            "Fake",
            "Real",
        )
    )

    predictions_frame["correct"] = (
        predictions == labels
    )

    predictions_frame.to_parquet(
        predictions_parquet,
        index=False,
    )

    predictions_frame.to_csv(
        predictions_csv,
        index=False,
    )

    metric_columns = [
        "classifier",
        "n_samples",
        "n_real",
        "n_fake",
        "accuracy",
        "balanced_accuracy",
        "precision_fake",
        "recall_fake",
        "f1_fake",
        "auroc",
        "average_precision",
        "tn",
        "fp",
        "fn",
        "tp",
    ]

    pd.DataFrame(
        [metrics]
    )[metric_columns].to_csv(
        metrics_csv,
        index=False,
    )

    with metrics_json.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            metrics,
            handle,
            indent=2,
        )

    feature_norms = np.linalg.norm(
        features,
        axis=1,
    )

    run_info = {
        "detector": "UniversalFakeDetect",
        "architecture": ARCHITECTURE,
        "classifier": "linear",
        "repository_path": str(
            UFD_REPO
        ),
        "checkpoint_path": str(
            CHECKPOINT_PATH
        ),
        "checkpoint_sha256": sha256_file(
            CHECKPOINT_PATH
        ),
        "jsonl_path": str(
            JSONL_PATH
        ),
        "jsonl_sha256": sha256_file(
            JSONL_PATH
        ),
        "output_dir": str(
            OUTPUT_DIR
        ),
        "path_field": path_field,
        "label_field": label_field,
        "n_samples": len(records),
        "feature_shape": list(
            features.shape
        ),
        "feature_dtype": str(
            features.dtype
        ),
        "feature_l2_norm_mean": float(
            feature_norms.mean()
        ),
        "feature_l2_norm_std": float(
            feature_norms.std()
        ),
        "preprocessing": {
            "resize": False,
            "center_crop": 224,
            "to_tensor": True,
            "normalize_mean": CLIP_MEAN,
            "normalize_std": CLIP_STD,
        },
        "feature_definition": (
            "Official CLIP ViT-L/14 encode_image "
            "output before the official linear layer"
        ),
        "score_definition": (
            "sigmoid of the official linear "
            "classifier logit"
        ),
        "decision_rule": (
            "fake_score > 0.5"
        ),
        "label_mapping": {
            "ground_truth": "0=Real, 1=Fake",
            "prediction": "0=Real, 1=Fake",
        },
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
    }

    with run_info_json.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            run_info,
            handle,
            indent=2,
        )

    print(
        "============================================================"
    )
    print(
        "DETECTOR:             UniversalFakeDetect"
    )
    print(
        f"ARCHITECTURE:         {ARCHITECTURE}"
    )
    print(
        "CLASSIFIER:           linear"
    )
    print(
        f"ACCURACY:             "
        f"{metrics['accuracy']:.6f}"
    )
    print(
        f"BALANCED ACCURACY:    "
        f"{metrics['balanced_accuracy']:.6f}"
    )
    print(
        f"PRECISION FAKE:       "
        f"{metrics['precision_fake']:.6f}"
    )
    print(
        f"RECALL FAKE:          "
        f"{metrics['recall_fake']:.6f}"
    )
    print(
        f"F1 FAKE:              "
        f"{metrics['f1_fake']:.6f}"
    )
    print(
        f"AUROC:                "
        f"{metrics['auroc']:.6f}"
    )
    print(
        f"AVERAGE PRECISION:    "
        f"{metrics['average_precision']:.6f}"
    )
    print(
        "CONFUSION MATRIX:     "
        f"TN={metrics['tn']} "
        f"FP={metrics['fp']} "
        f"FN={metrics['fn']} "
        f"TP={metrics['tp']}"
    )
    print(
        f"OUTPUT:               {OUTPUT_DIR}"
    )
    print(
        "============================================================"
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(
            f"[ERROR] {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        raise