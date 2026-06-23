#!/usr/bin/env python3

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Any

import huggingface_hub
import joblib
import numpy as np
import pandas as pd
import sklearn
import torch
import torch.nn as nn
import transformers
from huggingface_hub import hf_hub_download
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

MODEL_REPO = "aimagelab/CoDE"

JSONL_PATH = Path(
    "/work/cvcs2026/resnet_gang/datasets/json_standardized/json/openfake.jsonl"
)

OUTPUT_ROOT = Path(
    "/work/cvcs2026/resnet_gang/outputs/CoDE/detection_from_paper"
)

BATCH_SIZE = 256
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

CLASSIFIER_FILES = {
    "linear": "sklearn/linear_tot_classifier_epoch-32.sav",
    "knn": "sklearn/knn_tot_classifier_epoch-32.sav",
    "svm": "sklearn/ocsvm_kernel_poly_gamma_auto_nu_0_1_crop.joblib",
}

# Preprocessing riportato nella model card ufficiale CoDE.
OFFICIAL_TRANSFORM = transforms.Compose(
    [
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)

ImageFile.LOAD_TRUNCATED_IMAGES = True


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Esegue il detector CoDE ufficiale su OpenFake con "
            "preprocessing, backbone e classificatore ufficiali."
        )
    )
    parser.add_argument(
        "--classifier",
        required=True,
        choices=sorted(CLASSIFIER_FILES),
    )
    parser.add_argument(
        "--revision",
        required=True,
        help="Commit SHA Hugging Face fissato dal launcher.",
    )
    return parser.parse_args()


# ============================================================
# UTILITIES
# ============================================================

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
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

    with JSONL_PATH.open("r", encoding="utf-8") as handle:
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
        raise RuntimeError(f"Nessun record trovato in {JSONL_PATH}")

    assert detected_path_field is not None
    assert detected_label_field is not None

    return records, detected_path_field, detected_label_field


# ============================================================
# DATASET
# ============================================================

class OpenFakeDataset(Dataset):
    def __init__(self, records: list[dict[str, Any]]) -> None:
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
                f"Errore durante la lettura di {image_path}: {exc}"
            ) from exc

        return (
            image_tensor,
            int(record["label"]),
            int(record["row_id"]),
            record["image_path"],
            record["original_json"],
        )


# ============================================================
# MODEL AND CLASSIFIER
# ============================================================

def load_backbone(
    revision: str,
    device: torch.device,
):
    model = transformers.AutoModel.from_pretrained(
        MODEL_REPO,
        revision=revision,
    )

    # Replica la model card. Il pooler non viene comunque usato:
    # le feature sono last_hidden_state[:, 0, :].
    if hasattr(model, "pooler"):
        model.pooler = nn.Identity()

    model.eval()
    model.to(device)

    for parameter in model.parameters():
        parameter.requires_grad_(False)

    return model


def load_official_classifier(
    classifier_name: str,
    revision: str,
):
    filename = CLASSIFIER_FILES[classifier_name]

    classifier_path = Path(
        hf_hub_download(
            repo_id=MODEL_REPO,
            filename=filename,
            revision=revision,
        )
    )

    classifier = joblib.load(classifier_path)

    if hasattr(classifier, "n_jobs"):
        try:
            classifier.n_jobs = 8
        except Exception:
            pass

    return classifier, classifier_path


def extract_features(
    model,
    loader: DataLoader,
    device: torch.device,
):
    feature_batches: list[np.ndarray] = []
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
            desc="CoDE feature extraction",
            unit="batch",
        ):
            images = images.to(
                device,
                non_blocking=True,
            )

            outputs = model(pixel_values=images)

            # Feature ufficiale CoDE:
            # token CLS, senza L2 normalization aggiuntiva.
            features = outputs.last_hidden_state[:, 0, :]

            feature_batches.append(
                features.cpu().numpy().astype(np.float32)
            )
            label_batches.append(
                labels.numpy().astype(np.int64)
            )
            row_id_batches.append(
                row_ids.numpy().astype(np.int64)
            )
            image_paths.extend(batch_paths)
            original_json_rows.extend(batch_original_json)

    return (
        np.concatenate(feature_batches, axis=0),
        np.concatenate(label_batches, axis=0),
        np.concatenate(row_id_batches, axis=0),
        image_paths,
        original_json_rows,
    )


def convert_predictions(
    classifier_name: str,
    raw_predictions: np.ndarray,
) -> np.ndarray:
    raw_predictions = np.asarray(raw_predictions).reshape(-1)

    if classifier_name in {"linear", "knn"}:
        if not np.isin(raw_predictions, [0, 1]).all():
            raise ValueError(
                f"Predizioni inattese per {classifier_name}: "
                f"{np.unique(raw_predictions)}"
            )

        return raw_predictions.astype(np.int64)

    if classifier_name == "svm":
        if not np.isin(raw_predictions, [-1, 1]).all():
            raise ValueError(
                f"Predizioni inattese per svm: "
                f"{np.unique(raw_predictions)}"
            )

        # Mapping ufficiale della model card:
        # -1 = Real
        # +1 = Fake
        return (raw_predictions == 1).astype(np.int64)

    raise ValueError(classifier_name)


def extract_fake_score(
    classifier_name: str,
    classifier,
    features: np.ndarray,
) -> np.ndarray | None:
    """
    Score continuo crescente verso Fake.
    Non usa le label del test per orientare lo score.
    """
    if hasattr(classifier, "predict_proba"):
        probabilities = np.asarray(
            classifier.predict_proba(features)
        )
        classes = list(
            np.asarray(classifier.classes_).reshape(-1)
        )

        if 1 not in classes:
            raise ValueError(
                f"Classe Fake=1 assente in classes_: {classes}"
            )

        return probabilities[:, classes.index(1)].astype(np.float64)

    if hasattr(classifier, "decision_function"):
        scores = np.asarray(
            classifier.decision_function(features)
        ).reshape(-1).astype(np.float64)

        if classifier_name == "svm":
            # OneClassSVM:
            # decision_function > 0 corrisponde a predict() = +1.
            # Nella model card CoDE +1 = Fake.
            return scores

        classes = list(
            np.asarray(
                getattr(classifier, "classes_", [])
            ).reshape(-1)
        )

        if classes == [0, 1]:
            return scores

        if classes == [1, 0]:
            return -scores

        raise ValueError(
            "Impossibile orientare decision_function verso Fake. "
            f"classes_={classes}"
        )

    return None


# ============================================================
# METRICS
# ============================================================

def compute_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    scores: np.ndarray | None,
) -> dict[str, Any]:
    tn, fp, fn, tp = confusion_matrix(
        labels,
        predictions,
        labels=[0, 1],
    ).ravel()

    metrics: dict[str, Any] = {
        "n_samples": int(len(labels)),
        "n_real": int(np.sum(labels == 0)),
        "n_fake": int(np.sum(labels == 1)),
        "accuracy": float(
            accuracy_score(labels, predictions)
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(labels, predictions)
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
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }

    if scores is not None and len(np.unique(labels)) == 2:
        metrics["auroc"] = float(
            roc_auc_score(labels, scores)
        )
        metrics["average_precision"] = float(
            average_precision_score(labels, scores)
        )
    else:
        metrics["auroc"] = None
        metrics["average_precision"] = None

    return metrics


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    args = parse_args()

    classifier_name = args.classifier
    revision = args.revision.strip()

    if not revision:
        raise ValueError("Revision Hugging Face vuota.")

    if not JSONL_PATH.is_file():
        raise FileNotFoundError(
            f"JSONL non trovato: {JSONL_PATH}"
        )

    if REQUIRE_CUDA and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA non disponibile. Il job deve richiedere una GPU."
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    output_dir = OUTPUT_ROOT / classifier_name
    output_dir.mkdir(parents=True, exist_ok=True)

    features_path = output_dir / "features_test.npy"
    metadata_path = output_dir / "metadata_test.parquet"
    predictions_parquet = output_dir / "predictions_test.parquet"
    predictions_csv = output_dir / "predictions_test.csv"
    metrics_csv = output_dir / "metrics_test.csv"
    metrics_json = output_dir / "metrics_test.json"
    run_info_json = output_dir / "run_info.json"

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

    records, path_field, label_field = load_jsonl_records()

    missing_paths = [
        record["image_path"]
        for record in records
        if not Path(record["image_path"]).is_file()
    ]

    if missing_paths:
        preview = "\n".join(missing_paths[:10])
        raise FileNotFoundError(
            f"{len(missing_paths)} immagini non trovate. "
            f"Prime occorrenze:\n{preview}"
        )

    dataset = OpenFakeDataset(records)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY and device.type == "cuda",
        persistent_workers=NUM_WORKERS > 0,
    )

    print(f"[INFO] JSONL: {JSONL_PATH}")
    print(f"[INFO] samples: {len(records)}")
    print(f"[INFO] path field: {path_field}")
    print(f"[INFO] label field: {label_field}")
    print(f"[INFO] device: {device}")
    print(f"[INFO] classifier: {classifier_name}")
    print(f"[INFO] HF revision: {revision}")

    model = load_backbone(
        revision=revision,
        device=device,
    )

    classifier, classifier_path = load_official_classifier(
        classifier_name=classifier_name,
        revision=revision,
    )

    (
        features,
        labels,
        row_ids,
        image_paths,
        original_json_rows,
    ) = extract_features(
        model=model,
        loader=loader,
        device=device,
    )

    if features.ndim != 2:
        raise ValueError(
            f"Feature non 2D: {features.shape}"
        )

    if not np.isfinite(features).all():
        raise ValueError(
            "Le feature contengono NaN o infiniti."
        )

    expected_features = getattr(
        classifier,
        "n_features_in_",
        None,
    )

    if (
        expected_features is not None
        and int(expected_features) != features.shape[1]
    ):
        raise ValueError(
            "Dimensione feature incompatibile: "
            f"ottenuta={features.shape[1]}, "
            f"attesa={expected_features}"
        )

    raw_predictions = np.asarray(
        classifier.predict(features)
    ).reshape(-1)

    predictions = convert_predictions(
        classifier_name,
        raw_predictions,
    )

    scores = extract_fake_score(
        classifier_name,
        classifier,
        features,
    )

    metrics = compute_metrics(
        labels,
        predictions,
        scores,
    )
    metrics["classifier"] = classifier_name

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
    predictions_frame["raw_prediction"] = raw_predictions
    predictions_frame["prediction"] = predictions
    predictions_frame["prediction_name"] = np.where(
        predictions == 1,
        "Fake",
        "Real",
    )
    predictions_frame["correct"] = predictions == labels

    if scores is not None:
        predictions_frame["fake_score"] = scores

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

    pd.DataFrame([metrics])[metric_columns].to_csv(
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
        "model_repo": MODEL_REPO,
        "model_revision": revision,
        "classifier": classifier_name,
        "classifier_hf_file": CLASSIFIER_FILES[classifier_name],
        "classifier_local_path": str(classifier_path),
        "classifier_sha256": sha256_file(classifier_path),
        "jsonl_path": str(JSONL_PATH),
        "jsonl_sha256": sha256_file(JSONL_PATH),
        "output_dir": str(output_dir),
        "path_field": path_field,
        "label_field": label_field,
        "n_samples": len(records),
        "feature_shape": list(features.shape),
        "feature_dtype": str(features.dtype),
        "feature_l2_norm_mean": float(feature_norms.mean()),
        "feature_l2_norm_std": float(feature_norms.std()),
        "preprocessing": {
            "resize": False,
            "center_crop": 224,
            "to_tensor": True,
            "normalize_mean": [0.485, 0.456, 0.406],
            "normalize_std": [0.229, 0.224, 0.225],
        },
        "feature_definition": (
            "outputs.last_hidden_state[:, 0, :] "
            "without additional L2 normalization"
        ),
        "label_mapping": {
            "ground_truth": "0=Real, 1=Fake",
            "linear_knn_raw": "0=Real, 1=Fake",
            "svm_raw": "-1=Real, +1=Fake",
        },
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "torchvision": __import__("torchvision").__version__,
            "transformers": transformers.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
            "huggingface_hub": huggingface_hub.__version__,
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

    print("============================================================")
    print(f"CLASSIFIER:          {classifier_name}")
    print(f"ACCURACY:            {metrics['accuracy']:.6f}")
    print(f"BALANCED ACCURACY:   {metrics['balanced_accuracy']:.6f}")
    print(f"PRECISION FAKE:      {metrics['precision_fake']:.6f}")
    print(f"RECALL FAKE:         {metrics['recall_fake']:.6f}")
    print(f"F1 FAKE:             {metrics['f1_fake']:.6f}")
    print(f"AUROC:               {metrics['auroc']}")
    print(f"AVERAGE PRECISION:   {metrics['average_precision']}")
    print(
        f"CONFUSION MATRIX:    "
        f"TN={metrics['tn']} "
        f"FP={metrics['fp']} "
        f"FN={metrics['fn']} "
        f"TP={metrics['tp']}"
    )
    print(f"OUTPUT:              {output_dir}")
    print("============================================================")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(
            f"[ERROR] {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        raise