#!/usr/bin/env python3

import argparse
import hashlib
import json
import os
import random
import shutil
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
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

GENERATOR_FIELDS = (
    "generator",
    "generator_name",
    "model",
    "source",
    "fake_source",
    "method",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Valuta un singolo checkpoint Effort direttamente sulle immagini "
            "complete indicate dal JSONL OpenFake, senza face detection o face crop."
        )
    )

    parser.add_argument("--detector-id", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--deepfakebench-root", type=Path, required=True)
    parser.add_argument("--jsonl", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--overwrite", action="store_true")

    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def find_value(
    record: dict[str, Any],
    candidates: tuple[str, ...],
    required: bool = True,
) -> tuple[str | None, Any]:
    for field in candidates:
        if field in record and record[field] is not None:
            return field, record[field]

    if required:
        raise KeyError(
            f"Nessun campo tra {candidates}. "
            f"Campi presenti: {sorted(record.keys())}"
        )

    return None, None


def normalize_label(value: Any) -> int:
    if isinstance(value, (bool, np.bool_)):
        return int(value)

    if isinstance(value, (int, np.integer)):
        numeric = int(value)
        if numeric in (0, 1):
            return numeric

    if isinstance(value, (float, np.floating)):
        numeric = float(value)
        if numeric in (0.0, 1.0):
            return int(numeric)

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


def resolve_image_path(
    value: Any,
    jsonl_path: Path,
    image_root: Path | None,
) -> Path:
    path = Path(str(value))

    if path.is_absolute():
        return path

    if image_root is not None:
        return image_root / path

    return jsonl_path.parent / path


def load_records(
    jsonl_path: Path,
    image_root: Path | None,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"JSON non valido alla riga {line_number}: {exc}"
                ) from exc

            _, raw_path = find_value(raw, PATH_FIELDS, required=True)
            _, raw_label = find_value(raw, LABEL_FIELDS, required=True)
            _, raw_generator = find_value(
                raw,
                GENERATOR_FIELDS,
                required=False,
            )

            image_path = resolve_image_path(
                raw_path,
                jsonl_path,
                image_root,
            )

            records.append(
                {
                    "row_id": len(records),
                    "image_path": str(image_path),
                    "label": normalize_label(raw_label),
                    "generator": (
                        str(raw_generator)
                        if raw_generator is not None
                        else "unknown"
                    ),
                    "original_json": json.dumps(
                        raw,
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                }
            )

    if not records:
        raise RuntimeError(f"Nessun record trovato in {jsonl_path}")

    return records


class OpenFakeDataset(Dataset):
    def __init__(
        self,
        records: list[dict[str, Any]],
        resolution: int,
        mean: list[float],
        std: list[float],
    ) -> None:
        self.records = records
        self.resolution = int(resolution)
        self.mean = [float(value) for value in mean]
        self.std = [float(value) for value in std]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        image_path = record["image_path"]

        image = cv2.imread(image_path, cv2.IMREAD_COLOR)

        if image is None:
            raise RuntimeError(f"Impossibile leggere immagine: {image_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(
            image,
            (self.resolution, self.resolution),
            interpolation=cv2.INTER_CUBIC,
        )

        image_tensor = TF.to_tensor(image)
        image_tensor = TF.normalize(
            image_tensor,
            mean=self.mean,
            std=self.std,
        )

        return {
            "image": image_tensor,
            "label": int(record["label"]),
            "row_id": int(record["row_id"]),
            "image_path": image_path,
            "generator": record["generator"],
            "original_json": record["original_json"],
        }


def collate_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "image": torch.stack([item["image"] for item in batch], dim=0),
        "label": torch.tensor(
            [item["label"] for item in batch],
            dtype=torch.long,
        ),
        "row_id": torch.tensor(
            [item["row_id"] for item in batch],
            dtype=torch.long,
        ),
        "image_path": [item["image_path"] for item in batch],
        "generator": [item["generator"] for item in batch],
        "original_json": [item["original_json"] for item in batch],
    }


def import_detector_registry(root: Path):
    root = root.resolve()
    training_path = root / "training"

    if not training_path.is_dir():
        raise FileNotFoundError(
            f"Directory training non trovata: {training_path}"
        )

    os.chdir(root)

    for path in (str(training_path), str(root)):
        if path not in sys.path:
            sys.path.insert(0, path)

    import detectors  # noqa: F401
    from detectors import DETECTOR

    return DETECTOR


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if not isinstance(config, dict):
        raise TypeError(f"Configurazione YAML non valida: {path}")

    required = ("model_name", "resolution", "mean", "std")
    missing = [key for key in required if key not in config]

    if missing:
        raise KeyError(f"Campi mancanti nel config {path}: {missing}")

    return config


def get_model_class(registry, model_name: str):
    if hasattr(registry, "get"):
        model_class = registry.get(model_name)
        if model_class is not None:
            return model_class

    if hasattr(registry, "module_dict"):
        available = registry.module_dict
        if model_name in available:
            return available[model_name]

    try:
        return registry[model_name]
    except (KeyError, TypeError) as exc:
        available = (
            sorted(registry.module_dict.keys())
            if hasattr(registry, "module_dict")
            else "non disponibili"
        )
        raise KeyError(
            f"Detector '{model_name}' non registrato. "
            f"Disponibili: {available}"
        ) from exc


def extract_raw_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    current = checkpoint

    if isinstance(current, dict):
        for key in (
            "state_dict",
            "model_state_dict",
            "model",
            "net",
            "detector",
        ):
            value = current.get(key)
            if isinstance(value, dict):
                current = value
                break

    if not isinstance(current, dict):
        raise TypeError("Formato del checkpoint non riconosciuto")

    state_dict = {
        str(key): value
        for key, value in current.items()
        if torch.is_tensor(value)
    }

    if not state_dict:
        raise RuntimeError("Il checkpoint non contiene tensori validi")

    return state_dict


def strip_prefix(
    state_dict: dict[str, torch.Tensor],
    prefix: str,
) -> dict[str, torch.Tensor]:
    if not all(key.startswith(prefix) for key in state_dict):
        return state_dict

    return {
        key[len(prefix):]: value
        for key, value in state_dict.items()
    }


ALLOWED_REBUILT_SUFFIXES = (
    ".U_r",
    ".S_r",
    ".V_r",
)


def choose_compatible_state_dict(
    raw_state_dict: dict[str, torch.Tensor],
    model_state: dict[str, torch.Tensor],
) -> tuple[
    dict[str, torch.Tensor],
    str,
    list[str],
    list[str],
    list[str],
]:
    candidates: list[tuple[str, dict[str, torch.Tensor]]] = [
        ("original", raw_state_dict),
        ("strip_module", strip_prefix(raw_state_dict, "module.")),
        ("strip_model", strip_prefix(raw_state_dict, "model.")),
    ]

    module_stripped = strip_prefix(raw_state_dict, "module.")
    model_stripped = strip_prefix(raw_state_dict, "model.")

    candidates.extend(
        [
            (
                "strip_module_then_model",
                strip_prefix(module_stripped, "model."),
            ),
            (
                "strip_model_then_module",
                strip_prefix(model_stripped, "module."),
            ),
        ]
    )

    model_keys = set(model_state.keys())
    best_report: dict[str, Any] | None = None

    for name, candidate in candidates:
        candidate_keys = set(candidate.keys())

        missing = sorted(model_keys - candidate_keys)
        unexpected = sorted(candidate_keys - model_keys)
        shape_mismatch = sorted(
            key
            for key in candidate_keys & model_keys
            if tuple(candidate[key].shape)
            != tuple(model_state[key].shape)
        )

        invalid_missing = [
            key
            for key in missing
            if not key.endswith(ALLOWED_REBUILT_SUFFIXES)
        ]

        if (
            not invalid_missing
            and not unexpected
            and not shape_mismatch
        ):
            return (
                candidate,
                name,
                missing,
                unexpected,
                shape_mismatch,
            )

        overlap = len(candidate_keys & model_keys)

        if best_report is None or overlap > best_report["overlap"]:
            best_report = {
                "name": name,
                "overlap": overlap,
                "missing": missing,
                "invalid_missing": invalid_missing,
                "unexpected": unexpected,
                "shape_mismatch": shape_mismatch,
            }

    assert best_report is not None

    raise RuntimeError(
        "Checkpoint incompatibile con il modello Effort.\n"
        f"Migliore trasformazione: {best_report['name']}\n"
        f"Chiavi coincidenti: "
        f"{best_report['overlap']}/{len(model_keys)}\n"
        f"Missing totali ({len(best_report['missing'])}): "
        f"{best_report['missing'][:20]}\n"
        f"Missing non consentite "
        f"({len(best_report['invalid_missing'])}): "
        f"{best_report['invalid_missing'][:20]}\n"
        f"Unexpected ({len(best_report['unexpected'])}): "
        f"{best_report['unexpected'][:20]}\n"
        f"Shape mismatch "
        f"({len(best_report['shape_mismatch'])}): "
        f"{best_report['shape_mismatch'][:20]}"
    )


def load_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: Path,
) -> dict[str, Any]:
    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
        )

    raw_state_dict = extract_raw_state_dict(checkpoint)
    model_state = model.state_dict()

    (
        state_dict,
        transformation,
        expected_missing,
        expected_unexpected,
        shape_mismatch,
    ) = choose_compatible_state_dict(
        raw_state_dict,
        model_state,
    )

    load_result = model.load_state_dict(
        state_dict,
        strict=False,
    )

    actual_missing = sorted(load_result.missing_keys)
    actual_unexpected = sorted(load_result.unexpected_keys)

    invalid_missing = [
        key
        for key in actual_missing
        if not key.endswith(ALLOWED_REBUILT_SUFFIXES)
    ]

    if invalid_missing:
        raise RuntimeError(
            "Mancano parametri Effort non ricostruibili:\n"
            f"{invalid_missing[:50]}"
        )

    if actual_unexpected:
        raise RuntimeError(
            "Il checkpoint contiene chiavi inattese:\n"
            f"{actual_unexpected[:50]}"
        )

    if actual_missing != expected_missing:
        raise RuntimeError(
            "Incoerenza nel caricamento del checkpoint.\n"
            f"Missing previste: {expected_missing[:20]}\n"
            f"Missing effettive: {actual_missing[:20]}"
        )

    loaded_elements = sum(
        tensor.numel()
        for tensor in state_dict.values()
    )
    model_elements = sum(
        tensor.numel()
        for tensor in model_state.values()
    )

    return {
        "checkpoint_tensor_count": len(raw_state_dict),
        "loaded_tensor_count": len(state_dict),
        "loaded_parameter_elements": int(loaded_elements),
        "model_parameter_elements": int(model_elements),
        "loaded_fraction": float(
            loaded_elements / model_elements
            if model_elements > 0
            else 0.0
        ),
        "key_transformation": transformation,
        "missing_keys": actual_missing,
        "unexpected_keys": expected_unexpected,
        "shape_mismatch": shape_mismatch,
        "reconstructed_svd_keys": actual_missing,
    }


def get_fake_scores(output: Any) -> torch.Tensor:
    if torch.is_tensor(output):
        logits = output
    elif isinstance(output, dict):
        if "prob" in output:
            scores = output["prob"]

            if not torch.is_tensor(scores):
                scores = torch.as_tensor(scores)

            if scores.ndim == 2 and scores.shape[1] == 2:
                scores = scores[:, 1]

            return scores.reshape(-1)

        if "cls" in output:
            logits = output["cls"]
        elif "logits" in output:
            logits = output["logits"]
        else:
            raise KeyError(
                "Output detector senza prob, cls o logits. "
                f"Chiavi: {sorted(output.keys())}"
            )
    else:
        raise TypeError(
            f"Tipo output detector non supportato: {type(output)}"
        )

    if logits.ndim == 2 and logits.shape[1] == 2:
        return torch.softmax(logits, dim=1)[:, 1]

    return torch.sigmoid(logits.reshape(-1))


def compute_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> tuple[dict[str, Any], np.ndarray]:
    predictions = (scores >= threshold).astype(np.int64)

    tn, fp, fn, tp = confusion_matrix(
        labels,
        predictions,
        labels=[0, 1],
    ).ravel()

    result: dict[str, Any] = {
        "n_samples": int(len(labels)),
        "n_real": int(np.sum(labels == 0)),
        "n_fake": int(np.sum(labels == 1)),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(
            balanced_accuracy_score(labels, predictions)
        ),
        "precision": float(
            precision_score(labels, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(labels, predictions, zero_division=0)
        ),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }

    if len(np.unique(labels)) == 2:
        result["auroc"] = float(roc_auc_score(labels, scores))
        result["average_precision"] = float(
            average_precision_score(labels, scores)
        )
    else:
        result["auroc"] = None
        result["average_precision"] = None

    return result, predictions


def compute_generator_metrics(
    dataframe: pd.DataFrame,
    threshold: float,
) -> pd.DataFrame:
    real_rows = dataframe[dataframe["label"] == 0]
    fake_rows = dataframe[dataframe["label"] == 1]
    rows: list[dict[str, Any]] = []

    for generator in sorted(fake_rows["generator"].astype(str).unique()):
        generator_fake = fake_rows[
            fake_rows["generator"].astype(str) == generator
        ]

        group = pd.concat(
            [real_rows, generator_fake],
            ignore_index=True,
        )

        labels = group["label"].to_numpy(dtype=np.int64)
        scores = group["score_fake"].to_numpy(dtype=np.float64)

        metrics, _ = compute_metrics(labels, scores, threshold)
        metrics["generator"] = generator
        rows.append(metrics)

    columns = [
        "generator",
        "n_samples",
        "n_real",
        "n_fake",
        "threshold",
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
        "tn",
        "fp",
        "fn",
        "tp",
        "auroc",
        "average_precision",
    ]

    if not rows:
        return pd.DataFrame(columns=columns)

    return pd.DataFrame(rows)[columns]


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    deepfakebench_root = args.deepfakebench_root.resolve()
    config_path = args.config.resolve()
    weights_path = args.weights.resolve()
    jsonl_path = args.jsonl.resolve()
    output_root = args.output_root.resolve()
    image_root = (
        args.image_root.resolve()
        if args.image_root is not None
        else None
    )

    for required_path in (
        deepfakebench_root,
        config_path,
        weights_path,
        jsonl_path,
    ):
        if not required_path.exists():
            raise FileNotFoundError(f"Percorso non trovato: {required_path}")

    output_dir = output_root / args.detector_id

    if output_dir.exists():
        if args.overwrite:
            shutil.rmtree(output_dir)
        elif (output_dir / "metrics.json").exists():
            raise FileExistsError(
                f"Output già esistente: {output_dir}. "
                "Usare --overwrite per sostituirlo."
            )

    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(config_path)
    resolution = int(config["resolution"])
    mean = config["mean"]
    std = config["std"]
    model_name = str(config["model_name"])

    print(
        f"Detector={args.detector_id} "
        f"model_name={model_name} "
        f"resolution={resolution}"
    )

    records = load_records(jsonl_path, image_root)
    print(f"Record OpenFake: {len(records)}")
    print("Preprocessing: immagine completa, resize e normalizzazione; nessun face crop")

    missing_images = [
        record["image_path"]
        for record in records
        if not Path(record["image_path"]).is_file()
    ]

    if missing_images:
        examples = "\n".join(missing_images[:10])
        raise FileNotFoundError(
            f"{len(missing_images)} immagini non trovate. "
            f"Prime occorrenze:\n{examples}"
        )

    dataset = OpenFakeDataset(
        records=records,
        resolution=resolution,
        mean=mean,
        std=std,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.workers > 0,
        collate_fn=collate_batch,
    )

    registry = import_detector_registry(deepfakebench_root)
    model_class = get_model_class(registry, model_name)

    print("Costruzione modello Effort...")
    model = model_class(config)

    print("Caricamento checkpoint...")
    checkpoint_report = load_checkpoint(model, weights_path)

    print(
        "Checkpoint caricato: "
        f"{checkpoint_report['loaded_fraction']:.2%}, "
        f"trasformazione={checkpoint_report['key_transformation']}"
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    if device.type != "cuda":
        raise RuntimeError("CUDA non disponibile: eseguire tramite job GPU")

    model.to(device)
    model.eval()

    all_rows: list[dict[str, Any]] = []
    start_time = time.time()
    amp_enabled = args.amp and device.type == "cuda"

    with torch.inference_mode():
        for batch in tqdm(
            loader,
            desc=f"Inference {args.detector_id}",
        ):
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)

            data_dict = {
                "image": images,
                "label": labels,
            }

            amp_context = (
                torch.autocast(
                    device_type="cuda",
                    dtype=torch.float16,
                )
                if amp_enabled
                else nullcontext()
            )

            with amp_context:
                output = model(data_dict, inference=True)
                scores = get_fake_scores(output)

            scores_np = scores.detach().float().cpu().numpy()
            labels_np = labels.detach().cpu().numpy().astype(np.int64)
            row_ids = batch["row_id"].cpu().numpy().astype(np.int64)

            if len(scores_np) != len(labels_np):
                raise RuntimeError(
                    "Numero di score diverso dal numero di immagini: "
                    f"{len(scores_np)} contro {len(labels_np)}"
                )

            for index in range(len(scores_np)):
                all_rows.append(
                    {
                        "row_id": int(row_ids[index]),
                        "image_path": batch["image_path"][index],
                        "generator": batch["generator"][index],
                        "label": int(labels_np[index]),
                        "score_fake": float(scores_np[index]),
                        "original_json": batch["original_json"][index],
                    }
                )

    elapsed_seconds = time.time() - start_time

    dataframe = pd.DataFrame(all_rows)
    dataframe = dataframe.sort_values("row_id").reset_index(drop=True)

    metrics, predictions = compute_metrics(
        labels=dataframe["label"].to_numpy(dtype=np.int64),
        scores=dataframe["score_fake"].to_numpy(dtype=np.float64),
        threshold=args.threshold,
    )

    dataframe["prediction"] = predictions
    dataframe["correct"] = (
        dataframe["prediction"] == dataframe["label"]
    )

    generator_metrics = compute_generator_metrics(
        dataframe,
        args.threshold,
    )

    metrics.update(
        {
            "detector_id": args.detector_id,
            "model_name": model_name,
            "config_path": str(config_path),
            "weights_path": str(weights_path),
            "weights_sha256": sha256_file(weights_path),
            "jsonl_path": str(jsonl_path),
            "jsonl_sha256": sha256_file(jsonl_path),
            "deepfakebench_root": str(deepfakebench_root),
            "resolution": resolution,
            "mean": mean,
            "std": std,
            "batch_size": args.batch_size,
            "workers": args.workers,
            "seed": args.seed,
            "amp": amp_enabled,
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(0),
            "preprocessing": "whole_image_rgb_resize_normalize_no_face_detection",
            "elapsed_seconds": float(elapsed_seconds),
            "samples_per_second": float(
                len(dataframe) / elapsed_seconds
                if elapsed_seconds > 0
                else 0.0
            ),
        }
    )

    try:
        dataframe.to_parquet(
            output_dir / "predictions.parquet",
            index=False,
        )
    except (ImportError, ModuleNotFoundError, ValueError) as exc:
        print(f"Parquet non salvato: {exc}")

    dataframe.to_csv(
        output_dir / "predictions.csv",
        index=False,
    )

    generator_metrics.to_csv(
        output_dir / "metrics_by_generator.csv",
        index=False,
    )

    with (output_dir / "metrics.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(metrics, handle, indent=2, ensure_ascii=False)

    with (output_dir / "checkpoint_report.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            checkpoint_report,
            handle,
            indent=2,
            ensure_ascii=False,
        )

    with (output_dir / "resolved_config.yaml").open(
        "w",
        encoding="utf-8",
    ) as handle:
        yaml.safe_dump(
            config,
            handle,
            sort_keys=False,
            allow_unicode=True,
        )

    print(
        f"Completato {args.detector_id}: "
        f"AUROC={metrics.get('auroc')} "
        f"AP={metrics.get('average_precision')} "
        f"ACC={metrics.get('accuracy')} "
        f"output={output_dir}"
    )


if __name__ == "__main__":
    main()