#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

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


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Valuta un classificatore sullo split test usando "
            "la soglia scelta sulla validation."
        )
    )

    parser.add_argument(
        "--architecture",
        type=str,
        required=True,
        help="Architettura delle feature, ad esempio CLIP o CoDE.",
    )

    parser.add_argument(
        "--protocol",
        type=str,
        required=True,
        help="Protocollo sperimentale, ad esempio test1 o test2.",
    )

    parser.add_argument(
        "--features-dir",
        type=Path,
        default=None,
        help=(
            "Cartella contenente features_test.npy e "
            "metadata_test.parquet."
        ),
    )

    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Path del model.joblib salvato dal training.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Cartella dove salvare metriche e predizioni.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Permette di sovrascrivere output esistenti.",
    )

    return parser.parse_args()


# ============================================================
# LOAD
# ============================================================

def load_test_split(features_dir: Path):
    features_path = features_dir / "features_test.npy"
    metadata_path = features_dir / "metadata_test.parquet"

    if not features_path.exists():
        raise FileNotFoundError(
            f"Feature test non trovate: {features_path}"
        )

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Metadata test non trovati: {metadata_path}"
        )

    print("[LOAD TEST]")
    print(f"  Features: {features_path}")
    print(f"  Metadata: {metadata_path}")

    X = np.load(features_path)
    metadata = pd.read_parquet(metadata_path)

    if X.ndim != 2:
        raise ValueError(
            f"Le feature devono essere una matrice 2D. "
            f"Shape trovata: {X.shape}"
        )

    if len(X) != len(metadata):
        raise ValueError(
            "Mismatch tra feature e metadata: "
            f"features={len(X)}, metadata={len(metadata)}"
        )

    required_columns = {
        "label",
        "generator",
        "image_path",
    }

    missing_columns = required_columns - set(metadata.columns)

    if missing_columns:
        raise ValueError(
            "Nel metadata mancano le colonne: "
            f"{sorted(missing_columns)}. "
            f"Colonne disponibili: {list(metadata.columns)}"
        )

    y = metadata["label"].astype(int).to_numpy()

    unique_labels = sorted(np.unique(y).tolist())

    if unique_labels != [0, 1]:
        raise ValueError(
            f"Il test deve contenere entrambe le label [0, 1]. "
            f"Trovate: {unique_labels}"
        )

    if not np.isfinite(X).all():
        raise ValueError(
            "Le feature test contengono NaN o Inf."
        )

    X = X.astype(np.float32, copy=False)

    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")
    print(
        "  Label counts:",
        dict(pd.Series(y).value_counts().sort_index()),
    )

    return X, y, metadata


# ============================================================
# METRICS
# ============================================================

def safe_roc_auc(y_true, y_score):
    if len(np.unique(y_true)) < 2:
        return np.nan

    return float(
        roc_auc_score(y_true, y_score)
    )


def safe_average_precision(y_true, y_score):
    if len(np.unique(y_true)) < 2:
        return np.nan

    return float(
        average_precision_score(y_true, y_score)
    )


def compute_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
):
    y_pred = (
        y_score >= threshold
    ).astype(np.int64)

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    )

    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "n_samples": int(len(y_true)),
        "n_real": int((y_true == 0).sum()),
        "n_fake": int((y_true == 1).sum()),
        "threshold": float(threshold),
        "accuracy": float(
            accuracy_score(y_true, y_pred)
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, y_pred)
        ),
        "auroc": safe_roc_auc(
            y_true,
            y_score,
        ),
        "average_precision": safe_average_precision(
            y_true,
            y_score,
        ),
        "precision": float(
            precision_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }

    return metrics, y_pred


# ============================================================
# PER-GENERATOR METRICS
# ============================================================

def compute_generator_metrics(
    predictions: pd.DataFrame,
    threshold: float,
):
    """
    Per ogni generatore fake:
      - prende tutte le fake di quel generatore;
      - prende tutte le real del test;
      - calcola metriche binarie real-vs-generatore.

    Questo permette di calcolare AUROC e AP correttamente.
    """

    real_rows = predictions[
        predictions["y_true"] == 0
    ].copy()

    fake_rows = predictions[
        predictions["y_true"] == 1
    ].copy()

    if real_rows.empty:
        raise ValueError(
            "Nessuna immagine real disponibile per "
            "le metriche per generatore."
        )

    generator_rows = []

    fake_generators = sorted(
        fake_rows["generator"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    for generator in fake_generators:
        generator_fake_rows = fake_rows[
            fake_rows["generator"].astype(str)
            == generator
        ].copy()

        evaluation_rows = pd.concat(
            [
                real_rows,
                generator_fake_rows,
            ],
            ignore_index=True,
        )

        y_true = (
            evaluation_rows["y_true"]
            .astype(int)
            .to_numpy()
        )

        y_score = (
            evaluation_rows["score_fake"]
            .astype(float)
            .to_numpy()
        )

        metrics, _ = compute_metrics(
            y_true=y_true,
            y_score=y_score,
            threshold=threshold,
        )

        fake_scores = (
            generator_fake_rows["score_fake"]
            .astype(float)
            .to_numpy()
        )

        fake_predictions = (
            fake_scores >= threshold
        ).astype(np.int64)

        fake_detection_rate = float(
            fake_predictions.mean()
        )

        row = {
            "generator": generator,
            "n_real_reference": int(
                len(real_rows)
            ),
            "n_fake_generator": int(
                len(generator_fake_rows)
            ),
            "n_samples": metrics["n_samples"],
            "accuracy": metrics["accuracy"],
            "balanced_accuracy": (
                metrics["balanced_accuracy"]
            ),
            "auroc": metrics["auroc"],
            "average_precision": (
                metrics["average_precision"]
            ),
            "precision": metrics["precision"],
            "recall_fake": metrics["recall"],
            "f1": metrics["f1"],
            "fake_detection_rate": (
                fake_detection_rate
            ),
            "mean_fake_score": float(
                np.mean(fake_scores)
            ),
            "median_fake_score": float(
                np.median(fake_scores)
            ),
            "std_fake_score": float(
                np.std(fake_scores)
            ),
            "tn": metrics["tn"],
            "fp": metrics["fp"],
            "fn": metrics["fn"],
            "tp": metrics["tp"],
        }

        generator_rows.append(row)

    return pd.DataFrame(generator_rows)


# ============================================================
# PER-DATASET METRICS
# ============================================================

def compute_dataset_metrics(
    predictions: pd.DataFrame,
    threshold: float,
):
    rows = []

    if "dataset" not in predictions.columns:
        return pd.DataFrame()

    for dataset_name, group in predictions.groupby(
        "dataset",
        dropna=False,
    ):
        y_true = (
            group["y_true"]
            .astype(int)
            .to_numpy()
        )

        y_score = (
            group["score_fake"]
            .astype(float)
            .to_numpy()
        )

        metrics, _ = compute_metrics(
            y_true=y_true,
            y_score=y_score,
            threshold=threshold,
        )

        metrics["dataset"] = str(dataset_name)
        rows.append(metrics)

    return pd.DataFrame(rows)


# ============================================================
# SAVE
# ============================================================

def save_json(path: Path, data: dict):
    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=2,
            allow_nan=True,
        )


def check_outputs(
    paths: list[Path],
    overwrite: bool,
):
    existing = [
        path
        for path in paths
        if path.exists()
    ]

    if existing and not overwrite:
        text = "\n".join(
            str(path)
            for path in existing
        )

        raise FileExistsError(
            "Esistono già alcuni output:\n"
            f"{text}\n"
            "Usa --overwrite per sovrascriverli."
        )


# ============================================================
# MAIN
# ============================================================

def main():
    args = parse_args()

    if args.features_dir is None:
        args.features_dir = (
            Path(
                "/work/cvcs2026/resnet_gang/results"
            )
            / args.architecture
            / "features"
            / args.protocol
        )

    if args.model_path is None:
        args.model_path = (
            Path(
                "/work/cvcs2026/resnet_gang/results"
            )
            / args.architecture
            / "classifiers"
            / args.protocol
            / "logreg"
            / "model.joblib"
        )

    if args.output_dir is None:
        args.output_dir = (
            Path(
                "/work/cvcs2026/resnet_gang/results"
            )
            / args.architecture
            / "classifiers"
            / args.protocol
            / "logreg"
            / "test"
        )

    if not args.model_path.exists():
        raise FileNotFoundError(
            f"Modello non trovato: {args.model_path}"
        )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    predictions_path = (
        args.output_dir
        / "test_predictions.parquet"
    )
    test_scores_path = (
        args.output_dir
        / "test_scores.parquet"
    )

    metrics_json_path = (
        args.output_dir
        / "test_metrics.json"
    )

    metrics_csv_path = (
        args.output_dir
        / "test_metrics.csv"
    )

    generator_metrics_path = (
        args.output_dir
        / "test_metrics_by_generator.csv"
    )

    dataset_metrics_path = (
        args.output_dir
        / "test_metrics_by_dataset.csv"
    )

    check_outputs(
        paths=[
            predictions_path,
            test_scores_path,
            metrics_json_path,
            metrics_csv_path,
            generator_metrics_path,
            dataset_metrics_path,
        ],
        overwrite=args.overwrite,
    )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    print("[MODEL]")
    print(f"  Path:         {args.model_path}")
    print(f"  Architecture: {args.architecture}")
    print(f"  Protocol:     {args.protocol}")

    bundle = joblib.load(args.model_path)

    if "model" not in bundle:
        raise ValueError(
            "Nel bundle manca la chiave 'model'."
        )

    if "threshold" not in bundle:
        raise ValueError(
            "Nel bundle manca la soglia scelta "
            "sulla validation."
        )

    model = bundle["model"]
    threshold = float(bundle["threshold"])

    model_architecture = bundle.get(
        "architecture"
    )

    model_protocol = bundle.get(
        "protocol"
    )

    if (
        model_architecture is not None
        and model_architecture != args.architecture
    ):
        raise ValueError(
            "Architettura non coerente: "
            f"argomento={args.architecture}, "
            f"modello={model_architecture}"
        )

    if (
        model_protocol is not None
        and model_protocol != args.protocol
    ):
        raise ValueError(
            "Protocollo non coerente: "
            f"argomento={args.protocol}, "
            f"modello={model_protocol}"
        )

    print(f"  Threshold:    {threshold:.6f}")
    print(f"  Best C:       {bundle.get('best_c')}")

    # --------------------------------------------------------
    # LOAD TEST
    # --------------------------------------------------------

    X_test, y_test, metadata = load_test_split(
        features_dir=args.features_dir
    )

    expected_dimension = bundle.get(
        "feature_dimension"
    )

    if (
        expected_dimension is not None
        and X_test.shape[1] != expected_dimension
    ):
        raise ValueError(
            "Dimensione delle feature incompatibile: "
            f"test={X_test.shape[1]}, "
            f"modello={expected_dimension}"
        )

    # --------------------------------------------------------
    # INFERENCE
    # --------------------------------------------------------

    print()
    print("[INFERENCE]")

    if not hasattr(model, "predict_proba"):
        raise TypeError(
            "Il modello non supporta predict_proba."
        )

    y_score = model.predict_proba(
        X_test
    )[:, 1]

    metrics, y_pred = compute_metrics(
        y_true=y_test,
        y_score=y_score,
        threshold=threshold,
    )

    # --------------------------------------------------------
    # PREDICTIONS
    # --------------------------------------------------------
    test_sample_idx = (
    metadata["sample_idx"].to_numpy()
    if "sample_idx" in metadata.columns
    else np.arange(len(y_test))
    )

    test_scores_df = pd.DataFrame({
        "sample_idx": test_sample_idx,
        "score": y_score.astype(np.float32),
        "label": y_test.astype(np.int64),
        "generator": (
            metadata["generator"]
            .fillna("unknown")
            .astype(str)
            .to_numpy()
        ),
    })

    test_scores_df.to_parquet(
        test_scores_path,
        index=False,
    )
    predictions = metadata.copy()

    predictions["y_true"] = (
        y_test.astype(np.int64)
    )

    predictions["y_pred"] = (
        y_pred.astype(np.int64)
    )

    predictions["score_fake"] = (
        y_score.astype(np.float32)
    )

    predictions["decision_threshold"] = (
        threshold
    )

    predictions["correct"] = (
        predictions["y_true"]
        == predictions["y_pred"]
    )

    predictions["architecture"] = (
        args.architecture
    )

    predictions["protocol"] = (
        args.protocol
    )

    predictions.to_parquet(
        predictions_path,
        index=False,
    )

    # --------------------------------------------------------
    # GENERATOR METRICS
    # --------------------------------------------------------

    generator_metrics = (
        compute_generator_metrics(
            predictions=predictions,
            threshold=threshold,
        )
    )

    generator_metrics.insert(
        0,
        "protocol",
        args.protocol,
    )

    generator_metrics.insert(
        0,
        "architecture",
        args.architecture,
    )

    generator_metrics.to_csv(
        generator_metrics_path,
        index=False,
    )

    # --------------------------------------------------------
    # DATASET METRICS
    # --------------------------------------------------------

    dataset_metrics = compute_dataset_metrics(
        predictions=predictions,
        threshold=threshold,
    )

    if not dataset_metrics.empty:
        dataset_metrics.insert(
            0,
            "protocol",
            args.protocol,
        )

        dataset_metrics.insert(
            0,
            "architecture",
            args.architecture,
        )

        dataset_metrics.to_csv(
            dataset_metrics_path,
            index=False,
        )

    # --------------------------------------------------------
    # GLOBAL METRICS
    # --------------------------------------------------------

    global_result = {
        "architecture": args.architecture,
        "protocol": args.protocol,
        "model_path": str(args.model_path),
        "features_dir": str(args.features_dir),
        "threshold_source": "validation",
        "best_c": bundle.get("best_c"),
        **metrics,
    }

    save_json(
        metrics_json_path,
        global_result,
    )

    pd.DataFrame(
        [global_result]
    ).to_csv(
        metrics_csv_path,
        index=False,
    )

    # --------------------------------------------------------
    # PRINT
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("TEST METRICS")
    print("=" * 80)

    print(f"Samples:            {metrics['n_samples']}")
    print(f"Real:               {metrics['n_real']}")
    print(f"Fake:               {metrics['n_fake']}")
    print(f"Threshold:          {metrics['threshold']:.6f}")
    print(f"Accuracy:           {metrics['accuracy']:.6f}")
    print(
        f"Balanced accuracy:  "
        f"{metrics['balanced_accuracy']:.6f}"
    )
    print(f"AUROC:              {metrics['auroc']:.6f}")
    print(
        f"Average Precision:  "
        f"{metrics['average_precision']:.6f}"
    )
    print(f"Precision:          {metrics['precision']:.6f}")
    print(f"Recall:             {metrics['recall']:.6f}")
    print(f"F1:                 {metrics['f1']:.6f}")
    print(
        "Confusion matrix:   "
        f"TN={metrics['tn']} "
        f"FP={metrics['fp']} "
        f"FN={metrics['fn']} "
        f"TP={metrics['tp']}"
    )

    print()
    print("[SAVED]")
    print(f"Predizioni:          {predictions_path}")
    print(f"Metriche JSON:       {metrics_json_path}")
    print(f"Metriche CSV:        {metrics_csv_path}")
    print(
        f"Metriche generatori: "
        f"{generator_metrics_path}"
    )

    if not dataset_metrics.empty:
        print(
            f"Metriche dataset:    "
            f"{dataset_metrics_path}"
        )


if __name__ == "__main__":
    main()