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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer
from sklearn.svm import LinearSVC


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Allena un LinearSVC sul train, ottimizza C e loss "
            "sulla validation e seleziona la soglia sulla validation."
        )
    )

    parser.add_argument(
        "--architecture",
        type=str,
        required=True,
        help="Architettura delle feature: CLIP oppure CoDE.",
    )

    parser.add_argument(
        "--protocol",
        type=str,
        required=True,
        help="Protocollo sperimentale: test1 oppure test2.",
    )

    parser.add_argument(
        "--features-dir",
        type=Path,
        default=None,
        help=(
            "Cartella contenente features_train.npy, "
            "metadata_train.parquet, features_val.npy "
            "e metadata_val.parquet."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Cartella in cui salvare modello e risultati.",
    )

    parser.add_argument(
        "--c-values",
        nargs="+",
        type=float,
        default=[
            1e-4,
            1e-3,
            1e-2,
            1e-1,
            1.0,
            10.0,
            100.0,
        ],
        help="Valori del parametro C da provare.",
    )

    parser.add_argument(
        "--losses",
        nargs="+",
        type=str,
        default=[
            "squared_hinge",
            "hinge",
        ],
        choices=[
            "squared_hinge",
            "hinge",
        ],
        help="Loss LinearSVC da provare.",
    )

    parser.add_argument(
        "--class-weight",
        type=str,
        default="none",
        choices=[
            "none",
            "balanced",
        ],
        help="Pesi delle classi.",
    )

    parser.add_argument(
        "--threshold-steps",
        type=int,
        default=2001,
        help="Numero di soglie da provare sulla validation.",
    )

    parser.add_argument(
        "--max-iter",
        type=int,
        default=10000,
        help="Numero massimo di iterazioni del LinearSVC.",
    )

    parser.add_argument(
        "--tol",
        type=float,
        default=1e-4,
        help="Tolleranza di convergenza.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Permette di sovrascrivere output già esistenti.",
    )

    return parser.parse_args()


# ============================================================
# LOAD DATA
# ============================================================

def load_split(split_name: str, features_dir: Path):
    features_path = (
        features_dir / f"features_{split_name}.npy"
    )

    metadata_path = (
        features_dir / f"metadata_{split_name}.parquet"
    )

    if not features_path.exists():
        raise FileNotFoundError(
            f"Feature non trovate: {features_path}"
        )

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Metadata non trovati: {metadata_path}"
        )

    print()
    print(f"[LOAD] split={split_name}")
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
            f"Mismatch nello split {split_name}: "
            f"features={len(X)}, metadata={len(metadata)}"
        )

    if "label" not in metadata.columns:
        raise ValueError(
            f"Colonna 'label' assente in {metadata_path}. "
            f"Colonne disponibili: {list(metadata.columns)}"
        )

    y = metadata["label"].astype(int).to_numpy()

    unique_labels = sorted(
        np.unique(y).tolist()
    )

    if unique_labels != [0, 1]:
        raise ValueError(
            f"Le label devono essere [0, 1]. "
            f"Trovate: {unique_labels}"
        )

    if not np.isfinite(X).all():
        raise ValueError(
            f"Le feature dello split {split_name} "
            "contengono NaN o Inf."
        )

    X = X.astype(
        np.float32,
        copy=False,
    )

    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")
    print(
        "  Label counts:",
        dict(
            pd.Series(y)
            .value_counts()
            .sort_index()
        ),
    )

    norms = np.linalg.norm(
        X,
        axis=1,
    )

    print(
        "  Norme feature prima della Pipeline: "
        f"mean={norms.mean():.6f}, "
        f"min={norms.min():.6f}, "
        f"max={norms.max():.6f}"
    )

    return X, y, metadata


# ============================================================
# MODEL
# ============================================================

def build_model(
    c_value: float,
    loss_value: str,
    class_weight,
    max_iter: int,
    tol: float,
    seed: int,
):
    classifier = LinearSVC(
        C=c_value,
        penalty="l2",
        loss=loss_value,
        dual="auto",
        class_weight=class_weight,
        max_iter=max_iter,
        tol=tol,
        random_state=seed,
    )

    model = Pipeline([
        (
            "normalizer",
            Normalizer(norm="l2"),
        ),
        (
            "classifier",
            classifier,
        ),
    ])

    return model


# ============================================================
# METRICS
# ============================================================

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
        "threshold": float(threshold),
        "accuracy": float(
            accuracy_score(
                y_true,
                y_pred,
            )
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                y_true,
                y_pred,
            )
        ),
        "auroc": float(
            roc_auc_score(
                y_true,
                y_score,
            )
        ),
        "average_precision": float(
            average_precision_score(
                y_true,
                y_score,
            )
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
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
    }

    return metrics, y_pred


def find_best_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold_steps: int,
):

    #LinearSVC produce score non limitati a [0, 1]. Le soglie vengono quindi cercate tra il minimo e il massimo degli score sulla validation.

    score_min = float(
        np.min(y_score)
    )

    score_max = float(
        np.max(y_score)
    )

    thresholds = np.linspace(
        score_min,
        score_max,
        threshold_steps,
        dtype=np.float64,
    )

    best_threshold = 0.0
    best_balanced_accuracy = -np.inf
    best_accuracy = -np.inf

    threshold_rows = []

    for threshold in thresholds:
        y_pred = (
            y_score >= threshold
        ).astype(np.int64)

        balanced_accuracy = (
            balanced_accuracy_score(
                y_true,
                y_pred,
            )
        )

        accuracy = accuracy_score(
            y_true,
            y_pred,
        )

        threshold_rows.append({
            "threshold": float(threshold),
            "balanced_accuracy": float(
                balanced_accuracy
            ),
            "accuracy": float(accuracy),
        })

        better_balanced_accuracy = (
            balanced_accuracy
            > best_balanced_accuracy
        )

        same_balanced_better_accuracy = (
            np.isclose(
                balanced_accuracy,
                best_balanced_accuracy,
            )
            and accuracy > best_accuracy
        )

        same_metrics_closer_to_zero = (
            np.isclose(
                balanced_accuracy,
                best_balanced_accuracy,
            )
            and np.isclose(
                accuracy,
                best_accuracy,
            )
            and abs(threshold)
            < abs(best_threshold)
        )

        if (
            better_balanced_accuracy
            or same_balanced_better_accuracy
            or same_metrics_closer_to_zero
        ):
            best_threshold = float(
                threshold
            )

            best_balanced_accuracy = float(
                balanced_accuracy
            )

            best_accuracy = float(
                accuracy
            )

    return (
        best_threshold,
        pd.DataFrame(threshold_rows),
    )


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
            indent=2,
            ensure_ascii=False,
        )


def check_output_files(
    output_paths,
    overwrite: bool,
):
    existing = [
        path
        for path in output_paths
        if path.exists()
    ]

    if existing and not overwrite:
        existing_text = "\n".join(
            str(path)
            for path in existing
        )

        raise FileExistsError(
            "Esistono già alcuni output:\n"
            f"{existing_text}\n"
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

    if args.output_dir is None:
        args.output_dir = (
            Path(
                "/work/cvcs2026/resnet_gang/results"
            )
            / args.architecture
            / "classifiers"
            / args.protocol
            / "linear_svc"
        )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_path = (
        args.output_dir / "model.joblib"
    )

    validation_metrics_path = (
        args.output_dir
        / "validation_metrics.json"
    )

    search_path = (
        args.output_dir
        / "hyperparameter_search.csv"
    )

    threshold_path = (
        args.output_dir
        / "threshold_search.csv"
    )

    predictions_path = (
        args.output_dir
        / "validation_predictions.parquet"
    )

    config_path = (
        args.output_dir
        / "config.json"
    )

    output_paths = [
        model_path,
        validation_metrics_path,
        search_path,
        threshold_path,
        predictions_path,
        config_path,
    ]

    check_output_files(
        output_paths=output_paths,
        overwrite=args.overwrite,
    )

    # --------------------------------------------------------
    # CLASS WEIGHT
    # --------------------------------------------------------

    if args.class_weight == "none":
        class_weight = None
    else:
        class_weight = "balanced"

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    X_train, y_train, train_metadata = load_split(
        split_name="train",
        features_dir=args.features_dir,
    )

    X_val, y_val, val_metadata = load_split(
        split_name="val",
        features_dir=args.features_dir,
    )

    if X_train.shape[1] != X_val.shape[1]:
        raise ValueError(
            "Dimensione delle feature differente: "
            f"train={X_train.shape[1]}, "
            f"val={X_val.shape[1]}"
        )

    # --------------------------------------------------------
    # HYPERPARAMETER SEARCH
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("RICERCA IPERPARAMETRI LINEAR SVC")
    print("=" * 80)

    search_rows = []

    best_model = None
    best_c = None
    best_loss = None
    best_val_auc = -np.inf
    best_val_ap = -np.inf
    best_val_scores = None

    for loss_value in args.losses:
        for c_value in args.c_values:
            print()
            print(
                f"[TRAIN] "
                f"C={c_value}, "
                f"loss={loss_value}"
            )

            model = build_model(
                c_value=c_value,
                loss_value=loss_value,
                class_weight=class_weight,
                max_iter=args.max_iter,
                tol=args.tol,
                seed=args.seed,
            )

            model.fit(
                X_train,
                y_train,
            )

            train_scores = (
                model.decision_function(
                    X_train
                )
            )

            val_scores = (
                model.decision_function(
                    X_val
                )
            )

            train_auc = roc_auc_score(
                y_train,
                train_scores,
            )

            train_ap = average_precision_score(
                y_train,
                train_scores,
            )

            val_auc = roc_auc_score(
                y_val,
                val_scores,
            )

            val_ap = average_precision_score(
                y_val,
                val_scores,
            )

            search_rows.append({
                "C": float(c_value),
                "loss": loss_value,
                "class_weight": (
                    args.class_weight
                ),
                "normalization": "l2",
                "train_auroc": float(
                    train_auc
                ),
                "train_average_precision": float(
                    train_ap
                ),
                "validation_auroc": float(
                    val_auc
                ),
                "validation_average_precision": float(
                    val_ap
                ),
            })

            print(
                f"  Train AUROC: {train_auc:.6f}"
            )
            print(
                f"  Train AP:    {train_ap:.6f}"
            )
            print(
                f"  Val AUROC:   {val_auc:.6f}"
            )
            print(
                f"  Val AP:      {val_ap:.6f}"
            )

            better_auc = (
                val_auc > best_val_auc
            )

            same_auc_better_ap = (
                np.isclose(
                    val_auc,
                    best_val_auc,
                )
                and val_ap > best_val_ap
            )

            same_metrics_smaller_c = (
                np.isclose(
                    val_auc,
                    best_val_auc,
                )
                and np.isclose(
                    val_ap,
                    best_val_ap,
                )
                and (
                    best_c is None
                    or c_value < best_c
                )
            )

            if (
                better_auc
                or same_auc_better_ap
                or same_metrics_smaller_c
            ):
                best_model = model
                best_c = float(c_value)
                best_loss = loss_value
                best_val_auc = float(val_auc)
                best_val_ap = float(val_ap)
                best_val_scores = (
                    val_scores.copy()
                )

    search_df = pd.DataFrame(
        search_rows
    )

    search_df.to_csv(
        search_path,
        index=False,
    )

    if best_model is None:
        raise RuntimeError(
            "Nessun modello allenato."
        )

    # --------------------------------------------------------
    # THRESHOLD SEARCH
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("RICERCA SOGLIA SU VALIDATION")
    print("=" * 80)

    best_threshold, threshold_df = (
        find_best_threshold(
            y_true=y_val,
            y_score=best_val_scores,
            threshold_steps=args.threshold_steps,
        )
    )

    threshold_df.to_csv(
        threshold_path,
        index=False,
    )

    validation_metrics, val_predictions = (
        compute_metrics(
            y_true=y_val,
            y_score=best_val_scores,
            threshold=best_threshold,
        )
    )

    print()
    print(f"Best C:         {best_c}")
    print(f"Best loss:      {best_loss}")
    print(
        f"Best threshold: "
        f"{best_threshold:.8f}"
    )

    print()
    print("[VALIDATION METRICS]")
    print(
        f"Accuracy:          "
        f"{validation_metrics['accuracy']:.6f}"
    )
    print(
        f"Balanced accuracy: "
        f"{validation_metrics['balanced_accuracy']:.6f}"
    )
    print(
        f"AUROC:             "
        f"{validation_metrics['auroc']:.6f}"
    )
    print(
        f"Average Precision: "
        f"{validation_metrics['average_precision']:.6f}"
    )
    print(
        f"Precision:          "
        f"{validation_metrics['precision']:.6f}"
    )
    print(
        f"Recall:             "
        f"{validation_metrics['recall']:.6f}"
    )
    print(
        f"F1:                 "
        f"{validation_metrics['f1']:.6f}"
    )
    print(
        "Confusion matrix:",
        validation_metrics[
            "confusion_matrix"
        ],
    )

    # --------------------------------------------------------
    # VALIDATION PREDICTIONS
    # --------------------------------------------------------

    validation_predictions = (
        val_metadata.copy()
    )

    validation_predictions[
        "decision_score"
    ] = best_val_scores.astype(
        np.float32
    )

    validation_predictions[
        "predicted_label"
    ] = val_predictions.astype(
        np.int64
    )

    validation_predictions[
        "decision_threshold"
    ] = best_threshold

    validation_predictions[
        "correct"
    ] = (
        validation_predictions[
            "label"
        ].astype(int)
        == validation_predictions[
            "predicted_label"
        ].astype(int)
    )

    validation_predictions.to_parquet(
        predictions_path,
        index=False,
    )

    # --------------------------------------------------------
    # SAVE MODEL
    # --------------------------------------------------------

    model_bundle = {
        "model": best_model,
        "model_type": "LinearSVC",
        "architecture": args.architecture,
        "protocol": args.protocol,
        "features_dir": str(
            args.features_dir
        ),
        "best_c": best_c,
        "best_loss": best_loss,
        "threshold": best_threshold,
        "score_method": "decision_function",
        "selection_metric": (
            "validation_auroc"
        ),
        "threshold_metric": (
            "validation_balanced_accuracy"
        ),
        "normalization": "l2",
        "class_weight": (
            args.class_weight
        ),
        "feature_dimension": int(
            X_train.shape[1]
        ),
        "train_samples": int(
            len(X_train)
        ),
        "validation_samples": int(
            len(X_val)
        ),
        "label_mapping": {
            "0": "real",
            "1": "fake",
        },
        "validation_metrics": (
            validation_metrics
        ),
    }

    joblib.dump(
        model_bundle,
        model_path,
    )

    result = {
        "architecture": args.architecture,
        "protocol": args.protocol,
        "model_type": "LinearSVC",
        "best_c": best_c,
        "best_loss": best_loss,
        "threshold": best_threshold,
        "normalization": "l2",
        "class_weight": (
            args.class_weight
        ),
        "train_shape": list(
            X_train.shape
        ),
        "validation_shape": list(
            X_val.shape
        ),
        "validation_metrics": (
            validation_metrics
        ),
    }

    save_json(
        validation_metrics_path,
        result,
    )

    config = {
        "architecture": args.architecture,
        "protocol": args.protocol,
        "features_dir": str(
            args.features_dir
        ),
        "output_dir": str(
            args.output_dir
        ),
        "c_values": args.c_values,
        "losses": args.losses,
        "class_weight": (
            args.class_weight
        ),
        "normalization": "l2",
        "threshold_steps": (
            args.threshold_steps
        ),
        "max_iter": args.max_iter,
        "tol": args.tol,
        "seed": args.seed,
    }

    save_json(
        config_path,
        config,
    )

    print()
    print("=" * 80)
    print("SALVATAGGIO COMPLETATO")
    print("=" * 80)
    print(f"Modello:        {model_path}")
    print(
        f"Metriche val:   "
        f"{validation_metrics_path}"
    )
    print(f"Ricerca C/loss: {search_path}")
    print(f"Ricerca soglia: {threshold_path}")
    print(
        f"Predizioni val: "
        f"{predictions_path}"
    )
    print(f"Config:         {config_path}")


if __name__ == "__main__":
    main()