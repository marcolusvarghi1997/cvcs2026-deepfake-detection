#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
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
from sklearn.preprocessing import StandardScaler


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Allena una Logistic Regression sul train, "
            "sceglie C e soglia sulla validation e salva il modello."
        )
    )

    parser.add_argument(
        "--architecture",
        type=str,
        required=True,
        help="Nome architettura, ad esempio CLIP o CoDE.",
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
            "Cartella contenente features_train.npy, "
            "metadata_train.parquet, features_val.npy "
            "e metadata_val.parquet. "
            "Se omessa viene costruita da architecture e protocol."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Cartella di output. Se omessa viene costruita "
            "automaticamente."
        ),
    )

    parser.add_argument(
        "--c-values",
        nargs="+",
        type=float,
        default=[0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0],
        help="Valori di C da provare.",
    )

    parser.add_argument(
        "--threshold-steps",
        type=int,
        default=1001,
        help="Numero di soglie tra 0 e 1 da provare.",
    )

    parser.add_argument(
        "--standardize",
        action="store_true",
        help=(
            "Applica StandardScaler usando solo il train. "
            "Di default non viene applicato."
        ),
    )

    parser.add_argument(
        "--max-iter",
        type=int,
        default=2000,
        help="Numero massimo di iterazioni della Logistic Regression.",
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

    features_path = features_dir / f"features_{split_name}.npy"
    metadata_path = features_dir / f"metadata_{split_name}.parquet"
    
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
            f"Le feature devono essere una matrice 2D, "
            f"trovata shape={X.shape}"
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

    unique_labels = sorted(np.unique(y).tolist())

    if unique_labels != [0, 1]:
        raise ValueError(
            f"Le label devono essere [0, 1], trovate: "
            f"{unique_labels}"
        )

    if not np.isfinite(X).all():
        raise ValueError(
            f"Feature contenenti NaN o Inf nello split {split_name}"
        )

    X = X.astype(np.float32, copy=False)

    print(f"  Shape X: {X.shape}")
    print(f"  Shape y: {y.shape}")
    print(
        "  Label counts:",
        dict(pd.Series(y).value_counts().sort_index()),
    )

    return X, y, metadata


# ============================================================
# MODEL
# ============================================================

def build_model(
    c_value: float,
    standardize: bool,
    max_iter: int,
    seed: int,
):
    classifier = LogisticRegression(
        C=c_value,
        max_iter=max_iter,
        class_weight="balanced",
        random_state=seed,
        solver="lbfgs",
    )

    if standardize:
        return Pipeline([
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "classifier",
                classifier,
            ),
        ])

    return classifier


# ============================================================
# METRICS
# ============================================================

def compute_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
):
    y_pred = (y_score >= threshold).astype(np.int64)

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    )

    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "threshold": float(threshold),
        "accuracy": float(
            accuracy_score(y_true, y_pred)
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, y_pred)
        ),
        "auroc": float(
            roc_auc_score(y_true, y_score)
        ),
        "average_precision": float(
            average_precision_score(y_true, y_score)
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
    thresholds = np.linspace(
        0.0,
        1.0,
        threshold_steps,
        dtype=np.float64,
    )

    best_threshold = 0.5
    best_balanced_accuracy = -np.inf
    best_accuracy = -np.inf

    threshold_rows = []

    for threshold in thresholds:
        y_pred = (
            y_score >= threshold
        ).astype(np.int64)

        balanced_accuracy = balanced_accuracy_score(
            y_true,
            y_pred,
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

        same_metrics_closer_to_half = (
            np.isclose(
                balanced_accuracy,
                best_balanced_accuracy,
            )
            and np.isclose(
                accuracy,
                best_accuracy,
            )
            and abs(threshold - 0.5)
            < abs(best_threshold - 0.5)
        )

        if (
            better_balanced_accuracy
            or same_balanced_better_accuracy
            or same_metrics_closer_to_half
        ):
            best_threshold = float(threshold)
            best_balanced_accuracy = float(
                balanced_accuracy
            )
            best_accuracy = float(accuracy)

    threshold_df = pd.DataFrame(threshold_rows)

    return best_threshold, threshold_df


# ============================================================
# SAVE HELPERS
# ============================================================

def save_json(path: Path, data: dict):
    with path.open("w", encoding="utf-8") as file:
        json.dump(
            data,
            file,
            indent=2,
            ensure_ascii=False,
        )


def check_output_files(
    output_paths: list[Path],
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
        args.features_dir = Path(
            "/work/cvcs2026/resnet_gang/results"
        ) / args.architecture / "features" / args.protocol

    if args.output_dir is None:
        args.output_dir = Path(
            "/work/cvcs2026/resnet_gang/results"
        ) / args.architecture / "classifiers" / args.protocol / "logreg"

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_path = (
        args.output_dir / "model.joblib"
    )

    metrics_path = (
        args.output_dir / "validation_metrics.json"
    )

    search_path = (
        args.output_dir / "hyperparameter_search.csv"
    )

    threshold_path = (
        args.output_dir / "threshold_search.csv"
    )

    predictions_path = (
        args.output_dir
        / "validation_predictions.parquet"
    )

    config_path = (
        args.output_dir / "config.json"
    )
    train_predictions_path = (
        args.output_dir
        / "train_scores.parquet"
    )
    val_scores_path = (
        args.output_dir
        / "val_scores.parquet"
    )

    output_paths = [
        model_path,
        metrics_path,
        search_path,
        threshold_path,
        predictions_path,
        config_path,
        train_predictions_path,
        val_scores_path,
    ]

    check_output_files(
        output_paths=output_paths,
        overwrite=args.overwrite,
    )

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
            "Dimensione feature differente tra train e val: "
            f"train={X_train.shape[1]}, "
            f"val={X_val.shape[1]}"
        )

    # --------------------------------------------------------
    # HYPERPARAMETER SEARCH
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("RICERCA IPERPARAMETRO C")
    print("=" * 80)

    search_rows = []

    best_model = None
    best_c = None
    best_val_auc = -np.inf
    best_val_ap = -np.inf
    best_val_scores = None

    for c_value in args.c_values:
        print()
        print(f"[TRAIN] C={c_value}")

        model = build_model(
            c_value=c_value,
            standardize=args.standardize,
            max_iter=args.max_iter,
            seed=args.seed,
        )

        model.fit(
            X_train,
            y_train,
        )

        train_scores = model.predict_proba(
            X_train
        )[:, 1]

        val_scores = model.predict_proba(
            X_val
        )[:, 1]

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
            "train_auroc": float(train_auc),
            "train_average_precision": float(
                train_ap
            ),
            "validation_auroc": float(val_auc),
            "validation_average_precision": float(
                val_ap
            ),
        })

        print(f"  Train AUROC: {train_auc:.6f}")
        print(f"  Train AP:    {train_ap:.6f}")
        print(f"  Val AUROC:   {val_auc:.6f}")
        print(f"  Val AP:      {val_ap:.6f}")

        better_auc = val_auc > best_val_auc

        same_auc_better_ap = (
            np.isclose(val_auc, best_val_auc)
            and val_ap > best_val_ap
        )

        same_metrics_smaller_c = (
            np.isclose(val_auc, best_val_auc)
            and np.isclose(val_ap, best_val_ap)
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
            best_val_auc = float(val_auc)
            best_val_ap = float(val_ap)
            best_val_scores = val_scores.copy()
            best_train_scores = train_scores.copy()

    search_df = pd.DataFrame(search_rows)
    search_df.to_csv(
        search_path,
        index=False,
    )

    if best_model is None:
        raise RuntimeError(
            "Nessun modello è stato allenato."
        )

    # --------------------------------------------------------
    # THRESHOLD SEARCH
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("RICERCA SOGLIA SU VALIDATION")
    print("=" * 80)

    best_threshold, threshold_df = find_best_threshold(
        y_true=y_val,
        y_score=best_val_scores,
        threshold_steps=args.threshold_steps,
    )

    threshold_df.to_csv(
        threshold_path,
        index=False,
    )

    validation_metrics, val_predictions = compute_metrics(
        y_true=y_val,
        y_score=best_val_scores,
        threshold=best_threshold,
    )

    print()
    print(f"Best C:         {best_c}")
    print(f"Best threshold: {best_threshold:.6f}")
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
        validation_metrics["confusion_matrix"],
    )

    

    # --------------------------------------------------------
    #  PREDICTIONS
    # --------------------------------------------------------

    best_train_scores = best_model.predict_proba(X_train)[:, 1]
    
    train_sample_idx = (
    train_metadata["sample_idx"].to_numpy()
    if "sample_idx" in train_metadata.columns
    else np.arange(len(y_train))
    )

    train_scores_df = pd.DataFrame({
        "sample_idx": train_sample_idx,
        "score": best_train_scores.astype(np.float32),
        "label": y_train.astype(np.int64),
        "generator": (
            train_metadata["generator"]
            .fillna("unknown")
            .astype(str)
            .to_numpy()
        ),
    })

    train_scores_df.to_parquet(
        train_predictions_path,
        index=False,
    )

    validation_predictions = (
        val_metadata.copy()
    )

    validation_predictions[
        "fake_score"
    ] = best_val_scores.astype(np.float32)

    validation_predictions[
        "predicted_label"
    ] = val_predictions.astype(np.int64)

    validation_predictions[
        "decision_threshold"
    ] = best_threshold

    validation_predictions[
        "correct"
    ] = (
        validation_predictions["label"].astype(int)
        == validation_predictions[
            "predicted_label"
        ].astype(int)
    )

    validation_predictions.to_parquet(
        predictions_path,
        index=False,
    )

    val_sample_idx = (
        val_metadata["sample_idx"].to_numpy()
        if "sample_idx" in val_metadata.columns
        else np.arange(len(y_val))
    )

    best_validation_scores_df = pd.DataFrame({
        "sample_idx": val_sample_idx,
        "score": best_val_scores.astype(np.float32),
        "label": y_val.astype(np.int64),
        "generator": (
            val_metadata["generator"]
            .fillna("unknown")
            .astype(str)
            .to_numpy()
        ),
    })

    best_validation_scores_df.to_parquet(
        val_scores_path,
        index=False,
    )
    # --------------------------------------------------------
    # SAVE MODEL
    # --------------------------------------------------------

    model_bundle = {
        "model": best_model,
        "architecture": args.architecture,
        "protocol": args.protocol,
        "features_dir": str(args.features_dir),
        "best_c": best_c,
        "threshold": best_threshold,
        "selection_metric": "validation_auroc",
        "threshold_metric": (
            "validation_balanced_accuracy"
        ),
        "standardize": args.standardize,
        "feature_dimension": int(
            X_train.shape[1]
        ),
        "train_samples": int(len(X_train)),
        "validation_samples": int(len(X_val)),
        "label_mapping": {
            "0": "real",
            "1": "fake",
        },
        "validation_metrics": validation_metrics,
    }

    joblib.dump(
        model_bundle,
        model_path,
    )

    results = {
        "architecture": args.architecture,
        "protocol": args.protocol,
        "best_c": best_c,
        "threshold": best_threshold,
        "selection_metric": "validation_auroc",
        "threshold_metric": (
            "validation_balanced_accuracy"
        ),
        "standardize": args.standardize,
        "train_shape": list(X_train.shape),
        "validation_shape": list(X_val.shape),
        "validation_metrics": validation_metrics,
    }

    save_json(
        metrics_path,
        results,
    )

    config = {
        "architecture": args.architecture,
        "protocol": args.protocol,
        "features_dir": str(args.features_dir),
        "output_dir": str(args.output_dir),
        "c_values": args.c_values,
        "threshold_steps": args.threshold_steps,
        "standardize": args.standardize,
        "max_iter": args.max_iter,
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
    print(f"Modello:          {model_path}")
    print(f"Metriche val:     {metrics_path}")
    print(f"Ricerca C:        {search_path}")
    print(f"Ricerca soglia:   {threshold_path}")
    print(f"Predizioni val:   {predictions_path}")
    print(f"Configurazione:   {config_path}")


if __name__ == "__main__":
    main()