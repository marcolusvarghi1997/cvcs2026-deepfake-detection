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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, Normalizer

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Allena un k-NN sul train, seleziona gli iperparametri "
            "e la soglia sulla validation."
        )
    )
    parser.add_argument("--architecture", type=str, required=True)
    parser.add_argument("--protocol", type=str, required=True)

    parser.add_argument("--features-dir", type=Path, default=None)

    parser.add_argument("--output-dir", type=Path, default=None)

    parser.add_argument(
        "--k-values",
        nargs="+",
        type=int,
        default=[1, 3, 5, 10, 20, 50, 100],
    )

    parser.add_argument(
        "--weights-values",
        nargs="+",
        type=str,
        default=["uniform", "distance"],
        choices=["uniform", "distance"],
    )

    parser.add_argument(
        "--metrics",
        nargs="+",
        type=str,
        default=["cosine", "euclidean"],
        choices=["cosine", "euclidean", "manhattan"],
    )

    parser.add_argument(
        "--threshold-steps",
        type=int,
        default=1001,
    )

    parser.add_argument(
        "--standardize",
        action="store_true",
        help="Applica l2 norm",
    )

    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
    )


    parser.add_argument("--overwrite", action="store_true")

    return parser.parse_args()


# ============================================================
# LOAD DATA
# ============================================================

def load_split(split_name: str, features_dir: Path):
    features_path = features_dir / f"features_{split_name}.npy"
    metadata_path = features_dir / f"metadata_{split_name}.parquet"

    if not features_path.exists():
        raise FileNotFoundError(f"Feature non trovate: {features_path}")

    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata non trovati: {metadata_path}")

    print()
    print(f"[LOAD] split={split_name}")
    print(f"  Features: {features_path}")
    print(f"  Metadata: {metadata_path}")

    X = np.load(features_path)
    metadata = pd.read_parquet(metadata_path)

    if X.ndim != 2:
        raise ValueError(f"Le feature devono essere 2D, trovata shape={X.shape}")

    if len(X) != len(metadata):
        raise ValueError(
            f"Mismatch split {split_name}: features={len(X)}, metadata={len(metadata)}"
        )

    if "label" not in metadata.columns:
        raise ValueError(f"Colonna 'label' assente in {metadata_path}")

    y = metadata["label"].astype(int).to_numpy()

    unique_labels = sorted(np.unique(y).tolist())
    if unique_labels != [0, 1]:
        raise ValueError(f"Le label devono essere [0, 1], trovate: {unique_labels}")

    if not np.isfinite(X).all():
        raise ValueError(f"Feature contenenti NaN o Inf nello split {split_name}")

    X = X.astype(np.float32, copy=False)

    print(f"  Shape X: {X.shape}")
    print(f"  Shape y: {y.shape}")
    print("  Label counts:", dict(pd.Series(y).value_counts().sort_index()))

    return X, y, metadata


# ============================================================
# MODEL
# ============================================================

def build_model(
    k: int,
    weights: str,
    metric: str,
    standardize: bool,
    n_jobs: int,
):
    classifier = KNeighborsClassifier(
        n_neighbors=k,
        weights=weights,
        metric=metric,
        algorithm="brute",
        n_jobs=n_jobs,
    )

    if standardize:
        return Pipeline([
            ("normalizer", Normalizer("l2")),
            ("classifier", classifier),
        ])

    return classifier


# ============================================================
# METRICS
# ============================================================

def compute_metrics(y_true, y_score, threshold):
    y_pred = (y_score >= threshold).astype(np.int64)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "auroc": float(roc_auc_score(y_true, y_score)),
        "average_precision": float(average_precision_score(y_true, y_score)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
    }

    return metrics, y_pred


def find_best_threshold(y_true, y_score, threshold_steps):
    thresholds = np.linspace(0.0, 1.0, threshold_steps)

    best_threshold = 0.5
    best_balanced_accuracy = -np.inf
    best_accuracy = -np.inf

    rows = []

    for threshold in thresholds:
        y_pred = (y_score >= threshold).astype(np.int64)

        balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
        accuracy = accuracy_score(y_true, y_pred)

        rows.append({
            "threshold": float(threshold),
            "balanced_accuracy": float(balanced_accuracy),
            "accuracy": float(accuracy),
        })

        if (
            balanced_accuracy > best_balanced_accuracy
            or (
                np.isclose(balanced_accuracy, best_balanced_accuracy)
                and accuracy > best_accuracy
            )
            or (
                np.isclose(balanced_accuracy, best_balanced_accuracy)
                and np.isclose(accuracy, best_accuracy)
                and abs(threshold - 0.5) < abs(best_threshold - 0.5)
            )
        ):
            best_threshold = float(threshold)
            best_balanced_accuracy = float(balanced_accuracy)
            best_accuracy = float(accuracy)

    return best_threshold, pd.DataFrame(rows)


def save_json(path: Path, data: dict):
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


def check_output_files(output_paths, overwrite):
    existing = [path for path in output_paths if path.exists()]

    if existing and not overwrite:
        existing_text = "\n".join(str(path) for path in existing)
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
            Path("/work/cvcs2026/resnet_gang/outputs")
            / args.architecture
            / "features"
            / args.protocol
        )

    if args.output_dir is None:
        args.output_dir = (
            Path("/work/cvcs2026/resnet_gang/results")
            / args.architecture
            / "classifiers"
            / args.protocol
            / "knn"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    model_path = args.output_dir / "model.joblib"
    metrics_path = args.output_dir / "validation_metrics.json"
    search_path = args.output_dir / "hyperparameter_search.csv"
    threshold_path = args.output_dir / "threshold_search.csv"
    predictions_path = args.output_dir / "validation_predictions.parquet"
    config_path = args.output_dir / "config.json"
    val_scores_path = args.output_dir / "val_scores.parquet"
    train_scores_path = args.output_dir / "train_scores.parquet"

    output_paths = [
        model_path,
        metrics_path,
        search_path,
        threshold_path,
        predictions_path,
        config_path,
        val_scores_path,
    ]


    check_output_files(output_paths, args.overwrite)

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    X_train, y_train, train_metadata = load_split("train", args.features_dir)
    X_val, y_val, val_metadata = load_split("val", args.features_dir)

    if X_train.shape[1] != X_val.shape[1]:
        raise ValueError(
            "Dimensione feature differente tra train e val: "
            f"train={X_train.shape[1]}, val={X_val.shape[1]}"
        )

    # --------------------------------------------------------
    # HYPERPARAMETER SEARCH
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("RICERCA IPERPARAMETRI k-NN")
    print("=" * 80)

    search_rows = []

    best_model = None
    best_k = None
    best_weights = None
    best_metric = None
    best_val_auc = -np.inf
    best_val_ap = -np.inf
    best_val_scores = None

    for metric in args.metrics:
        for weights in args.weights_values:
            for k in args.k_values:
                print()
                print(f"[TRAIN] k={k}, weights={weights}, metric={metric}")

                model = build_model(
                    k=k,
                    weights=weights,
                    metric=metric,
                    standardize=args.standardize,
                    n_jobs=args.n_jobs,
                )

                model.fit(X_train, y_train)

                val_scores = model.predict_proba(X_val)[:, 1]

                val_auc = roc_auc_score(y_val, val_scores)
                val_ap = average_precision_score(y_val, val_scores)

                search_rows.append({
                    "k": int(k),
                    "weights": weights,
                    "metric": metric,
                    "validation_auroc": float(val_auc),
                    "validation_average_precision": float(val_ap),
                })

                print(f"  Val AUROC: {val_auc:.6f}")
                print(f"  Val AP:    {val_ap:.6f}")

                better_auc = val_auc > best_val_auc

                same_auc_better_ap = (
                    np.isclose(val_auc, best_val_auc)
                    and val_ap > best_val_ap
                )

                same_metrics_smaller_k = (
                    np.isclose(val_auc, best_val_auc)
                    and np.isclose(val_ap, best_val_ap)
                    and (best_k is None or k < best_k)
                )

                if better_auc or same_auc_better_ap or same_metrics_smaller_k:
                    best_model = model
                    best_k = int(k)
                    best_weights = weights
                    best_metric = metric
                    best_val_auc = float(val_auc)
                    best_val_ap = float(val_ap)
                    best_val_scores = val_scores.copy()

    pd.DataFrame(search_rows).to_csv(search_path, index=False)

    if best_model is None:
        raise RuntimeError("Nessun modello è stato allenato.")

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

    threshold_df.to_csv(threshold_path, index=False)

    validation_metrics, val_predictions = compute_metrics(
        y_true=y_val,
        y_score=best_val_scores,
        threshold=best_threshold,
    )

    print()
    print(f"Best k:         {best_k}")
    print(f"Best weights:   {best_weights}")
    print(f"Best metric:    {best_metric}")
    print(f"Best threshold: {best_threshold:.6f}")

    # --------------------------------------------------------
    # SAVE PREDICTIONS
    # --------------------------------------------------------

    validation_predictions = val_metadata.copy()
    validation_predictions["fake_score"] = best_val_scores.astype(np.float32)
    validation_predictions["predicted_label"] = val_predictions.astype(np.int64)
    validation_predictions["decision_threshold"] = best_threshold
    validation_predictions["correct"] = (
        validation_predictions["label"].astype(int)
        == validation_predictions["predicted_label"].astype(int)
    )

    validation_predictions.to_parquet(predictions_path, index=False)

    val_scores_df = pd.DataFrame({
        "sample_idx": np.arange(len(y_val)),
        "score": best_val_scores.astype(np.float32),
        "label": y_val.astype(np.int64),
    })

    val_scores_df.to_parquet(val_scores_path, index=False)

 
    # --------------------------------------------------------
    # SAVE MODEL
    # --------------------------------------------------------

    model_bundle = {
        "model": best_model,
        "architecture": args.architecture,
        "protocol": args.protocol,
        "features_dir": str(args.features_dir),
        "best_k": best_k,
        "best_weights": best_weights,
        "best_metric": best_metric,
        "threshold": best_threshold,
        "selection_metric": "validation_auroc",
        "threshold_metric": "validation_balanced_accuracy",
        "standardize": args.standardize,
        "feature_dimension": int(X_train.shape[1]),
        "train_samples": int(len(X_train)),
        "validation_samples": int(len(X_val)),
        "label_mapping": {
            "0": "real",
            "1": "fake",
        },
        "validation_metrics": validation_metrics,
    }

    joblib.dump(model_bundle, model_path)

    results = {
        "architecture": args.architecture,
        "protocol": args.protocol,
        "best_k": best_k,
        "best_weights": best_weights,
        "best_metric": best_metric,
        "threshold": best_threshold,
        "selection_metric": "validation_auroc",
        "threshold_metric": "validation_balanced_accuracy",
        "standardize": args.standardize,
        "train_shape": list(X_train.shape),
        "validation_shape": list(X_val.shape),
        "validation_metrics": validation_metrics,
    }

    save_json(metrics_path, results)

    config = {
        "architecture": args.architecture,
        "protocol": args.protocol,
        "features_dir": str(args.features_dir),
        "output_dir": str(args.output_dir),
        "k_values": args.k_values,
        "weights_values": args.weights_values,
        "metrics": args.metrics,
        "threshold_steps": args.threshold_steps,
        "standardize": args.standardize,
        "n_jobs": args.n_jobs,
    }

    save_json(config_path, config)

    print()
    print("=" * 80)
    print("SALVATAGGIO COMPLETATO")
    print("=" * 80)
    print(f"Modello:          {model_path}")
    print(f"Metriche val:     {metrics_path}")
    print(f"Ricerca k-NN:     {search_path}")
    print(f"Ricerca soglia:   {threshold_path}")
    print(f"Predizioni val:   {predictions_path}")
    print(f"Score val:        {val_scores_path}")
    print(f"Configurazione:   {config_path}")



if __name__ == "__main__":
    main()