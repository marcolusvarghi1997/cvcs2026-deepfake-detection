#!/usr/bin/env python3

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
)


# Modifica Case 
MODEL_PATH = Path("/work/cvcs2026/resnet_gang/results/CoDE_CLIP/classifiers/case1/logreg/model.joblib")
FEATURES_PATH = Path("/work/cvcs2026/resnet_gang/outputs/CoDE_CLIP/features/case1/features_test.npy")
METADATA_PATH = Path("/work/cvcs2026/resnet_gang/outputs/CoDE_CLIP/features/case1/metadata_test.parquet")
THRESHOLD = 0.543

OUTPUT_DIR = Path("/work/cvcs2026/resnet_gang/results/CoDE_CLIP/permutation_blocks/logreg/case1")




def load_features(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Feature non trovate: {path}")

    if path.suffix == ".npy":
        return np.asarray(np.load(path, mmap_mode="r"), dtype=np.float32)

    if path.suffix == ".parquet":
        return pd.read_parquet(path).to_numpy(dtype=np.float32)

    raise ValueError("Sbagliato file .npy oppure .parquet")


def extract_model(bundle):
    if isinstance(bundle, dict) and "model" in bundle:
        return bundle["model"], bundle
    return bundle, {}



def get_scores(model, X):
    if not hasattr(model, "predict_proba"):
        raise TypeError("Il modello non supporta predict_proba. (SVC serve decision_function)")

    probabilities = model.predict_proba(X)

    return np.asarray(
        probabilities[:, 1],
        dtype=np.float64,
    ).reshape(-1)

def compute_metrics(y_true, scores, threshold):
    y_pred = (scores >= threshold).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision_fake": precision_score(y_true, y_pred, zero_division=0),
        "recall_fake": recall_score(y_true, y_pred, zero_division=0),
        "f1_fake": f1_score(y_true, y_pred, zero_division=0),
    }

    try:
        metrics["auroc"] = roc_auc_score(y_true, scores)
    except ValueError:
        metrics["auroc"] = np.nan

    try:
        metrics["average_precision"] = average_precision_score(y_true, scores)
    except ValueError:
        metrics["average_precision"] = np.nan

    return {k: float(v) for k, v in metrics.items()}


def permute_block(X, start, end, rng):
    X_perm = X.copy()
    perm_idx = rng.permutation(X.shape[0])
    X_perm[:, start:end] = X_perm[perm_idx, start:end]
    return X_perm


def summarize_results(results_df, baseline_metrics):
    rows = []

    for block, group in results_df.groupby("permuted_block"):
        row = {"permuted_block": block}

        for metric in baseline_metrics:
            values = group[metric].to_numpy(dtype=np.float64)
            drops = baseline_metrics[metric] - values

            row[f"{metric}_baseline"] = baseline_metrics[metric]
            row[f"{metric}_mean"] = float(np.nanmean(values))
            row[f"{metric}_std"] = float(np.nanstd(values))
            row[f"{metric}_drop_mean"] = float(np.nanmean(drops))
            row[f"{metric}_drop_std"] = float(np.nanstd(drops))

        rows.append(row)

    return pd.DataFrame(rows)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    raw_path = OUTPUT_DIR / "block_permutation_importance_raw.csv"
    summary_path = OUTPUT_DIR / "block_permutation_importance_summary.csv"
    json_path = OUTPUT_DIR / "block_permutation_importance_summary.json"

    print("=" * 80)
    print("BLOCK PERMUTATION IMPORTANCE: CoDE + CLIP")
    print("=" * 80)

    bundle = joblib.load(MODEL_PATH)
    model, bundle_dict = extract_model(bundle)

    X = load_features(FEATURES_PATH)
    metadata = pd.read_parquet(METADATA_PATH)
    y_true = metadata["label"].astype(int).to_numpy()

    if X.ndim != 2:
        raise ValueError(f"Le feature devono essere 2D. Shape trovata: {X.shape}")

    if len(X) != len(y_true):
        raise ValueError(f"Mismatch X/y: X={len(X)}, y={len(y_true)}")

    if not np.isfinite(X).all():
        raise ValueError("Le feature contengono NaN oppure Inf.")

    code_dim = 192
    clip_dim = X.shape[1] - code_dim

    print(f"Model:       {type(model).__name__}")
    print(f"X shape:     {X.shape}")
    print(f"CoDE dim:    {code_dim}")
    print(f"CLIP dim:    {clip_dim}")
    print(f"Threshold:   {THRESHOLD}")

    print("\n[BASELINE]")
    baseline_scores = get_scores(model, X)
    baseline_metrics = compute_metrics(y_true, baseline_scores, THRESHOLD)

    for key, value in baseline_metrics.items():
        print(f"  {key}: {value:.6f}")

    rows = []
    rng = np.random.default_rng(42)
    
    blocks = {
        "CoDE": (0, code_dim),
        "CLIP": (code_dim, X.shape[1]),
    }

    print("\n[PERMUTATION]")
    for block_name, (start, end) in blocks.items():
        print(f"  Permuting {block_name}...")

        for repeat in range(20):
            X_perm = permute_block(X, start, end, rng)
            scores_perm = get_scores(model, X_perm)
            metrics_perm = compute_metrics(y_true, scores_perm, THRESHOLD)

            row = {
                "permuted_block": block_name,
                "repeat": repeat,
                **metrics_perm,
            }

            for metric, base_value in baseline_metrics.items():
                row[f"{metric}_drop"] = float(base_value - metrics_perm[metric])

            rows.append(row)

    raw_df = pd.DataFrame(rows)
    summary_df = summarize_results(raw_df, baseline_metrics)

    raw_df.to_csv(raw_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    summary_json = {
        "model_path": str(MODEL_PATH),
        "features_path": str(FEATURES_PATH),
        "metadata_path": str(METADATA_PATH),
        "n_samples": int(len(X)),
        "feature_dim_total": int(X.shape[1]),
        "code_dim": int(code_dim),
        "clip_dim": int(clip_dim),
        "feature_order": "[CoDE | CLIP]",
        "threshold": float(THRESHOLD),
        "bundle_architecture": bundle_dict.get("architecture"),
        "bundle_protocol": bundle_dict.get("protocol"),
        "bundle_model_type": bundle_dict.get("model_type"),
        "bundle_normalization": bundle_dict.get("normalization"),
        "baseline_metrics": baseline_metrics,
        "permutation_summary": summary_df.to_dict(orient="records"),
    }

    with json_path.open("w", encoding="utf-8") as file:
        json.dump(summary_json, file, indent=2, ensure_ascii=False, allow_nan=True)

    print("\n[SAVE]")
    print(f"Raw:     {raw_path}")
    print(f"Summary: {summary_path}")
    print(f"JSON:    {json_path}")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
