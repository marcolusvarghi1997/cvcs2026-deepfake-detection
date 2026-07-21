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

def parse_args():
    parser = argparse.ArgumentParser(
        description="Allena una score-level fusion usando gli score di più classificatori."
    )

    parser.add_argument(
        "--protocol",
        type=str,
        required=True,
        help="Protocollo/case, es. case1 o case2.",
    )

    parser.add_argument(
        "--score-dirs",
        nargs="+",
        type=Path,
        required=True,
        help=(
            "Cartelle contenenti train_scores.parquet, val_scores.parquet, "
            "test_scores.parquet. Es: .../CLIP/.../logreg .../CoDE/.../logreg"
        ),
    )

    parser.add_argument(
        "--score-names",
        nargs="+",
        type=str,
        required=True,
        help="Nomi degli score, es. CLIP CoDE.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Cartella output della fusion.",
    )

    parser.add_argument(
        "--c-values",
        nargs="+",
        type=float,
        default=[0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
    )

    parser.add_argument(
        "--threshold-steps",
        type=int,
        default=1001,
    )

    parser.add_argument(
        "--max-iter",
        type=int,
        default=2000,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    return parser.parse_args()


def load_scores(score_dirs, score_names, split_name):
    dfs = []

    for score_dir, score_name in zip(score_dirs, score_names):
        if split_name == "test":
            path = (
                score_dir
                / "test"
                / "test_scores.parquet"
            )
        else:
            path = (
                score_dir
                / f"{split_name}_scores.parquet"
            )
        if not path.exists():
            raise FileNotFoundError(f"File score non trovato: {path}")

        df = pd.read_parquet(path)

        required = {"sample_idx", "score", "label"}
        missing = required - set(df.columns)

        if missing:
            raise ValueError(
                f"In {path} mancano colonne {missing}. "
                f"Colonne disponibili: {list(df.columns)}"
            )

        df = df[["sample_idx", "score", "label"]].copy()
        df = df.rename(columns={"score": f"score_{score_name}"})
        dfs.append(df)

    merged = dfs[0]

    for df in dfs[1:]:
        merged = merged.merge(
            df,
            on=["sample_idx", "label"],
            how="inner",
            validate="one_to_one",
        )

    expected_len = len(dfs[0])
    if len(merged) != expected_len:
        raise ValueError(
            f"Merge non allineato nello split {split_name}: "
            f"prima tabella={expected_len}, dopo merge={len(merged)}"
        )

    score_columns = [f"score_{name}" for name in score_names]

    X = merged[score_columns].to_numpy(dtype=np.float32)
    y = merged["label"].astype(int).to_numpy()

    if sorted(np.unique(y).tolist()) != [0, 1]:
        raise ValueError(
            f"Split {split_name}: label attese [0, 1], trovate {sorted(np.unique(y).tolist())}"
        )

    if not np.isfinite(X).all():
        raise ValueError(f"Split {split_name}: score con NaN o Inf")

    return X, y, merged


def compute_metrics(y_true, y_score, threshold):
    y_pred = (y_score >= threshold).astype(np.int64)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    metrics = {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "auroc": float(roc_auc_score(y_true, y_score)),
        "average_precision": float(average_precision_score(y_true, y_score)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
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

        bal_acc = balanced_accuracy_score(y_true, y_pred)
        acc = accuracy_score(y_true, y_pred)

        rows.append({
            "threshold": float(threshold),
            "balanced_accuracy": float(bal_acc),
            "accuracy": float(acc),
        })

        if (
            bal_acc > best_balanced_accuracy
            or (
                np.isclose(bal_acc, best_balanced_accuracy)
                and acc > best_accuracy
            )
            or (
                np.isclose(bal_acc, best_balanced_accuracy)
                and np.isclose(acc, best_accuracy)
                and abs(threshold - 0.5) < abs(best_threshold - 0.5)
            )
        ):
            best_threshold = float(threshold)
            best_balanced_accuracy = float(bal_acc)
            best_accuracy = float(acc)

    return best_threshold, pd.DataFrame(rows)


def save_json(path, data):
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def check_outputs(paths, overwrite):
    existing = [p for p in paths if p.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Esistono già output:\n"
            + "\n".join(str(p) for p in existing)
            + "\nUsa --overwrite."
        )


def main():
    args = parse_args()

    if len(args.score_dirs) != len(args.score_names):
        raise ValueError(
            "--score-dirs e --score-names devono avere la stessa lunghezza."
        )

    if args.output_dir is None:
        args.output_dir = (
            Path("/work/cvcs2026/resnet_gang/results")
            / "Fusion"
            / "classifiers"
            / args.protocol
            / "logreg"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    model_path = args.output_dir / "model.joblib"
    metrics_path = args.output_dir / "metrics.json"
    search_path = args.output_dir / "hyperparameter_search.csv"
    threshold_path = args.output_dir / "threshold_search.csv"
    val_predictions_path = args.output_dir / "val_predictions.parquet"
    test_predictions_path = args.output_dir / "test_predictions.parquet"
    config_path = args.output_dir / "config.json"

    check_outputs(
        [
            model_path,
            metrics_path,
            search_path,
            threshold_path,
            val_predictions_path,
            test_predictions_path,
            config_path,
        ],
        args.overwrite,
    )

    X_train, y_train, train_df = load_scores(
        args.score_dirs,
        args.score_names,
        "train",
    )

    X_val, y_val, val_df = load_scores(
        args.score_dirs,
        args.score_names,
        "val",
    )

    X_test, y_test, test_df = load_scores(
        args.score_dirs,
        args.score_names,
        "test",
    )

    print("[DATA]")
    print(f"Train: {X_train.shape}")
    print(f"Val:   {X_val.shape}")
    print(f"Test:  {X_test.shape}")

    best_model = None
    best_c = None
    best_val_auc = -np.inf
    best_val_ap = -np.inf
    best_val_scores = None

    search_rows = []

    for c in args.c_values:
        print(f"[TRAIN] C={c}")

        model = Pipeline([
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "classifier",
                LogisticRegression(
                    C=c,
                    max_iter=args.max_iter,
                    class_weight="balanced",
                    random_state=args.seed,
                    solver="lbfgs",
                ),
            ),
        ])

        model.fit(X_train, y_train)

        train_scores = model.predict_proba(X_train)[:, 1]
        val_scores = model.predict_proba(X_val)[:, 1]

        train_auc = roc_auc_score(y_train, train_scores)
        train_ap = average_precision_score(y_train, train_scores)
        val_auc = roc_auc_score(y_val, val_scores)
        val_ap = average_precision_score(y_val, val_scores)

        search_rows.append({
            "C": float(c),
            "train_auroc": float(train_auc),
            "train_average_precision": float(train_ap),
            "validation_auroc": float(val_auc),
            "validation_average_precision": float(val_ap),
        })

        print(f"  Train AUROC: {train_auc:.6f}")
        print(f"  Val AUROC:   {val_auc:.6f}")

        if (
            val_auc > best_val_auc
            or (
                np.isclose(val_auc, best_val_auc)
                and val_ap > best_val_ap
            )
            or (
                np.isclose(val_auc, best_val_auc)
                and np.isclose(val_ap, best_val_ap)
                and (best_c is None or c < best_c)
            )
        ):
            best_model = model
            best_c = float(c)
            best_val_auc = float(val_auc)
            best_val_ap = float(val_ap)
            best_val_scores = val_scores.copy()

    pd.DataFrame(search_rows).to_csv(search_path, index=False)

    best_threshold, threshold_df = find_best_threshold(
        y_val,
        best_val_scores,
        args.threshold_steps,
    )

    threshold_df.to_csv(threshold_path, index=False)

    val_metrics, val_pred = compute_metrics(
        y_val,
        best_val_scores,
        best_threshold,
    )

    test_scores = best_model.predict_proba(X_test)[:, 1]

    test_metrics, test_pred = compute_metrics(
        y_test,
        test_scores,
        best_threshold,
    )

    classifier = best_model.named_steps["classifier"]

    fusion_weights = {
        name: float(weight)
        for name, weight in zip(
            args.score_names,
            classifier.coef_[0],
        )
    }

    fusion_intercept = float(classifier.intercept_[0])

    val_out = val_df.copy()
    val_out["fusion_score"] = best_val_scores.astype(np.float32)
    val_out["fusion_pred"] = val_pred.astype(np.int64)
    val_out["decision_threshold"] = best_threshold
    val_out["correct"] = val_out["label"].astype(int) == val_out["fusion_pred"].astype(int)
    val_out.to_parquet(val_predictions_path, index=False)

    test_out = test_df.copy()
    test_out["fusion_score"] = test_scores.astype(np.float32)
    test_out["fusion_pred"] = test_pred.astype(np.int64)
    test_out["decision_threshold"] = best_threshold
    test_out["correct"] = test_out["label"].astype(int) == test_out["fusion_pred"].astype(int)
    test_out.to_parquet(test_predictions_path, index=False)

    bundle = {
        "model": best_model,
        "protocol": args.protocol,
        "score_names": args.score_names,
        "score_dirs": [str(p) for p in args.score_dirs],
        "best_c": best_c,
        "threshold": best_threshold,
        "selection_metric": "validation_auroc",
        "threshold_metric": "validation_balanced_accuracy",
        "input_dimension": int(X_train.shape[1]),
        "label_mapping": {
            "0": "real",
            "1": "fake",
        },
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        "fusion_weights": fusion_weights,
        "fusion_intercept": fusion_intercept,
    }

    joblib.dump(bundle, model_path)

    results = {
        "protocol": args.protocol,
        "score_names": args.score_names,
        "best_c": best_c,
        "threshold": best_threshold,
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        "fusion_weights": fusion_weights,
        "fusion_intercept": fusion_intercept,
    }

    save_json(metrics_path, results)

    config = {
        "protocol": args.protocol,
        "score_dirs": [str(p) for p in args.score_dirs],
        "score_names": args.score_names,
        "output_dir": str(args.output_dir),
        "c_values": args.c_values,
        "threshold_steps": args.threshold_steps,
        "max_iter": args.max_iter,
        "seed": args.seed,
    }

    save_json(config_path, config)

    print()
    print("=" * 80)
    print("FUSION COMPLETATA")
    print("=" * 80)
    print(f"Best C:        {best_c}")
    print(f"Threshold:     {best_threshold:.6f}")
    print(f"Val AUROC:     {val_metrics['auroc']:.6f}")
    print(f"Val AP:        {val_metrics['average_precision']:.6f}")
    print(f"Test AUROC:    {test_metrics['auroc']:.6f}")
    print(f"Test AP:       {test_metrics['average_precision']:.6f}")
    print(f"Test Acc:      {test_metrics['accuracy']:.6f}")
    print()
    print("[SAVED]")
    print(f"Modello:       {model_path}")
    print(f"Metriche:      {metrics_path}")
    print(f"Search C:      {search_path}")
    print(f"Search soglia: {threshold_path}")
    print(f"Val pred:      {val_predictions_path}")
    print(f"Test pred:     {test_predictions_path}")
    print(f"Config:        {config_path}")


if __name__ == "__main__":
    main()