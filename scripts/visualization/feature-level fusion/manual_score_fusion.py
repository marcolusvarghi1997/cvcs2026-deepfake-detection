#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Valuta fusioni manuali pesate tra due score."
    )

    parser.add_argument("--score-dir-a", type=Path, required=True)
    parser.add_argument("--score-dir-b", type=Path, required=True)

    parser.add_argument("--name-a", type=str, default="CLIP")
    parser.add_argument("--name-b", type=str, default="CoDE")

    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Soglia da usare sugli score fusi.",
    )

    parser.add_argument(
        "--weights-a",
        nargs="+",
        type=float,
        default=[1.0, 0.95, 0.9, 0.8, 0.7, 0.5],
        help="Pesi da assegnare al detector A. Il peso B sarà 1 - peso A.",
    )

    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")

    return parser.parse_args()


def get_score_path(score_dir: Path, split: str):
    if split == "test":
        return score_dir / "test" / "test_scores.parquet"
    return score_dir / f"{split}_scores.parquet"


def load_scores(score_dir, name, split):
    path = get_score_path(score_dir, split)

    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_parquet(path)

    required = {"sample_idx", "score", "label"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"In {path} mancano colonne {missing}. "
            f"Colonne disponibili: {list(df.columns)}"
        )

    return df[["sample_idx", "score", "label"]].rename(
        columns={"score": f"score_{name}"}
    )


def compute_metrics(y_true, y_score, threshold):
    y_pred = (y_score >= threshold).astype(np.int64)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    return {
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


def save_json(path, data):
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def main():
    args = parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    output_csv = args.output_dir / f"manual_fusion_{args.split}.csv"
    output_json = args.output_dir / f"manual_fusion_{args.split}.json"

    for path in [output_csv, output_json]:
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"Output già esistente: {path}. Usa --overwrite.")

    df_a = load_scores(args.score_dir_a, args.name_a, args.split)
    df_b = load_scores(args.score_dir_b, args.name_b, args.split)

    df = df_a.merge(
        df_b,
        on=["sample_idx", "label"],
        how="inner",
        validate="one_to_one",
    )

    if len(df) != len(df_a) or len(df) != len(df_b):
        raise ValueError(
            f"Merge non allineato: {args.name_a}={len(df_a)}, "
            f"{args.name_b}={len(df_b)}, merged={len(df)}"
        )

    y = df["label"].astype(int).to_numpy()
    score_a = df[f"score_{args.name_a}"].to_numpy(dtype=np.float64)
    score_b = df[f"score_{args.name_b}"].to_numpy(dtype=np.float64)

    rows = []

    for weight_a in args.weights_a:
        weight_b = 1.0 - weight_a

        fusion_score = (
            weight_a * score_a
            + weight_b * score_b
        )

        metrics = compute_metrics(
            y_true=y,
            y_score=fusion_score,
            threshold=args.threshold,
        )

        row = {
            "split": args.split,
            "name_a": args.name_a,
            "name_b": args.name_b,
            "weight_a": float(weight_a),
            "weight_b": float(weight_b),
            **metrics,
        }

        rows.append(row)

    results = pd.DataFrame(rows)
    results.to_csv(output_csv, index=False)

    save_json(
        output_json,
        {
            "split": args.split,
            "name_a": args.name_a,
            "name_b": args.name_b,
            "score_dir_a": str(args.score_dir_a),
            "score_dir_b": str(args.score_dir_b),
            "threshold": args.threshold,
            "results": rows,
        },
    )

    print("=" * 80)
    print("MANUAL FUSION RESULTS")
    print("=" * 80)
    print(results)
    print()
    print(f"CSV:  {output_csv}")
    print(f"JSON: {output_json}")


if __name__ == "__main__":
    main()


"""
python3 manual_score_fusion.py \
    --score-dir-a /work/cvcs2026/resnet_gang/results/CLIP/classifiers/case1/logreg \
    --score-dir-b /work/cvcs2026/resnet_gang/results/CoDE/classifiers/case1/logreg \
    --name-a CLIP \
    --name-b CoDE \
    --split test \
    --output-dir /work/cvcs2026/resnet_gang/results/Fusion/manual/case1 \
    --overwrite
"""