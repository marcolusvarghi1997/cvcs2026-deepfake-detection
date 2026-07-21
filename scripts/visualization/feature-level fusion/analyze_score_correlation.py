#!/usr/bin/env python3

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analizza correlazione tra score di due detector."
    )

    parser.add_argument(
        "--score-dir-a",
        type=Path,
        required=True,
        help="Cartella score primo detector, es. CLIP/logreg",
    )

    parser.add_argument(
        "--score-dir-b",
        type=Path,
        required=True,
        help="Cartella score secondo detector, es. CoDE/logreg",
    )

    parser.add_argument(
        "--name-a",
        type=str,
        default="A",
        help="Nome primo detector, es. CLIP",
    )

    parser.add_argument(
        "--name-b",
        type=str,
        default="B",
        help="Nome secondo detector, es. CoDE",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Cartella output.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    return parser.parse_args()


def get_score_path(score_dir: Path, split: str):
    if split == "test":
        return score_dir / "test" / "test_scores.parquet"
    return score_dir / f"{split}_scores.parquet"


def load_pair(score_dir_a, score_dir_b, name_a, name_b, split):
    path_a = get_score_path(score_dir_a, split)
    path_b = get_score_path(score_dir_b, split)

    if not path_a.exists():
        raise FileNotFoundError(path_a)

    if not path_b.exists():
        raise FileNotFoundError(path_b)

    a = pd.read_parquet(path_a)
    b = pd.read_parquet(path_b)

    required = {"sample_idx", "score", "label"}

    for path, df in [(path_a, a), (path_b, b)]:
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"In {path} mancano colonne: {missing}. "
                f"Colonne disponibili: {list(df.columns)}"
            )

    a = a[["sample_idx", "score", "label"]].rename(
        columns={"score": f"score_{name_a}"}
    )

    b = b[["sample_idx", "score", "label"]].rename(
        columns={"score": f"score_{name_b}"}
    )

    merged = a.merge(
        b,
        on=["sample_idx", "label"],
        how="inner",
        validate="one_to_one",
    )

    if len(merged) != len(a) or len(merged) != len(b):
        raise ValueError(
            f"Merge non allineato per split={split}: "
            f"{name_a}={len(a)}, {name_b}={len(b)}, merged={len(merged)}"
        )

    return merged


def compute_stats(df, name_a, name_b, split):
    col_a = f"score_{name_a}"
    col_b = f"score_{name_b}"

    pearson = df[[col_a, col_b]].corr(method="pearson").iloc[0, 1]
    spearman = df[[col_a, col_b]].corr(method="spearman").iloc[0, 1]

    rows = []

    rows.append({
        "split": split,
        "subset": "all",
        "n_samples": len(df),
        "pearson": float(pearson),
        "spearman": float(spearman),
        f"mean_{name_a}": float(df[col_a].mean()),
        f"std_{name_a}": float(df[col_a].std()),
        f"mean_{name_b}": float(df[col_b].mean()),
        f"std_{name_b}": float(df[col_b].std()),
    })

    for label_value, label_name in [(0, "real"), (1, "fake")]:
        sub = df[df["label"] == label_value]

        if len(sub) < 2:
            continue

        pearson = sub[[col_a, col_b]].corr(method="pearson").iloc[0, 1]
        spearman = sub[[col_a, col_b]].corr(method="spearman").iloc[0, 1]

        rows.append({
            "split": split,
            "subset": label_name,
            "n_samples": len(sub),
            "pearson": float(pearson),
            "spearman": float(spearman),
            f"mean_{name_a}": float(sub[col_a].mean()),
            f"std_{name_a}": float(sub[col_a].std()),
            f"mean_{name_b}": float(sub[col_b].mean()),
            f"std_{name_b}": float(sub[col_b].std()),
        })

    return rows


def main():
    args = parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = args.output_dir / "score_correlation_summary.csv"
    merged_output_dir = args.output_dir / "merged_scores"
    merged_output_dir.mkdir(parents=True, exist_ok=True)

    if summary_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output già esistente: {summary_path}. Usa --overwrite."
        )

    all_rows = []

    for split in ["train", "val", "test"]:
        print(f"[LOAD] split={split}")

        df = load_pair(
            args.score_dir_a,
            args.score_dir_b,
            args.name_a,
            args.name_b,
            split,
        )

        merged_path = merged_output_dir / f"{split}_merged_scores.parquet"

        if merged_path.exists() and not args.overwrite:
            raise FileExistsError(
                f"Output già esistente: {merged_path}. Usa --overwrite."
            )

        df.to_parquet(merged_path, index=False)

        rows = compute_stats(
            df=df,
            name_a=args.name_a,
            name_b=args.name_b,
            split=split,
        )

        all_rows.extend(rows)

        print(df[[f"score_{args.name_a}", f"score_{args.name_b}"]].corr())
        print()

    summary = pd.DataFrame(all_rows)
    summary.to_csv(summary_path, index=False)

    print("=" * 80)
    print("CORRELAZIONE COMPLETATA")
    print("=" * 80)
    print(summary)
    print()
    print(f"Summary: {summary_path}")
    print(f"Merged scores: {merged_output_dir}")


if __name__ == "__main__":
    main()

