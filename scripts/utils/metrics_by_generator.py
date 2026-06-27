#!/usr/bin/env python3

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


# ============================================================
# CONFIGURAZIONE
# ============================================================

INPUT_CSV = Path(
    "/work/cvcs2026/resnet_gang/outputs/CoDE/detection_from_paper/knn/predictions_test.csv"
)

OUTPUT_CSV = Path(
    "/work/cvcs2026/resnet_gang/outputs/CoDE/"
    "detection_from_paper/knn/metrics_test_by_generator.csv"
)

ARCHITECTURE = "CoDE"
PROTOCOL = "openfake"
CLASSIFIER = "knn"

# Colonne presenti nel file predictions_test.csv
LABEL_COLUMN = "label"
SCORE_COLUMN = "fake_score"
PREDICTION_COLUMN = "prediction"
JSON_COLUMN = "original_json"

# Usata solo se PREDICTION_COLUMN non è presente.
THRESHOLD = 0.5


# ============================================================
# FUNZIONI
# ============================================================

def parse_original_json(value):
    """Converte original_json in dizionario."""
    if isinstance(value, dict):
        return value

    if pd.isna(value):
        return {}

    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return {}


def safe_auroc(y_true, scores):
    """Calcola AUROC solo se sono presenti entrambe le classi."""
    if np.unique(y_true).size < 2:
        return np.nan
    return roc_auc_score(y_true, scores)


def safe_average_precision(y_true, scores):
    """Calcola AP solo se esistono campioni positivi."""
    if np.sum(y_true == 1) == 0:
        return np.nan
    return average_precision_score(y_true, scores)


def calculate_metrics(data, generator, n_real_reference):
    y_true = data[LABEL_COLUMN].astype(int).to_numpy()
    scores = data[SCORE_COLUMN].astype(float).to_numpy()

    if PREDICTION_COLUMN in data.columns:
        y_pred = data[PREDICTION_COLUMN].astype(int).to_numpy()
    else:
        y_pred = (scores >= THRESHOLD).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    fake_mask = y_true == 1
    fake_scores = scores[fake_mask]

    n_fake_generator = int(fake_mask.sum())

    return {
        "architecture": ARCHITECTURE,
        "protocol": PROTOCOL,
        "classifier": CLASSIFIER,
        "generator": generator,
        "n_real_reference": n_real_reference,
        "n_fake_generator": n_fake_generator,
        "n_samples": len(data),
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "auroc": safe_auroc(y_true, scores),
        "average_precision": safe_average_precision(y_true, scores),
        "precision": precision_score(
            y_true,
            y_pred,
            pos_label=1,
            zero_division=0,
        ),
        "recall_fake": recall_score(
            y_true,
            y_pred,
            pos_label=1,
            zero_division=0,
        ),
        "f1": f1_score(
            y_true,
            y_pred,
            pos_label=1,
            zero_division=0,
        ),
        "fake_detection_rate": (
            float(tp / (tp + fn))
            if (tp + fn) > 0
            else np.nan
        ),
        "mean_fake_score": (
            float(np.mean(fake_scores))
            if len(fake_scores) > 0
            else np.nan
        ),
        "median_fake_score": (
            float(np.median(fake_scores))
            if len(fake_scores) > 0
            else np.nan
        ),
        "std_fake_score": (
            float(np.std(fake_scores, ddof=0))
            if len(fake_scores) > 0
            else np.nan
        ),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


# ============================================================
# MAIN
# ============================================================

def main():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"File non trovato: {INPUT_CSV}"
        )

    df = pd.read_csv(INPUT_CSV)

    required_columns = {
        LABEL_COLUMN,
        SCORE_COLUMN,
        JSON_COLUMN,
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            "Colonne mancanti nel CSV: "
            + ", ".join(sorted(missing_columns))
        )

    # Estrazione dei metadati dal JSON contenuto nella colonna.
    metadata = df[JSON_COLUMN].apply(parse_original_json)

    df["generator"] = metadata.apply(
        lambda item: item.get(
            "generator",
            item.get("source_model", "unknown"),
        )
    )

    df["source_model"] = metadata.apply(
        lambda item: item.get("source_model", "unknown")
    )

    # Tutte le immagini reali vengono usate come riferimento
    # per ciascun generatore fake.
    real_df = df[df[LABEL_COLUMN] == 0].copy()
    fake_df = df[df[LABEL_COLUMN] == 1].copy()

    n_real_reference = len(real_df)

    if n_real_reference == 0:
        raise ValueError(
            "Il file non contiene immagini reali con label 0."
        )

    if fake_df.empty:
        raise ValueError(
            "Il file non contiene immagini fake con label 1."
        )

    generators = sorted(
        fake_df["generator"]
        .dropna()
        .astype(str)
        .unique()
    )

    results = []

    for generator in generators:
        generator_fake_df = fake_df[
            fake_df["generator"].astype(str) == generator
        ].copy()

        evaluation_df = pd.concat(
            [real_df, generator_fake_df],
            ignore_index=True,
        )

        metrics = calculate_metrics(
            data=evaluation_df,
            generator=generator,
            n_real_reference=n_real_reference,
        )

        results.append(metrics)

        print(
            f"{generator:<30} "
            f"fake={metrics['n_fake_generator']:<6} "
            f"AUROC={metrics['auroc']:.4f} "
            f"AP={metrics['average_precision']:.4f} "
            f"Recall={metrics['recall_fake']:.4f}"
        )

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        by="generator",
        ascending=True,
    ).reset_index(drop=True)

    OUTPUT_CSV.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_df.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    print()
    print(f"Generatori analizzati: {len(results_df)}")
    print(f"Immagini reali di riferimento: {n_real_reference}")
    print(f"Risultati salvati in: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()