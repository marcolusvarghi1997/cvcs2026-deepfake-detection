#!/usr/bin/env python3

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
from sklearn.model_selection import train_test_split
from sklearn.neighbors import (
    KNeighborsClassifier,
    NearestNeighbors,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ============================================================
# CONFIGURAZIONE GLOBALE
# ============================================================

ARCHITECTURE = "CLIP"  # "CLIP" o "CoDE"
PROTOCOL = "test1"

ROOT_RESULTS_DIR = Path(
    "/work/cvcs2026/resnet_gang/results"
)

# Se None, viene costruita automaticamente come:
# results/<ARCHITECTURE>/features/<PROTOCOL>
FEATURES_DIR = None

# Se None, viene costruita automaticamente come:
# results/<ARCHITECTURE>/classifiers/<PROTOCOL>/knn
OUTPUT_DIR = None


# ============================================================
# CAMPIONAMENTO TRAIN
# ============================================================

# Percentuale del train da usare:
#
# 1.0  = 100%
# 0.5  = 50%
# 0.1  = 10%
# 0.05 = 5%
#
# Il validation set viene sempre usato interamente.
TRAIN_FRACTION = 1

# Mantiene la proporzione tra label real e fake.
STRATIFY_BY_LABEL = True

RANDOM_SEED = 42


# ============================================================
# IPERPARAMETRI K-NN
# ============================================================

K_VALUES = [
    1,
    3,
    5,
    11,
    21,
]

WEIGHTS_VALUES = [
    "uniform",
    "distance",
]

# Per embedding CLIP e CoDE è consigliata cosine.
METRIC = "cosine"

# Con cosine viene utilizzata ricerca brute-force.
ALGORITHM = "brute"

# Numero di CPU.
# -1 significa tutte quelle disponibili.
N_JOBS = 8

# Generalmente False per embedding CLIP e CoDE.
STANDARDIZE = False


# ============================================================
# SOGLIA
# ============================================================

THRESHOLD_STEPS = 1001


# ============================================================
# OUTPUT
# ============================================================

OVERWRITE = False

MODEL_FILENAME = "model.joblib"
METRICS_FILENAME = "validation_metrics.json"
SEARCH_FILENAME = "hyperparameter_search.csv"
THRESHOLD_FILENAME = "threshold_search.csv"
PREDICTIONS_FILENAME = "validation_predictions.parquet"
CONFIG_FILENAME = "config.json"


# ============================================================
# LOAD DATA
# ============================================================

def load_split(
    split_name: str,
    features_dir: Path,
):
    features_path = (
        features_dir
        / f"features_{split_name}.npy"
    )

    metadata_path = (
        features_dir
        / f"metadata_{split_name}.parquet"
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

    X = np.load(
        features_path,
        mmap_mode=None,
    )

    metadata = pd.read_parquet(
        metadata_path
    )

    if X.ndim != 2:
        raise ValueError(
            "Le feature devono essere una matrice 2D. "
            f"Shape trovata: {X.shape}"
        )

    if len(X) != len(metadata):
        raise ValueError(
            f"Mismatch nello split {split_name}: "
            f"features={len(X)}, "
            f"metadata={len(metadata)}"
        )

    if "label" not in metadata.columns:
        raise ValueError(
            f"Colonna 'label' assente in {metadata_path}. "
            f"Colonne disponibili: "
            f"{list(metadata.columns)}"
        )

    y = (
        metadata["label"]
        .astype(int)
        .to_numpy()
    )

    unique_labels = sorted(
        np.unique(y).tolist()
    )

    if unique_labels != [0, 1]:
        raise ValueError(
            "Le label devono essere [0, 1]. "
            f"Valori trovati: {unique_labels}"
        )

    if not np.isfinite(X).all():
        raise ValueError(
            "Le feature contengono NaN o Inf "
            f"nello split {split_name}."
        )

    X = X.astype(
        np.float32,
        copy=False,
    )

    print(f"  Shape X: {X.shape}")
    print(f"  Shape y: {y.shape}")
    print(
        "  Label counts:",
        dict(
            pd.Series(y)
            .value_counts()
            .sort_index()
        ),
    )

    return X, y, metadata


# ============================================================
# TRAIN SAMPLING
# ============================================================

def sample_training_set(
    X: np.ndarray,
    y: np.ndarray,
    metadata: pd.DataFrame,
    fraction: float,
    seed: int,
    stratify_by_label: bool,
):
    if not 0.0 < fraction <= 1.0:
        raise ValueError(
            "TRAIN_FRACTION deve essere "
            "maggiore di 0 e minore o uguale a 1."
        )

    if np.isclose(fraction, 1.0):
        print()
        print("[TRAIN SAMPLING]")
        print("  Uso il 100% del train.")

        return (
            X,
            y,
            metadata.reset_index(drop=True),
            np.arange(len(y)),
        )

    all_indices = np.arange(
        len(y),
        dtype=np.int64,
    )

    stratify_values = (
        y
        if stratify_by_label
        else None
    )

    sampled_indices, _ = train_test_split(
        all_indices,
        train_size=fraction,
        random_state=seed,
        shuffle=True,
        stratify=stratify_values,
    )

    sampled_indices = np.sort(
        sampled_indices
    )

    X_sampled = X[
        sampled_indices
    ]

    y_sampled = y[
        sampled_indices
    ]

    metadata_sampled = (
        metadata
        .iloc[sampled_indices]
        .reset_index(drop=True)
    )

    print()
    print("[TRAIN SAMPLING]")
    print(
        f"  Frazione richiesta: {fraction}"
    )
    print(
        f"  Percentuale:        "
        f"{fraction * 100:.2f}%"
    )
    print(
        f"  Campioni originali: {len(X)}"
    )
    print(
        f"  Campioni utilizzati:{len(X_sampled)}"
    )
    print(
        "  Label originali:",
        dict(
            pd.Series(y)
            .value_counts()
            .sort_index()
        ),
    )
    print(
        "  Label campionate:",
        dict(
            pd.Series(y_sampled)
            .value_counts()
            .sort_index()
        ),
    )

    sampled_unique_labels = sorted(
        np.unique(y_sampled).tolist()
    )

    if sampled_unique_labels != [0, 1]:
        raise ValueError(
            "Il campionamento non contiene entrambe "
            "le classi 0 e 1. "
            "Aumenta TRAIN_FRACTION."
        )

    return (
        X_sampled,
        y_sampled,
        metadata_sampled,
        sampled_indices,
    )


# ============================================================
# STANDARDIZZAZIONE PER LA RICERCA
# ============================================================

def prepare_search_features(
    X_train: np.ndarray,
    X_val: np.ndarray,
    standardize: bool,
):
    if not standardize:
        return X_train, X_val

    print()
    print("[STANDARDIZE]")
    print(
        "  Fit StandardScaler sul train campionato."
    )

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train
    ).astype(
        np.float32,
        copy=False,
    )

    X_val_scaled = scaler.transform(
        X_val
    ).astype(
        np.float32,
        copy=False,
    )

    return (
        X_train_scaled,
        X_val_scaled,
    )


# ============================================================
# MODEL
# ============================================================

def build_model(
    n_neighbors: int,
    weights: str,
):
    classifier = KNeighborsClassifier(
        n_neighbors=n_neighbors,
        weights=weights,
        metric=METRIC,
        algorithm=ALGORITHM,
        n_jobs=N_JOBS,
    )

    if STANDARDIZE:
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
# K-NN SCORE
# ============================================================

def compute_knn_scores(
    neighbor_distances: np.ndarray,
    neighbor_indices: np.ndarray,
    train_labels: np.ndarray,
    n_neighbors: int,
    weights: str,
):
    distances = neighbor_distances[
        :,
        :n_neighbors,
    ]

    indices = neighbor_indices[
        :,
        :n_neighbors,
    ]

    neighbor_labels = train_labels[
        indices
    ].astype(
        np.float64,
        copy=False,
    )

    if weights == "uniform":
        return neighbor_labels.mean(
            axis=1
        )

    if weights != "distance":
        raise ValueError(
            f"Weights non supportati: {weights}"
        )

    scores = np.empty(
        len(distances),
        dtype=np.float64,
    )

    zero_distance_mask = (
        distances == 0.0
    )

    rows_with_zero_distance = (
        zero_distance_mask.any(axis=1)
    )

    if rows_with_zero_distance.any():
        zero_rows = np.where(
            rows_with_zero_distance
        )[0]

        for row in zero_rows:
            mask = zero_distance_mask[row]

            scores[row] = (
                neighbor_labels[row][mask].mean()
            )

    normal_rows = np.where(
        ~rows_with_zero_distance
    )[0]

    if len(normal_rows) > 0:
        normal_distances = distances[
            normal_rows
        ]

        normal_labels = neighbor_labels[
            normal_rows
        ]

        inverse_distances = (
            1.0 / normal_distances
        )

        scores[normal_rows] = (
            (
                inverse_distances
                * normal_labels
            ).sum(axis=1)
            / inverse_distances.sum(axis=1)
        )

    return scores


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
):
    if THRESHOLD_STEPS < 2:
        raise ValueError(
            "THRESHOLD_STEPS deve essere almeno 2."
        )

    thresholds = np.linspace(
        0.0,
        1.0,
        THRESHOLD_STEPS,
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
# SAVE HELPERS
# ============================================================

def save_json(
    path: Path,
    data: dict,
):
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
    output_paths: list[Path],
):
    existing = [
        path
        for path in output_paths
        if path.exists()
    ]

    if existing and not OVERWRITE:
        existing_text = "\n".join(
            str(path)
            for path in existing
        )

        raise FileExistsError(
            "Esistono già alcuni output:\n"
            f"{existing_text}\n"
            "Imposta OVERWRITE = True "
            "per sovrascriverli."
        )


# ============================================================
# MAIN
# ============================================================

def main():
    features_dir = FEATURES_DIR

    if features_dir is None:
        features_dir = (
            ROOT_RESULTS_DIR
            / ARCHITECTURE
            / "features"
            / PROTOCOL
        )

    output_dir = OUTPUT_DIR

    if output_dir is None:
        output_dir = (
            ROOT_RESULTS_DIR
            / ARCHITECTURE
            / "classifiers"
            / PROTOCOL
            / "knn"
        )

    if not K_VALUES:
        raise ValueError(
            "K_VALUES non può essere vuoto."
        )

    if any(k <= 0 for k in K_VALUES):
        raise ValueError(
            "Tutti i valori di K_VALUES "
            "devono essere maggiori di zero."
        )

    invalid_weights = [
        value
        for value in WEIGHTS_VALUES
        if value not in {
            "uniform",
            "distance",
        }
    ]

    if invalid_weights:
        raise ValueError(
            "Valori weights non validi: "
            f"{invalid_weights}"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_path = (
        output_dir / MODEL_FILENAME
    )

    metrics_path = (
        output_dir / METRICS_FILENAME
    )

    search_path = (
        output_dir / SEARCH_FILENAME
    )

    threshold_path = (
        output_dir / THRESHOLD_FILENAME
    )

    predictions_path = (
        output_dir / PREDICTIONS_FILENAME
    )

    config_path = (
        output_dir / CONFIG_FILENAME
    )

    output_paths = [
        model_path,
        metrics_path,
        search_path,
        threshold_path,
        predictions_path,
        config_path,
    ]

    check_output_files(
        output_paths
    )

    print()
    print("=" * 80)
    print("CONFIGURAZIONE")
    print("=" * 80)
    print(
        f"Architecture:   {ARCHITECTURE}"
    )
    print(
        f"Protocol:       {PROTOCOL}"
    )
    print(
        f"Features dir:   {features_dir}"
    )
    print(
        f"Output dir:     {output_dir}"
    )
    print(
        f"Train fraction: {TRAIN_FRACTION}"
    )
    print(
        f"Metric:         {METRIC}"
    )
    print(
        f"K values:       {K_VALUES}"
    )
    print(
        f"Weights:        {WEIGHTS_VALUES}"
    )

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    X_train_full, y_train_full, train_metadata_full = (
        load_split(
            split_name="train",
            features_dir=features_dir,
        )
    )

    X_val, y_val, val_metadata = load_split(
        split_name="val",
        features_dir=features_dir,
    )

    if (
        X_train_full.shape[1]
        != X_val.shape[1]
    ):
        raise ValueError(
            "Dimensione feature differente "
            "tra train e validation: "
            f"train={X_train_full.shape[1]}, "
            f"val={X_val.shape[1]}"
        )

    # --------------------------------------------------------
    # TRAIN SAMPLING
    # --------------------------------------------------------

    (
        X_train,
        y_train,
        train_metadata,
        sampled_indices,
    ) = sample_training_set(
        X=X_train_full,
        y=y_train_full,
        metadata=train_metadata_full,
        fraction=TRAIN_FRACTION,
        seed=RANDOM_SEED,
        stratify_by_label=STRATIFY_BY_LABEL,
    )

    del X_train_full
    del y_train_full
    del train_metadata_full

    max_k = max(K_VALUES)

    if max_k > len(X_train):
        raise ValueError(
            f"Il valore massimo di k ({max_k}) "
            "è maggiore del numero di campioni "
            f"di train utilizzati ({len(X_train)})."
        )

    # --------------------------------------------------------
    # PREPARE FEATURES
    # --------------------------------------------------------

    (
        X_train_search,
        X_val_search,
    ) = prepare_search_features(
        X_train=X_train,
        X_val=X_val,
        standardize=STANDARDIZE,
    )

    # --------------------------------------------------------
    # CALCOLO DEI VICINI UNA SOLA VOLTA
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("RICERCA VICINI")
    print("=" * 80)
    print(
        f"Calcolo i {max_k} vicini più vicini "
        "per tutti i campioni di validation."
    )

    nearest_neighbors = NearestNeighbors(
        n_neighbors=max_k,
        metric=METRIC,
        algorithm=ALGORITHM,
        n_jobs=N_JOBS,
    )

    nearest_neighbors.fit(
        X_train_search
    )

    (
        neighbor_distances,
        neighbor_indices,
    ) = nearest_neighbors.kneighbors(
        X_val_search,
        return_distance=True,
    )

    # --------------------------------------------------------
    # HYPERPARAMETER SEARCH
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("RICERCA IPERPARAMETRI K-NN")
    print("=" * 80)

    search_rows = []

    best_k = None
    best_weights = None
    best_val_auc = -np.inf
    best_val_ap = -np.inf
    best_val_scores = None

    for k_value in K_VALUES:
        for weights_value in WEIGHTS_VALUES:
            val_scores = compute_knn_scores(
                neighbor_distances=neighbor_distances,
                neighbor_indices=neighbor_indices,
                train_labels=y_train,
                n_neighbors=k_value,
                weights=weights_value,
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
                "n_neighbors": int(k_value),
                "weights": weights_value,
                "metric": METRIC,
                "validation_auroc": float(
                    val_auc
                ),
                "validation_average_precision": float(
                    val_ap
                ),
            })

            print()
            print(
                f"[KNN] k={k_value}, "
                f"weights={weights_value}"
            )
            print(
                f"  Val AUROC: {val_auc:.6f}"
            )
            print(
                f"  Val AP:    {val_ap:.6f}"
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

            same_metrics_smaller_k = (
                np.isclose(
                    val_auc,
                    best_val_auc,
                )
                and np.isclose(
                    val_ap,
                    best_val_ap,
                )
                and (
                    best_k is None
                    or k_value < best_k
                )
            )

            same_metrics_same_k_distance = (
                np.isclose(
                    val_auc,
                    best_val_auc,
                )
                and np.isclose(
                    val_ap,
                    best_val_ap,
                )
                and best_k is not None
                and k_value == best_k
                and weights_value == "distance"
                and best_weights != "distance"
            )

            if (
                better_auc
                or same_auc_better_ap
                or same_metrics_smaller_k
                or same_metrics_same_k_distance
            ):
                best_k = int(k_value)
                best_weights = weights_value
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

    if best_k is None:
        raise RuntimeError(
            "Nessuna configurazione k-NN selezionata."
        )

    # --------------------------------------------------------
    # MODELLO FINALE
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("FIT MODELLO FINALE")
    print("=" * 80)

    best_model = build_model(
        n_neighbors=best_k,
        weights=best_weights,
    )

    best_model.fit(
        X_train,
        y_train,
    )

    # Ricalcola gli score usando esattamente
    # il modello che verrà salvato.
    best_val_scores = (
        best_model
        .predict_proba(X_val)[:, 1]
    )

    # --------------------------------------------------------
    # THRESHOLD SEARCH
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("RICERCA SOGLIA SU VALIDATION")
    print("=" * 80)

    (
        best_threshold,
        threshold_df,
    ) = find_best_threshold(
        y_true=y_val,
        y_score=best_val_scores,
    )

    threshold_df.to_csv(
        threshold_path,
        index=False,
    )

    (
        validation_metrics,
        val_predictions,
    ) = compute_metrics(
        y_true=y_val,
        y_score=best_val_scores,
        threshold=best_threshold,
    )

    print()
    print(f"Best k:         {best_k}")
    print(
        f"Best weights:   {best_weights}"
    )
    print(f"Metric:         {METRIC}")
    print(
        f"Best threshold: {best_threshold:.6f}"
    )

    print()
    print("[VALIDATION METRICS]")
    print(
        "Accuracy:          "
        f"{validation_metrics['accuracy']:.6f}"
    )
    print(
        "Balanced accuracy: "
        f"{validation_metrics['balanced_accuracy']:.6f}"
    )
    print(
        "AUROC:             "
        f"{validation_metrics['auroc']:.6f}"
    )
    print(
        "Average Precision: "
        f"{validation_metrics['average_precision']:.6f}"
    )
    print(
        "Precision:          "
        f"{validation_metrics['precision']:.6f}"
    )
    print(
        "Recall:             "
        f"{validation_metrics['recall']:.6f}"
    )
    print(
        "F1:                 "
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
        "fake_score"
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
        "classifier_type": "knn",
        "architecture": ARCHITECTURE,
        "protocol": PROTOCOL,
        "features_dir": str(
            features_dir
        ),
        "best_k": best_k,
        "best_n_neighbors": best_k,
        "best_weights": best_weights,
        "metric": METRIC,
        "threshold": best_threshold,
        "selection_metric": (
            "validation_auroc"
        ),
        "threshold_metric": (
            "validation_balanced_accuracy"
        ),
        "standardize": STANDARDIZE,
        "feature_dimension": int(
            X_train.shape[1]
        ),
        "original_train_samples": int(
            len(sampled_indices)
            / TRAIN_FRACTION
        ) if TRAIN_FRACTION < 1.0 else int(
            len(X_train)
        ),
        "train_samples": int(
            len(X_train)
        ),
        "train_fraction": float(
            TRAIN_FRACTION
        ),
        "validation_samples": int(
            len(X_val)
        ),
        "random_seed": RANDOM_SEED,
        "label_mapping": {
            "0": "real",
            "1": "fake",
        },
        "validation_metrics": (
            validation_metrics
        ),

        # Compatibilità con eventuali script
        # originariamente scritti per logreg.
        "best_c": None,
    }

    joblib.dump(
        model_bundle,
        model_path,
        compress=3,
    )

    results = {
        "classifier_type": "knn",
        "architecture": ARCHITECTURE,
        "protocol": PROTOCOL,
        "best_k": best_k,
        "best_n_neighbors": best_k,
        "best_weights": best_weights,
        "metric": METRIC,
        "threshold": best_threshold,
        "selection_metric": (
            "validation_auroc"
        ),
        "threshold_metric": (
            "validation_balanced_accuracy"
        ),
        "standardize": STANDARDIZE,
        "train_fraction": float(
            TRAIN_FRACTION
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
        "best_c": None,
    }

    save_json(
        metrics_path,
        results,
    )

    config = {
        "classifier_type": "knn",
        "architecture": ARCHITECTURE,
        "protocol": PROTOCOL,
        "features_dir": str(
            features_dir
        ),
        "output_dir": str(
            output_dir
        ),
        "train_fraction": float(
            TRAIN_FRACTION
        ),
        "stratify_by_label": (
            STRATIFY_BY_LABEL
        ),
        "random_seed": RANDOM_SEED,
        "k_values": K_VALUES,
        "weights_values": WEIGHTS_VALUES,
        "metric": METRIC,
        "algorithm": ALGORITHM,
        "threshold_steps": (
            THRESHOLD_STEPS
        ),
        "standardize": STANDARDIZE,
        "n_jobs": N_JOBS,
    }

    save_json(
        config_path,
        config,
    )

    print()
    print("=" * 80)
    print("SALVATAGGIO COMPLETATO")
    print("=" * 80)
    print(
        f"Modello:          {model_path}"
    )
    print(
        f"Metriche val:     {metrics_path}"
    )
    print(
        f"Ricerca parametri:{search_path}"
    )
    print(
        f"Ricerca soglia:   {threshold_path}"
    )
    print(
        f"Predizioni val:   {predictions_path}"
    )
    print(
        f"Configurazione:   {config_path}"
    )


if __name__ == "__main__":
    main()