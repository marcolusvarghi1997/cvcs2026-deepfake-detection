#!/usr/bin/env python3

import argparse
import inspect
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def parse_args():
    parser = argparse.ArgumentParser(
        description="Calcola il t-SNE delle feature e salva un'immagine con label e generator."
    )

    parser.add_argument("--architecture", type=str, required=True)
    parser.add_argument("--protocol", type=str, required=True)
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--features-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-samples", type=int, default=10000)
    parser.add_argument("--pca-components", type=int, default=50)
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--top-generators", type=int, default=10)
    parser.add_argument("--point-size", type=float, default=7.0)
    parser.add_argument("--alpha", type=float, default=0.65)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")

    return parser.parse_args()


def load_split(features_dir: Path, split_name: str):
    features_path = features_dir / f"features_{split_name}.npy"
    metadata_path = features_dir / f"metadata_{split_name}.parquet"

    if not features_path.exists():
        raise FileNotFoundError(f"Feature non trovate: {features_path}")

    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata non trovati: {metadata_path}")

    print("[LOAD]")
    print(f"  Features: {features_path}")
    print(f"  Metadata: {metadata_path}")

    X = np.load(features_path, mmap_mode="r")
    metadata = pd.read_parquet(metadata_path)

    if X.ndim != 2:
        raise ValueError(f"Le feature devono essere 2D. Shape: {X.shape}")

    if len(X) != len(metadata):
        raise ValueError(
            f"Mismatch feature/metadata: features={len(X)}, metadata={len(metadata)}"
        )

    if "label" not in metadata.columns:
        raise ValueError("Colonna 'label' assente nei metadata.")

    labels = metadata["label"].astype(int)

    if not labels.isin([0, 1]).all():
        raise ValueError("La colonna label deve contenere solo 0 e 1.")

    print(f"  Shape: {X.shape}")
    print("  Label counts:", labels.value_counts().sort_index().to_dict())

    return X, metadata


def make_label_categories(metadata: pd.DataFrame):
    return (
        metadata["label"]
        .astype(int)
        .map({0: "real", 1: "fake"})
        .fillna("unknown")
        .astype(str)
    )


def make_generator_categories(metadata: pd.DataFrame, top_generators: int):
    if "generator" not in metadata.columns:
        return pd.Series(["unknown"] * len(metadata), index=metadata.index)

    labels = metadata["label"].astype(int)

    generators = (
        metadata["generator"]
        .astype("string")
        .fillna("unknown")
        .astype(str)
    )

    fake_generators = generators[labels == 1]

    top_generator_names = set(
        fake_generators
        .value_counts()
        .head(top_generators)
        .index
        .tolist()
    )

    categories = pd.Series(index=metadata.index, dtype="object")

    real_mask = labels == 0
    fake_mask = labels == 1

    categories.loc[real_mask] = "real"

    top_fake_mask = fake_mask & generators.isin(top_generator_names)

    categories.loc[top_fake_mask] = generators.loc[top_fake_mask]

    categories.loc[
        fake_mask & ~generators.isin(top_generator_names)
    ] = "other_fake"

    return categories.fillna("unknown").astype(str)


def sample_indices(metadata: pd.DataFrame, max_samples: int, top_generators: int, seed: int):
    n_samples = len(metadata)

    if max_samples <= 0:
        raise ValueError("--max-samples deve essere maggiore di zero.")

    if n_samples <= max_samples:
        return np.arange(n_samples, dtype=np.int64)

    categories = make_generator_categories(
        metadata,
        top_generators=top_generators,
    )

    rng = np.random.default_rng(seed)
    values = categories.to_numpy()
    counts = categories.value_counts()

    allocations = (counts / counts.sum() * max_samples).astype(int)
    allocations = allocations.clip(lower=1)

    while allocations.sum() > max_samples:
        reducible = allocations[allocations > 1]

        if reducible.empty:
            break

        largest_category = reducible.idxmax()
        allocations.loc[largest_category] -= 1

    selected = []

    for category, amount in allocations.items():
        category_indices = np.flatnonzero(values == category)

        amount = min(int(amount), len(category_indices))

        chosen = rng.choice(
            category_indices,
            size=amount,
            replace=False,
        )

        selected.append(chosen)

    selected = np.concatenate(selected)

    if len(selected) < max_samples:
        remaining_indices = np.setdiff1d(
            np.arange(n_samples),
            selected,
            assume_unique=False,
        )

        extra_count = min(
            max_samples - len(selected),
            len(remaining_indices),
        )

        if extra_count > 0:
            extra = rng.choice(
                remaining_indices,
                size=extra_count,
                replace=False,
            )

            selected = np.concatenate([selected, extra])

    rng.shuffle(selected)

    return selected.astype(np.int64)


def run_pca(X: np.ndarray, requested_components: int, seed: int):
    n_components = min(
        requested_components,
        X.shape[1],
        X.shape[0] - 1,
    )

    if n_components < 2:
        raise ValueError("Campioni insufficienti per PCA.")

    print(f"[PCA] {X.shape[1]} -> {n_components}")

    pca = PCA(
        n_components=n_components,
        random_state=seed,
    )

    return pca.fit_transform(X)


def run_tsne(X: np.ndarray, perplexity: float, iterations: int, seed: int):
    if perplexity >= len(X):
        raise ValueError("La perplexity deve essere minore del numero di campioni.")

    kwargs = {
        "n_components": 2,
        "perplexity": perplexity,
        "init": "pca",
        "learning_rate": "auto",
        "random_state": seed,
        "verbose": 1,
    }

    parameters = inspect.signature(TSNE).parameters

    if "max_iter" in parameters:
        kwargs["max_iter"] = iterations
    else:
        kwargs["n_iter"] = iterations

    print("[t-SNE]")
    print(f"  Samples:    {len(X)}")
    print(f"  Perplexity: {perplexity}")
    print(f"  Iterations: {iterations}")

    tsne = TSNE(**kwargs)

    return tsne.fit_transform(X)


def plot_panel(
    axis,
    coordinates: np.ndarray,
    categories: pd.Series,
    point_size: float,
    alpha: float,
):
    values = categories.astype(str).to_numpy()
    unique_categories = sorted(np.unique(values).tolist())

    for category in unique_categories:
        mask = values == category

        axis.scatter(
            coordinates[mask, 0],
            coordinates[mask, 1],
            s=point_size,
            alpha=alpha,
            rasterized=True,
        )


    axis.set_xlabel("t-SNE 1")
    axis.set_ylabel("t-SNE 2")


def main():
    args = parse_args()

    root = Path("/work/cvcs2026/resnet_gang/results")

    if args.features_dir is None:
        args.features_dir = (
            root
            / args.architecture
            / "features"
            / args.protocol
        )

    if args.output_dir is None:
        args.output_dir = (
            root
            / args.architecture
            / "visualizations"
            / args.protocol
            / args.split
            / "tsne"
        )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = args.output_dir / "tsne_summary.png"

    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output già esistente: {output_path}\n"
            "Usa --overwrite per sovrascriverlo."
        )

    X_all, metadata_all = load_split(
        features_dir=args.features_dir,
        split_name=args.split,
    )

    selected_indices = sample_indices(
        metadata=metadata_all,
        max_samples=args.max_samples,
        top_generators=args.top_generators,
        seed=args.seed,
    )

    print("[SAMPLING]")
    print(f"  Totali:      {len(metadata_all)}")
    print(f"  Selezionati: {len(selected_indices)}")

    X_sample = np.asarray(
        X_all[selected_indices],
        dtype=np.float32,
    )

    metadata_sample = (
        metadata_all
        .iloc[selected_indices]
        .reset_index(drop=True)
    )

    if not np.isfinite(X_sample).all():
        raise ValueError("Le feature campionate contengono NaN o Inf.")

    X_pca = run_pca(
        X=X_sample,
        requested_components=args.pca_components,
        seed=args.seed,
    )

    coordinates = run_tsne(
        X=X_pca,
        perplexity=args.perplexity,
        iterations=args.iterations,
        seed=args.seed,
    )

    label_categories = make_label_categories(metadata_sample)

    generator_categories = make_generator_categories(
        metadata_sample,
        top_generators=args.top_generators,
    )







    # Plot 1: Real vs Fake
    label_output_path = args.output_dir / "tsne_real_vs_fake.png"

    if label_output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output già esistente: {label_output_path}\n"
            "Usa --overwrite per sovrascriverlo."
        )

    figure, axis = plt.subplots(figsize=(8, 8))

    plot_panel(
        axis=axis,
        coordinates=coordinates,
        categories=label_categories,
        point_size=args.point_size,
        alpha=args.alpha,
    )

    figure.tight_layout()

    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_xlabel("")
    axis.set_ylabel("")

    for spine in axis.spines.values():
        spine.set_visible(False)


    figure.savefig(
        label_output_path,
        dpi=args.dpi,
        bbox_inches="tight",
    )

    plt.close(figure)







    # Plot 2: Generator
    generator_output_path = args.output_dir / "tsne_generators.png"

    if generator_output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output già esistente: {generator_output_path}\n"
            "Usa --overwrite per sovrascriverlo."
        )

    figure, axis = plt.subplots(figsize=(8, 8))

    plot_panel(
        axis=axis,
        coordinates=coordinates,
        categories=generator_categories,
        title=f"Generator - {args.architecture} - {args.protocol}",
        point_size=args.point_size,
        alpha=args.alpha,
    )
    

    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_xlabel("")
    axis.set_ylabel("")

    for spine in axis.spines.values():
        spine.set_visible(False)

    figure.tight_layout()

    figure.savefig(
        generator_output_path,
        dpi=args.dpi,
        bbox_inches="tight",
    )

    plt.close(figure)











    print()
    print("=" * 80)
    print("t-SNE COMPLETATO")
    print("=" * 80)
    print(f"Real vs Fake: {label_output_path}")
    print(f"Generators:   {generator_output_path}")

    print()
    print("=" * 80)
    print("t-SNE COMPLETATO")
    print("=" * 80)
    print(f"Immagine unica: {output_path}")


if __name__ == "__main__":
    main()