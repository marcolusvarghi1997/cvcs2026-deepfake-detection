#!/usr/bin/env python3

import argparse
import hashlib
import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


# ============================================================
# PATH
# ============================================================

STANDARD_JSON_DIR = Path(
    "/work/cvcs2026/resnet_gang/datasets/json_standardized/json"
)

TEST_JSON_DIR = Path(
    "/work/cvcs2026/resnet_gang/test_jsons"
)

D3_JSON = STANDARD_JSON_DIR / "d3_train_100k.jsonl"
CNN_JSON = STANDARD_JSON_DIR / "cnndetection.jsonl"
DF40_JSON = STANDARD_JSON_DIR / "df40.jsonl"
OPENFAKE_JSON = STANDARD_JSON_DIR / "openfake.jsonl"

OPENFAKE_NO_FLUX2_JSON = (
    STANDARD_JSON_DIR / "openfake_no_flux2_balanced.jsonl"
)

OPENFAKE_FLUX2_JSON = (
    STANDARD_JSON_DIR / "openfake_flux2_balanced.jsonl"
)

VAL_RATIO = 0.20
DEFAULT_SEED = 42


# ============================================================
# JSONL
# ============================================================

def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"File non trovato: {path}")

    rows: list[dict] = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"JSON non valido in {path}, "
                    f"riga {line_number}: {error}"
                ) from error

            required = {
                "image_path",
                "label",
                "generator",
                "dataset",
                "source_id",
            }

            missing = required - row.keys()

            if missing:
                raise ValueError(
                    f"{path}, riga {line_number}: "
                    f"campi mancanti {sorted(missing)}"
                )

            row["label"] = int(row["label"])
            rows.append(row)

    return rows


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(
                json.dumps(row, ensure_ascii=False) + "\n"
            )


# ============================================================
# GROUPING
# ============================================================

def normalize_text(value: object) -> str:
    return str(value).strip().replace("\\", "/")


def df40_video_group(source_id: str) -> str:
    """
    Esempio:

    fake/blendface/blendface/cdf/frames/id0_id16_0003/153.png

    diventa:

    fake/blendface/blendface/cdf/frames/id0_id16_0003
    """
    source_path = Path(normalize_text(source_id))
    return normalize_text(source_path.parent)


def get_group_id(row: dict) -> str:
    dataset = normalize_text(row["dataset"]).lower()
    source_id = normalize_text(row["source_id"])

    if dataset == "d3":
        group_value = source_id

    elif dataset == "df40":
        group_value = df40_video_group(source_id)

    elif dataset == "cnndetection":
        group_value = source_id

    elif dataset == "openfake":
        group_value = source_id

    else:
        group_value = source_id

    return f"{dataset}::{group_value}"


def get_group_stratum(group_rows: list[dict]) -> str:
    """
    Costruisce uno strato per mantenere una distribuzione simile
    di dataset, label e generatori tra train e validation.

    Per D3, una coppia real/fake viene stratificata usando
    il generatore fake della coppia.
    """
    datasets = sorted({
        normalize_text(row["dataset"]).lower()
        for row in group_rows
    })

    labels = sorted({
        int(row["label"])
        for row in group_rows
    })

    fake_generators = sorted({
        normalize_text(row["generator"]).lower()
        for row in group_rows
        if int(row["label"]) == 1
    })

    real_generators = sorted({
        normalize_text(row["generator"]).lower()
        for row in group_rows
        if int(row["label"]) == 0
    })

    if fake_generators:
        generator_part = ",".join(fake_generators)
    else:
        generator_part = ",".join(real_generators)

    return (
        f"datasets={','.join(datasets)}"
        f"|labels={','.join(map(str, labels))}"
        f"|generators={generator_part}"
    )


# ============================================================
# SPLIT
# ============================================================

def stable_seed(base_seed: int, text: str) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], byteorder="big")
    return base_seed ^ value


def grouped_stratified_split(
    rows: list[dict],
    val_ratio: float,
    seed: int,
) -> tuple[list[dict], list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)

    for row in rows:
        groups[get_group_id(row)].append(row)

    strata: dict[str, list[tuple[str, list[dict]]]] = defaultdict(list)

    for group_id, group_rows in groups.items():
        stratum = get_group_stratum(group_rows)
        strata[stratum].append((group_id, group_rows))

    train_group_ids: set[str] = set()
    val_group_ids: set[str] = set()

    for stratum, stratum_groups in sorted(strata.items()):
        rng = random.Random(stable_seed(seed, stratum))
        shuffled_groups = stratum_groups.copy()
        rng.shuffle(shuffled_groups)

        total_rows = sum(
            len(group_rows)
            for _, group_rows in shuffled_groups
        )

        target_val_rows = round(total_rows * val_ratio)
        current_val_rows = 0

        for index, (group_id, group_rows) in enumerate(
            shuffled_groups
        ):
            groups_remaining = len(shuffled_groups) - index

            can_assign_to_val = (
                len(shuffled_groups) == 1
                or groups_remaining > 1
                or current_val_rows == 0
            )

            distance_if_train = abs(
                target_val_rows - current_val_rows
            )

            distance_if_val = abs(
                target_val_rows
                - (current_val_rows + len(group_rows))
            )

            assign_to_val = (
                can_assign_to_val
                and distance_if_val <= distance_if_train
            )

            if assign_to_val:
                val_group_ids.add(group_id)
                current_val_rows += len(group_rows)
            else:
                train_group_ids.add(group_id)

        stratum_ids = [
            group_id
            for group_id, _ in shuffled_groups
        ]

        stratum_train = [
            group_id
            for group_id in stratum_ids
            if group_id in train_group_ids
        ]

        stratum_val = [
            group_id
            for group_id in stratum_ids
            if group_id in val_group_ids
        ]

        if len(stratum_ids) >= 2 and not stratum_val:
            moved = stratum_train[-1]
            train_group_ids.remove(moved)
            val_group_ids.add(moved)

        elif len(stratum_ids) >= 2 and not stratum_train:
            moved = stratum_val[-1]
            val_group_ids.remove(moved)
            train_group_ids.add(moved)

    train_rows: list[dict] = []
    val_rows: list[dict] = []

    for row in rows:
        group_id = get_group_id(row)

        if group_id in val_group_ids:
            val_rows.append(row)

        elif group_id in train_group_ids:
            train_rows.append(row)

        else:
            raise RuntimeError(
                f"Gruppo non assegnato: {group_id}"
            )

    random.Random(seed).shuffle(train_rows)
    random.Random(seed + 1).shuffle(val_rows)

    return train_rows, val_rows


# ============================================================
# VALIDATION
# ============================================================

def image_paths(rows: Iterable[dict]) -> set[str]:
    return {
        normalize_text(row["image_path"])
        for row in rows
    }


def group_ids(rows: Iterable[dict]) -> set[str]:
    return {
        get_group_id(row)
        for row in rows
    }


def validate_protocol(
    train_rows: list[dict],
    val_rows: list[dict],
    test_rows: list[dict],
) -> None:
    train_groups = group_ids(train_rows)
    val_groups = group_ids(val_rows)

    train_val_group_overlap = train_groups & val_groups

    if train_val_group_overlap:
        raise ValueError(
            "Leakage tra train e validation: "
            f"{len(train_val_group_overlap)} gruppi"
        )

    train_paths = image_paths(train_rows)
    val_paths = image_paths(val_rows)
    test_paths = image_paths(test_rows)

    overlaps = {
        "train-validation": train_paths & val_paths,
        "train-test": train_paths & test_paths,
        "validation-test": val_paths & test_paths,
    }

    for name, overlap in overlaps.items():
        if overlap:
            raise ValueError(
                f"Leakage {name}: "
                f"{len(overlap)} image_path in comune"
            )

    train_openfake_sources = {
        normalize_text(row["source_id"])
        for row in train_rows
        if normalize_text(row["dataset"]).lower() == "openfake"
    }

    val_openfake_sources = {
        normalize_text(row["source_id"])
        for row in val_rows
        if normalize_text(row["dataset"]).lower() == "openfake"
    }

    test_openfake_sources = {
        normalize_text(row["source_id"])
        for row in test_rows
        if normalize_text(row["dataset"]).lower() == "openfake"
    }

    openfake_overlap = (
        (train_openfake_sources & test_openfake_sources)
        | (val_openfake_sources & test_openfake_sources)
    )

    if openfake_overlap:
        raise ValueError(
            "Leakage OpenFake tra sviluppo e test: "
            f"{len(openfake_overlap)} source_id"
        )


# ============================================================
# REPORT
# ============================================================

def summarize(rows: list[dict]) -> dict:
    labels = Counter(
        str(int(row["label"]))
        for row in rows
    )

    datasets = Counter(
        normalize_text(row["dataset"])
        for row in rows
    )

    generators = Counter(
        normalize_text(row["generator"])
        for row in rows
    )

    return {
        "total": len(rows),
        "labels": dict(sorted(labels.items())),
        "datasets": dict(sorted(datasets.items())),
        "generators": dict(
            sorted(
                generators.items(),
                key=lambda item: (-item[1], item[0]),
            )
        ),
        "groups": len(group_ids(rows)),
    }


def print_summary(
    protocol_name: str,
    train_rows: list[dict],
    val_rows: list[dict],
    test_rows: list[dict],
) -> None:
    print()
    print("=" * 80)
    print(protocol_name)
    print("=" * 80)

    for split_name, rows in (
        ("TRAIN", train_rows),
        ("VALIDATION", val_rows),
        ("TEST", test_rows),
    ):
        summary = summarize(rows)

        print(f"\n{split_name}")
        print(f"  Totale:  {summary['total']}")
        print(f"  Gruppi:  {summary['groups']}")
        print(f"  Label:   {summary['labels']}")
        print(f"  Dataset: {summary['datasets']}")


# ============================================================
# OUTPUT
# ============================================================

def prepare_output_directory(
    output_dir: Path,
    overwrite: bool,
) -> None:
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"La cartella esiste già: {output_dir}\n"
                "Usa --overwrite per ricrearla."
            )

        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=False)


def create_protocol(
    protocol_name: str,
    development_rows: list[dict],
    test_rows: list[dict],
    seed: int,
    overwrite: bool,
    description: str,
) -> None:
    output_dir = TEST_JSON_DIR / protocol_name

    prepare_output_directory(
        output_dir=output_dir,
        overwrite=overwrite,
    )

    train_rows, val_rows = grouped_stratified_split(
        rows=development_rows,
        val_ratio=VAL_RATIO,
        seed=seed,
    )

    validate_protocol(
        train_rows=train_rows,
        val_rows=val_rows,
        test_rows=test_rows,
    )

    train_path = output_dir / "train.jsonl"
    val_path = output_dir / "val.jsonl"
    test_path = output_dir / "test.jsonl"
    manifest_path = output_dir / "manifest.json"

    write_jsonl(train_path, train_rows)
    write_jsonl(val_path, val_rows)
    write_jsonl(test_path, test_rows)

    manifest = {
        "protocol": protocol_name,
        "description": description,
        "seed": seed,
        "validation_ratio": VAL_RATIO,
        "grouping": {
            "d3": "dataset + source_id",
            "cnndetection": "dataset + source_id",
            "df40": (
                "dataset + directory video "
                "ricavata da source_id"
            ),
            "openfake": "dataset + source_id",
        },
        "source_files": {
            "d3": str(D3_JSON),
            "cnndetection": str(CNN_JSON),
            "df40": str(DF40_JSON),
            "openfake_complete": str(OPENFAKE_JSON),
            "openfake_no_flux2_balanced": str(
                OPENFAKE_NO_FLUX2_JSON
            ),
            "openfake_flux2_balanced": str(
                OPENFAKE_FLUX2_JSON
            ),
        },
        "files": {
            "train": str(train_path),
            "validation": str(val_path),
            "test": str(test_path),
        },
        "statistics": {
            "train": summarize(train_rows),
            "validation": summarize(val_rows),
            "test": summarize(test_rows),
        },
    }

    with manifest_path.open("w", encoding="utf-8") as file:
        json.dump(
            manifest,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print_summary(
        protocol_name=protocol_name,
        train_rows=train_rows,
        val_rows=val_rows,
        test_rows=test_rows,
    )

    print("\nFile creati:")
    print(f"  {train_path}")
    print(f"  {val_path}")
    print(f"  {test_path}")
    print(f"  {manifest_path}")


# ============================================================
# MAIN
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Crea i protocolli test1 e test2 con split "
            "train/validation senza leakage."
        )
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Elimina e ricrea le cartelle "
            "test_jsons/test1 e test_jsons/test2."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("Caricamento dataset...")

    d3_rows = load_jsonl(D3_JSON)
    cnn_rows = load_jsonl(CNN_JSON)
    df40_rows = load_jsonl(DF40_JSON)
    openfake_all_rows = load_jsonl(OPENFAKE_JSON)

    openfake_no_flux2_rows = load_jsonl(
        OPENFAKE_NO_FLUX2_JSON
    )

    openfake_flux2_rows = load_jsonl(
        OPENFAKE_FLUX2_JSON
    )

    base_development_rows = (
        d3_rows
        + cnn_rows
        + df40_rows
    )

    print(f"D3:                         {len(d3_rows)}")
    print(f"CNNDetection:               {len(cnn_rows)}")
    print(f"DF40:                       {len(df40_rows)}")
    print(f"OpenFake completo:          {len(openfake_all_rows)}")
    print(
        "OpenFake senza FLUX2:      "
        f"{len(openfake_no_flux2_rows)}"
    )
    print(
        "OpenFake FLUX2 bilanciato: "
        f"{len(openfake_flux2_rows)}"
    )

    create_protocol(
        protocol_name="test1",
        development_rows=base_development_rows,
        test_rows=openfake_all_rows,
        seed=args.seed,
        overwrite=args.overwrite,
        description=(
            "Training e validation su D3 + CNNDetection + DF40; "
            "test su tutto OpenFake."
        ),
    )

    create_protocol(
        protocol_name="test2",
        development_rows=(
            base_development_rows
            + openfake_no_flux2_rows
        ),
        test_rows=openfake_flux2_rows,
        seed=args.seed,
        overwrite=args.overwrite,
        description=(
            "Training e validation su D3 + CNNDetection + DF40 "
            "+ OpenFake senza FLUX2; test su OpenFake FLUX2 "
            "con real disgiunte."
        ),
    )

    print()
    print("=" * 80)
    print("TUTTI I PROTOCOLLI CREATI CORRETTAMENTE")
    print("=" * 80)


if __name__ == "__main__":
    main()