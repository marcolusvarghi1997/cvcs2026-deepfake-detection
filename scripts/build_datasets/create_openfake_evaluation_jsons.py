#!/usr/bin/env python3

import json
import random
from pathlib import Path


INPUT_JSONL = Path(
    "/work/cvcs2026/resnet_gang/datasets/json_standardized/json/openfake.jsonl"
)

OUTPUT_DIR = Path(
    "/work/cvcs2026/resnet_gang/data_jsons"
)

OUTPUT_FLUX2 = OUTPUT_DIR / "openfake_flux2_balanced.jsonl"
OUTPUT_NO_FLUX2 = OUTPUT_DIR / "openfake_no_flux2_balanced.jsonl"

FLUX2_GENERATOR = "flux.2-klein-9b"
SEED = 42


def load_jsonl(path: Path) -> list[dict]:
    rows = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"JSON non valido alla riga {line_number}: {error}"
                ) from error

            required_fields = {
                "image_path",
                "label",
                "generator",
                "source_id",
            }

            missing_fields = required_fields - row.keys()

            if missing_fields:
                raise ValueError(
                    f"Riga {line_number}: campi mancanti {sorted(missing_fields)}"
                )

            rows.append(row)

    return rows


def save_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def image_paths(rows: list[dict]) -> set[str]:
    return {row["image_path"] for row in rows}


def source_ids(rows: list[dict]) -> set[str]:
    return {str(row["source_id"]) for row in rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = load_jsonl(INPUT_JSONL)

    real_rows = [
        row
        for row in rows
        if int(row["label"]) == 0
    ]

    flux2_fake_rows = [
        row
        for row in rows
        if int(row["label"]) == 1
        and row["generator"] == FLUX2_GENERATOR
    ]

    no_flux2_fake_rows = [
        row
        for row in rows
        if int(row["label"]) == 1
        and row["generator"] != FLUX2_GENERATOR
    ]

    print("Input")
    print(f"  Totale:             {len(rows)}")
    print(f"  Real:               {len(real_rows)}")
    print(f"  Fake FLUX2:         {len(flux2_fake_rows)}")
    print(f"  Fake senza FLUX2:   {len(no_flux2_fake_rows)}")

    total_fake = len(flux2_fake_rows) + len(no_flux2_fake_rows)

    if len(real_rows) != total_fake:
        raise ValueError(
            "Il numero di real deve coincidere con il numero totale di fake: "
            f"real={len(real_rows)}, fake={total_fake}"
        )

    if not flux2_fake_rows:
        raise ValueError(
            f"Nessuna immagine trovata per il generatore {FLUX2_GENERATOR}"
        )

    rng = random.Random(SEED)

    shuffled_real_rows = real_rows.copy()
    rng.shuffle(shuffled_real_rows)

    flux2_real_count = len(flux2_fake_rows)

    flux2_real_rows = shuffled_real_rows[:flux2_real_count]
    no_flux2_real_rows = shuffled_real_rows[flux2_real_count:]

    if len(flux2_real_rows) != len(flux2_fake_rows):
        raise ValueError("Il dataset FLUX2 non risulta bilanciato")

    if len(no_flux2_real_rows) != len(no_flux2_fake_rows):
        raise ValueError("Il dataset senza FLUX2 non risulta bilanciato")

    flux2_output_rows = flux2_real_rows + flux2_fake_rows
    no_flux2_output_rows = no_flux2_real_rows + no_flux2_fake_rows

    rng.shuffle(flux2_output_rows)
    rng.shuffle(no_flux2_output_rows)

    real_path_overlap = (
        image_paths(flux2_real_rows)
        & image_paths(no_flux2_real_rows)
    )

    if real_path_overlap:
        raise ValueError(
            f"{len(real_path_overlap)} immagini real presenti in entrambi gli output"
        )

    total_path_overlap = (
        image_paths(flux2_output_rows)
        & image_paths(no_flux2_output_rows)
    )

    if total_path_overlap:
        raise ValueError(
            f"{len(total_path_overlap)} immagini presenti in entrambi gli output"
        )

    total_source_overlap = (
        source_ids(flux2_output_rows)
        & source_ids(no_flux2_output_rows)
    )

    if total_source_overlap:
        print(
            f"ATTENZIONE: {len(total_source_overlap)} source_id compaiono "
            "in entrambi gli output."
        )
        print(
            "Questo può essere normale se source_id non identifica "
            "univocamente immagini real e fake."
        )

    save_jsonl(OUTPUT_FLUX2, flux2_output_rows)
    save_jsonl(OUTPUT_NO_FLUX2, no_flux2_output_rows)

    print()
    print("Output creati")

    print(f"\n{OUTPUT_FLUX2}")
    print(f"  Real:   {len(flux2_real_rows)}")
    print(f"  Fake:   {len(flux2_fake_rows)}")
    print(f"  Totale: {len(flux2_output_rows)}")

    print(f"\n{OUTPUT_NO_FLUX2}")
    print(f"  Real:   {len(no_flux2_real_rows)}")
    print(f"  Fake:   {len(no_flux2_fake_rows)}")
    print(f"  Totale: {len(no_flux2_output_rows)}")

    print()
    print(f"Overlap image_path: {len(total_path_overlap)}")
    print("Completato correttamente.")


if __name__ == "__main__":
    main()