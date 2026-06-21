#!/usr/bin/env python3

from pathlib import Path
import json
import sys
import time
from collections import Counter

try:
    from PIL import Image, ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = False
except Exception:
    Image = None


# ============================================================
# CONFIG DA MODIFICARE
# ============================================================

JSONL_NAME = "d3_train_100k.jsonl"
# Esempi:
# JSONL_NAME = "cnndetection.jsonl"
# JSONL_NAME = "df40.jsonl"
# JSONL_NAME = "openfake.jsonl"
# JSONL_NAME = "d3_train_100k.jsonl"

ROOT = Path("/work/cvcs2026/resnet_gang/datasets/json_standardized")
JSON_DIR = ROOT / "json"
IMAGES_DIR = ROOT / "images"

OUT_DIR = Path("/work/cvcs2026/resnet_gang/outputs/integrity")

VERIFY_IMAGES = "header"
# "none"   -> controlla solo esistenza file
# "header" -> apre immagine + img.verify(), consigliato
# "full"   -> carica tutta l'immagine, più lento

CHECK_ORPHANS = True

PROGRESS_EVERY = 1_000
MAX_SAMPLES = 2_000

STRICT_REAL_GENERATOR = False
WARN_STRING_LABEL = False

# In D3 real/fake possono avere stesso source_id.
# Quindi meglio lasciarlo False.
STRICT_SOURCE_ID_UNIQUE = False


# ============================================================
# COSTANTI
# ============================================================

IMAGE_EXTS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp",
    ".tif", ".tiff", ".gif"
}

REQUIRED_FIELDS = {
    "image_path",
    "label",
    "generator",
    "dataset",
    "split",
    "source_id",
}

REAL_NAMES = {"real", "authentic", "original", "true"}
BAD_REAL_SOURCES = {"", "none", "null", "unknown"}


# ============================================================
# UTILS
# ============================================================

def now():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def norm_name(x):
    return str(x).strip().lower()


def coerce_label(x):
    """
    Output:
        label: 0/1 oppure None
        issue: None se ok, altrimenti categoria warning/errore
    """
    if isinstance(x, bool):
        return None, "bool_label"

    if isinstance(x, int):
        if x in (0, 1):
            return x, None
        return None, "label_not_0_1"

    if isinstance(x, float):
        if x in (0.0, 1.0):
            return int(x), None
        return None, "label_not_0_1"

    if isinstance(x, str):
        s = x.strip().lower()

        if s in {"0", "real", "authentic", "true"}:
            return 0, "string_label"

        if s in {"1", "fake", "generated", "synthetic"}:
            return 1, "string_label"

        return None, "invalid_label"

    return None, "invalid_label"


def infer_dataset_name_from_jsonl(jsonl_name):
    stem = Path(jsonl_name).stem

    for suffix in [
        "_train_100k",
        "_train",
        "_test_balanced",
        "_balanced",
    ]:
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]

    return stem


def resolve_image_path(raw_path, dataset):
    p = Path(str(raw_path))

    if p.is_absolute():
        return p

    candidates = [
        ROOT / p,
        IMAGES_DIR / p,
        IMAGES_DIR / dataset / p,
        ROOT / str(raw_path).lstrip("/"),
    ]

    seen = set()
    unique = []

    for c in candidates:
        cs = str(c)
        if cs not in seen:
            seen.add(cs)
            unique.append(c)

    for c in unique:
        if c.exists():
            return c

    return unique[0] if unique else p


def verify_image(path):
    if VERIFY_IMAGES == "none":
        return None

    if Image is None:
        return "pillow_import_error"

    try:
        with Image.open(path) as img:
            _ = img.size
            _ = img.mode
            _ = img.format

            if VERIFY_IMAGES == "header":
                img.verify()
            elif VERIFY_IMAGES == "full":
                img.load()
            else:
                return f"unknown_verify_mode: {VERIFY_IMAGES}"

        return None

    except Exception as e:
        return f"image_decode_error: {type(e).__name__}: {e}"


def read_image_meta(path):
    if Image is None:
        return "unknown", 0, 0

    try:
        with Image.open(path) as im:
            return im.format or "unknown", im.width, im.height
    except Exception:
        return "unknown", 0, 0


def limited_append(lst, item):
    if len(lst) < MAX_SAMPLES:
        lst.append(item)


def counter_to_sorted_dict(counter):
    return dict(sorted(counter.items(), key=lambda kv: str(kv[0])))


def counter_to_top_dict(counter, n=100):
    return dict(counter.most_common(n))


# ============================================================
# MAIN
# ============================================================

def main():
    jsonl_path = JSON_DIR / JSONL_NAME

    if not jsonl_path.exists():
        print(f"ERROR: JSONL non trovato: {jsonl_path}", flush=True)
        return 2

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    report_path = OUT_DIR / f"{jsonl_path.stem}_integrity_report.json"

    expected_dataset_from_filename = infer_dataset_name_from_jsonl(JSONL_NAME)

    errors = []
    warnings = []

    error_counts = Counter()
    warning_counts = Counter()

    json_records = 0
    valid_image_files = 0

    label_counts = Counter()
    split_counts = Counter()
    subset_counts = Counter()
    dataset_counts = Counter()

    generator_counts = Counter()
    fake_generator_counts = Counter()
    source_model_counts = Counter()

    split_label_counts = Counter()
    generator_label_counts = Counter()

    image_format_counts = Counter()
    image_size_counts = Counter()

    file_size_total = 0

    seen_image_paths = set()
    seen_source_ids = set()
    seen_record_keys = set()

    referenced_abs_paths = set()

    first_dataset_value = None
    first_split_value = None

    def add_error(line_no, category, message):
        error_counts[category] += 1
        limited_append(errors, {
            "line": line_no,
            "category": category,
            "message": str(message),
        })

    def add_warning(line_no, category, message):
        warning_counts[category] += 1
        limited_append(warnings, {
            "line": line_no,
            "category": category,
            "message": str(message),
        })

    print(f"[{now()}] START integrity check", flush=True)
    print(f"JSONL_NAME={JSONL_NAME}", flush=True)
    print(f"JSONL_PATH={jsonl_path}", flush=True)
    print(f"ROOT={ROOT}", flush=True)
    print(f"IMAGES_DIR={IMAGES_DIR}", flush=True)
    print(f"VERIFY_IMAGES={VERIFY_IMAGES}", flush=True)
    print(f"CHECK_ORPHANS={CHECK_ORPHANS}", flush=True)
    print(f"STRICT_SOURCE_ID_UNIQUE={STRICT_SOURCE_ID_UNIQUE}", flush=True)

    with jsonl_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):

            if PROGRESS_EVERY and line_no % PROGRESS_EVERY == 0:
                print(f"[{now()}] processed {line_no}", flush=True)

            line = line.strip()

            if not line:
                add_error(line_no, "empty_line", "empty JSONL line")
                continue

            try:
                rec = json.loads(line)
            except Exception as e:
                add_error(line_no, "json_parse_error", e)
                continue

            missing_fields = REQUIRED_FIELDS - set(rec.keys())
            if missing_fields:
                add_error(line_no, "missing_fields", sorted(missing_fields))

            dataset = norm_name(rec.get("dataset", expected_dataset_from_filename))
            split = norm_name(rec.get("split", "unknown"))
            subset = norm_name(rec.get("subset", "unknown"))
            generator = norm_name(rec.get("generator", "unknown"))
            source_model = norm_name(rec.get("source_model", generator))
            source_id = str(rec.get("source_id", ""))

            if first_dataset_value is None:
                first_dataset_value = dataset

            if first_split_value is None:
                first_split_value = split

            dataset_counts[dataset] += 1
            split_counts[split] += 1
            subset_counts[subset] += 1
            generator_counts[generator] += 1
            source_model_counts[source_model] += 1

            label, label_issue = coerce_label(rec.get("label"))

            if label_issue:
                if label_issue == "string_label" and not WARN_STRING_LABEL:
                    add_warning(line_no, label_issue, f"label={rec.get('label')!r}")
                else:
                    add_error(line_no, label_issue, f"label={rec.get('label')!r}")

            if label is None:
                continue

            if generator in {"", "none", "null"}:
                add_error(line_no, "empty_generator", generator)
                generator = "unknown"

            if label == 0:
                if generator in BAD_REAL_SOURCES:
                    add_warning(line_no, "real_with_unknown_source", generator)

                if STRICT_REAL_GENERATOR and generator not in REAL_NAMES:
                    add_error(line_no, "real_with_nonreal_generator", generator)

            if label == 1 and generator in REAL_NAMES:
                add_error(line_no, "fake_with_real_generator", generator)

            json_records += 1

            label_counts[label] += 1
            split_label_counts[(split, label)] += 1
            generator_label_counts[(split, generator, label)] += 1

            if label == 1:
                fake_generator_counts[generator] += 1

            image_raw = rec.get("image_path")

            if not image_raw:
                add_error(line_no, "empty_image_path", image_raw)
                continue

            img_path = resolve_image_path(image_raw, dataset)
            img_key = str(img_path)

            if img_key in seen_image_paths:
                add_error(line_no, "duplicate_image_path", img_key)
            else:
                seen_image_paths.add(img_key)

            if source_id:
                if source_id in seen_source_ids:
                    if STRICT_SOURCE_ID_UNIQUE:
                        add_error(line_no, "duplicate_source_id", source_id)
                    else:
                        add_warning(line_no, "duplicate_source_id", source_id)
                else:
                    seen_source_ids.add(source_id)

            # Chiave utile per trovare veri duplicati logici.
            # Non blocca D3 real/fake con stesso source_id, perché label/generator cambiano.
            record_key = (
                dataset,
                split,
                label,
                generator,
                source_id,
                img_key,
            )

            if record_key in seen_record_keys:
                add_error(line_no, "duplicate_record_key", record_key)
            else:
                seen_record_keys.add(record_key)

            if not img_path.exists():
                add_error(line_no, "missing_image", img_path)
                continue

            if not img_path.is_file():
                add_error(line_no, "image_not_file", img_path)
                continue

            try:
                abs_key = str(img_path.resolve())
            except Exception:
                abs_key = str(img_path.absolute())

            referenced_abs_paths.add(abs_key)

            try:
                size = img_path.stat().st_size
                file_size_total += size

                if size <= 0:
                    add_error(line_no, "zero_byte_image", img_path)

            except Exception as e:
                add_error(line_no, "stat_error", f"{img_path}: {e}")

            img_err = verify_image(img_path)

            if img_err:
                add_error(line_no, "bad_image", f"{img_path}: {img_err}")
            else:
                valid_image_files += 1

                if VERIFY_IMAGES != "none":
                    fmt, w, h = read_image_meta(img_path)
                    image_format_counts[fmt] += 1
                    image_size_counts[f"{w}x{h}"] += 1

    orphan_summary = None

    if CHECK_ORPHANS:
        dataset_for_orphans = first_dataset_value or expected_dataset_from_filename
        split_for_orphans = first_split_value or "unknown"

        split_base = IMAGES_DIR / dataset_for_orphans / split_for_orphans
        dataset_base = IMAGES_DIR / dataset_for_orphans

        orphan_base = split_base if split_base.exists() else dataset_base

        print(f"[{now()}] CHECK ORPHANS in {orphan_base}", flush=True)

        total_files = 0
        orphan_files = 0
        orphan_samples = []

        if not orphan_base.exists():
            add_error(0, "missing_images_dir", orphan_base)
            orphan_summary = {
                "images_dir": str(orphan_base),
                "total_files": 0,
                "referenced_files": len(referenced_abs_paths),
                "orphan_files": 0,
                "sample": [],
            }
        else:
            for p in orphan_base.rglob("*"):
                if not p.is_file() or p.suffix.lower() not in IMAGE_EXTS:
                    continue

                total_files += 1

                try:
                    k = str(p.resolve())
                except Exception:
                    k = str(p.absolute())

                if k not in referenced_abs_paths:
                    orphan_files += 1

                    if len(orphan_samples) < 50:
                        orphan_samples.append(str(p))

            orphan_summary = {
                "images_dir": str(orphan_base),
                "total_files": total_files,
                "referenced_files": len(referenced_abs_paths),
                "orphan_files": orphan_files,
                "sample": orphan_samples,
            }

            if orphan_files:
                add_error(
                    0,
                    "orphan_images",
                    f"{orphan_files} files under {orphan_base} not referenced by JSONL"
                )

    total = label_counts[0] + label_counts[1]

    report = {
        "finished_at": now(),
        "ok": sum(error_counts.values()) == 0,

        "jsonl_name": JSONL_NAME,
        "jsonl": str(jsonl_path),
        "root": str(ROOT),
        "images_dir": str(IMAGES_DIR),

        "verify_images": VERIFY_IMAGES,
        "check_orphans": CHECK_ORPHANS,
        "strict_source_id_unique": STRICT_SOURCE_ID_UNIQUE,

        "records": json_records,
        "valid_image_files": valid_image_files,

        "real": label_counts[0],
        "fake": label_counts[1],
        "fake_ratio": (label_counts[1] / total) if total else 0,

        "bytes_images": file_size_total,
        "gb_images": file_size_total / (1024 ** 3),

        "datasets": counter_to_sorted_dict(dataset_counts),
        "splits": counter_to_sorted_dict(split_counts),
        "subsets": counter_to_sorted_dict(subset_counts),

        "num_generators_total": len(generator_counts),
        "num_fake_generators": len(fake_generator_counts),

        "generators": counter_to_sorted_dict(generator_counts),
        "fake_generators": counter_to_sorted_dict(fake_generator_counts),
        "source_models": counter_to_sorted_dict(source_model_counts),

        "split_label_counts": {
            f"{split}|{label}": n
            for (split, label), n in sorted(split_label_counts.items())
        },

        "generator_label_counts": {
            f"{split}|{gen}|{label}": n
            for (split, gen, label), n in sorted(generator_label_counts.items())
        },

        "image_formats": counter_to_sorted_dict(image_format_counts),
        "image_sizes_top": counter_to_top_dict(image_size_counts, 100),

        "error_counts": counter_to_sorted_dict(error_counts),
        "warning_counts": counter_to_sorted_dict(warning_counts),

        "errors_sample": errors,
        "warnings_sample": warnings,

        "orphan_summary": orphan_summary,
    }

    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("\n================ SUMMARY ================", flush=True)
    print(f"jsonl: {jsonl_path}", flush=True)
    print(f"records: {json_records}", flush=True)
    print(f"valid_image_files: {valid_image_files}", flush=True)
    print(f"real: {label_counts[0]}", flush=True)
    print(f"fake: {label_counts[1]}", flush=True)

    if total:
        print(f"fake_ratio: {label_counts[1] / total:.4f}", flush=True)
    else:
        print("fake_ratio: 0", flush=True)

    print(f"num_fake_generators: {len(fake_generator_counts)}", flush=True)

    print("\n================ IMAGE FORMATS ================", flush=True)
    if image_format_counts:
        for fmt, n in image_format_counts.most_common():
            print(f"{fmt}: {n}", flush=True)
    else:
        print("NO IMAGE FORMAT INFO", flush=True)

    print("\n================ WARNINGS ================", flush=True)
    if warning_counts:
        for k, v in warning_counts.most_common():
            print(f"{k}: {v}", flush=True)
    else:
        print("NO WARNINGS", flush=True)

    print("\n================ ERRORS ================", flush=True)
    if error_counts:
        for k, v in error_counts.most_common():
            print(f"{k}: {v}", flush=True)
    else:
        print("NO ERRORS", flush=True)

    if orphan_summary:
        print("\n================ ORPHANS ================", flush=True)
        print(f"images_dir: {orphan_summary['images_dir']}", flush=True)
        print(f"total_files: {orphan_summary['total_files']}", flush=True)
        print(f"referenced_files: {orphan_summary['referenced_files']}", flush=True)
        print(f"orphan_files: {orphan_summary['orphan_files']}", flush=True)

    print(f"\nREPORT JSON: {report_path}", flush=True)

    return 0 if not error_counts else 2


if __name__ == "__main__":
    sys.exit(main())