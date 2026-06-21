from pathlib import Path
import json
import hashlib
import re
import subprocess
import zipfile
import shutil

from PIL import Image
from tqdm import tqdm


RAW_ROOT = Path("/work/cvcs2026/resnet_gang/datasets/raw/CNNDetection")
ZIP_PATH = RAW_ROOT / "CNN_synth_testset.zip"
TEST_ROOT = RAW_ROOT

OUT_ROOT = Path("/work/cvcs2026/resnet_gang/datasets/json_standardized")
IMG_ROOT = OUT_ROOT / "images" / "cnndetection"
JSON_PATH = OUT_ROOT / "json" / "cnndetection.jsonl"

DATASET_NAME = "cnndetection"
SUBSET = "cnn_synth_testset"
SPLIT = "test"

URL = "https://huggingface.co/datasets/sywang/CNNDetection/resolve/main/CNN_synth_testset.zip"

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def safe_name(x):
    x = str(x).strip()
    x = re.sub(r"[^a-zA-Z0-9._-]+", "_", x)
    return x[:120] if x else "unknown"


def make_hash(*parts):
    key = "|".join(str(p) for p in parts)
    return hashlib.sha1(key.encode()).hexdigest()[:16]


def run(cmd, cwd=None):
    print(f"$ {' '.join(str(x) for x in cmd)}", flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def ensure_testset():
    RAW_ROOT.mkdir(parents=True, exist_ok=True)

    if TEST_ROOT.exists() and any(TEST_ROOT.rglob("*")):
        print(f"Test set già presente: {TEST_ROOT}. Skip download.", flush=True)
        return

    if not ZIP_PATH.exists():
        run([
            "wget",
            "-O",
            str(ZIP_PATH),
            URL
        ])

    print(f"Unzipping {ZIP_PATH}", flush=True)

    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        zf.extractall(RAW_ROOT)

    ZIP_PATH.unlink(missing_ok=True)

    # Dopo unzip, a volte la cartella può avere un nome diverso o essere estratta direttamente.
    # Normalizziamo cercando una directory plausibile.
    if not TEST_ROOT.exists():
        candidates = [
            p for p in RAW_ROOT.iterdir()
            if p.is_dir() and "test" in p.name.lower()
        ]

        if len(candidates) == 1:
            candidates[0].rename(TEST_ROOT)

    if not TEST_ROOT.exists():
        raise FileNotFoundError(
            f"Non trovo la cartella test estratta: {TEST_ROOT}. "
            f"Controlla contenuto di {RAW_ROOT}"
        )


def find_label_dirs():
    """
    Cerca tutte le directory chiamate 0_real e 1_fake dentro TEST_ROOT.

    Strutture supportate:
    - CNN_synth_testset/<generator>/<class>/0_real/*.jpg
    - CNN_synth_testset/<generator>/<class>/1_fake/*.jpg
    - CNN_synth_testset/<generator>/0_real/*.jpg
    - CNN_synth_testset/<generator>/1_fake/*.jpg
    """
    real_dirs = []
    fake_dirs = []

    for d in TEST_ROOT.rglob("*"):
        if not d.is_dir():
            continue

        name = d.name.lower()

        if name == "0_real":
            real_dirs.append(d)

        elif name == "1_fake":
            fake_dirs.append(d)

    real_dirs = sorted(real_dirs, key=lambda x: str(x))
    fake_dirs = sorted(fake_dirs, key=lambda x: str(x))

    return real_dirs, fake_dirs


def infer_generator_and_class(label_dir):
    """
    label_dir esempio:
    TEST_ROOT/biggan/airplane/0_real
    TEST_ROOT/biggan/airplane/1_fake

    generator = primo livello sotto TEST_ROOT
    class_name = cartella subito prima di 0_real/1_fake, se diversa dal generator
    """
    rel = label_dir.relative_to(TEST_ROOT)
    parts = rel.parts

    if len(parts) >= 1:
        generator = parts[0]
    else:
        generator = "unknown"

    if len(parts) >= 2:
        class_name = parts[-2]
    else:
        class_name = "unknown"

    if class_name == generator:
        class_name = "unknown"

    return generator, class_name


def collect_images():
    real_items = []
    fake_items = []

    real_dirs, fake_dirs = find_label_dirs()

    print(f"real dirs found: {len(real_dirs)}", flush=True)
    print(f"fake dirs found: {len(fake_dirs)}", flush=True)

    for d in real_dirs:
        gen_from_path, class_name = infer_generator_and_class(d)

        for p in d.rglob("*"):
            if p.is_file() and p.suffix.lower() in IMG_EXTS:
                real_items.append({
                    "src": p,
                    "label": 0,
                    "generator": "real",
                    "source_model": "real",
                    "class_name": class_name,
                    "source_generator_dir": gen_from_path,
                })

    for d in fake_dirs:
        generator, class_name = infer_generator_and_class(d)

        for p in d.rglob("*"):
            if p.is_file() and p.suffix.lower() in IMG_EXTS:
                fake_items.append({
                    "src": p,
                    "label": 1,
                    "generator": generator,
                    "source_model": generator,
                    "class_name": class_name,
                    "source_generator_dir": generator,
                })

    real_items = sorted(real_items, key=lambda x: str(x["src"]))
    fake_items = sorted(fake_items, key=lambda x: str(x["src"]))

    return real_items, fake_items


def save_png(src_path, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and out_path.stat().st_size > 0:
        return

    with Image.open(src_path) as img:
        img.convert("RGB").save(out_path)


def write_records(items, jf):
    written = 0

    for item in tqdm(items, desc=f"label={items[0]['label'] if items else 'none'}"):

        src_path = item["src"]
        label = item["label"]
        generator = item["generator"]
        source_model = item["source_model"]

        rel = src_path.relative_to(TEST_ROOT)
        source_id = str(rel)

        h = make_hash(
            DATASET_NAME,
            SUBSET,
            SPLIT,
            generator,
            label,
            source_id,
        )

        out_img = (
            IMG_ROOT
            / SPLIT
            / safe_name(generator)
            / str(label)
            / f"{h}.png"
        )

        save_png(src_path, out_img)

        rec = {
            "image_path": str(out_img),
            "label": int(label),
            "generator": str(generator),
            "source_model": str(source_model),
            "dataset": DATASET_NAME,
            "subset": SUBSET,
            "split": SPLIT,
            "source_id": source_id
        }

        jf.write(json.dumps(rec) + "\n")
        written += 1

    return written


def main():
    IMG_ROOT.mkdir(parents=True, exist_ok=True)
    JSON_PATH.parent.mkdir(parents=True, exist_ok=True)

    ensure_testset()

    real_items, fake_items = collect_images()

    print(f"real found: {len(real_items)}", flush=True)
    print(f"fake found: {len(fake_items)}", flush=True)

    if not real_items:
        raise RuntimeError("Nessuna immagine real trovata. Controlla struttura del testset.")

    if not fake_items:
        raise RuntimeError("Nessuna immagine fake trovata. Controlla struttura del testset.")

    # Se vuoi ripartire pulito, elimina vecchio JSONL.
    if JSON_PATH.exists():
        print(f"Overwriting JSONL: {JSON_PATH}", flush=True)

    with open(JSON_PATH, "w") as jf:
        wr = write_records(real_items, jf)
        wf = write_records(fake_items, jf)

    print("Done", flush=True)
    print(f"written real: {wr}", flush=True)
    print(f"written fake: {wf}", flush=True)
    print(f"written total: {wr + wf}", flush=True)
    print(f"jsonl: {JSON_PATH}", flush=True)
    print(f"images: {IMG_ROOT / SPLIT}", flush=True)


if __name__ == "__main__":
    main()