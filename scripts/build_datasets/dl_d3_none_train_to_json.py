from pathlib import Path
import json
import hashlib
import re
from io import BytesIO
import os
import time
import traceback
import tempfile
import shutil

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from PIL import Image
from tqdm import tqdm
from datasets import load_dataset


# ============================================================
# CONFIG
# ============================================================

OUT_ROOT = Path("/work/cvcs2026/resnet_gang/datasets/json_standardized")

IMG_ROOT = OUT_ROOT / "images" / "d3"
JSON_PATH = OUT_ROOT / "json" / "d3_train_100k.jsonl"

DATASET_NAME = "d3"
SUBSET = "elsa_d3"
SPLIT = "train"

TARGET_PAIRS = 50_000

GEN_COLS = ["gen0", "gen1", "gen2", "gen3"]

# ============================================================
# HF TOKEN
# ============================================================
# Incolla il token tra le virgolette.
# Esempio:
# HF_TOKEN = "hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
#
# Non lasciare spazi prima/dopo.
# Se non vuoi usarlo, lascia stringa vuota.
HF_TOKEN = os.environ.get("HF_TOKEN")

# ============================================================
# PARAMETRI ROBUSTEZZA
# ============================================================

MAX_STREAM_RESTARTS = 30
STREAM_RESTART_SLEEP = 30

REAL_URL_TIMEOUT = 2
FLUSH_EVERY_PAIRS = 250
STATUS_EVERY_ROWS = 5000

HEADERS = {
    "User-Agent": "Mozilla/5.0"
}

IMG_EXTS = {".png"}


# ============================================================
# SETUP
# ============================================================

IMG_ROOT.mkdir(parents=True, exist_ok=True)
JSON_PATH.parent.mkdir(parents=True, exist_ok=True)

if HF_TOKEN.strip():
    HF_TOKEN = HF_TOKEN.strip()
    os.environ["HF_TOKEN"] = HF_TOKEN
    os.environ["HUGGINGFACE_HUB_TOKEN"] = HF_TOKEN
else:
    HF_TOKEN = None

os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")


# ============================================================
# UTILS
# ============================================================

def safe_name(x):
    x = str(x).strip()
    x = re.sub(r"[^a-zA-Z0-9._-]+", "_", x)
    return x[:120] if x else "unknown"


def make_hash(*parts):
    key = "|".join(str(p) for p in parts)
    return hashlib.sha1(key.encode()).hexdigest()[:16]


def pair_key(row_id, gen_suffix, fake_model, fake_filepath):
    return "|".join([
        str(row_id),
        str(gen_suffix),
        str(fake_model),
        str(fake_filepath),
    ])


def make_requests_session():
    session = requests.Session()

    retry = Retry(
        total=0,
        connect=0,
        read=0,
        backoff_factor=0,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
    )

    adapter = HTTPAdapter(max_retries=retry)

    session.mount("http://", adapter)
    session.mount("https://", adapter)

    session.headers.update(HEADERS)

    return session


SESSION = make_requests_session()


def load_real_image_from_url(url, timeout=REAL_URL_TIMEOUT):
    r = SESSION.get(
        url,
        timeout=(1, timeout),
        stream=False,
    )
    r.raise_for_status()

    content_type = r.headers.get("Content-Type", "").lower()
    if "image" not in content_type:
        raise RuntimeError(f"not image content-type: {content_type}")

    return Image.open(BytesIO(r.content)).convert("RGB")


def save_png_atomic(img, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and out_path.stat().st_size > 0:
        return

    tmp_path = out_path.with_suffix(".png.tmp")

    if tmp_path.exists():
        tmp_path.unlink()

    img.convert("RGB").save(tmp_path, format="PNG")
    tmp_path.replace(out_path)


def build_real_output_path(row_id, url):
    h = make_hash(
        DATASET_NAME,
        SUBSET,
        SPLIT,
        "real",
        row_id,
        url,
    )

    return (
        IMG_ROOT
        / SPLIT
        / "real"
        / "0"
        / f"{h}.png"
    )


def build_fake_output_path(row_id, fake_model, fake_filepath, gen_suffix):
    h = make_hash(
        DATASET_NAME,
        SUBSET,
        SPLIT,
        "fake",
        row_id,
        fake_model,
        fake_filepath,
        gen_suffix,
    )

    return (
        IMG_ROOT
        / SPLIT
        / safe_name(fake_model)
        / "1"
        / f"{h}.png"
    )


def open_d3_stream():
    kwargs = {
        "path": "elsaEU/ELSA_D3",
        "split": "train",
        "streaming": True,
    }

    if HF_TOKEN:
        kwargs["token"] = HF_TOKEN

    return load_dataset(**kwargs)


def read_jsonl_records(path):
    records = []

    if not path.exists():
        return records

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                records.append(json.loads(line))
            except Exception as e:
                raise RuntimeError(
                    f"JSONL corrotto alla riga {line_no}: {e}. "
                    f"File: {path}"
                )

    return records


def get_pair_key_from_record(rec):
    row_id = str(rec.get("source_id", ""))

    if int(rec.get("label")) == 0:
        return rec.get("pair_key")

    return rec.get("pair_key")


def clean_jsonl_keep_complete_pairs():
    """
    Resume sicuro:
    - legge il JSONL esistente;
    - tiene solo coppie complete real+fake;
    - elimina eventuali mezze coppie causate da crash;
    - riscrive il JSONL pulito.

    Se parti da zero, non fa nulla.
    """
    if not JSON_PATH.exists():
        return set(), 0, 0

    records = read_jsonl_records(JSON_PATH)

    by_pair = {}

    for rec in records:
        pk = rec.get("pair_key")

        if not pk:
            continue

        by_pair.setdefault(pk, []).append(rec)

    complete_pair_keys = set()
    cleaned_records = []

    for pk, recs in by_pair.items():
        real_recs = [r for r in recs if int(r.get("label")) == 0]
        fake_recs = [r for r in recs if int(r.get("label")) == 1]

        if len(real_recs) >= 1 and len(fake_recs) >= 1:
            # Prende il primo real e il primo fake.
            # Evita duplicati se un crash o rilancio ha duplicato righe.
            cleaned_records.append(real_recs[0])
            cleaned_records.append(fake_recs[0])
            complete_pair_keys.add(pk)

    if len(cleaned_records) != len(records):
        backup = JSON_PATH.with_suffix(".jsonl.bak")
        shutil.copy2(JSON_PATH, backup)

        tmp = JSON_PATH.with_suffix(".jsonl.tmp")

        with tmp.open("w", encoding="utf-8") as f:
            for rec in cleaned_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        tmp.replace(JSON_PATH)

        print(
            f"Cleaned JSONL: kept {len(cleaned_records)} records "
            f"from {len(records)}. Backup: {backup}",
            flush=True,
        )

    n_real = sum(1 for r in cleaned_records if int(r.get("label")) == 0)
    n_fake = sum(1 for r in cleaned_records if int(r.get("label")) == 1)

    return complete_pair_keys, n_real, n_fake


def write_pair(jf, rec_real, rec_fake):
    """
    Scrive sempre real+fake insieme.
    Il JSONL resta coerente: 1 real, 1 fake.
    """
    jf.write(json.dumps(rec_real, ensure_ascii=False) + "\n")
    jf.write(json.dumps(rec_fake, ensure_ascii=False) + "\n")


def verify_existing_png(path):
    if not path.exists() or path.stat().st_size <= 0:
        return False

    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False


# ============================================================
# MAIN
# ============================================================

def main():
    complete_pair_keys, n_real, n_fake = clean_jsonl_keep_complete_pairs()

    n_pairs = min(n_real, n_fake)

    print("START D3 DOWNLOAD", flush=True)
    print(f"JSON_PATH: {JSON_PATH}", flush=True)
    print(f"IMG_ROOT: {IMG_ROOT / SPLIT}", flush=True)
    print(f"HF_TOKEN: {'YES' if HF_TOKEN else 'NO'}", flush=True)
    print(f"existing complete pairs: {n_pairs}", flush=True)
    print(f"existing real: {n_real}", flush=True)
    print(f"existing fake: {n_fake}", flush=True)
    print(f"target pairs: {TARGET_PAIRS}", flush=True)

    if n_pairs >= TARGET_PAIRS:
        print("Dataset già completo.", flush=True)
        return

    pbar = tqdm(
        total=TARGET_PAIRS,
        initial=n_pairs,
        desc="D3 pairs 50k",
    )

    mode = "a" if JSON_PATH.exists() else "w"

    stream_restarts = 0
    rows_seen_total = 0
    written_pairs_this_run = 0
    skipped_rows = 0

    with JSON_PATH.open(mode, encoding="utf-8") as jf:

        while stream_restarts <= MAX_STREAM_RESTARTS:

            if n_pairs >= TARGET_PAIRS:
                break

            try:
                print(
                    f"\nSTREAM ATTEMPT {stream_restarts + 1}/{MAX_STREAM_RESTARTS + 1}",
                    flush=True,
                )

                ds = open_d3_stream()

                rows_seen_this_attempt = 0

                for row in ds:
                    rows_seen_total += 1
                    rows_seen_this_attempt += 1

                    if rows_seen_this_attempt % STATUS_EVERY_ROWS == 0:
                        print(
                            f"rows_seen_this_attempt={rows_seen_this_attempt} "
                            f"pairs={n_pairs}/{TARGET_PAIRS} "
                            f"written_pairs_this_run={written_pairs_this_run} "
                            f"skipped_rows={skipped_rows}",
                            flush=True,
                        )

                    if n_pairs >= TARGET_PAIRS:
                        break

                    row_id = str(row.get("id") or f"d3_train_row_{rows_seen_this_attempt}")
                    url = row.get("url")

                    if not url:
                        skipped_rows += 1
                        continue

                    # Una sola fake per real.
                    # Rotazione bilanciata tra gen0, gen1, gen2, gen3.
                    gen_idx = n_pairs % len(GEN_COLS)
                    gen_suffix = GEN_COLS[gen_idx]

                    fake_img_key = f"image_{gen_suffix}"
                    fake_model_key = f"model_{gen_suffix}"
                    fake_filepath_key = f"filepath_{gen_suffix}"

                    fake_img = row.get(fake_img_key)
                    fake_model = row.get(fake_model_key) or f"unknown_{gen_suffix}"
                    fake_filepath = row.get(fake_filepath_key) or ""

                    if fake_img is None:
                        skipped_rows += 1
                        continue

                    pk = pair_key(row_id, gen_suffix, fake_model, fake_filepath)

                    if pk in complete_pair_keys:
                        continue

                    real_out = build_real_output_path(row_id, url)
                    fake_out = build_fake_output_path(
                        row_id,
                        fake_model,
                        fake_filepath,
                        gen_suffix,
                    )

                    try:
                        # Prima salva entrambe le immagini.
                        # Solo se entrambe riescono, scrive entrambe le righe JSON.
                        if not verify_existing_png(real_out):
                            real_img = load_real_image_from_url(url)
                            save_png_atomic(real_img, real_out)

                        if not verify_existing_png(fake_out):
                            save_png_atomic(fake_img, fake_out)

                        rec_real = {
                            "image_path": str(real_out),
                            "label": 0,
                            "generator": "real",
                            "source_model": "real",
                            "dataset": DATASET_NAME,
                            "subset": SUBSET,
                            "split": SPLIT,
                            "source_id": row_id,

                            # campi extra utili
                            "pair_key": pk,
                            "url": url,
                            "generator_slot": "real",
                            "source_filepath": "",
                        }

                        rec_fake = {
                            "image_path": str(fake_out),
                            "label": 1,
                            "generator": str(fake_model),
                            "source_model": str(fake_model),
                            "dataset": DATASET_NAME,
                            "subset": SUBSET,
                            "split": SPLIT,
                            "source_id": row_id,

                            # campi extra utili
                            "pair_key": pk,
                            "url": url,
                            "generator_slot": gen_suffix,
                            "source_filepath": fake_filepath,
                        }

                        write_pair(jf, rec_real, rec_fake)

                        complete_pair_keys.add(pk)

                        n_pairs += 1
                        n_real += 1
                        n_fake += 1
                        written_pairs_this_run += 1

                        pbar.update(1)

                        if written_pairs_this_run > 0 and written_pairs_this_run % FLUSH_EVERY_PAIRS == 0:
                            jf.flush()
                            os.fsync(jf.fileno())

                    except Exception as e:
                        skipped_rows += 1
                        print(
                            f"WARNING row failed row_id={row_id}: "
                            f"{type(e).__name__}: {e}",
                            flush=True,
                        )
                        continue

                # Se lo stream finisce naturalmente, esce.
                break

            except Exception as e:
                stream_restarts += 1

                print("\nSTREAM ERROR", flush=True)
                print(f"type: {type(e).__name__}", flush=True)
                print(f"error: {e}", flush=True)
                traceback.print_exc()

                jf.flush()
                os.fsync(jf.fileno())

                if stream_restarts > MAX_STREAM_RESTARTS:
                    print("MAX_STREAM_RESTARTS reached. Stop.", flush=True)
                    break

                print(f"Sleeping {STREAM_RESTART_SLEEP}s before restart...", flush=True)
                time.sleep(STREAM_RESTART_SLEEP)

        jf.flush()
        os.fsync(jf.fileno())

    pbar.close()

    print("\nDONE", flush=True)
    print(f"pairs: {n_pairs}", flush=True)
    print(f"real: {n_real}", flush=True)
    print(f"fake: {n_fake}", flush=True)
    print(f"written_pairs_this_run: {written_pairs_this_run}", flush=True)
    print(f"skipped_rows: {skipped_rows}", flush=True)
    print(f"rows_seen_total: {rows_seen_total}", flush=True)
    print(f"jsonl: {JSON_PATH}", flush=True)
    print(f"images: {IMG_ROOT / SPLIT}", flush=True)

    if n_pairs < TARGET_PAIRS:
        raise RuntimeError(
            f"Dataset incompleto: pairs={n_pairs}/{TARGET_PAIRS}, "
            f"real={n_real}, fake={n_fake}"
        )


if __name__ == "__main__":
    main()