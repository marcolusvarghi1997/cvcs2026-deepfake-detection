from pathlib import Path
import json
import hashlib
import re
import shutil
from collections import defaultdict, Counter
from tqdm import tqdm


RAW_ROOT = Path("/work/cvcs2026/resnet_gang/datasets/DF40")

OUT_ROOT = Path("/work/cvcs2026/resnet_gang/datasets/json_standardized")

IMG_ROOT = OUT_ROOT / "images" / "df40"
JSON_PATH = OUT_ROOT / "json" / "df40.jsonl"

IMG_ROOT.mkdir(parents=True, exist_ok=True)
JSON_PATH.parent.mkdir(parents=True, exist_ok=True)

DATASET_NAME = "df40"
SPLIT = "test"

IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def safe_name(x):
    x = str(x).strip()
    x = re.sub(r"[^a-zA-Z0-9._-]+", "_", x)
    return x[:120] if x else "unknown"


def make_hash(*parts):
    key = "|".join(str(p) for p in parts)
    return hashlib.sha1(key.encode()).hexdigest()[:16]


def is_image(p):
    return p.is_file() and p.suffix.lower() in IMG_EXTS


def get_domain_from_path(p):
    parts = p.parts

    if "cdf" in parts or "cdf_real" in parts:
        return "cdf"

    if "ff" in parts or "ff_real" in parts:
        return "ff"

    raise ValueError(f"Dominio non trovato nel path: {p}")


def get_video_id(p):
    return p.parent.name


def collect_real():
    real_root = RAW_ROOT / "real"

    if not real_root.exists():
        raise FileNotFoundError(f"Cartella real non trovata: {real_root}")

    items = []

    for p in sorted(real_root.rglob("*")):
        if not is_image(p):
            continue

        domain = get_domain_from_path(p)
        rel = p.relative_to(RAW_ROOT)

        generator = f"real_{domain}"
        source_model = "real"
        source_id = str(rel)

        items.append({
            "src": p,
            "label": 0,
            "generator": generator,
            "source_model": source_model,
            "dataset": DATASET_NAME,
            "subset": domain,
            "split": SPLIT,
            "source_id": source_id,
            "domain": domain,
            "video_id": get_video_id(p),
        })

    return items


def extract_generator_from_fake_path(rel_parts):
    # fake/blendface/blendface/cdf/frames/...
    # fake/mobileswap/mobileswap/ff/frames/...
    # fake/simswap/cdf/frames/...
    return rel_parts[1]


def collect_fake_by_domain_generator():
    fake_root = RAW_ROOT / "fake"

    if not fake_root.exists():
        raise FileNotFoundError(f"Cartella fake non trovata: {fake_root}")

    by_domain_gen = defaultdict(list)

    for p in sorted(fake_root.rglob("*")):
        if not is_image(p):
            continue

        rel = p.relative_to(RAW_ROOT)
        parts = rel.parts

        if "frames" not in parts:
            continue

        domain = get_domain_from_path(p)
        source_model = extract_generator_from_fake_path(parts)

        generator = f"{source_model}_{domain}"
        source_id = str(rel)

        item = {
            "src": p,
            "label": 1,
            "generator": generator,
            "source_model": source_model,
            "dataset": DATASET_NAME,
            "subset": domain,
            "split": SPLIT,
            "source_id": source_id,
            "domain": domain,
            "video_id": get_video_id(p),
        }

        by_domain_gen[(domain, source_model)].append(item)

    for k in by_domain_gen:
        by_domain_gen[k] = sorted(by_domain_gen[k], key=lambda x: x["source_id"])

    return by_domain_gen


def take_evenly(items, n):
    if n <= 0:
        return []

    if n >= len(items):
        return list(items)

    step = len(items) / n
    selected = []
    used = set()

    for i in range(n):
        idx = int(i * step)

        while idx in used and idx + 1 < len(items):
            idx += 1

        used.add(idx)
        selected.append(items[idx])

    return selected


def select_balanced_fake(by_domain_gen, real_counts_by_domain):
    selected = []

    for domain, quota in sorted(real_counts_by_domain.items()):
        gens = sorted([
            gen for (dom, gen) in by_domain_gen.keys()
            if dom == domain
        ])

        if not gens:
            raise RuntimeError(f"Nessun fake trovato per dominio {domain}")

        base = quota // len(gens)
        rem = quota % len(gens)

        domain_selected = []

        for i, gen in enumerate(gens):
            n = base + (1 if i < rem else 0)
            pool = by_domain_gen[(domain, gen)]

            if len(pool) < n:
                print(
                    f"WARNING: fake {domain}/{gen} ha solo {len(pool)} immagini, "
                    f"richieste {n}. Prendo tutte quelle disponibili."
                )
                n = len(pool)

            domain_selected.extend(take_evenly(pool, n))

        missing = quota - len(domain_selected)

        if missing > 0:
            already = {x["source_id"] for x in domain_selected}
            leftovers = []

            for gen in gens:
                for item in by_domain_gen[(domain, gen)]:
                    if item["source_id"] not in already:
                        leftovers.append(item)

            leftovers = sorted(leftovers, key=lambda x: x["source_id"])
            domain_selected.extend(take_evenly(leftovers, missing))

        if len(domain_selected) != quota:
            raise RuntimeError(
                f"Bilanciamento impossibile per dominio {domain}: "
                f"richieste {quota}, ottenute {len(domain_selected)}"
            )

        selected.extend(domain_selected)

    return selected


def move_file(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)

    if dst.exists() and dst.stat().st_size > 0:
        return

    shutil.move(str(src), str(dst))


def write_jsonl_and_move(items):
    with open(JSON_PATH, "w") as jf:

        for item in tqdm(items, desc="DF40 test balanced"):

            src = item["src"]
            label = item["label"]
            generator = item["generator"]
            source_model = item["source_model"]
            subset = item["subset"]
            source_id = item["source_id"]

            h = make_hash(
                DATASET_NAME,
                subset,
                SPLIT,
                generator,
                label,
                source_id
            )

            out_img = (
                IMG_ROOT
                / SPLIT
                / safe_name(generator)
                / str(label)
                / f"{h}{src.suffix.lower()}"
            )

            move_file(src, out_img)

            rec = {
                "image_path": str(out_img),
                "label": int(label),
                "generator": str(generator),
                "source_model": str(source_model),
                "dataset": DATASET_NAME,
                "subset": subset,
                "split": SPLIT,
                "source_id": source_id
            }

            jf.write(json.dumps(rec) + "\n")


def main():
    real_items = collect_real()
    by_domain_gen = collect_fake_by_domain_generator()

    real_counts_by_domain = Counter(x["domain"] for x in real_items)

    print("REAL COUNTS BY DOMAIN")
    for domain, n in sorted(real_counts_by_domain.items()):
        print(f"{domain}: {n}")

    print("\nFAKE COUNTS BY DOMAIN / GENERATOR")
    for (domain, gen), items in sorted(by_domain_gen.items()):
        print(f"{domain:3s} {gen:15s} {len(items)}")

    fake_items = select_balanced_fake(by_domain_gen, real_counts_by_domain)

    all_items = real_items + fake_items

    print("\nFINAL COUNTS")
    print(f"real:  {len(real_items)}")
    print(f"fake:  {len(fake_items)}")
    print(f"total: {len(all_items)}")

    print("\nFINAL BY LABEL")
    c_label = Counter(x["label"] for x in all_items)
    for label, n in sorted(c_label.items()):
        print(f"{label}: {n}")

    print("\nFINAL BY GENERATOR")
    c_gen = Counter(x["generator"] for x in all_items)
    for gen, n in sorted(c_gen.items()):
        print(f"{gen}: {n}")

    write_jsonl_and_move(all_items)

    print("\nDone")
    print(f"jsonl: {JSON_PATH}")
    print(f"images: {IMG_ROOT / SPLIT}")


if __name__ == "__main__":
    main()