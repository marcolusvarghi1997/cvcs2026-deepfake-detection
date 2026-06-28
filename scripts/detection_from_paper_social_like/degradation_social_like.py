#!/usr/bin/env python3

"""Protocollo deterministico di degradazione social-like condiviso tra detector."""

from __future__ import annotations

import hashlib
import io
import random
from pathlib import Path

from PIL import Image, ImageFilter


SOCIAL_PROBABILITY = 0.7
SOCIAL_SEED = 42
SOCIAL_PROTOCOL_NAME = "openfake_social_like_v1"

# L'ordine è fisso e fa parte del protocollo sperimentale.
SOCIAL_TRANSFORM_ORDER = (
    "resize_downsampling",
    "crop",
    "gaussian_blur",
    "jpeg_compression",
)

# Intensità fissate centralmente.
RESIZE_SCALE_RANGE = (0.50, 0.90)
CROP_AREA_RANGE = (0.75, 0.95)
CROP_ASPECT_RATIO_RANGE = (0.90, 1.10)
BLUR_RADIUS_RANGE = (0.40, 1.60)
JPEG_QUALITY_RANGE = (45, 85)

# Resampling espliciti per rendere il protocollo riproducibile.
_DOWNSAMPLE_RESAMPLE = Image.Resampling.BILINEAR
_UPSAMPLE_RESAMPLE = Image.Resampling.BICUBIC
_CROP_RESAMPLE = Image.Resampling.BICUBIC


def _canonical_image_path(image_path: str) -> str:
    """Normalizza il path senza richiedere che il file esista."""
    return Path(image_path).expanduser().resolve(strict=False).as_posix()


def _rng_for_image(image_path: str) -> random.Random:
    """Crea un RNG locale deterministico per la singola immagine."""
    material = (
        f"{SOCIAL_PROTOCOL_NAME}\0"
        f"{SOCIAL_SEED}\0"
        f"{_canonical_image_path(image_path)}"
    ).encode("utf-8")

    digest = hashlib.sha256(material).digest()
    seed = int.from_bytes(digest[:16], byteorder="big", signed=False)
    return random.Random(seed)


def _resize_downsampling(image: Image.Image, rng: random.Random) -> Image.Image:
    original_size = image.size
    scale = rng.uniform(*RESIZE_SCALE_RANGE)

    width = max(1, round(original_size[0] * scale))
    height = max(1, round(original_size[1] * scale))

    image = image.resize((width, height), resample=_DOWNSAMPLE_RESAMPLE)
    return image.resize(original_size, resample=_UPSAMPLE_RESAMPLE)


def _crop(image: Image.Image, rng: random.Random) -> Image.Image:
    original_width, original_height = image.size
    original_area = original_width * original_height

    target_area = original_area * rng.uniform(*CROP_AREA_RANGE)
    aspect_ratio = rng.uniform(*CROP_ASPECT_RATIO_RANGE)

    crop_width = min(
        original_width,
        max(1, round((target_area * aspect_ratio) ** 0.5)),
    )
    crop_height = min(
        original_height,
        max(1, round((target_area / aspect_ratio) ** 0.5)),
    )

    max_left = original_width - crop_width
    max_top = original_height - crop_height
    left = rng.randint(0, max_left) if max_left > 0 else 0
    top = rng.randint(0, max_top) if max_top > 0 else 0

    cropped = image.crop(
        (left, top, left + crop_width, top + crop_height)
    )

    # Si ripristina la dimensione originale: il detector applicherà poi
    # esclusivamente il proprio preprocessing ufficiale.
    return cropped.resize(
        (original_width, original_height),
        resample=_CROP_RESAMPLE,
    )


def _gaussian_blur(image: Image.Image, rng: random.Random) -> Image.Image:
    radius = rng.uniform(*BLUR_RADIUS_RANGE)
    return image.filter(ImageFilter.GaussianBlur(radius=radius))


def _jpeg_compression(image: Image.Image, rng: random.Random) -> Image.Image:
    quality = rng.randint(*JPEG_QUALITY_RANGE)

    buffer = io.BytesIO()
    image.save(
        buffer,
        format="JPEG",
        quality=quality,
        subsampling=2,
        optimize=False,
        progressive=False,
    )
    buffer.seek(0)

    with Image.open(buffer) as decoded:
        return decoded.convert("RGB").copy()


def apply_social_like(
    image: Image.Image,
    image_path: str,
) -> Image.Image:
    """
    Applica il protocollo social-like in modo deterministico per image_path.

    - SOCIAL_PROBABILITY decide se l'immagine resta clean o viene degradata.
    - Se degradata, viene scelta almeno una trasformazione.
    - Ogni trasformazione è selezionata indipendentemente con probabilità 0.5.
    - Le trasformazioni selezionate sono sempre applicate nell'ordine definito
      da SOCIAL_TRANSFORM_ORDER.
    """
    if not 0.0 <= SOCIAL_PROBABILITY <= 1.0:
        raise ValueError(
            "SOCIAL_PROBABILITY deve essere compresa tra 0 e 1."
        )

    rng = _rng_for_image(image_path)
    result = image.convert("RGB")

    if rng.random() >= SOCIAL_PROBABILITY:
        return result.copy()

    selected = {
        name: (rng.random() < 0.5)
        for name in SOCIAL_TRANSFORM_ORDER
    }

    if not any(selected.values()):
        selected[rng.choice(SOCIAL_TRANSFORM_ORDER)] = True

    if selected["resize_downsampling"]:
        result = _resize_downsampling(result, rng)

    if selected["crop"]:
        result = _crop(result, rng)

    if selected["gaussian_blur"]:
        result = _gaussian_blur(result, rng)

    if selected["jpeg_compression"]:
        result = _jpeg_compression(result, rng)

    return result