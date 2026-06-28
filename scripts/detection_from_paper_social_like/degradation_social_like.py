import hashlib
import io
import random

from PIL import Image, ImageFilter


def deterministic_seed(path: str, base_seed: int = 42) -> int:
    value = f"{base_seed}:{path}".encode("utf-8")
    digest = hashlib.sha256(value).digest()
    return int.from_bytes(digest[:8], "big")


def jpeg_compression(
    image: Image.Image,
    quality: int,
) -> Image.Image:
    buffer = io.BytesIO()

    image.convert("RGB").save(
        buffer,
        format="JPEG",
        quality=quality,
    )

    buffer.seek(0)

    with Image.open(buffer) as compressed:
        return compressed.convert("RGB").copy()


def resize_roundtrip(
    image: Image.Image,
    scale: float,
) -> Image.Image:
    original_width, original_height = image.size

    reduced_width = max(1, round(original_width * scale))
    reduced_height = max(1, round(original_height * scale))

    image = image.resize(
        (reduced_width, reduced_height),
        Image.Resampling.LANCZOS,
    )

    return image.resize(
        (original_width, original_height),
        Image.Resampling.LANCZOS,
    )


def random_crop_roundtrip(
    image: Image.Image,
    retained_ratio: float,
    rng: random.Random,
) -> Image.Image:
    width, height = image.size

    crop_width = max(1, round(width * retained_ratio))
    crop_height = max(1, round(height * retained_ratio))

    max_left = width - crop_width
    max_top = height - crop_height

    left = rng.randint(0, max_left) if max_left > 0 else 0
    top = rng.randint(0, max_top) if max_top > 0 else 0

    image = image.crop(
        (
            left,
            top,
            left + crop_width,
            top + crop_height,
        )
    )

    return image.resize(
        (width, height),
        Image.Resampling.LANCZOS,
    )


def apply_social_like(
    image: Image.Image,
    image_path: str,
    base_seed: int = 42,
    social_probability: float = 0.7,
) -> tuple[Image.Image, dict]:
    rng = random.Random(
        deterministic_seed(image_path, base_seed)
    )

    image = image.convert("RGB")

    metadata = {
        "social_applied": False,
        "resize_scale": None,
        "crop_ratio": None,
        "blur_radius": None,
        "jpeg_quality": None,
    }

    if rng.random() >= social_probability:
        return image, metadata

    metadata["social_applied"] = True

    # Riduzione della risoluzione.
    if rng.random() < 0.7:
        scale = rng.choice([0.4, 0.5, 0.65, 0.8])
        image = resize_roundtrip(image, scale)
        metadata["resize_scale"] = scale

    # Crop moderato.
    if rng.random() < 0.25:
        crop_ratio = rng.choice([0.7, 0.8, 0.9])
        image = random_crop_roundtrip(
            image,
            crop_ratio,
            rng,
        )
        metadata["crop_ratio"] = crop_ratio

    # Blur leggero.
    if rng.random() < 0.15:
        radius = rng.choice([0.4, 0.7, 1.0])
        image = image.filter(
            ImageFilter.GaussianBlur(radius)
        )
        metadata["blur_radius"] = radius

    # Compressione quasi sempre presente.
    if rng.random() < 0.8:
        quality = rng.choice([40, 50, 60, 70, 80])
        image = jpeg_compression(image, quality)
        metadata["jpeg_quality"] = quality

    return image, metadata