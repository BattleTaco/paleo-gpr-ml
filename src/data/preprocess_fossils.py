"""Preprocessing utilities for fossil / bone image data."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np


def load_image(path: Path | str) -> np.ndarray:
    """Load an image from *path* as a NumPy array (H, W, C) in uint8.

    Parameters
    ----------
    path:
        Path to the image file.

    Returns
    -------
    np.ndarray
        Image array with shape ``(H, W, C)`` for colour images or
        ``(H, W)`` for grayscale.
    """
    from PIL import Image  # lazy import

    img = Image.open(Path(path))
    return np.asarray(img)


def resize_images(
    images: Sequence[np.ndarray], size: tuple[int, int] = (224, 224)
) -> list[np.ndarray]:
    """Resize a collection of images to a common *size*.

    Parameters
    ----------
    images:
        Sequence of image arrays.
    size:
        Target ``(width, height)`` in pixels.

    Returns
    -------
    list[np.ndarray]
        Resized image arrays.
    """
    from PIL import Image  # lazy import

    resized = []
    for img in images:
        pil_img = Image.fromarray(img)
        resized.append(np.asarray(pil_img.resize(size, Image.BILINEAR)))
    return resized
