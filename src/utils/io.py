"""I/O helper utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def save_numpy(array: np.ndarray, path: Path | str) -> None:
    """Save a NumPy array to *path* using ``np.save``.

    Creates parent directories if they do not exist.

    Parameters
    ----------
    array:
        Array to save.
    path:
        Destination ``.npy`` file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array)


def load_numpy(path: Path | str) -> np.ndarray:
    """Load a NumPy array from *path*.

    Parameters
    ----------
    path:
        Source ``.npy`` file path.

    Returns
    -------
    np.ndarray
    """
    return np.load(Path(path), allow_pickle=False)


def load_yaml(path: Path | str) -> Any:
    """Load a YAML file and return the parsed content.

    Parameters
    ----------
    path:
        Path to the ``.yaml`` / ``.yml`` file.

    Returns
    -------
    Any
        Parsed YAML content (typically a ``dict``).
    """
    import yaml  # lazy import

    with open(Path(path)) as fh:
        return yaml.safe_load(fh)


def save_yaml(data: Any, path: Path | str) -> None:
    """Serialise *data* to a YAML file at *path*.

    Parameters
    ----------
    data:
        Python object to serialise.
    path:
        Destination ``.yaml`` / ``.yml`` file path.
    """
    import yaml  # lazy import

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        yaml.safe_dump(data, fh, default_flow_style=False)
