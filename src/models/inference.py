"""Inference utilities for trained models."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def load_model(checkpoint_path: Path | str):
    """Load a serialised PyTorch model from *checkpoint_path*.

    Parameters
    ----------
    checkpoint_path:
        Path to the ``.pt`` / ``.pth`` checkpoint file.

    Returns
    -------
    torch.nn.Module
        Loaded model in evaluation mode.
    """
    import torch  # lazy import

    checkpoint_path = Path(checkpoint_path)
    model = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.eval()
    return model


def predict(model, inputs: np.ndarray) -> np.ndarray:
    """Run *model* on *inputs* and return predictions as a NumPy array.

    Parameters
    ----------
    model:
        A callable model (e.g. ``torch.nn.Module``).
    inputs:
        Input array.  Will be converted to a ``torch.Tensor``.

    Returns
    -------
    np.ndarray
        Model predictions as a NumPy array.
    """
    import torch  # lazy import

    tensor = torch.from_numpy(inputs).float()
    with torch.no_grad():
        output = model(tensor)
    return output.numpy()
