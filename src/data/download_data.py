"""Utilities for downloading raw datasets."""

from __future__ import annotations

import urllib.request
from pathlib import Path


def download_file(url: str, dest: Path, overwrite: bool = False) -> Path:
    """Download *url* to *dest*, skipping if the file already exists.

    Parameters
    ----------
    url:
        Remote URL to download.
    dest:
        Local destination path (file, not directory).
    overwrite:
        When *True*, re-download even if *dest* exists.

    Returns
    -------
    Path
        The destination path.
    """
    dest = Path(dest)
    if dest.exists() and not overwrite:
        print(f"[download] Already exists, skipping: {dest}")
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"[download] {url} -> {dest}")
    urllib.request.urlretrieve(url, dest)
    return dest
