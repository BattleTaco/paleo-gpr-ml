"""Polarity-aware matched filter for GPR traces. The non-ML physics baseline (Cbase).

A buried target reflects the source pulse, and I know the pulse shape (Ricker at the survey
frequency). So I slide that template along each trace and compute a normalized cross
correlation (NCC). The peak tells me where a reflector sits and how well it matches the pulse.
The sign of the NCC at that peak gives the polarity, which is what separates high permittivity
targets (bone, negative top reflection) from low permittivity targets (voids, positive). The
physics behind the polarity split is in docs/notes/08_forward_model_1d_validation.md.

I compare learned detectors against this baseline in experiment_03. If a CNN cannot beat a
plain matched filter, the extra machinery is not earning its place.

NCC is bounded to [-1, 1], so the score is easy to threshold and read.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.data.forward_model_1d import ricker_wavelet


@dataclass(frozen=True)
class Detection:
    """Result of running the matched filter on one trace."""

    detected: bool
    index: int  # sample index of the best match
    score: float  # NCC at the best match, in [-1, 1]
    polarity: int  # sign of score: -1 bone-like, +1 void-like, 0 if nothing
    label: str  # "bone-like", "void-like", or "none"


def normalized_cross_correlation(trace: np.ndarray, template: np.ndarray) -> np.ndarray:
    """Sliding normalized cross correlation of *template* over *trace*.

    At each position the template is compared against the local window of the trace after
    removing the mean of each, then dividing by the product of their norms. The result is in
    [-1, 1]: +1 is a perfect match to the template, -1 a perfect polarity-flipped match.
    """
    n = template.size
    if n == 0 or trace.size < n:
        return np.zeros_like(trace, dtype=float)

    t = template - template.mean()
    t_norm = np.linalg.norm(t)
    if t_norm < 1e-12:
        return np.zeros_like(trace, dtype=float)

    half = n // 2
    padded = np.pad(trace.astype(float), (half, n - half - 1))
    out = np.zeros(trace.size, dtype=float)
    for i in range(trace.size):
        w = padded[i : i + n]
        w = w - w.mean()
        denom = np.linalg.norm(w) * t_norm
        out[i] = float(np.dot(w, t) / denom) if denom > 1e-12 else 0.0
    return out


def detect_trace(
    trace: np.ndarray,
    center_freq_mhz: float = 400.0,
    dt_ns: float = 0.02,
    wavelet_length_ns: float = 8.0,
    threshold: float = 0.3,
) -> Detection:
    """Run the polarity matched filter on a single trace.

    Builds a Ricker template at the survey frequency and reads the dominant reflection: the
    sample with the strongest |NCC|. Detection means that peak clears the threshold, and the
    sign of the NCC there gives the polarity. High eps targets (bone) reflect negative at the
    top, low eps targets (voids) positive, so the sign separates them. This reads the top
    reflection cleanly when it is the dominant feature, which is the case at a target's
    hyperbola apex. For a resolved slab with a strong bottom reflection you would window to the
    top first, see docs/notes/11_polarity_baseline.md.
    """
    template = ricker_wavelet(center_freq_mhz, dt_ns, wavelet_length_ns)
    ncc = normalized_cross_correlation(trace, template)
    k = int(np.argmax(np.abs(ncc)))
    score = float(ncc[k])
    if abs(score) < threshold:
        return Detection(False, k, score, 0, "none")
    polarity = -1 if score < 0 else 1
    label = "bone-like" if polarity < 0 else "void-like"
    return Detection(True, k, score, polarity, label)


def detect_bscan(
    bscan: np.ndarray,
    center_freq_mhz: float = 400.0,
    dt_ns: float = 0.02,
    wavelet_length_ns: float = 8.0,
    threshold: float = 0.3,
) -> list[Detection]:
    """Apply the matched filter to every trace (column) of a B-scan.

    Expects shape (n_samples, n_traces): each column is one A-scan down depth. Returns one
    Detection per trace, which a caller can aggregate into boxes or a detection map later.
    """
    if bscan.ndim != 2:
        raise ValueError(f"Expected a 2D B-scan (n_samples, n_traces), got shape {bscan.shape}.")
    return [
        detect_trace(bscan[:, j], center_freq_mhz, dt_ns, wavelet_length_ns, threshold)
        for j in range(bscan.shape[1])
    ]
