"""1D GPR forward model. A cheap physics check before I spend FDTD compute.

Why this exists
---------------
Before I generate hundreds of gprMax (FDTD) B-scans of fossil-like targets, I want to be
sure I actually understand the physics the papers describe. That means the polarity-triplet
signature of buried bone from Peredo et al. (2024) and the dielectric contrasts I plan to
bake into the synthetic data. FDTD is the real thing but it is heavy. A 1D convolutional
model runs in milliseconds and lets me check the core idea first.

The model
---------
This is the standard 1D convolutional (reflectivity) model that seismic and GPR people use:

    trace(t) = wavelet(t)  *  reflectivity(t)

A buried target is a layer with permittivity different from the host. At each interface the
wave partially reflects, with coefficient (normal incidence, non-magnetic media):

    r = (sqrt(eps_above) - sqrt(eps_below)) / (sqrt(eps_above) + sqrt(eps_below))

I place each interface at its two-way travel time (TWT), give it a spike of height r, and
convolve the reflectivity series with a Ricker wavelet (the standard GPR source pulse). The
resulting trace is what a GPR antenna would record over that 1D column.

What I'm checking
-----------------
1. A high-permittivity target (bone) and a low-permittivity target (air-filled cavity)
   reflect with opposite polarity. This matters, because our real dataset's anomalies are
   cavities, so to a detector bone is the polarity-flipped version of a cavity.
2. A bone layer produces the positive-negative-positive triplet Peredo describes. How
   strongly it shows up depends on layer thickness versus wavelength (tuning).

Dielectric values (from the reading notes):
    Fossilized/mineralized bone : eps ~ 7-12   (Peredo et al. 2024)
    Dry sand / sediment         : eps ~ 3-5    (Peredo et al. 2024)
    Limestone matrix            : eps ~ 4-8    (Catanzariti et al. 2023)
    Air-filled void (cavity)    : eps = 1
    Water                       : eps ~ 80

This is a 1D model on purpose. It does not produce diffraction hyperbolas, which need 2D
geometry and are gprMax's job. It only tells me about reflection amplitude and polarity down
a single column.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Speed of light in vacuum, in m/ns (GPR people work in nanoseconds).
C_M_PER_NS: float = 0.299792458


def velocity(eps_r: float) -> float:
    """EM wave velocity in a non-magnetic medium of relative permittivity *eps_r* (m/ns)."""
    if eps_r <= 0:
        raise ValueError(f"Relative permittivity must be positive, got {eps_r}.")
    return C_M_PER_NS / np.sqrt(eps_r)


def reflection_coefficient(eps_above: float, eps_below: float) -> float:
    """Normal-incidence reflection coefficient for a wave crossing one interface.

    Convention: the wave travels downward from the medium with permittivity
    *eps_above* into the medium with permittivity *eps_below*.

        r = (sqrt(eps_above) - sqrt(eps_below)) / (sqrt(eps_above) + sqrt(eps_below))

    A jump to *higher* permittivity (e.g. sand -> bone) gives r < 0; a jump to *lower*
    permittivity (e.g. sand -> air void) gives r > 0. r is bounded to [-1, 1].
    """
    na, nb = np.sqrt(eps_above), np.sqrt(eps_below)
    return float((na - nb) / (na + nb))


def ricker_wavelet(center_freq_mhz: float, dt_ns: float, length_ns: float) -> np.ndarray:
    """Zero-phase Ricker (Mexican-hat) wavelet, the standard GPR source pulse.

    Parameters
    ----------
    center_freq_mhz:
        Center frequency in MHz (e.g. 400 for Peredo's antenna, 2000 for Catanzariti).
    dt_ns:
        Sample interval in nanoseconds.
    length_ns:
        Total wavelet length in nanoseconds (it is centered and symmetric).

    Returns
    -------
    np.ndarray
        The wavelet samples. Central lobe is positive.
    """
    f = center_freq_mhz / 1000.0  # MHz -> GHz, i.e. cycles per ns
    # Symmetric grid centered exactly on t=0 (odd sample count) so the wavelet is
    # zero-phase. np.arange(-L/2, L/2, dt) would drop the endpoint and leave the pulse
    # off-center, which the symmetry test catches.
    half = int(round((length_ns / 2) / dt_ns))
    t = np.arange(-half, half + 1) * dt_ns
    a = (np.pi * f * t) ** 2
    return (1.0 - 2.0 * a) * np.exp(-a)


@dataclass(frozen=True)
class Layer:
    """A horizontal layer: everything from ``top_m`` down to the next layer's top.

    The deepest layer extends to the bottom of the model (its thickness is ignored).
    """

    name: str
    eps_r: float
    top_m: float


def two_way_times(layers: list[Layer]) -> list[tuple[float, float]]:
    """Return (twt_ns, reflection_coefficient) for every interface between *layers*.

    Layers must be ordered top-to-bottom by ``top_m``. TWT for an interface is twice the
    one-way travel time from the surface down to that interface, accumulating through
    every layer above it at that layer's own velocity.
    """
    layers = sorted(layers, key=lambda x: x.top_m)
    if layers[0].top_m != 0:
        raise ValueError("The first (top) layer must start at depth 0 m.")

    interfaces: list[tuple[float, float]] = []
    one_way_ns = 0.0
    for i in range(1, len(layers)):
        upper, lower = layers[i - 1], layers[i]
        thickness = lower.top_m - upper.top_m  # thickness of the upper layer
        one_way_ns += thickness / velocity(upper.eps_r)
        r = reflection_coefficient(upper.eps_r, lower.eps_r)
        interfaces.append((2.0 * one_way_ns, r))
    return interfaces


def synthesize_trace(
    layers: list[Layer],
    center_freq_mhz: float = 400.0,
    dt_ns: float = 0.02,
    record_length_ns: float = 40.0,
    wavelet_length_ns: float = 8.0,
) -> tuple[np.ndarray, np.ndarray, list[tuple[float, float]]]:
    """Forward-model a single GPR trace over a layered 1D column.

    Returns
    -------
    (time_ns, trace, interfaces)
        ``time_ns`` is the time axis, ``trace`` is the synthetic amplitude, and
        ``interfaces`` is the list of ``(twt_ns, r)`` reflectors used.
    """
    time_ns = np.arange(0.0, record_length_ns, dt_ns)
    reflectivity = np.zeros_like(time_ns)

    interfaces = two_way_times(layers)
    for twt, r in interfaces:
        idx = int(round(twt / dt_ns))
        if 0 <= idx < reflectivity.size:
            reflectivity[idx] += r

    wavelet = ricker_wavelet(center_freq_mhz, dt_ns, wavelet_length_ns)
    trace = np.convolve(reflectivity, wavelet, mode="same")
    return time_ns, trace, interfaces


# Convenience scene builders for the validation experiment.


def buried_target_scene(
    host_eps: float,
    target_eps: float,
    target_top_m: float,
    target_thickness_m: float,
    host_name: str = "host",
    target_name: str = "target",
) -> list[Layer]:
    """A host medium with a single buried slab target of a different permittivity."""
    return [
        Layer(host_name, host_eps, 0.0),
        Layer(target_name, target_eps, target_top_m),
        Layer(host_name, host_eps, target_top_m + target_thickness_m),
    ]
