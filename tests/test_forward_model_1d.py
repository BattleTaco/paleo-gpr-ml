"""Physics invariants for the 1D GPR forward model.

These aren't accuracy tests against field data (I don't have ground truth for that yet).
They check that the model obeys the physics it claims to: reflection coefficients are
bounded and have the right signs, the bone/cavity polarity inversion holds, travel times
are monotonic, and the Ricker wavelet is well-formed. If any of these break, the synthetic
data built on top of this model would be physically wrong.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.data.forward_model_1d import (
    Layer,
    buried_target_scene,
    reflection_coefficient,
    ricker_wavelet,
    synthesize_trace,
    two_way_times,
    velocity,
)

# Dielectric values from the reading notes (Peredo 2024, Catanzariti 2023).
EPS_SAND = 4.0
EPS_BONE = 9.0
EPS_AIR = 1.0


def test_velocity_decreases_with_permittivity():
    assert velocity(1.0) == pytest.approx(0.299792458)  # vacuum
    assert velocity(EPS_SAND) > velocity(EPS_BONE)  # bone is slower (denser dielectric)


def test_velocity_rejects_nonpositive():
    with pytest.raises(ValueError):
        velocity(0.0)


def test_reflection_coefficient_bounded():
    for ea in (1.0, 4.0, 9.0, 80.0):
        for eb in (1.0, 4.0, 9.0, 80.0):
            assert -1.0 <= reflection_coefficient(ea, eb) <= 1.0


def test_reflection_coefficient_signs():
    # Sand -> bone: wave enters HIGHER permittivity -> negative reflection.
    assert reflection_coefficient(EPS_SAND, EPS_BONE) < 0
    # Sand -> air void: wave enters LOWER permittivity -> positive reflection.
    assert reflection_coefficient(EPS_SAND, EPS_AIR) > 0
    # No contrast -> no reflection.
    assert reflection_coefficient(EPS_SAND, EPS_SAND) == pytest.approx(0.0)


def test_reflection_coefficient_antisymmetric():
    # Flipping which medium is on top flips the sign (energy bookkeeping).
    assert reflection_coefficient(EPS_SAND, EPS_BONE) == pytest.approx(
        -reflection_coefficient(EPS_BONE, EPS_SAND)
    )


def test_bone_and_cavity_have_opposite_polarity():
    """The headline result: bone (high eps) and cavity (low eps) are polarity-inverted.

    Bone slab in sand -> (top: negative, bottom: positive).
    Air cavity in sand -> (top: positive, bottom: negative).
    """
    bone = two_way_times(buried_target_scene(EPS_SAND, EPS_BONE, 0.5, 0.1))
    cavity = two_way_times(buried_target_scene(EPS_SAND, EPS_AIR, 0.5, 0.1))

    (_, r_bone_top), (_, r_bone_bot) = bone
    (_, r_cav_top), (_, r_cav_bot) = cavity

    assert r_bone_top < 0 < r_bone_bot
    assert r_cav_top > 0 > r_cav_bot
    # Same |contrast geometry|, opposite sign at each interface.
    assert np.sign(r_bone_top) == -np.sign(r_cav_top)


def test_two_way_times_monotonic_and_positive():
    scene = buried_target_scene(EPS_SAND, EPS_BONE, 0.5, 0.1)
    twts = [t for t, _ in two_way_times(scene)]
    assert all(t > 0 for t in twts)
    assert twts == sorted(twts)  # deeper interface arrives later


def test_deeper_target_arrives_later():
    shallow = two_way_times(buried_target_scene(EPS_SAND, EPS_BONE, 0.3, 0.1))[0][0]
    deep = two_way_times(buried_target_scene(EPS_SAND, EPS_BONE, 1.0, 0.1))[0][0]
    assert deep > shallow


def test_first_layer_must_start_at_surface():
    with pytest.raises(ValueError):
        two_way_times([Layer("a", EPS_SAND, 0.5), Layer("b", EPS_BONE, 1.0)])


def test_ricker_wavelet_shape():
    w = ricker_wavelet(center_freq_mhz=400.0, dt_ns=0.02, length_ns=8.0)
    assert w.size > 0
    # Zero-phase: the global maximum is the central positive lobe.
    assert np.argmax(w) == pytest.approx(w.size // 2, abs=1)
    # Ricker integrates to ~0 (it's the 2nd derivative of a Gaussian).
    assert np.trapezoid(w) == pytest.approx(0.0, abs=1e-2)
    # Symmetric about its center.
    assert np.allclose(w, w[::-1], atol=1e-6)


def test_synthesize_trace_runs_and_responds_to_target():
    """A trace with a target carries more energy than a homogeneous (no-contrast) one."""
    t, trace_bone, interfaces = synthesize_trace(buried_target_scene(EPS_SAND, EPS_BONE, 0.5, 0.1))
    _, trace_flat, _ = synthesize_trace([Layer("sand", EPS_SAND, 0.0)])
    assert t.shape == trace_bone.shape
    assert len(interfaces) == 2
    assert np.sum(trace_bone**2) > np.sum(trace_flat**2)
