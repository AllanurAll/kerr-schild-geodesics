import numpy as np

from kerr_schild_geodesics import (
    CAPTURED,
    ESCAPED,
    normalization,
    parallel_ray_state,
    classify_parallel_ray,
)


def test_null_initial_state_schwarzschild():
    state = parallel_ray_state(
        -40.0, 7.0, 0.0,
        M=1.0,
        a=0.0,
    )
    assert abs(normalization(state, 1.0, 0.0)) < 1.0e-12


def test_null_initial_state_kerr():
    state = parallel_ray_state(
        -40.0, -5.0, 0.0,
        M=1.0,
        a=0.8,
    )
    assert abs(normalization(state, 1.0, 0.8)) < 1.0e-12


def test_schwarzschild_capture_and_scatter():
    captured = classify_parallel_ray(
        5.0, 0.0,
        M=1.0,
        a=0.0,
        x0=-40.0,
        h=0.03,
        n_steps=4000,
    )
    scattered = classify_parallel_ray(
        7.0, 0.0,
        M=1.0,
        a=0.0,
        x0=-40.0,
        h=0.03,
        n_steps=4000,
    )

    assert captured == CAPTURED
    assert scattered == ESCAPED


def test_kerr_prograde_retrograde_capture_asymmetry():
    # Incoming rays move along +x. At x<0:
    # y<0 gives L_z>0 (prograde for a>0),
    # y>0 gives L_z<0 (retrograde).
    prograde = classify_parallel_ray(
        -5.0, 0.0,
        M=1.0,
        a=0.8,
        x0=-40.0,
        h=0.03,
        n_steps=4000,
    )
    retrograde = classify_parallel_ray(
        5.0, 0.0,
        M=1.0,
        a=0.8,
        x0=-40.0,
        h=0.03,
        n_steps=4000,
    )

    assert prograde == ESCAPED
    assert retrograde == CAPTURED
