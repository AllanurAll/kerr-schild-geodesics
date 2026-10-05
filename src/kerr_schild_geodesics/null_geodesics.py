"""Null geodesic helpers for Schwarzschild and Kerr spacetimes.

The routines in this module use the same Cartesian Kerr-Schild metric and
geodesic kernel as the timelike calculations.  Photon trajectories differ
only in their initial normalization,

    g_{mu nu} k^mu k^nu = 0.

Outcome codes
-------------
0 : escaped back to the launch radius
1 : crossed the outer event horizon
2 : integration limit reached before classification
"""

import numpy as np
from numba import njit

from .kerr_schild import (
    metric_cov,
    kerr_radius,
    horizon_radius,
)
from .integrators import rk4_step


ESCAPED = 0
CAPTURED = 1
UNRESOLVED = 2


@njit(cache=True)
def solve_kt_null(x, y, z, kx, ky, kz, M=1.0, a=0.0):
    """Solve the null normalization for the future-directed k^t root."""
    g = metric_cov(x, y, z, M, a)

    A = g[0, 0]
    B = 2.0 * (
        g[0, 1] * kx
        + g[0, 2] * ky
        + g[0, 3] * kz
    )
    C = (
        g[1, 1] * kx * kx
        + g[2, 2] * ky * ky
        + g[3, 3] * kz * kz
        + 2.0 * g[1, 2] * kx * ky
        + 2.0 * g[1, 3] * kx * kz
        + 2.0 * g[2, 3] * ky * kz
    )

    disc = B * B - 4.0 * A * C

    if disc < 0.0 and disc > -1.0e-13:
        disc = 0.0

    if disc < 0.0:
        return np.nan

    root = np.sqrt(disc)

    r1 = (-B + root) / (2.0 * A)
    r2 = (-B - root) / (2.0 * A)

    if r1 > 0.0 and r2 > 0.0:
        return max(r1, r2)
    if r1 > 0.0:
        return r1
    if r2 > 0.0:
        return r2

    return max(r1, r2)


@njit(cache=True)
def parallel_ray_state(
    x0,
    y0,
    z0,
    M=1.0,
    a=0.0,
):
    """Construct a future-directed null ray initially moving along +x."""
    kt = solve_kt_null(
        x0, y0, z0,
        1.0, 0.0, 0.0,
        M, a,
    )

    return np.array(
        [
            0.0,
            x0, y0, z0,
            kt,
            1.0, 0.0, 0.0,
        ],
        dtype=np.float64,
    )


@njit(cache=True)
def trace_null_ray(
    state0,
    h,
    n_steps,
    M=1.0,
    a=0.0,
    horizon_buffer=1.001,
):
    """Trace one photon until capture, escape, or the step limit.

    Escape is declared after the ray first approaches the hole and later
    returns to at least its initial Kerr radial coordinate.
    """
    trajectory = np.empty(
        (n_steps + 1, 8),
        dtype=np.float64,
    )

    trajectory[0] = state0
    state = state0.copy()

    r_initial = kerr_radius(
        state[1],
        state[2],
        state[3],
        a,
    )

    r_plus = horizon_radius(M, a)

    approached = False

    for i in range(n_steps):
        state = rk4_step(
            state,
            h,
            M,
            a,
        )

        trajectory[i + 1] = state

        r_now = kerr_radius(
            state[1],
            state[2],
            state[3],
            a,
        )

        if not np.isfinite(r_now):
            return trajectory, i + 2, CAPTURED

        if r_now <= horizon_buffer * r_plus:
            return trajectory, i + 2, CAPTURED

        if r_now < 0.90 * r_initial:
            approached = True

        if approached and r_now >= r_initial:
            return trajectory, i + 2, ESCAPED

    return trajectory, n_steps + 1, UNRESOLVED


@njit(cache=True)
def classify_parallel_ray(
    y0,
    z0,
    M=1.0,
    a=0.0,
    x0=-30.0,
    h=0.03,
    n_steps=4000,
):
    """Classify a parallel ray without storing its full trajectory."""
    state = parallel_ray_state(
        x0,
        y0,
        z0,
        M,
        a,
    )

    r_initial = kerr_radius(
        state[1],
        state[2],
        state[3],
        a,
    )

    r_plus = horizon_radius(M, a)

    approached = False

    for _ in range(n_steps):
        state = rk4_step(
            state,
            h,
            M,
            a,
        )

        r_now = kerr_radius(
            state[1],
            state[2],
            state[3],
            a,
        )

        if not np.isfinite(r_now):
            return CAPTURED

        if r_now <= 1.001 * r_plus:
            return CAPTURED

        if r_now < 0.90 * r_initial:
            approached = True

        if approached and r_now >= r_initial:
            return ESCAPED

    return UNRESOLVED


@njit(cache=True)
def parallel_capture_map(
    y_values,
    z_values,
    M=1.0,
    a=0.0,
    x0=-30.0,
    h=0.03,
    n_steps=4000,
):
    """Return a 2D capture/escape map for initially parallel null rays."""
    result = np.empty(
        (len(z_values), len(y_values)),
        dtype=np.int8,
    )

    for iz in range(len(z_values)):
        for iy in range(len(y_values)):
            result[iz, iy] = classify_parallel_ray(
                y_values[iy],
                z_values[iz],
                M,
                a,
                x0,
                h,
                n_steps,
            )

    return result
