"""Numerical integrators for geodesic evolution."""

import numpy as np
from numba import njit

from .kerr_schild import geodesic_rhs


@njit(cache=True)
def rk4_step(state, h, M=1.0, a=0.0):
    """Advance one affine-parameter step with classical fourth-order RK4."""
    k1 = geodesic_rhs(state, M, a)
    k2 = geodesic_rhs(state + 0.5 * h * k1, M, a)
    k3 = geodesic_rhs(state + 0.5 * h * k2, M, a)
    k4 = geodesic_rhs(state + h * k3, M, a)
    return state + (h / 6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4)


@njit(cache=True)
def integrate_rk4(state0, h, n_steps, M=1.0, a=0.0):
    """Integrate an eight-component geodesic state with fixed-step RK4."""
    trajectory = np.empty((n_steps + 1, 8), dtype=np.float64)
    trajectory[0] = state0

    state = state0.copy()
    for i in range(n_steps):
        state = rk4_step(state, h, M, a)
        trajectory[i + 1] = state

    return trajectory
