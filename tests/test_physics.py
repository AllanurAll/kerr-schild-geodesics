import numpy as np

from kerr_schild_geodesics import (
    kerr_radius,
    metric_cov,
    normalization,
    energy_lz,
    circular_equatorial_state,
    zamo_omega_equator,
)
from kerr_schild_geodesics.integrators import integrate_rk4


def test_flat_metric():
    g = metric_cov(8.0, 1.0, 0.5, M=0.0, a=0.5)
    np.testing.assert_allclose(g, np.diag([-1.0, 1.0, 1.0, 1.0]),
                               rtol=0.0, atol=1e-14)


def test_kerr_radius_equatorial_ring():
    r0 = 10.0
    a = 0.5
    x = np.sqrt(r0*r0 + a*a)
    assert abs(kerr_radius(x, 0.0, 0.0, a) - r0) < 1e-13


def _integrate_circular(r0, M, a, sense, periods=1.0, h_target=0.02):
    state0 = circular_equatorial_state(r0, M, a, sense=sense)

    # Coordinate angular velocity from initial Cartesian motion.
    x, y = state0[1], state0[2]
    ux, uy = state0[5], state0[6]
    ut = state0[4]
    R2 = x*x + y*y
    omega = (x*uy - y*ux) / (R2*ut)
    u_phi = omega * ut

    proper_period = 2.0*np.pi / abs(u_phi)
    lam_end = periods * proper_period
    n_steps = int(np.ceil(lam_end / h_target))
    h = lam_end / n_steps

    traj = integrate_rk4(state0, h, n_steps, M, a)
    return state0, traj


def _check_circular_invariants(state0, traj, M, a, r0):
    rvals = np.array([
        kerr_radius(s[1], s[2], s[3], a)
        for s in traj
    ])
    norms = np.array([normalization(s, M, a) for s in traj[::20]])
    EL = np.array([energy_lz(s, M, a) for s in traj[::20]])

    assert np.max(np.abs(rvals - r0)) < 1e-9
    assert np.max(np.abs(norms + 1.0)) < 1e-10
    assert np.max(np.abs(EL[:, 0] - EL[0, 0])) < 1e-10
    assert np.max(np.abs(EL[:, 1] - EL[0, 1])) < 1e-9


def test_schwarzschild_circular_orbit():
    M, a, r0 = 1.0, 0.0, 10.0
    state0, traj = _integrate_circular(r0, M, a, sense=+1)
    _check_circular_invariants(state0, traj, M, a, r0)


def test_kerr_prograde_circular_orbit():
    M, a, r0 = 1.0, 0.5, 10.0
    state0, traj = _integrate_circular(r0, M, a, sense=+1)
    _check_circular_invariants(state0, traj, M, a, r0)


def test_kerr_retrograde_circular_orbit():
    M, a, r0 = 1.0, 0.5, 10.0
    state0, traj = _integrate_circular(r0, M, a, sense=-1)
    _check_circular_invariants(state0, traj, M, a, r0)


def test_frame_dragging_increases_inward():
    M, a = 1.0, 0.5
    w10 = zamo_omega_equator(10.0, M, a)
    w5 = zamo_omega_equator(5.0, M, a)
    w3 = zamo_omega_equator(3.0, M, a)

    assert 0.0 < w10 < w5 < w3
