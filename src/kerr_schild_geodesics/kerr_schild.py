"""Fast numerical Kerr-Schild geodesic kernel.

The formulas in this file are generated from the symbolic derivation documented
in notebooks/02_geodesic_rhs.ipynb.

Conventions
-----------
- Geometrized units: G = c = 1
- Metric signature: (-,+,+,+)
- Cartesian Kerr-Schild coordinates
- No SymPy work occurs during trajectory integration.
"""

import math
import numpy as np
from numba import njit


@njit(cache=True)
def kerr_radius(x, y, z, a):
    """Return the physical Kerr radial coordinate r(x,y,z)."""
    rho2 = x*x + y*y + z*z
    A = rho2 - a*a
    disc = math.sqrt(A*A + 4.0*a*a*z*z)
    r2 = 0.5*(A + disc)
    return math.sqrt(r2)


@njit(cache=True)
def acceleration_core(x, y, z, r, ut, ux, uy, uz, M, a):
    """Generated contracted geodesic acceleration A^mu."""
    q0 = r**3
    q1 = r**4
    q2 = a**2
    q3 = z**2
    q4 = 1/(q1 + q2*q3)
    q5 = M*q4
    q6 = 2*q5
    q7 = q0*q6
    q8 = r**2
    q9 = 2*q8
    q10 = -q2 + q3 - q9 + x**2 + y**2
    q11 = 1/q10
    q12 = q1*q4
    q13 = 4*q12 - 3
    q14 = q11*q13
    q15 = q14*x
    q16 = ut*ux
    q17 = q14*y
    q18 = ut*uy
    q19 = ux*uz
    q20 = -1/q10
    q21 = q20*x
    q22 = q9*z
    q23 = q22*(2*q12 - 1)
    q24 = uy*uz
    q25 = q23*y
    q26 = q2*q4*q9
    q27 = -q13
    q28 = q2 + q8
    q29 = q20*q28
    q30 = q26 - q27*q29
    q31 = ut*uz
    q32 = r*z
    q33 = r*x
    q34 = a*y + q33
    q35 = q34*y
    q36 = q14*q35
    q37 = r*y
    q38 = 1/q28
    q39 = 2*r
    q40 = q38*q39
    q41 = q34*q40 - x
    q42 = q11*q41
    q43 = q37*q42
    q44 = a + q43
    q45 = q0*q38
    q46 = ux*uy
    q47 = q45*q46
    q48 = ux**2
    q49 = r*(q42*x + 1)
    q50 = q15*q34
    q51 = q49 + q50
    q52 = uy**2
    q53 = a*x - r*y
    q54 = -q53
    q55 = q40*q54 - y
    q56 = q11*q55
    q57 = q56*y + 1
    q58 = q57*r
    q59 = q17*q53
    q60 = -q59
    q61 = q58 + q60
    q62 = q15*q53
    q63 = -q55
    q64 = q20*q63
    q65 = q33*q64
    q66 = -q65
    q67 = a + q66
    q68 = uz**2
    q69 = q11*q28
    q70 = q3/q8
    q71 = q69*q70 + 1
    q72 = q3*q30
    q73 = q71*q8 - q72
    q74 = q42*r
    q75 = q30*q38
    q76 = q34*q75
    q77 = q19*q32
    q78 = q24*q32
    q79 = q0*q15*q16 + q0*q17*q18 - q19*q21*q23 - q20*q24*q25 - q30*q31*q32 + q45*q48*q51 + q45*q52*q61 + q47*(q36 + q44) - q47*(q62 + q67) + q68*q73 + q77*(q74 - q76) + q78*(q53*q75 + q56*r)
    q80 = ut**2
    q81 = q0*q80
    q82 = q11*q31
    q83 = q20*q35
    q84 = q27*q83
    q85 = -q41
    q86 = q20*q85
    q87 = q37*q86
    q88 = a + q87
    q89 = q16*q45
    q90 = q64*y + 1
    q91 = q90*r
    q92 = q20*q54
    q93 = q18*q45
    q94 = q28**(-2)
    q95 = 1/r
    q96 = -q20*q54*q95*y + q90
    q97 = -q96
    q98 = q8*z
    q99 = q24*q38*q98
    q100 = q44*q53
    q101 = 3*a
    q102 = 2*q53
    q103 = q101 + q102*q15 + q36 - 2*q65
    q104 = q11*q37
    q105 = q104*q13
    q106 = 2*q3
    q107 = q38*q72
    q108 = q38*q71
    q109 = -q108*q53 + q11*q3*q55*q95
    q110 = -q83
    q111 = q21*q54
    q112 = q34**2
    q113 = 2*q34
    q114 = q34*q67
    q115 = q0*q94
    q116 = q64*r
    q117 = 4*q54
    q118 = q54**2
    q119 = -q30
    q120 = q119*q38
    q121 = 2*q120
    q122 = 2*q27
    q123 = -q67
    q124 = q115*q46
    q125 = q110 + q88
    q126 = q102*q76 + q11*q39*(q34*q55 - q41*q53)
    q127 = q0*q38*q61*ut*uy + q0*q52*q53*q94*(-2*q58 - q60) + q0*q94*ux*uy*(-q100 + q34*q57*r - q36*q53) - q115*q48*(q102*q49 + q112*q17 + q113*q44 + q113*q62 + 2*q114) - q124*(-q117*q123 - q118*q122*q21 + q34*q91 + q54*q84 + q54*q88) - q17*q81 - q25*q82 - q38*q78*(-q116*q117 - q118*q121 + q27*q37*q92 + q8*q96) + q38*q8*ux*uz*z*(-q103 - q110 - 2*q111 - q87) + q38*r*ux*uz*z*(-q105*q34 - q125*r + q126) + q68*(q102*q107 - q104*q106 - q105*q3 + q109*q9) - q89*(q103 + q43) - q89*(q84 + q88) - q93*(q27*q92*y + q91) - q99*(q59 + q97*r) + r*ut*uz*z*(-q104 - q105 + 2*q11*q55*r + 2*q30*q38*q53)
    q128 = q45*q5
    q129 = q128*q53
    q130 = q86*x + 1
    q131 = q130*r
    q132 = q111*q27
    q133 = q130 - q20*q34*q95*x
    q134 = -q133
    q135 = q11*q33
    q136 = q13*q135
    q137 = q101 - q20*q27*q54*x + q66 + 2*q84 + 2*q87
    q138 = q3*q95
    q139 = q108*q34 + q138*q42
    q140 = q53**2
    q141 = q20*q33
    q142 = q86*r
    q143 = 4*q34
    q144 = 2*q20
    q145 = q111 + q67
    q146 = q0*q34*q48*q94*(2*q49 + q50) + q0*q38*q51*ut*ux + q0*q52*q94*(-2*q100 - q102*q36 - q102*q67 - q140*q15 + 2*q34*q57*r) - q124*(q114 + q49*q53 + q50*q53) - q124*(-q112*q144*q27*y + q123*q34 + q131*q54 + q132*q34 - q143*q88) + q137*q93 - q15*q81 - q23*q82*x - q38*q77*(-q112*q121 + q133*q8 + q141*q27*q34 - q142*q143) + q38*q8*ux*uz*z*(q11*q13*q34*x - q134*r) + q38*r*uy*uz*z*(q126 + q136*q53 + q145*r) + q68*(-q106*q135 - q107*q113 - q136*q3 + 2*q139*q8) - q89*(q131 + q21*q27*q34) - q93*(-a + q132 + q65) - q99*(-q111 - q137 + 2*q20*q34*y) + r*ut*uz*z*(2*q11*q41*r - q135 - q136 - 2*q76)
    q147 = -q29*q70 + 1
    q148 = q119*q3
    q149 = q120*q34 + q142
    q150 = q32*ut
    q151 = q150*ux
    q152 = q116 + q120*q54
    q153 = q150*uy
    q154 = q144*q33
    q155 = q154*q27
    q156 = q144*q37
    q157 = q156*q27
    q158 = 4*q3
    q159 = q148*q38
    q160 = q147*q38
    q161 = q20*q37
    q162 = -q119*q34*q38*q54 - q20*r*(q34*q63 + q54*q85)
    q163 = q32*q38*q46
    q164 = -q149*q151 - q151*(q149 + q154 - q155) - q152*q153 - q153*(q152 + q156 - q157) - q163*(-q113*q161*q27 - q125*q39 - q162) - q163*(-q122*q33*q92 + q145*q39 - q162) - q19*(q141*q158 - q155*q3 + q159*q34 + q8*(q138*q86 + q160*q34)) - q24*(-q157*q3 + q158*q161 + q159*q54 + q8*(q138*q64 + q160*q54)) - q31*(q147*q8 + q148) + q38*q48*r*z*(2*q11*q13*q34*r*x + q112*q30*q38 - q113*q74 - q134*q9) + q38*q52*r*z*(-q102*q105 + 2*q11*q53*q55*r + q140*q30*q38 - q9*q97) + q68*z*(q39*q71 - q72*q95) + q73*ut*uz + q80*r*z*(-q13*q69 + q26) + ux*uz*(-q107*q34 + q139*q8) + uy*uz*(q107*q53 + q109*q8)
    q165 = q5*q98
    q166 = q7*q94
    q167 = q5*q53
    q168 = q127*q167
    q169 = q113*q115
    q170 = 4*q79
    q171 = q22*q38
    A0 = q6*(q127*q129 - q128*q146*q34 - q164*q165 + q79*(q7 + 1))
    A1 = q5*(2*M*q164*q34*q38*q4*q8*z - q128*q143*q79 + q146*(q112*q166 - 1) - q168*q169)
    A2 = q5*(q127*(q118*q166 - 1) + q129*q170 - q146*q167*q169 - q164*q167*q171)
    A3 = q5*(2*M*q146*q34*q38*q4*q8*z + q164*(q3*q39*q5 - 1) - q165*q170 - q168*q171)
    return A0, A1, A2, A3


@njit(cache=True)
def acceleration(x, y, z, ut, ux, uy, uz, M=1.0, a=0.5):
    """Evaluate A^mu = -Gamma^mu_ab u^a u^b."""
    r = kerr_radius(x, y, z, a)
    return acceleration_core(x, y, z, r, ut, ux, uy, uz, M, a)


@njit(cache=True)
def geodesic_rhs(state, M=1.0, a=0.5):
    """Eight-dimensional first-order affine geodesic system."""
    _, x, y, z, ut, ux, uy, uz = state
    A0, A1, A2, A3 = acceleration(x, y, z, ut, ux, uy, uz, M, a)

    out = np.empty(8, dtype=np.float64)
    out[0] = ut
    out[1] = ux
    out[2] = uy
    out[3] = uz
    out[4] = A0
    out[5] = A1
    out[6] = A2
    out[7] = A3
    return out


@njit(cache=True)
def metric_cov(x, y, z, M=1.0, a=0.5):
    """Covariant Kerr-Schild metric g_mu_nu."""
    r = kerr_radius(x, y, z, a)
    r2 = r*r
    den = r2 + a*a

    l0 = 1.0
    l1 = (r*x + a*y)/den
    l2 = (r*y - a*x)/den
    l3 = z/r

    H = M*r*r*r/(r2*r2 + a*a*z*z)

    ell = np.array((l0, l1, l2, l3), dtype=np.float64)
    g = np.zeros((4, 4), dtype=np.float64)

    g[0, 0] = -1.0
    g[1, 1] = 1.0
    g[2, 2] = 1.0
    g[3, 3] = 1.0

    for i in range(4):
        for j in range(4):
            g[i, j] += 2.0*H*ell[i]*ell[j]

    return g


@njit(cache=True)
def normalization(state, M=1.0, a=0.5):
    """Return g_mu_nu u^mu u^nu."""
    x, y, z = state[1], state[2], state[3]
    u = state[4:8]
    g = metric_cov(x, y, z, M, a)

    value = 0.0
    for i in range(4):
        for j in range(4):
            value += g[i, j]*u[i]*u[j]
    return value


@njit(cache=True)
def energy_lz(state, M=1.0, a=0.5):
    """Return Killing energy E=-p_t and axial angular momentum L_z."""
    x, y, z = state[1], state[2], state[3]
    u = state[4:8]
    g = metric_cov(x, y, z, M, a)

    p = np.empty(4, dtype=np.float64)
    for i in range(4):
        value = 0.0
        for j in range(4):
            value += g[i, j]*u[j]
        p[i] = value

    E = -p[0]

    # Rotational Killing vector about z:
    # xi^mu = (0, -y, x, 0)
    Lz = -y*p[1] + x*p[2]

    return E, Lz


@njit(cache=True)
def solve_ut_timelike(x, y, z, ux, uy, uz, M=1.0, a=0.5):
    """Solve g_mu_nu u^mu u^nu = -1 for the future-directed u^t root."""
    g = metric_cov(x, y, z, M, a)

    A = g[0, 0]
    B = 2.0*(g[0, 1]*ux + g[0, 2]*uy + g[0, 3]*uz)
    C = (
        g[1, 1]*ux*ux
        + g[2, 2]*uy*uy
        + g[3, 3]*uz*uz
        + 2.0*g[1, 2]*ux*uy
        + 2.0*g[1, 3]*ux*uz
        + 2.0*g[2, 3]*uy*uz
        + 1.0
    )

    disc = B*B - 4.0*A*C
    if disc < 0.0 and disc > -1.0e-14:
        disc = 0.0
    if disc < 0.0:
        return np.nan

    root = math.sqrt(disc)
    r1 = (-B + root)/(2.0*A)
    r2 = (-B - root)/(2.0*A)

    if r1 > 0.0 and r2 > 0.0:
        return max(r1, r2)
    if r1 > 0.0:
        return r1
    if r2 > 0.0:
        return r2
    return max(r1, r2)


@njit(cache=True)
def horizon_radius(M=1.0, a=0.5):
    """Outer Kerr horizon r_+ for |a| <= M."""
    if abs(a) > M:
        return np.nan
    return M + math.sqrt(M*M - a*a)
