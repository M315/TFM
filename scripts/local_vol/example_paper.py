"""
Paper dummy experiment (Lakhal et al. 2010).

True local vol: σ(y) = 0.4 − 0.1·arctan(y), time-constant.
Goal: recover σ from noisy call prices at two maturities T0 and T1.

Uses the LM (TRF) solver.  N_CAL_SLICES is kept small so the
finite-difference Jacobian remains tractable.
"""

from mpi4py import MPI

import numpy as np
from dolfinx import fem, mesh

from calibration import Vol, run_calibration
from a_f import compute_Af
from utils import add_noise, plot_calibration_result_3d


# --- experiment-specific parameters ---
Y0, Y1     = -1.0, 1.0    # log-moneyness domain
T0, T1     = 0.3,  0.6    # calibration window [T0, T1]
R, Q       = 0.0,  0.0    # no rates for the paper dummy
NOISE      = 5e-4          # Gaussian noise level on call prices
N_SPACE    = 70            # spatial elements
M_TIME     = 100           # time steps for the forward/adjoint solve
N_CAL_SLICES = 3           # calibration slices; keep low for LM tractability


def setup(V):
    """
    Build the true call-price snapshots f = u(T0) and g = u(T1) from the
    known local vol, add noise, and return the initial guess a_init.

    Returns (f, g, a_init, psi_1, psi_2).
    """
    # true vol: time-constant, single slice suffices
    a_true_func = fem.Function(V)
    a_true_func.interpolate(lambda x: 0.5 * (0.4 - 0.1 * np.arctan(x[0])) ** 2)
    a_true = Vol(V, a_true_func, 0.0, T1, N=1)

    # boundary conditions: deep-ITM call on the left, zero on the right
    psi_1 = lambda t: 0.67
    psi_2 = lambda t: 0.0

    # intrinsic payoff u_0 = max(1 − exp(y), 0)
    u_0 = fem.Function(V)
    u_0.interpolate(lambda x: np.maximum(1.0 - np.exp(x[0]), 0.0))

    # propagate 0 → T0 to get f, then T0 → T1 to get g
    dt_pre = T0 / M_TIME
    f, _ = compute_Af(a_true, u_0, dt_pre, M_TIME, V, R, Q, Y0, Y1, 0.0, psi_1, psi_2)

    dt = (T1 - T0) / M_TIME
    g, _ = compute_Af(a_true, f.copy(), dt, M_TIME, V, R, Q, Y0, Y1, T0, psi_1, psi_2)

    f = add_noise(f, NOISE)
    g = add_noise(g, NOISE)

    # initial guess: flat vol at σ = 0.3, i.e. a_0 = ½·0.3²
    a0_func = fem.Function(V)
    a0_func.interpolate(lambda x: np.full_like(x[0], 0.5 * 0.3 ** 2))
    a_init = Vol(V, a0_func, T0, T1, N=N_CAL_SLICES)

    return f, g, a_init, psi_1, psi_2


def run():
    msh = mesh.create_interval(MPI.COMM_WORLD, N_SPACE, points=(Y0, Y1))
    V   = fem.functionspace(msh, ("Lagrange", 1))

    f, g, a_init, psi_1, psi_2 = setup(V)

    # LM is more robust for the ill-conditioned paper experiment.
    # N_CAL_SLICES=3 keeps N_params = 3×71 ≈ 213, making the FD Jacobian cheap.
    # ftol/xtol matched to the noise floor (NOISE=5e-4) so the solver stops
    # once it can no longer improve beyond what the noise allows.
    a_cal, _ = run_calibration(
        f, g, a_init, V, R, Q, Y0, Y1, T0, T1, M_TIME, psi_1, psi_2,
        method='LM', max_iter=20, ftol=1e-4, xtol=1e-4,
    )

    plot_calibration_result_3d(a_cal, V, T0, T1, title="Paper experiment — calibrated local vol")


if __name__ == "__main__":
    run()
