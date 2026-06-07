from mpi4py import MPI

import numpy as np
import scipy.optimize

from dolfinx import fem

from a_f import compute_Af
from adj_da_f import compute_adj_dAf
from utils import L2_norm


class Vol:
    """Piecewise-constant-in-time local volatility parameter a = ½σ²."""

    t_0: float
    t_1: float
    a: list[fem.Function]

    def __init__(self, V, const_func, t_0, t_1, N):
        self.t_0 = t_0
        self.t_1 = t_1
        self.a = [const_func.copy() for _ in range(N)]

    def get(self, t: float) -> fem.Function:
        return self.a[self.get_idx(t)]

    def get_idx(self, t: float) -> int:
        if t < self.t_0 or t > self.t_1:
            raise ValueError(f"t={t} outside Vol range [{self.t_0}, {self.t_1}]")
        if t == self.t_1:
            return len(self.a) - 1
        dt = (self.t_1 - self.t_0) / len(self.a)
        return int((t - self.t_0) / dt)

    def update(self, V, a_vec: np.ndarray) -> None:
        dof_size = V.dofmap.index_map.size_global
        for i, func in enumerate(self.a):
            func.x.array[:] = a_vec[i * dof_size:(i + 1) * dof_size]


# ---------------------------------------------------------------------------
# Cost functions
# ---------------------------------------------------------------------------

def cost_function_and_grad(a_vec, a_vol, f, g, V, r, q, y_0, y_1, t_0, t_1, M_time, psi_1, psi_2):
    """Scalar cost ||A_f(a)(f) - g||^2 / 2 and its adjoint gradient; used by L-BFGS-B."""
    a_vol.update(V, a_vec)

    dt = (t_1 - t_0) / M_time
    g_pred, traj = compute_Af(a_vol, f, dt, M_time, V, r, q, y_0, y_1, t_0, psi_1, psi_2)

    residual = fem.Function(V)
    residual.x.array[:] = g_pred.x.array - g.x.array
    cost = 0.5 * L2_norm(residual) ** 2

    grads, _ = compute_adj_dAf(a_vol, residual, traj, dt, M_time, V, r, q, y_0, y_1, t_0)
    return cost, np.concatenate([gr.x.array for gr in grads])


def _residual_vec(a_vec, a_vol, f, g, V, r, q, y_0, y_1, t_0, t_1, M_time, psi_1, psi_2):
    """DOF-level residual A_f(a)(f) - g; used by the LM (TRF) solver."""
    a_vol.update(V, a_vec)
    dt = (t_1 - t_0) / M_time
    g_pred, _ = compute_Af(a_vol, f, dt, M_time, V, r, q, y_0, y_1, t_0, psi_1, psi_2)
    return g_pred.x.array - g.x.array


# ---------------------------------------------------------------------------
# Unified calibration entry point
# ---------------------------------------------------------------------------

def run_calibration(
    f, g, a_init, V, r, q, y_0, y_1, t_0, t_1, M_time, psi_1, psi_2,
    method: str = 'L-BFGS-B',
    max_iter: int = 50,
):
    """
    Calibrate a_init so that A_f(a)(f) ≈ g.

    method='L-BFGS-B'  — uses the adjoint gradient; scales to many parameters.
    method='LM'        — uses scipy least_squares (TRF variant of LM); more robust
                         for ill-conditioned problems but uses finite-difference
                         Jacobian, so keep N_cal_slices small (≤ 10 recommended).

    Positivity a >= 1e-8 is enforced in both methods.
    Returns (a_calibrated, scipy_result).
    """
    a_vec_init = np.concatenate([func.x.array for func in a_init.a])
    args = (a_init, f, g, V, r, q, y_0, y_1, t_0, t_1, M_time, psi_1, psi_2)
    n = len(a_vec_init)

    print(f"Starting calibration  method={method}  N_params={n}")

    if method == 'L-BFGS-B':
        res = scipy.optimize.minimize(
            fun=cost_function_and_grad,
            x0=a_vec_init,
            args=args,
            method='L-BFGS-B',
            jac=True,
            bounds=[(1e-8, None)] * n,
            options={'disp': True, 'maxiter': max_iter},
        )
        print(f"Done.  cost={res.fun:.4e}  success={res.success}")

    elif method == 'LM':
        # TRF (Trust Region Reflective) is the bounded variant of Levenberg-Marquardt.
        # scipy's 'lm' does not support bounds, so we use 'trf' which has equivalent
        # convergence properties and is the standard recommendation for bounded problems.
        res = scipy.optimize.least_squares(
            fun=_residual_vec,
            x0=a_vec_init,
            args=args,
            method='trf',
            bounds=([1e-8] * n, [np.inf] * n),
            max_nfev=max_iter * n,
            verbose=2,
        )
        print(f"Done.  cost={0.5 * res.cost:.4e}  success={res.success}")

    else:
        raise ValueError(f"Unknown method '{method}'. Choose 'L-BFGS-B' or 'LM'.")

    a_init.update(V, res.x)
    return a_init, res
