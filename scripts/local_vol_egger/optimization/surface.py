r"""
Surface local-volatility calibration, extending Egger & Engl (2005) to a time-dependent a.

The unknown a(y,tau) = 1/2 sigma^2(y,tau) is held as N spatial slices, one per observed
maturity (a single slice reproduces Egger's time-independent fit). Prices come from one
forward Dupire solve to T_max, sampled at each maturity; the gradient comes from one backward
adjoint sweep. Egger's convergence-rate theorems 4.1-4.2 assume a time-independent a, so the
surface fit is a numerical extension beyond their proven setting.

  * Forward: one solve, sampled at each maturity tau_j and compared to its smile g_j on the band.
  * Adjoint: one backward sweep; each maturity's masked residual is injected as the sweep
    reaches tau_j (the adjoint of a sum-of-terminal-costs misfit) and feeds only the slices up
    to tau_j, so early maturities fix the near slices and later ones the far slices.
  * Regularization: H1 in y per slice plus an L2 penalty on the jump between slices (see
    _surface_reg), the discrete form of an H1-in-y, L2-in-tau penalty.
  * r and q enter as piecewise-constant term structures r(tau), q(tau).
  * `discounted` selects Egger & Engl's change of variable u = e^{\int_0^tau q} C, which
    removes the zeroth-order term and leaves -u_tau + a(u_yy - u_y) + (q-r)u_y = 0. The caller
    must then supply data and boundary values already scaled by e^{\int_0^tau q} and undo the
    scaling on the output. It is off by default (the two agree when q = 0); see the annex
    subsection on the discounted normalization for the comparison.

The single-step forward/adjoint kernels are reused from pde.forward / pde.adjoint.
"""

import numpy as np
import scipy.optimize
from dolfin import Function

from pde.forward import _step as fwd_step
from pde.adjoint import _step as adj_step
from utils import (create_bcs, L2_norm, H1_norm_sq, h1_reg_gradient,
                   l2_mass_apply, get_array, set_array)


# ---------------------------------------------------------------------------
# Forward / adjoint time-marching with a term structure
# ---------------------------------------------------------------------------

def compute_Af_surface(a_vol, u_0, dt, M_time, V, r_of_t, q_of_t,
                       y_0, y_1, psi_1, psi_2, discounted=False):
    r"""
    Forward Dupire solve on [0, M_time*dt] with time-dependent rates.

    Same implicit-Euler step as pde.forward, but r and q are read per step from r_of_t(t),
    q_of_t(t). Returns the terminal state and the full trajectory [u(0), u(dt), ..., u(T)]
    so it can be sampled at each observed maturity. With `discounted` the reaction coefficient
    is set to zero, i.e. the solve is for u = e^{\int_0^tau q} C.
    """
    u = u_0.copy(deepcopy=True)
    trajectory = [u.copy(deepcopy=True)]
    for t in range(M_time):
        ct = (t + 1) * dt
        bcs = create_bcs(V, y_0, y_1, psi_1(ct), psi_2(ct))
        a = a_vol.get(ct)
        u = fwd_step(u, a, dt, V, r_of_t(ct), q_of_t(ct), bcs,
                     c_rate=0.0 if discounted else None)
        trajectory.append(u.copy(deepcopy=True))
    return u, trajectory


def compute_adj_surface(a_vol, injections, trajectory, dt, M_time, V,
                        r_of_t, q_of_t, y_0, y_1, discounted=False):
    r"""
    Backward adjoint sweep for the multi-maturity data misfit.

    `injections` maps a trajectory step index n_j to the masked residual
    P_j(u(tau_j) - g_j) at maturity tau_j. The adjoint state m starts from the terminal
    maturity's residual and picks up each earlier maturity's residual as the sweep passes its
    time level. The residual at level n_j is added only after that interval's gradient has
    been taken, so maturity tau_j feeds a(y) only on [0, tau_j].

    Returns one gradient Function per time slice (a single entry when N=1).
    """
    m = Function(V)                                   # zero at the final time
    if M_time in injections:                          # terminal-maturity residual
        set_array(m, get_array(injections[M_time]))

    gradients = [Function(V) for _ in a_vol.a]
    bcs = create_bcs(V, y_0, y_1, 0.0, 0.0)

    for t in range(M_time - 1, -1, -1):
        ct = (t + 1) * dt
        a = a_vol.get(ct)
        u_t = trajectory[t + 1]

        m_new, g_inc = adj_step(m, u_t, a, dt, V, r_of_t(ct), q_of_t(ct), bcs,
                                c_rate=0.0 if discounted else None)

        idx = a_vol.get_idx(ct)
        set_array(gradients[idx], get_array(gradients[idx]) + get_array(g_inc))

        # m_new is the adjoint at time level t; add the residual observed there, after this
        # interval's gradient contribution, so tau_j does not feed a(y) on later intervals.
        if t in injections:
            set_array(m_new, get_array(m_new) + get_array(injections[t]))
        set_array(m, get_array(m_new))

    return gradients


# ---------------------------------------------------------------------------
# Space-time regularization
# ---------------------------------------------------------------------------

def _surface_reg(a_mat, a_prior_mat, beta_y, beta_tau, V):
    r"""
    Space-time Tikhonov penalty for the surface, held as N spatial slices.

        beta_y  sum_j ||a_j - a*_j||^2_{H1}  +  beta_tau  sum_j ||a_j - a_{j-1}||^2_{L2},

    the discrete form of an H1 penalty in y plus an L2 penalty on d a / d tau. `a_mat` and
    `a_prior_mat` are (N, n_dof). Returns (cost, grad) with grad of shape (N, n_dof). With
    N = 1 the second sum is empty, recovering the plain H1 penalty of the single-curve fit.
    """
    N = a_mat.shape[0]
    cost = 0.0
    grad = np.zeros_like(a_mat)

    for j in range(N):                                   # H1 smoothness in y, per slice
        da = Function(V)
        set_array(da, a_mat[j] - a_prior_mat[j])
        cost += beta_y * H1_norm_sq(da, V)
        grad[j] += h1_reg_gradient(da, beta_y, V)

    for j in range(1, N):                                # L2 smoothness across maturities
        d = Function(V)
        set_array(d, a_mat[j] - a_mat[j - 1])
        cost += beta_tau * L2_norm(d) ** 2
        md = 2.0 * beta_tau * l2_mass_apply(d, V)
        grad[j] += md
        grad[j - 1] -= md

    return cost, grad


# ---------------------------------------------------------------------------
# Cost / gradient and driver
# ---------------------------------------------------------------------------

def cost_and_grad(a_vec, a_vol, u_0, obs, a_prior_mat, beta_y, beta_tau,
                  V, r_of_t, q_of_t, y_0, y_1, dt, M_time, psi_1, psi_2,
                  discounted=False):
    r"""
    Evaluate the surface Tikhonov functional

        sum_j ||P_j(u(tau_j) - g_j)||^2 + (space-time penalty, see _surface_reg)

    and its Euclidean gradient. `obs` is a list of per-maturity dicts with keys:
        'idx'  : trajectory step index n_j of maturity tau_j
        'mask' : {0,1} DOF mask of that maturity's observed band
        'g'    : market call prices on the nodes for that maturity
    """
    a_vol.update(V, a_vec)
    _, traj = compute_Af_surface(a_vol, u_0, dt, M_time, V, r_of_t, q_of_t,
                                 y_0, y_1, psi_1, psi_2, discounted)

    # Per-maturity masked residuals -> data cost + adjoint injections.
    cost_data = 0.0
    injections = {}
    for ob in obs:
        res = Function(V)
        set_array(res, (get_array(traj[ob["idx"]]) - get_array(ob["g"])) * ob["mask"])
        cost_data += L2_norm(res) ** 2
        injections[ob["idx"]] = res

    grads = compute_adj_surface(a_vol, injections, traj, dt, M_time, V,
                                r_of_t, q_of_t, y_0, y_1, discounted)
    data_grad = 2.0 * np.concatenate([get_array(gr) for gr in grads])

    a_mat = a_vec.reshape(len(a_vol.a), V.dim())
    reg_cost, reg_grad = _surface_reg(a_mat, a_prior_mat, beta_y, beta_tau, V)

    return cost_data + reg_cost, data_grad + reg_grad.ravel()


def run(u_0, obs, a_init, a_prior_mat, beta_y, beta_tau, V, r_of_t, q_of_t,
        y_0, y_1, dt, M_time, psi_1, psi_2,
        max_iter=300, ftol=1e-12, gtol=1e-10, discounted=False):
    r"""Minimise the surface Tikhonov functional via L-BFGS-B (adjoint gradient)."""
    a_vec_init = a_init.to_vec()
    n = len(a_vec_init)
    args = (a_init, u_0, obs, a_prior_mat, beta_y, beta_tau, V, r_of_t, q_of_t,
            y_0, y_1, dt, M_time, psi_1, psi_2, discounted)

    print(rf"Surface Tikhonov  beta_y={beta_y:.1e} beta_tau={beta_tau:.1e}  "
          rf"N_slices={len(a_init.a)}  N_params={n}  N_maturities={len(obs)}  "
          rf"discounted={discounted}")

    res = scipy.optimize.minimize(
        fun=cost_and_grad,
        x0=a_vec_init,
        args=args,
        method="L-BFGS-B",
        jac=True,
        bounds=[(1e-8, None)] * n,
        options={"disp": True, "maxiter": max_iter, "ftol": ftol, "gtol": gtol},
    )

    print(f"Done.  cost={res.fun:.4e}  success={res.success}")
    a_init.update(V, res.x)
    return a_init, res
