r"""
Tikhonov regularization calibration - Egger & Engl (2005).

Minimises the functional

    J^\beta(a) = \|u(a)(\cdot,T) - u^\delta\|^2_{L^2}  +  \beta \|a - a*\|^2_{H^1}

via L-BFGS-B using the adjoint gradient.

Reference: Egger & Engl, "Tikhonov regularization applied to the inverse
problem of option pricing: convergence analysis and rates",
Inverse Problems 21 (2005) 1027-1045.
"""

import numpy as np
import scipy.optimize
from dolfin import Function

from pde.forward  import compute_Af
from pde.adjoint  import compute_adj_dAf
from utils        import L2_norm, H1_norm_sq, h1_reg_gradient, get_array, set_array


def cost_and_grad(a_vec, a_vol, u_0, u_obs, a_prior_vec, beta,
                  V, r, q, y_0, y_1, t_0, t_1, M_time, psi_1, psi_2,
                  obs_mask=None):
    r"""
    Evaluate J^\beta(a) and its Euclidean (DOF-level) gradient.

    Parameters
    ----------
    a_vec      : np.ndarray - current flat parameter vector
    a_vol      : Vol - parameter container (updated in place)
    u_0        : FEM Function - initial condition u(y,0)
    u_obs      : FEM Function - observed prices u^\delta(y,T)
    a_prior_vec: np.ndarray - DOF vector of the a priori guess a*
    beta       : float - regularisation parameter \beta
    obs_mask   : np.ndarray or None - {0,1} DOF mask selecting the observed
                 strikes.  None \to complete data (all nodes observed).  When set,
                 the residual is zeroed outside the observed nodes, so the data
                 term becomes the (quadrature-weighted) misfit over those strikes
                 only - the "incomplete data" setting of cases B / BD.

    Returns
    -------
    cost : float
    grad : np.ndarray - same shape as a_vec
    """
    a_vol.update(V, a_vec)

    dt = (t_1 - t_0) / M_time
    u_pred, traj = compute_Af(a_vol, u_0, dt, M_time, V, r, q, y_0, y_1, t_0, psi_1, psi_2)

    residual = Function(V)
    set_array(residual, get_array(u_pred) - get_array(u_obs))
    if obs_mask is not None:
        # Restrict the misfit to the observed strikes.  Masking the residual is
        # consistent for both the cost and the adjoint gradient: the projection
        # P is idempotent, so \partial_a \frac{1}{2}\|P(u-u^\delta)\|^2 = \int P(u-u^\delta)\cdot w(T), i.e. the adjoint
        # terminal condition is simply the masked residual.
        set_array(residual, get_array(residual) * obs_mask)

    # Data-fidelity term: \|u(a) - u^\delta\|^2_{L^2}
    cost_data = L2_norm(residual) ** 2

    # Regularisation term: \beta \|a - a*\|^2_{H^1}
    da = Function(V)
    set_array(da, a_vec - a_prior_vec)
    cost_reg = beta * H1_norm_sq(da, V)

    cost = cost_data + cost_reg

    # Adjoint gradient of \|u(a) - u^\delta\|^2_{L^2}
    # (compute_adj_dAf returns the gradient of \frac{1}{2}\|\cdot\|^2, so multiply by 2)
    grads, _ = compute_adj_dAf(a_vol, residual, traj, dt, M_time, V, r, q, y_0, y_1, t_0)
    data_grad = 2.0 * np.concatenate([get_array(gr) for gr in grads])

    # Gradient of \beta \|a - a*\|^2_{H^1}: assembles 2\beta(M+K)(a - a*)
    reg_grad = h1_reg_gradient(da, beta, V)

    return cost, data_grad + reg_grad


def run(u_0, u_obs, a_init, a_prior_vec, beta,
        V, r, q, y_0, y_1, t_0, t_1, M_time, psi_1, psi_2,
        max_iter=200, ftol=1e-10, gtol=1e-8, obs_mask=None):
    r"""
    Minimise J^\beta via L-BFGS-B.

    Parameters
    ----------
    u_0, u_obs  : FEM Functions - initial condition and observations
    a_init      : Vol - initial parameter guess (modified in place with result)
    a_prior_vec : np.ndarray - a* for the H^1 regularisation term
    beta        : float - regularisation parameter \beta
    max_iter    : int - L-BFGS-B iteration budget
    ftol, gtol  : float - convergence tolerances on cost and gradient

    Returns
    -------
    a_init : Vol - calibrated parameter (same object, updated in place)
    res    : scipy OptimizeResult
    """
    a_vec_init = a_init.to_vec()
    n = len(a_vec_init)

    args = (a_init, u_0, u_obs, a_prior_vec, beta,
            V, r, q, y_0, y_1, t_0, t_1, M_time, psi_1, psi_2, obs_mask)

    print(f"Tikhonov  β={beta:.1e}  N_params={n}")

    res = scipy.optimize.minimize(
        fun=cost_and_grad,
        x0=a_vec_init,
        args=args,
        method='L-BFGS-B',
        jac=True,
        bounds=[(1e-8, None)] * n,
        options={'disp': True, 'maxiter': max_iter, 'ftol': ftol, 'gtol': gtol},
    )

    print(f"Done.  cost={res.fun:.4e}  success={res.success}")
    a_init.update(V, res.x)
    return a_init, res
