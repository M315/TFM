"""
Gradient (adjoint) verification for the Egger & Engl (2005) calibration.

Justifies that the discrete adjoint gradient assembled in pde/adjoint.py is the
exact gradient of the discrete data-misfit functional.  Three independent checks:

  1. Tangent-linear <-> adjoint identity (exact, no finite differences).
     For the data term J_d(a) = ||A_f(a) - u_obs||^2_{L2} with residual
     r = A_f(a) - u_obs, the directional derivative computed two ways must agree:

         <grad J_d, h>_Euclid  ==  2 (r, A'_f(a)[h])_{L2}

     LHS uses the adjoint solve (pde/adjoint.py); RHS uses the tangent-linear /
     sensitivity solve (pde/linearized.py).  Agreement to solver tolerance proves
     the adjoint is the transpose of the linearised forward operator.

  2. Finite-difference Taylor test on the full Tikhonov functional J^beta:
         R0(t) = |J(a+t h) - J(a)|                ~ O(t)
         R1(t) = |J(a+t h) - J(a) - t <grad,h>|   ~ O(t^2)
     The second-order decay of R1 is the standard gradient-correctness signature.

  3. scipy.optimize.check_grad on J^beta (the course's idiom).

Run:  conda run -n fenics-legacy python derivative_test.py
"""

import numpy as np
import scipy.optimize as opt
from dolfin import Function, FunctionSpace, IntervalMesh, assemble, inner, dx, \
    set_log_level, LogLevel

from optimization.vol      import Vol
from optimization.tikhonov import cost_and_grad
from pde.forward           import compute_Af
from pde.linearized        import compute_dAf
from pde.adjoint           import compute_adj_dAf
from utils                 import interpolate_func, get_array, set_array, bs_call

set_log_level(LogLevel.ERROR)

# ---------------------------------------------------------------------------
# Small, fast test problem.  S0 = 1 keeps the prices (and hence the cost) O(1),
# so the finite-difference test is well scaled; the adjoint is scale-independent.
# ---------------------------------------------------------------------------
Y0, Y1     = -3.0, 3.0
T_END      = 1.0
R, Q       = 0.0, 0.0
S0         = 1.0
N_SPACE    = 50
M_TIME     = 50
SIGMA_TRUE = np.sqrt(2.0 * 0.15)
DT         = T_END / M_TIME


def _bc_left(tau):
    if tau <= 1e-14:
        return float(S0 * max(1.0 - np.exp(Y0), 0.0))
    return float(bs_call(S0, Y0, R, Q, SIGMA_TRUE, tau))


def _bc_right(tau):
    if tau <= 1e-14:
        return 0.0
    return float(bs_call(S0, Y1, R, Q, SIGMA_TRUE, tau))


def _setup():
    """Build the space, IC, observations (from a_true) and a test point a0 != a_true."""
    msh = IntervalMesh(N_SPACE, Y0, Y1)
    V   = FunctionSpace(msh, "Lagrange", 1)

    u_0 = interpolate_func(V, lambda y: S0 * np.maximum(1.0 - np.exp(y), 0.0))

    # Observations: forward solve at the true constant a_true = 0.15.
    a_true = Vol(V, interpolate_func(V, lambda y: np.full_like(y, 0.15)),
                 0.0, T_END, N=1)
    u_obs, _ = compute_Af(a_true, u_0, DT, M_TIME, V, R, Q, Y0, Y1, 0.0,
                          _bc_left, _bc_right)

    # Test point: a smooth field different from a_true, so the residual is nonzero
    # and the gradient is informative.
    a0 = Vol(V, interpolate_func(V, lambda y: 0.12 + 0.02 * np.exp(-y**2)),
             0.0, T_END, N=1)
    return V, u_0, u_obs, a0


def test_tlm_adjoint_identity(V, u_0, u_obs, a0):
    """Check <grad J_d, h>_Euclid == 2 (r, A'_f[h])_L2 (data term, exact)."""
    a_vec = a0.to_vec()
    a0.update(V, a_vec)

    # Forward solve and residual r = A_f(a) - u_obs.
    u_pred, traj = compute_Af(a0, u_0, DT, M_TIME, V, R, Q, Y0, Y1, 0.0,
                              _bc_left, _bc_right)
    residual = Function(V)
    set_array(residual, get_array(u_pred) - get_array(u_obs))

    # Adjoint Euclidean gradient of ||r||^2 (factor 2: compute_adj_dAf gives 1/2||.||^2).
    grads, _ = compute_adj_dAf(a0, residual, traj, DT, M_TIME, V, R, Q, Y0, Y1, 0.0)
    data_grad = 2.0 * np.concatenate([get_array(g) for g in grads])

    # A fixed smooth direction h.
    rng = np.random.default_rng(0)
    h_vec = rng.standard_normal(a_vec.size)
    h_func = Function(V)
    set_array(h_func, h_vec)

    lhs_val = float(np.dot(data_grad, h_vec))                  # adjoint side

    w = compute_dAf(a0, h_func, traj, DT, M_TIME, V, R, Q, Y0, Y1, 0.0)
    rhs_val = 2.0 * assemble(inner(residual, w) * dx)          # tangent-linear side

    rel = abs(lhs_val - rhs_val) / max(abs(rhs_val), 1e-300)
    print("\n[1] Tangent-linear <-> adjoint identity (data term)")
    print(f"    <grad, h>_adjoint = {lhs_val: .12e}")
    print(f"    2 (r, A'_f[h])_L2 = {rhs_val: .12e}")
    print(f"    relative error    = {rel: .3e}   "
          f"{'PASS' if rel < 1e-8 else 'CHECK'}")
    return rel


def _cost_args(V, u_0, u_obs, a0, beta):
    prior_vec = np.zeros(a0.to_vec().size)   # beta=0 -> prior irrelevant
    return (a0, u_0, u_obs, prior_vec, beta,
            V, R, Q, Y0, Y1, 0.0, T_END, M_TIME, _bc_left, _bc_right, None)


def test_taylor(V, u_0, u_obs, a0, beta=0.0):
    """Finite-difference Taylor test on J^beta (beta=0 isolates the adjoint)."""
    args = _cost_args(V, u_0, u_obs, a0, beta)
    a_vec = a0.to_vec()

    J0, grad0 = cost_and_grad(a_vec, *args)

    rng = np.random.default_rng(0)
    h_vec = rng.standard_normal(a_vec.size)
    dir_deriv = float(np.dot(grad0, h_vec))

    print(f"\n[2] Taylor test on J (beta={beta:g}),  <grad,h> = {dir_deriv: .6e}")
    print(f"    {'t':>8} {'R0=|dJ|':>14} {'R1=|dJ-t g.h|':>16} {'rate(R1)':>10}")
    prev = None
    for t in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]:
        Jt, _ = cost_and_grad(a_vec + t * h_vec, *args)
        R0 = abs(Jt - J0)
        R1 = abs(Jt - J0 - t * dir_deriv)
        rate = "" if prev is None else f"{np.log(prev / R1) / np.log(10):.2f}"
        print(f"    {t:>8.0e} {R0:>14.6e} {R1:>16.6e} {rate:>10}")
        prev = R1
    print("    expected rate(R1) ~ 2.0  (gradient correct)")


def test_check_grad(V, u_0, u_obs, a0, beta=0.0):
    """scipy.optimize.check_grad on J^beta (course idiom)."""
    args = _cost_args(V, u_0, u_obs, a0, beta)
    a_vec = a0.to_vec()

    f      = lambda x: cost_and_grad(x, *args)[0]
    grad_f = lambda x: cost_and_grad(x, *args)[1]

    err = opt.check_grad(f, grad_f, a_vec)
    gnorm = np.linalg.norm(grad_f(a_vec))
    print(f"\n[3] scipy.optimize.check_grad (beta={beta:g})")
    print(f"    ||grad_fd - grad_adj|| = {err: .3e}")
    print(f"    relative to ||grad||   = {err / gnorm: .3e}   "
          f"{'PASS' if err / gnorm < 1e-4 else 'CHECK'}")


if __name__ == "__main__":
    print("=" * 70)
    print("  Adjoint-gradient verification - Egger calibration (legacy dolfin)")
    print("=" * 70)
    V, u_0, u_obs, a0 = _setup()
    test_tlm_adjoint_identity(V, u_0, u_obs, a0)
    test_taylor(V, u_0, u_obs, a0, beta=0.0)
    test_check_grad(V, u_0, u_obs, a0, beta=0.0)
    print("\nDone.")
