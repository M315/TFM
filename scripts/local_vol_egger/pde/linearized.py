r"""
Gateaux (linearized) PDE solver - operator dA_f.

Computes the directional derivative A'_f(a)[h] by solving the
linearized (sensitivity) forward equation

    w_\tau = a(w_yy - w_y) - (r-q) w_y - qw  -  h(u_yy - u_y)

forward from t_0, with zero initial and boundary conditions.

Used for:
  - gradient verification (compare with finite differences / the adjoint)
  - analytical Jacobian construction (per-slice, via active_slice)
"""

from dolfin import (
    TrialFunction, TestFunction, Function, Constant,
    grad, dot, dx, lhs, rhs, solve,
)

from utils import create_bcs


def compute_dAf(a_vol, h_func, trajectory, dt, M_time, V, r, q, y_0, y_1, t_0,
                active_slice=None):
    """
    Compute A'_f(a)[h] - the Gateaux derivative in direction h.

    Parameters
    ----------
    a_vol        : Vol
    h_func       : Function - perturbation direction in parameter space
    trajectory   : list of Functions - forward trajectory from compute_Af
    active_slice : int or None - if set, h is applied only in the time steps
                   belonging to that Vol slice (isolates per-slice sensitivity);
                   None applies h uniformly across all steps

    Returns
    -------
    w : Function - A'_f(a)[h] at time t_0 + M_time*dt
    """
    w = Function(V)                 # w(t_0) = 0  (IC independent of a)

    bcs = create_bcs(V, y_0, y_1, 0.0, 0.0)

    zero_fn = Function(V)           # used when h is inactive on a slice

    for t in range(M_time):
        current_time = t_0 + (t + 1) * dt
        a   = a_vol.get(current_time)
        u_t = trajectory[t + 1]      # implicit-Euler new-time state u^n

        if active_slice is not None and a_vol.get_idx(current_time) != active_slice:
            h_active = zero_fn
        else:
            h_active = h_func

        w = _step(w, u_t, a, h_active, dt, V, r, q, bcs)

    return w


def _step(w_old, u_t, a_func, h_func, dt, V, r_rate, q_rate, bcs):
    r"""
    One backward-Euler step of the linearized equation.

    Linearising u_\tau = a(u_yy - u_y) - (r-q)u_y - qu in direction h gives:

        w_\tau = a(w_yy - w_y) - (r-q)w_y - qw  +  h(u_yy - u_y)

    Weak form (IBP on both the homogeneous and source terms):

        \int(w - w_old) v dy
        + dt [ \int a w_y v_y dy  +  \int \partial_y a \cdot w_y v dy
             + \int (a+r-q) w_y v dy  +  \int q w v dy
             + \int h u_y v_y dy  +  \int \partial_y h \cdot u_y v dy  +  \int h u_y v dy ] = 0
    """
    w = TrialFunction(V)
    v = TestFunction(V)

    dt_c = Constant(dt)
    r_c  = Constant(r_rate)
    q_c  = Constant(q_rate)

    F = (
        (w - w_old) * v * dx
        + dt_c * (
            # homogeneous part (same operator as forward PDE)
            a_func * dot(grad(w), grad(v)) * dx          # \int a w_y v_y
            + a_func.dx(0) * w.dx(0) * v * dx             # \int \partial_y a \cdot w_y v
            + (a_func + r_c - q_c) * w.dx(0) * v * dx     # \int (a+r-q) w_y v
            + q_c * w * v * dx                            # \int q w v
            # source h(u_yy - u_y): after IBP \to \int(h u_y v_y + \partial_y h \cdot u_y v + h u_y v)
            + h_func * dot(grad(u_t), grad(v)) * dx       # \int h u_y v_y
            + h_func.dx(0) * u_t.dx(0) * v * dx            # \int \partial_y h \cdot u_y v
            + h_func * u_t.dx(0) * v * dx                  # \int h u_y v
        )
    )

    w_sol = Function(V)
    solve(lhs(F) == rhs(F), w_sol, bcs,
          solver_parameters={"linear_solver": "lu"})
    return w_sol
