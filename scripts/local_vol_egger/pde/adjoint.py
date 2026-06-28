r"""
Adjoint PDE solver - operator adj_dA_f.

Solves the backward adjoint equation

    V_\tau + (aV)_yy + (aV)_y + (r-q) V_y - qV = 0,   V(y, T) = residual(y)

backward in \tau and accumulates the Euclidean gradient

    \partial J/\partial a_j = -dt \int [ u_y m_y \phi_j  +  u_y m \partial_y \phi_j  +  u_y m \phi_j ] dy

where m = V and \phi_j are the FEM basis functions.

The gradient is assembled directly as a linear 1-form (no mass-matrix solve),
which gives the correct Euclidean (DOF-level) gradient for L-BFGS-B.
"""

from dolfin import (
    TrialFunction, TestFunction, Function, Constant,
    grad, dot, dx, lhs, rhs, solve, assemble,
)

from utils import create_bcs, get_array, set_array


def compute_adj_dAf(a_vol, residual, trajectory, dt, M_time, V, r, q, y_0, y_1, t_0):
    r"""
    Compute the gradient of \frac{1}{2}\|A_f(a) - g\|^2_{L^2} w.r.t. the DOF vector of a.

    Parameters
    ----------
    a_vol     : Vol
    residual  : Function - A_f(a) - g  (adjoint terminal condition)
    trajectory: list of Functions - forward trajectory from compute_Af
    dt        : float
    M_time    : int

    Returns
    -------
    gradients : list of Functions - one per Vol time slice, each of length n_dof
    m_at_t0   : Function - adjoint state at t = t_0 (for diagnostics)
    """
    m = Function(V)
    set_array(m, get_array(residual))     # m(T) = A_f(a) - g

    gradients = [Function(V) for _ in a_vol.a]   # zero-initialised

    bcs = create_bcs(V, y_0, y_1, 0.0, 0.0)

    for t in range(M_time - 1, -1, -1):
        current_time = t_0 + (t + 1) * dt
        a   = a_vol.get(current_time)
        u_t = trajectory[t + 1]      # implicit-Euler new-time state u^n

        m_prev, g_inc = _step(m, u_t, a, dt, V, r, q, bcs)

        idx = a_vol.get_idx(current_time)
        set_array(gradients[idx], get_array(gradients[idx]) + get_array(g_inc))
        set_array(m, get_array(m_prev))

    return gradients, m


def _step(m_curr, u_t, a_func, dt, V, r_rate, q_rate, bcs):
    r"""
    One backward-Euler step of the adjoint equation, plus the gradient contribution.

    Adjoint PDE (in reverse time \tau' = T - \tau):

        m_{\tau'} = -(am)_yy - (am)_y - (r-q) m_y + qm

    Backward-Euler weak form (m unknown, m_curr = previous adjoint state):

        \int(m - m_curr) v dy
        + dt [ \int (am)_y v_y dy
             + \int a m v_y dy
             + \int (r-q) m v_y dy
             + \int q m v dy ] = 0

    Gradient contribution assembled as a linear 1-form:

        g_j = -dt \int [ u_y m_y \phi_j   (from B1: \int a u_y v_y)
                    + u_y m \partial_y \phi_j  (from B2: \int \partial_y a \cdot u_y v)
                    + u_y m \phi_j ]    (from B3: \int (a+r-q) u_y v)   dy
    """
    m = TrialFunction(V)
    v = TestFunction(V)

    dt_c = Constant(dt)
    r_c  = Constant(r_rate)
    q_c  = Constant(q_rate)

    adj_form = (
        (m - m_curr) * v * dx
        + dt_c * (
            a_func * dot(grad(m), grad(v)) * dx     # \int a m_y v_y        ] together these
            + m * dot(grad(a_func), grad(v)) * dx   # \int \partial_y a m v_y  ] give \int (am)_y v_y
            + (a_func + r_c - q_c) * m * v.dx(0) * dx   # \int (a+r-q) m v_y
            + q_c * m * v * dx                          # \int q m v
        )
    )

    m_new = Function(V)
    solve(lhs(adj_form) == rhs(adj_form), m_new, bcs,
          solver_parameters={"linear_solver": "lu"})

    # Gradient \partial J / \partial a_j assembled directly as a 1-form - no mass-matrix solve needed.
    # Captures all three terms from differentiating the forward bilinear form w.r.t. a_j.
    h_test = TestFunction(V)
    grad_form = (
        -dt_c * (
            dot(grad(u_t), grad(m_new)) * h_test    # B1: \int u_y m_y \phi_j
            + u_t.dx(0) * m_new * h_test.dx(0)      # B2: \int u_y m \partial_y \phi_j
            + u_t.dx(0) * m_new * h_test            # B3: \int u_y m \phi_j
        ) * dx
    )
    g_func = Function(V)
    set_array(g_func, assemble(grad_form).get_local())

    return m_new, g_func
