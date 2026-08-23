r"""
Forward Dupire PDE solver - operator A_f.

Solves the log-moneyness form of the Dupire forward equation

    u_\tau = a(y,\tau)(u_yy - u_y) - (r-q) u_y - q u

by the implicit (backward) Euler scheme on a 1-D dolfin mesh.
"""

from dolfin import (
    TrialFunction, TestFunction, Function, Constant,
    grad, dot, dx, lhs, rhs, solve,
)

from utils import create_bcs


def compute_Af(a_vol, u_initial, dt, M_time, V, r, q, y_0, y_1, t_0, psi_1, psi_2):
    r"""
    Apply the propagation operator A_f over M_time steps of size dt.

    Parameters
    ----------
    a_vol    : Vol - piecewise-constant-in-time diffusion coefficient a = \frac{1}{2}\sigma^2
    u_initial: Function - call-price profile at time t_0
    dt       : float - time step
    M_time   : int - number of time steps
    V        : FunctionSpace
    r, q     : float - interest and dividend rates
    y_0, y_1 : float - left/right boundary of the log-moneyness domain
    t_0      : float - start time of this interval
    psi_1    : callable(t) \to float - Dirichlet BC at y_0
    psi_2    : callable(t) \to float - Dirichlet BC at y_1

    Returns
    -------
    u        : Function - call-price profile at t_0 + M_time*dt
    trajectory: list of Functions - snapshots [u(t_0), u(t_0+dt), ..., u(T)]
                (needed by the adjoint solver to compute the gradient)
    """
    u = u_initial.copy(deepcopy=True)
    trajectory = [u.copy(deepcopy=True)]

    for t in range(M_time):
        current_time = t_0 + (t + 1) * dt
        bcs = create_bcs(V, y_0, y_1, psi_1(current_time), psi_2(current_time))
        a = a_vol.get(current_time)
        u = _step(u, a, dt, V, r, q, bcs)
        trajectory.append(u.copy(deepcopy=True))

    return u, trajectory


def _step(u_old, a_func, dt, V, r_rate, q_rate, bcs, c_rate=None):
    r"""
    One backward-Euler step of the forward Dupire PDE.

    Weak form (IBP applied to the second-order term \int a u_yy v dy):

        \int(u - u_old) v dy
        + dt [ \int a u_y v_y dy
             + \int \partial_y a \cdot u_y v dy
             + \int (a + r - q) u_y v dy
             + \int c u v dy ] = 0

    `c_rate` is the zeroth-order (reaction) coefficient, q by default. Passing
    c_rate = 0 solves instead for the discounted variable u = e^{\int_0^\tau q} C of
    Egger & Engl, whose equation -u_\tau + a(u_yy - u_y) + (q-r)u_y = 0 has no
    reaction term. See the annex subsection on the discounted normalization.
    """
    u = TrialFunction(V)
    v = TestFunction(V)

    dt_c = Constant(dt)
    r_c  = Constant(r_rate)
    q_c  = Constant(q_rate)
    c_c  = Constant(q_rate if c_rate is None else c_rate)

    F = (
        (u - u_old) * v * dx
        + dt_c * (
            a_func * dot(grad(u), grad(v)) * dx          # \int a u_y v_y
            + a_func.dx(0) * u.dx(0) * v * dx             # \int \partial_y a \cdot u_y v
            + (a_func + r_c - q_c) * u.dx(0) * v * dx     # \int (a+r-q) u_y v
            + c_c * u * v * dx                            # \int c u v  (c = q, or 0 if discounted)
        )
    )

    u_sol = Function(V)
    solve(lhs(F) == rhs(F), u_sol, bcs,
          solver_parameters={"linear_solver": "lu"})
    return u_sol
