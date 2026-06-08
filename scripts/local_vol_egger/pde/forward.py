"""
Forward Dupire PDE solver — operator A_f.

Solves the log-moneyness form of the Dupire forward equation

    u_τ = a(y,τ)(u_yy − u_y) − (r−q) u_y − q u

by the implicit (backward) Euler scheme on a 1-D FEniCSx mesh.
"""

from petsc4py.PETSc import ScalarType
import ufl
from dolfinx import fem
from dolfinx.fem.petsc import LinearProblem

from utils import create_bcs


def compute_Af(a_vol, u_initial, dt, M_time, V, r, q, y_0, y_1, t_0, psi_1, psi_2):
    """
    Apply the propagation operator A_f over M_time steps of size dt.

    Parameters
    ----------
    a_vol    : Vol — piecewise-constant-in-time diffusion coefficient a = ½σ²
    u_initial: FEM Function — call-price profile at time t_0
    dt       : float — time step
    M_time   : int — number of time steps
    V        : FunctionSpace
    r, q     : float — interest and dividend rates
    y_0, y_1 : float — left/right boundary of the log-moneyness domain
    t_0      : float — start time of this interval
    psi_1    : callable(t) → float — Dirichlet BC at y_0
    psi_2    : callable(t) → float — Dirichlet BC at y_1

    Returns
    -------
    u        : FEM Function — call-price profile at t_0 + M_time*dt
    trajectory: list of FEM Functions — snapshots [u(t_0), u(t_0+dt), ..., u(T)]
                (needed by the adjoint solver to compute the gradient)
    """
    u = fem.Function(V)
    u.x.array[:] = u_initial.x.array[:]
    trajectory = [u.copy()]

    for t in range(M_time):
        current_time = t_0 + (t + 1) * dt
        bcs = create_bcs(V, y_0, y_1, psi_1(current_time), psi_2(current_time))
        a = a_vol.get(current_time)
        u = _step(u, a, dt, V, r, q, bcs)
        trajectory.append(u.copy())

    return u, trajectory


def _step(u_old, a_func, dt, V, r_rate, q_rate, bcs):
    """
    One backward-Euler step of the forward Dupire PDE.

    Weak form (IBP applied to the second-order term ∫ a u_yy v dy):

        ∫(u − u_old) v dy
        + dt [ ∫ a u_y v_y dy
             + ∫ ∂_y a · u_y v dy
             + ∫ (a + r − q) u_y v dy
             + ∫ q u v dy ] = 0
    """
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    dt_c = fem.Constant(V.mesh, ScalarType(dt))
    r_c  = fem.Constant(V.mesh, ScalarType(r_rate))
    q_c  = fem.Constant(V.mesh, ScalarType(q_rate))

    F = (
        (u - u_old) * v * ufl.dx
        + dt_c * (
            a_func * ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx       # ∫ a u_y v_y
            + a_func.dx(0) * u.dx(0) * v * ufl.dx                      # ∫ ∂_y a · u_y v
            + (a_func + r_c - q_c) * u.dx(0) * v * ufl.dx              # ∫ (a+r−q) u_y v
            + q_c * u * v * ufl.dx                                      # ∫ q u v
        )
    )

    return LinearProblem(
        ufl.lhs(F), ufl.rhs(F), bcs=bcs,
        petsc_options_prefix="fwd",
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
    ).solve()
