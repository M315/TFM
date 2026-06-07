from petsc4py.PETSc import ScalarType
import ufl
from dolfinx import fem
from dolfinx.fem.petsc import LinearProblem

from utils import create_bcs


def compute_Af(a_vol, u_initial, dt, M_time, V, r, q, y_0, y_1, t_0, psi_1_func, psi_2_func):
    """
    Applies the propagation operator A_f over M_time steps of size dt.

    Returns the terminal price function u(t_0 + M_time*dt) and the full
    trajectory [u(t_0), u(t_0+dt), ..., u(t_0+M_time*dt)].
    """
    u = fem.Function(V)
    u.x.array[:] = u_initial.x.array[:]
    trajectory = [u.copy()]

    for t in range(M_time):
        current_time = t_0 + (t + 1) * dt
        bcs = create_bcs(V, y_0, y_1, psi_1_func(current_time), psi_2_func(current_time))
        a = a_vol.get(current_time)
        u = _propagation_step(u, a, dt, V, r, q, bcs)
        trajectory.append(u.copy())

    return u, trajectory


def _propagation_step(u_old, a_func, dt, V, r_rate, q_rate, bcs):
    """
    One backward-Euler step of the forward Dupire PDE:

        u_τ = a (u_yy - u_y) - (r-q) u_y - q u

    Weak form (all spatial terms on the LHS, using IBP on the second-order term):

        ∫(u - u_old) v dy
        + dt [ ∫ a u_y v_y dy
             + ∫ a_y u_y v dy
             + ∫ (a + r - q) u_y v dy
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
            a_func * ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx   # ∫ a u_y v_y
            + a_func.dx(0) * u.dx(0) * v * ufl.dx                  # ∫ a_y u_y v
            + (a_func + r_c - q_c) * u.dx(0) * v * ufl.dx          # ∫ (a+r-q) u_y v
            + q_c * u * v * ufl.dx                                  # ∫ q u v
        )
    )

    problem = LinearProblem(
        ufl.lhs(F), ufl.rhs(F), bcs=bcs,
        petsc_options_prefix="A_f",
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
    )
    return problem.solve()
