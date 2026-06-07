from petsc4py.PETSc import ScalarType
import ufl
from dolfinx import fem
from dolfinx.fem.petsc import LinearProblem

from utils import create_bcs


def compute_dAf(a_vol, h_func, trajectory, dt, M_time, V, r, q, y_0, y_1, t_0):
    """
    Computes the Gâteaux derivative A'_f(a) · h  by integrating the linearized
    PDE forward from t_0.

    Used for gradient checking (compare with finite-difference on A_f);
    not called in the calibration loop.
    """
    w = fem.Function(V)
    w.x.array[:] = 0.0     # w(t_0) = 0  (IC does not depend on a)

    zero = fem.Constant(V.mesh, 0.0)
    bcs  = create_bcs(V, y_0, y_1, zero, zero)

    for t in range(M_time):
        current_time = t_0 + (t + 1) * dt
        a   = a_vol.get(current_time)
        u_t = trajectory[t]           # u at the START of this step
        w   = _linearized_step(w, u_t, a, h_func, dt, V, r, q, bcs)

    return w


def _linearized_step(w_old, u_t, a_func, h_func, dt, V, r_rate, q_rate, bcs):
    """
    One backward-Euler step of the linearized (Gâteaux) equation.

    Linearising  u_τ = a(u_yy - u_y) - (r-q)u_y - qu  in direction h gives:

        w_τ = a(w_yy - w_y) - (r-q)w_y - qw  +  h(u_yy - u_y)

    Weak form (F = 0 residual; all terms on the left, IBP applied):

        F = ∫(w - w_old)v dy
          + dt [ ∫ a w_y v_y dy  +  ∫ a_y w_y v dy
               + ∫ (a+r-q) w_y v dy  +  ∫ q w v dy
               + ∫ h u_y v_y dy  +  ∫ h_y u_y v dy  +  ∫ h u_y v dy ] = 0

    The source terms enter F with a positive sign (IBP of -h(u_yy-u_y) produces
    +h u_y v_y + h_y u_y v + h u_y v), so ufl.rhs(F) moves them to the RHS
    with the correct minus sign.
    """
    w = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    dt_c = fem.Constant(V.mesh, ScalarType(dt))
    r_c  = fem.Constant(V.mesh, ScalarType(r_rate))
    q_c  = fem.Constant(V.mesh, ScalarType(q_rate))

    F = (
        (w - w_old) * v * ufl.dx
        + dt_c * (
            # homogeneous (PDE) part
            a_func * ufl.dot(ufl.grad(w), ufl.grad(v)) * ufl.dx   # ∫ a w_y v_y
            + a_func.dx(0) * w.dx(0) * v * ufl.dx                  # ∫ a_y w_y v
            + (a_func + r_c - q_c) * w.dx(0) * v * ufl.dx          # ∫ (a+r-q) w_y v
            + q_c * w * v * ufl.dx                                  # ∫ q w v
            # source h(u_yy - u_y): after IBP becomes +∫(h u_y v_y + h_y u_y v + h u_y v)
            + h_func * ufl.dot(ufl.grad(u_t), ufl.grad(v)) * ufl.dx  # ∫ h u_y v_y
            + h_func.dx(0) * u_t.dx(0) * v * ufl.dx                   # ∫ h_y u_y v
            + h_func * u_t.dx(0) * v * ufl.dx                         # ∫ h u_y v
        )
    )

    return LinearProblem(
        ufl.lhs(F), ufl.rhs(F), bcs=bcs,
        petsc_options_prefix="da_f",
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
    ).solve()
