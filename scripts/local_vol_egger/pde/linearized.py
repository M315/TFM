"""
Gâteaux (linearized) PDE solver — operator dA_f.

Computes the directional derivative A'_f(a)[h] by solving the
linearized (sensitivity) forward equation

    w_τ = a(w_yy − w_y) − (r−q) w_y − qw  −  h(u_yy − u_y)

forward from t_0, with zero initial and boundary conditions.

Used for:
  - gradient verification (compare with finite differences)
  - analytical Jacobian construction (per-slice, via active_slice)
"""

from petsc4py.PETSc import ScalarType
import ufl
from dolfinx import fem
from dolfinx.fem.petsc import LinearProblem

from utils import create_bcs


def compute_dAf(a_vol, h_func, trajectory, dt, M_time, V, r, q, y_0, y_1, t_0,
                active_slice=None):
    """
    Compute A'_f(a)[h] — the Gâteaux derivative in direction h.

    Parameters
    ----------
    a_vol        : Vol
    h_func       : FEM Function — perturbation direction in parameter space
    trajectory   : list of FEM Functions — forward trajectory from compute_Af
    active_slice : int or None — if set, h is applied only in the time steps
                   belonging to that Vol slice (isolates per-slice sensitivity);
                   None applies h uniformly across all steps

    Returns
    -------
    w : FEM Function — A'_f(a)[h] at time t_0 + M_time*dt
    """
    w = fem.Function(V)
    w.x.array[:] = 0.0     # w(t_0) = 0  (IC independent of a)

    zero = fem.Constant(V.mesh, 0.0)
    bcs  = create_bcs(V, y_0, y_1, zero, zero)

    zero_fn = fem.Function(V)
    zero_fn.x.array[:] = 0.0

    for t in range(M_time):
        current_time = t_0 + (t + 1) * dt
        a   = a_vol.get(current_time)
        u_t = trajectory[t]

        if active_slice is not None and a_vol.get_idx(current_time) != active_slice:
            h_active = zero_fn
        else:
            h_active = h_func

        w = _step(w, u_t, a, h_active, dt, V, r, q, bcs)

    return w


def _step(w_old, u_t, a_func, h_func, dt, V, r_rate, q_rate, bcs):
    """
    One backward-Euler step of the linearized equation.

    Linearising u_τ = a(u_yy − u_y) − (r−q)u_y − qu in direction h gives:

        w_τ = a(w_yy − w_y) − (r−q)w_y − qw  +  h(u_yy − u_y)

    Weak form (IBP on both the homogeneous and source terms):

        ∫(w − w_old) v dy
        + dt [ ∫ a w_y v_y dy  +  ∫ ∂_y a · w_y v dy
             + ∫ (a+r−q) w_y v dy  +  ∫ q w v dy
             + ∫ h u_y v_y dy  +  ∫ ∂_y h · u_y v dy  +  ∫ h u_y v dy ] = 0
    """
    w = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    dt_c = fem.Constant(V.mesh, ScalarType(dt))
    r_c  = fem.Constant(V.mesh, ScalarType(r_rate))
    q_c  = fem.Constant(V.mesh, ScalarType(q_rate))

    F = (
        (w - w_old) * v * ufl.dx
        + dt_c * (
            # homogeneous part (same operator as forward PDE)
            a_func * ufl.dot(ufl.grad(w), ufl.grad(v)) * ufl.dx         # ∫ a w_y v_y
            + a_func.dx(0) * w.dx(0) * v * ufl.dx                        # ∫ ∂_y a · w_y v
            + (a_func + r_c - q_c) * w.dx(0) * v * ufl.dx                # ∫ (a+r−q) w_y v
            + q_c * w * v * ufl.dx                                        # ∫ q w v
            # source h(u_yy − u_y): after IBP → ∫(h u_y v_y + ∂_y h · u_y v + h u_y v)
            + h_func * ufl.dot(ufl.grad(u_t), ufl.grad(v)) * ufl.dx      # ∫ h u_y v_y
            + h_func.dx(0) * u_t.dx(0) * v * ufl.dx                      # ∫ ∂_y h · u_y v
            + h_func * u_t.dx(0) * v * ufl.dx                            # ∫ h u_y v
        )
    )

    return LinearProblem(
        ufl.lhs(F), ufl.rhs(F), bcs=bcs,
        petsc_options_prefix="lin",
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
    ).solve()
