from petsc4py.PETSc import ScalarType
import ufl
from dolfinx import fem
from dolfinx.fem.petsc import LinearProblem

from utils import create_bcs


def compute_adj_dAf(a_vol, residual, trajectory, dt, M_time, V, r, q, y_0, y_1, t_0):
    """
    Computes the L2 gradient of ½||A_f(a) - g||² w.r.t. a via the adjoint method.

    The adjoint state m is initialized with (A_f(a) - g) at the terminal time and
    propagated backward. At each step the gradient contribution is accumulated into
    the corresponding time-slice bucket of a_vol.

    Returns (gradients, m_at_t0).
    """
    m = fem.Function(V)
    m.x.array[:] = residual.x.array[:]   # m(T) = A_f(a) - g

    gradients = [fem.Function(V) for _ in a_vol.a]
    for g in gradients:
        g.x.array[:] = 0.0

    zero = fem.Constant(V.mesh, 0.0)
    bcs  = create_bcs(V, y_0, y_1, zero, zero)

    for t in range(M_time - 1, -1, -1):
        current_time = t_0 + (t + 1) * dt
        a = a_vol.get(current_time)

        # u and m evaluated at the same discrete time τ_t (continuous formula:
        # gradient ∝ m(τ)·(u(τ)_yy - u(τ)_y), both at the same node)
        u_t = trajectory[t]

        m_prev, g_inc = _adjoint_step(m, u_t, a, dt, V, r, q, bcs)

        gradients[a_vol.get_idx(current_time)].x.array[:] += g_inc.x.array[:]
        m.x.array[:] = m_prev.x.array[:]

    return gradients, m


def _adjoint_step(m_curr, u_t, a_func, dt, V, r_rate, q_rate, bcs):
    """
    One backward-Euler step of the time-reversed adjoint equation.

    The adjoint PDE (written in the forward-reversed-time variable τ' = T - τ):

        M_{τ'} = (a M)_yy + (a M)_y + (r-q) M_y - q M

    Backward-Euler weak form (m = M^{n+1} unknown, m_1 = M^n = m_curr known):

        ∫(m - m_1) v dy
        + dt [ ∫ a m_y v_y dy
             + ∫ a_y m v_y dy        (two terms together: ∫(am)_y v_y dy)
             + ∫ (a + r - q) m v_y dy
             + ∫ q m v dy ] = 0

    Gradient contribution (L2 projection):

        g_inc(y)  such that  ∫ g_inc h dy = -dt ∫ (u_y m_y + u_y m) h dy  ∀h ∈ V

    This equals dt ∫ m (u_yy - u_y) h dy after IBP with h = 0 on ∂Ω.
    """
    m = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    dt_c = fem.Constant(V.mesh, ScalarType(dt))
    r_c  = fem.Constant(V.mesh, ScalarType(r_rate))
    q_c  = fem.Constant(V.mesh, ScalarType(q_rate))

    adj_form = (
        (m - m_curr) * v * ufl.dx
        + dt_c * (
            a_func * ufl.dot(ufl.grad(m), ufl.grad(v)) * ufl.dx   # ∫ a m_y v_y
            + m * ufl.dot(ufl.grad(a_func), ufl.grad(v)) * ufl.dx  # ∫ a_y m v_y
            + (a_func + r_c - q_c) * m * v.dx(0) * ufl.dx          # ∫ (a+r-q) m v_y
            + q_c * m * v * ufl.dx                                  # ∫ q m v
        )
    )

    m_new = LinearProblem(
        ufl.lhs(adj_form), ufl.rhs(adj_form), bcs=bcs,
        petsc_options_prefix="adj_da_f",
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
    ).solve()

    # --- gradient contribution ---
    # Pointwise expression: -dt * (u_y m_y + u_y m)
    # Projected onto V via mass-matrix solve so the result lives in the FEM space.
    g_expr = -dt_c * (
        ufl.dot(ufl.grad(u_t), ufl.grad(m_new))   # u_y m_y
        + u_t.dx(0) * m_new                        # u_y m
    )

    g_trial = ufl.TrialFunction(V)
    v_test  = ufl.TestFunction(V)
    g_func = LinearProblem(
        ufl.inner(g_trial, v_test) * ufl.dx,
        ufl.inner(g_expr,  v_test) * ufl.dx,
        petsc_options_prefix="grad_proj",
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
    ).solve()

    return m_new, g_func
