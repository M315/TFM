from petsc4py.PETSc import ScalarType
from petsc4py import PETSc
import ufl
from dolfinx import fem
from dolfinx.fem.petsc import LinearProblem, assemble_vector

from utils import create_bcs


def compute_adj_dAf(a_vol, residual, trajectory, dt, M_time, V, r, q, y_0, y_1, t_0):
    """
    Computes the Euclidean gradient ∂J/∂a_j of ½||A_f(a) - g||² w.r.t. the DOF
    vector of a, via the adjoint method.

    The adjoint state m is initialized with (A_f(a) - g) at the terminal time and
    propagated backward. At each step the gradient contribution ∂J/∂a_j is assembled
    as a linear 1-form and accumulated into the corresponding time-slice bucket of a_vol.

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
    One backward-Euler step of the adjoint equation, plus the gradient contribution.

    The adjoint PDE (in reverse time τ' = T - τ):

        M_{τ'} = (a M)_yy + (a M)_y + (r-q) M_y - q M

    Backward-Euler weak form (m unknown, m_curr = previous adjoint state):

        ∫(m - m_curr) v dy
        + dt [ ∫ a m_y v_y dy
             + ∫ a_y m v_y dy
             + ∫ (a + r - q) m v_y dy
             + ∫ q m v dy ] = 0

    Gradient contribution — Euclidean gradient ∂J/∂a_j, assembled as a 1-form:

        g_j = -dt ∫ [φ_j u_y m_y   (from B1 = ∫ a u_y v_y)
                   + φ_{j,y} u_y m   (from B2 = ∫ a_y u_y v)
                   + φ_j u_y m]      (from B3 = ∫ (a+r-q) u_y v)

    The B2 term involves φ_{j,y} (test-function derivative) and cannot be
    captured by projecting a pointwise expression via the mass matrix.
    Direct 1-form assembly gives the correct Euclidean gradient for L-BFGS-B.
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
    # Assemble ∂J/∂a_j directly as a linear 1-form (no mass-matrix solve).
    # Uses the test function φ_j and its derivative φ_{j,y} to capture all
    # three terms from differentiating the forward bilinear form w.r.t. a_j.
    h_test = ufl.TestFunction(V)
    grad_form = fem.form(
        -dt_c * (
            ufl.dot(ufl.grad(u_t), ufl.grad(m_new)) * h_test   # B1: ∫ u_y m_y φ_j
            + u_t.dx(0) * m_new * h_test.dx(0)                  # B2: ∫ u_y m φ_{j,y}
            + u_t.dx(0) * m_new * h_test                        # B3: ∫ u_y m φ_j
        ) * ufl.dx
    )
    b = assemble_vector(grad_form)
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

    g_func = fem.Function(V)
    g_func.x.array[:] = b.array_r

    return m_new, g_func
