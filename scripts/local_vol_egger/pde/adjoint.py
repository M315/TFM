"""
Adjoint PDE solver — operator adj_dA_f.

Solves the backward adjoint equation

    V_τ + (aV)_yy + (aV)_y + (r−q) V_y − qV = 0,   V(y, T) = residual(y)

backward in τ and accumulates the Euclidean gradient

    ∂J/∂a_j = −dt ∫ [ u_y m_y φ_j  +  u_y m ∂_y φ_j  +  u_y m φ_j ] dy

where m = V and φ_j are the FEM basis functions.

The gradient is assembled directly as a linear 1-form (no mass-matrix solve),
which gives the correct Euclidean (DOF-level) gradient for L-BFGS-B.
"""

from petsc4py.PETSc import ScalarType
from petsc4py import PETSc
import ufl
from dolfinx import fem
from dolfinx.fem.petsc import LinearProblem, assemble_vector

from utils import create_bcs


def compute_adj_dAf(a_vol, residual, trajectory, dt, M_time, V, r, q, y_0, y_1, t_0):
    """
    Compute the gradient of ½‖A_f(a) − g‖²_{L²} w.r.t. the DOF vector of a.

    Parameters
    ----------
    a_vol     : Vol
    residual  : FEM Function — A_f(a) − g  (adjoint terminal condition)
    trajectory: list of FEM Functions — forward trajectory from compute_Af
    dt        : float
    M_time    : int

    Returns
    -------
    gradients : list of FEM Functions — one per Vol time slice, each of length n_dof
    m_at_t0   : FEM Function — adjoint state at t = t_0 (for diagnostics)
    """
    m = fem.Function(V)
    m.x.array[:] = residual.x.array[:]   # m(T) = A_f(a) − g

    gradients = [fem.Function(V) for _ in a_vol.a]
    for g in gradients:
        g.x.array[:] = 0.0

    zero = fem.Constant(V.mesh, 0.0)
    bcs  = create_bcs(V, y_0, y_1, zero, zero)

    for t in range(M_time - 1, -1, -1):
        current_time = t_0 + (t + 1) * dt
        a   = a_vol.get(current_time)
        u_t = trajectory[t]

        m_prev, g_inc = _step(m, u_t, a, dt, V, r, q, bcs)

        gradients[a_vol.get_idx(current_time)].x.array[:] += g_inc.x.array[:]
        m.x.array[:] = m_prev.x.array[:]

    return gradients, m


def _step(m_curr, u_t, a_func, dt, V, r_rate, q_rate, bcs):
    """
    One backward-Euler step of the adjoint equation, plus the gradient contribution.

    Adjoint PDE (in reverse time τ' = T − τ):

        m_{τ'} = −(am)_yy − (am)_y − (r−q) m_y + qm

    Backward-Euler weak form (m unknown, m_curr = previous adjoint state):

        ∫(m − m_curr) v dy
        + dt [ ∫ (am)_y v_y dy
             + ∫ a m v_y dy
             + ∫ (r−q) m v_y dy
             + ∫ q m v dy ] = 0

    Gradient contribution assembled as a linear 1-form:

        g_j = −dt ∫ [ u_y m_y φ_j   (from B1: ∫ a u_y v_y)
                    + u_y m ∂_y φ_j  (from B2: ∫ ∂_y a · u_y v)
                    + u_y m φ_j ]    (from B3: ∫ (a+r−q) u_y v)   dy
    """
    m = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    dt_c = fem.Constant(V.mesh, ScalarType(dt))
    r_c  = fem.Constant(V.mesh, ScalarType(r_rate))
    q_c  = fem.Constant(V.mesh, ScalarType(q_rate))

    adj_form = (
        (m - m_curr) * v * ufl.dx
        + dt_c * (
            a_func * ufl.dot(ufl.grad(m), ufl.grad(v)) * ufl.dx     # ∫ a m_y v_y  ─┐ together:
            + m * ufl.dot(ufl.grad(a_func), ufl.grad(v)) * ufl.dx   # ∫ ∂_y a m v_y ─┘ ∫(am)_y v_y
            + (a_func + r_c - q_c) * m * v.dx(0) * ufl.dx           # ∫ (a+r−q) m v_y
            + q_c * m * v * ufl.dx                                   # ∫ q m v
        )
    )

    m_new = LinearProblem(
        ufl.lhs(adj_form), ufl.rhs(adj_form), bcs=bcs,
        petsc_options_prefix="adj",
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
    ).solve()

    # Gradient ∂J/∂a_j assembled directly as a 1-form — no mass-matrix solve needed.
    # Captures all three terms from differentiating the forward bilinear form w.r.t. a_j.
    h_test = ufl.TestFunction(V)
    grad_form = fem.form(
        -dt_c * (
            ufl.dot(ufl.grad(u_t), ufl.grad(m_new)) * h_test    # B1: ∫ u_y m_y φ_j
            + u_t.dx(0) * m_new * h_test.dx(0)                  # B2: ∫ u_y m ∂_y φ_j
            + u_t.dx(0) * m_new * h_test                        # B3: ∫ u_y m φ_j
        ) * ufl.dx
    )
    b = assemble_vector(grad_form)
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

    g_func = fem.Function(V)
    g_func.x.array[:] = b.array_r

    return m_new, g_func
