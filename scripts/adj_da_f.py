import ufl
import numpy as np
from petsc4py.PETSc import ScalarType
from dolfinx import fem, mesh
from dolfinx.fem.petsc import LinearProblem

from utils import create_bcs

def compute_adj_dAf(a, residual, trajectory, dt, M_time, V, r, y_0, y_1):
    """
    Computes the gradient of the objective function w.r.t 'a' using the Adjoint method.
    Returns the gradient function and the adjoint state at the beginning of the interval.
    """
    # Adjoint state m initialized with the residual at terminal time T
    m = fem.Function(V)
    m.x.array[:] = residual.x.array[:]
    
    gradient = fem.Function(V)
    gradient.x.array[:] = 0.0
    
    # Adjoint Boundary Conditions are homogeneous (0.0)
    zero = fem.Constant(V.mesh, 0.0)
    bcs = create_bcs(V, y_0, y_1, zero, zero)
    
    # Backward time stepping
    for t in range(M_time - 1, -1, -1):
        u_prev = trajectory[t]
        
        # Compute one step backward for adjoint m and get gradient contribution
        # Note: adjoint_gateaux returns (m_prev, grad_inc)
        m_prev, g_inc = adjoint_gateaux(m, u_prev, a, dt, V, r, bcs)
        
        # Accumulate gradient
        gradient.x.array[:] += g_inc.x.array[:]
        
        # Update adjoint state for next step
        m.x.array[:] = m_prev.x.array[:]
        
    return gradient, m


def adjoint_gateaux(m_1, u_1, a_func, dt, V, r_rate, bcs):
    """ Computes the Adjoint of the Gâteaux derivative operator (A'_f)^* """
    
    m = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    dt_c = fem.Constant(V.mesh, ScalarType(dt))
    r_c = fem.Constant(V.mesh, ScalarType(r_rate))
    
    adj_op = (m - m_1) * v * ufl.dx + \
            dt_c * (
                a_func * ufl.dot(ufl.grad(m), ufl.grad(v)) * ufl.dx + \
                (a_func + r_c) * m * v.dx(0) * ufl.dx
            )
    
    
    # Solve for Adjoint Variable p
    problem_adj = LinearProblem(
        ufl.lhs(adj_op),
        ufl.rhs(adj_op),
        bcs=bcs,
        petsc_options_prefix="ajd_da_f",
        petsc_options = {"ksp_type": "preonly", "pc_type": "lu"}
    )
    m_0 = problem_adj.solve()
    
    # Compute Gradient Contribution
    g_expr = - dt_c * (
            ufl.dot(ufl.grad(u_1), ufl.grad(m_0)) + \
            u_1.dx(0) * m_0
        )
    
    g_trial = ufl.TrialFunction(V)
    v_test = ufl.TestFunction(V)
    
    a_proj = ufl.inner(g_trial, v_test) * ufl.dx
    L_proj = ufl.inner(g_expr, v_test) * ufl.dx
    
    problem_proj = LinearProblem(
        a_proj,
        L_proj,
        petsc_options_prefix="grad_adj",
        petsc_options = {"ksp_type": "preonly", "pc_type": "lu"}
    )
    g_func = problem_proj.solve()
    
    return m_0, g_func
