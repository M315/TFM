import ufl
import numpy as np
from petsc4py.PETSc import ScalarType
from dolfinx import fem, mesh
from dolfinx.fem.petsc import LinearProblem

from utils import create_bcs


def compute_dAf(a, h, trajectory, dt, M_time, V, r, y_0, y_1):
    """
    Computes the Gateaux derivative A'_f(a) * h.
    Solves the linearized PDE forward.
    """
    w = fem.Function(V)
    w.x.array[:] = 0.0  # Initial condition for perturbation is 0

    zero = fem.Constant(V.mesh, 0.0)
    bcs = create_bcs(V, y_0, y_1, zero, zero)
    
    for t in range(M_time):
        u_prev = trajectory[t]
        w = gateaux_derivative(u_prev, a, h, dt, V, r, bcs)
        
    return w


def gateaux_derivative(u_1, a_func, h_func, dt, V, r_rate, bcs):
    """ Computes the Gâteaux derivative of the propagation operator A_f over a single step """
    w = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    dt_c = fem.Constant(V.mesh, ScalarType(dt))
    r_c = fem.Constant(V.mesh, ScalarType(r_rate))
   
    lhs = w * v * ufl.dx + \
        dt_c * (
            a_func * ufl.dot(ufl.grad(w), ufl.grad(v)) * ufl.dx + \
            (a_func + r_c) * w.dx(0) * v * ufl.dx
        )
    
    rhs = - dt_c * (
        h_func * ufl.dot(ufl.grad(u_1), ufl.grad(v)) * ufl.dx + \
        h_func * u_1.dx(0) * v * ufl.dx
    )

    problem = LinearProblem(
        lhs,
        rhs,
        bcs=bcs,
        petsc_options_prefix="da_f",
        petsc_options={ "ksp_type": "preonly", "pc_type": "lu" },
    )
    
    w_sol = problem.solve()
    
    return w_sol
