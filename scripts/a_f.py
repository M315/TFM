from mpi4py import MPI
from petsc4py.PETSc import ScalarType
import numpy as np
import ufl
from dolfinx import fem, mesh, plot
from dolfinx.fem.petsc import LinearProblem

from utils import create_bcs

def compute_Af(a, u_initial, dt, M_time, V, r, y_0, y_1, t_0, psi_1_func, psi_2_func):
    u = fem.Function(V)
    u.x.array[:] = u_initial.x.array[:]
    trajectory = [u.copy()]
    
    for t in range(M_time):
        current_time = t_0 + (t + 1) * dt
        bcs = create_bcs(V, y_0, y_1, psi_1_func(current_time), psi_2_func(current_time))
        u = propagation(u, a, dt, V, r, bcs)
        trajectory.append(u.copy())
        
    return u, trajectory


def propagation(u_0, a_func, dt, V, r_rate, bcs):
    """ Computes one step of the forward propagation operator A_f over a time interval. """
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    dt_c = fem.Constant(V.mesh, ScalarType(dt))
    r_c = fem.Constant(V.mesh, ScalarType(r_rate))

    # Variational Problem
    # 
    # PDE: (u - u_old)/dt = a*u_yy - (a + r)*u_y
    # Weak Form: (u - u_old) * v dx 
    #            + dt * ( a * u_y * v_y + (a + r) * u_y * v ) dx 
    #            = 0

    F = (u - u_0) * v * ufl.dx + \
        dt_c * (
            a_func * ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx + \
            (a_func + r_c) * u.dx(0) * v * ufl.dx
        )

    problem = LinearProblem(
        ufl.lhs(F),
        ufl.rhs(F),
        bcs=bcs,
        petsc_options_prefix="A_f",
        petsc_options={ "ksp_type": "preonly", "pc_type": "lu" },
    )
    
    return problem.solve()
