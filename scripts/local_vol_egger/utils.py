from mpi4py import MPI

import numpy as np
from scipy.stats import norm

import ufl
from dolfinx import fem, mesh


def create_bcs(V, y_min, y_max, psi_1, psi_2):
    facets_min = mesh.locate_entities_boundary(V.mesh, 0, lambda x: np.isclose(x[0], y_min))
    facets_max = mesh.locate_entities_boundary(V.mesh, 0, lambda x: np.isclose(x[0], y_max))
    dofs_min = fem.locate_dofs_topological(V, 0, facets_min)
    dofs_max = fem.locate_dofs_topological(V, 0, facets_max)
    bcs = [
        fem.dirichletbc(psi_1, dofs_min, V),
        fem.dirichletbc(psi_2, dofs_max, V),
    ]
    return bcs


def L2_norm(u):
    J_form = fem.form(ufl.inner(u, u) * ufl.dx)
    local_val = fem.assemble_scalar(J_form)
    global_val = u.function_space.mesh.comm.allreduce(local_val, op=MPI.SUM)
    return np.sqrt(global_val)


def H1_norm_sq(f, V):
    """||f||²_{H¹} = ∫(f² + f_y²) dy."""
    J_form = fem.form((f * f + ufl.dot(ufl.grad(f), ufl.grad(f))) * ufl.dx)
    local_val = fem.assemble_scalar(J_form)
    return f.function_space.mesh.comm.allreduce(local_val, op=MPI.SUM)


def h1_reg_gradient(da, beta, V):
    """DOF gradient of β‖da‖²_{H¹}: assembles 2β(M+K)da as a vector."""
    from petsc4py import PETSc
    from petsc4py.PETSc import ScalarType
    from dolfinx.fem.petsc import assemble_vector
    h = ufl.TestFunction(V)
    beta_c = fem.Constant(V.mesh, ScalarType(beta))
    form = fem.form(
        2.0 * beta_c * (da * h + ufl.dot(ufl.grad(da), ufl.grad(h))) * ufl.dx
    )
    b = assemble_vector(form)
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    return b.array_r.copy()


def bs_call(S0, y, r, q, sigma, tau):
    """
    Black-Scholes call price with log-moneyness y = log(K / S0).
    Vectorized: y and sigma may be numpy arrays.
    """
    K = S0 * np.exp(y)
    sqrtT = sigma * np.sqrt(tau)
    d1 = (-y + (r - q + 0.5 * sigma**2) * tau) / sqrtT
    d2 = d1 - sqrtT
    return S0 * np.exp(-q * tau) * norm.cdf(d1) - K * np.exp(-r * tau) * norm.cdf(d2)
