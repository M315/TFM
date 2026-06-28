"""
Shared helpers for the Egger & Engl (2005) calibration (legacy FEniCS / dolfin).
"""

import numpy as np
from scipy.stats import norm

from dolfin import (
    Function, Constant, DirichletBC, SubDomain, near,
    TestFunction, assemble, inner, dot, grad, dx,
)


def get_array(f):
    """Local DOF values of a Function as a numpy array."""
    return f.vector().get_local()


def set_array(f, arr):
    """Write a numpy array into a Function's DOF vector."""
    f.vector().set_local(np.asarray(arr, dtype=float))
    f.vector().apply("insert")


def dof_coordinates(V):
    """1-D coordinates of the DOFs of V, in the same order as get_array()."""
    gdim = V.mesh().geometry().dim()
    return V.tabulate_dof_coordinates().reshape(-1, gdim)[:, 0]


def interpolate_func(V, func):
    """
    Nodal interpolation of a Python function y -> values onto a P1 space.

    For P1 elements the DOFs are the mesh nodes, so evaluating `func` at the
    DOF coordinates and storing the result is exact nodal interpolation.
    Replaces dolfinx's `Function.interpolate(callable)`.
    """
    f = Function(V)
    set_array(f, func(dof_coordinates(V)))
    return f


class _Endpoint(SubDomain):
    """Boundary marker for a single interval endpoint x[0] == x0."""

    def __init__(self, x0, tol=1e-9):
        super().__init__()
        self.x0 = x0
        self.tol = tol

    def inside(self, x, on_boundary):
        return bool(on_boundary and near(x[0], self.x0, self.tol))


def create_bcs(V, y_min, y_max, psi_1, psi_2):
    """Dirichlet BCs u(y_min)=psi_1, u(y_max)=psi_2 (psi_i scalar values)."""
    return [
        DirichletBC(V, Constant(float(psi_1)), _Endpoint(y_min)),
        DirichletBC(V, Constant(float(psi_2)), _Endpoint(y_max)),
    ]


def L2_norm(u):
    """||u||_{L2} over the whole (1-D) domain."""
    return np.sqrt(assemble(inner(u, u) * dx))


def H1_norm_sq(f, V):
    """||f||^2_{H1} = int (f^2 + f_y^2) dy."""
    return assemble((f * f + dot(grad(f), grad(f))) * dx)


def h1_reg_gradient(da, beta, V):
    """DOF gradient of beta*||da||^2_{H1}: assembles 2 beta (M+K) da as a vector."""
    h = TestFunction(V)
    form = 2.0 * Constant(beta) * (da * h + dot(grad(da), grad(h))) * dx
    return assemble(form).get_local()


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
