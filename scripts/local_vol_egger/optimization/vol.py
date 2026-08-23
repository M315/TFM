r"""
Vol - piecewise-constant-in-time diffusion coefficient a(y,tau) = 1/2 sigma^2(y,tau).

Stored as a list of N spatial Functions, one per time slice. Slice boundaries are
either uniform (pass N) or given explicitly (pass edges), the latter used to align
slices with the observed maturities in the surface calibration.
"""

import numpy as np
from dolfin import Function


class Vol:

    def __init__(self, V, init_func: Function, t_0: float, t_1: float,
                 N: int = 1, edges=None):
        r"""
        Parameters
        ----------
        V         : FunctionSpace
        init_func : starting value copied into every slice (optimisation updates the copies)
        t_0, t_1  : time interval [t_0, t_1] covered by this object
        N         : number of uniform slices (used when edges is None; N=1 -> a is constant in time)
        edges     : optional sorted boundary times [t_0, e_1, ..., e_{N-1}, t_1]; slice j is
                    the half-open interval (edges[j], edges[j+1]]. Overrides N when given.
        """
        self.t_0, self.t_1 = t_0, t_1
        self.edges = np.linspace(t_0, t_1, N + 1) if edges is None \
            else np.asarray(edges, dtype=float)
        self.a = [init_func.copy(deepcopy=True) for _ in range(len(self.edges) - 1)]

    # ------------------------------------------------------------------
    # Time-slice look-up
    # ------------------------------------------------------------------

    def get_idx(self, t: float) -> int:
        """Index of the slice (edges[j], edges[j+1]] containing time t, clamped to the range."""
        j = int(np.searchsorted(self.edges, t, side="left")) - 1
        return min(max(j, 0), len(self.a) - 1)

    def get(self, t: float) -> Function:
        """Function for the slice containing time t."""
        return self.a[self.get_idx(t)]

    # ------------------------------------------------------------------
    # Flat DOF vector interface (used by scipy optimisers)
    # ------------------------------------------------------------------

    def update(self, V, a_vec: np.ndarray) -> None:
        """Write a flat DOF vector (slices concatenated) back into the per-slice functions."""
        dof_size = V.dim()
        for i, func in enumerate(self.a):
            func.vector().set_local(a_vec[i * dof_size:(i + 1) * dof_size])
            func.vector().apply("insert")

    def to_vec(self) -> np.ndarray:
        """Flat DOF vector concatenating all time slices."""
        return np.concatenate([func.vector().get_local() for func in self.a])
