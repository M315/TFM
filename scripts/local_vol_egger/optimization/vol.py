r"""
Vol - piecewise-constant-in-time parameter container.

Represents the diffusion coefficient a(y,\tau) = \frac{1}{2}\sigma^2(y,\tau) as a list of
N dolfin Functions, each constant over the time slice
[t_0 + i\cdot\Delta\tau, t_0 + (i+1)\cdot\Delta\tau].

Used by all calibration methods and the PDE solvers.
"""

import numpy as np
from dolfin import Function


class Vol:

    def __init__(self, V, init_func: Function, t_0: float, t_1: float, N: int):
        r"""
        Parameters
        ----------
        V         : FunctionSpace
        init_func : Function used as the starting value for every slice
                    (deep-copied N times; subsequent optimisation updates the copies)
        t_0, t_1  : float - time interval [t_0, t_1] covered by this Vol object
        N         : int - number of time slices (N=1 \to time-independent a)
        """
        self.t_0 = t_0
        self.t_1 = t_1
        self.a   = [init_func.copy(deepcopy=True) for _ in range(N)]

    # ------------------------------------------------------------------
    # Time-slice look-up
    # ------------------------------------------------------------------

    def get_idx(self, t: float) -> int:
        """Index of the slice that contains time t."""
        if t < self.t_0 or t > self.t_1:
            raise ValueError(f"t={t} outside Vol range [{self.t_0}, {self.t_1}]")
        if t == self.t_1:
            return len(self.a) - 1
        dt = (self.t_1 - self.t_0) / len(self.a)
        return int((t - self.t_0) / dt)

    def get(self, t: float) -> Function:
        """Function for the slice containing time t."""
        return self.a[self.get_idx(t)]

    # ------------------------------------------------------------------
    # Flat DOF vector interface (used by scipy optimisers)
    # ------------------------------------------------------------------

    def update(self, V, a_vec: np.ndarray) -> None:
        """Write a flat DOF vector back into the per-slice functions."""
        dof_size = V.dim()
        for i, func in enumerate(self.a):
            func.vector().set_local(a_vec[i * dof_size:(i + 1) * dof_size])
            func.vector().apply("insert")

    def to_vec(self) -> np.ndarray:
        """Return a flat DOF vector concatenating all time slices."""
        return np.concatenate([func.vector().get_local() for func in self.a])
