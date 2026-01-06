from mpi4py import MPI

import numpy as np
import scipy.optimize

from dolfinx import fem, mesh

from a_f import compute_Af
from adj_da_f import compute_adj_dAf
from utils import L2_norm, plot_calibration_result_3d, add_noise


def cost_function_and_grad(a_vec, a_vol, u_at_t0, g_target, V, r, y_0, y_1, t_0, t_1, M_time, psi_1, psi_2):
    a_vol.update(V, a_vec)
    
    dt = (t_1 - t_0) / M_time
    g_pred, traj = compute_Af(a_vol, u_at_t0, dt, M_time, V, r, y_0, y_1, t_0, psi_1, psi_2)
    
    residual = fem.Function(V)
    residual.x.array[:] = g_pred.x.array[:] - g_target.x.array[:]
    cost = L2_norm(residual)
    
    grads, _ = compute_adj_dAf(a_vol, residual, traj, dt, M_time, V, r, y_0, y_1, t_0)
    
    total_grads = np.concatenate([g.x.array for g in grads])
        
    return cost, total_grads


class Vol:
    t_0: float
    t_1: float
    a: list[fem.Function]

    def __init__(self, V, const_func, t_0, t_1, N):
        self.t_0 = t_0
        self.t_1 = t_1
        self.a = [const_func.copy() for _ in range(N)]

    def get(self, t: float):
        return self.a[self.get_idx(t)]
    
    def get_idx(self, t: float):
        if t < self.t_0 or t > self.t_1:
            raise ValueError("Time t is out of bounds for the Vol parameter.")
        if t == self.t_1:
            return len(self.a) - 1
        dt = (self.t_1 - self.t_0) / len(self.a)
        idx = int((t - self.t_0) / dt)
        return idx
    
    def update(self, V, a_vec):
        dof_size = V.dofmap.index_map.size_global
        for i, func in enumerate(self.a):
            start = i * dof_size
            end = (i + 1) * dof_size
            func.x.array[:] = a_vec[start:end]



if __name__ == "__main__":
    # Initial conditions
    y_0, y_1 = (-1.0, 1.0)
    t_0, t_1 = (0.3, 0.6)
    r = 0.0
    S_0 = 1.0

    # Discretization
    N_space = 70
    M_time = 100

    # Define Mesh and Function Space
    msh = mesh.create_interval(MPI.COMM_WORLD, N_space, points=(y_0, y_1))
    V = fem.functionspace(msh, ("Lagrange", 1))

    # True Vol
    sigma_true = lambda y: 0.4 + 0.1 * np.arctan(y[0])
    a_true = fem.Function(V)
    a_true.interpolate(lambda x: 0.5 * sigma_true(x)**2)
    a_true = Vol(V, a_true, 0.0, t_1, N= 2 * M_time)

    # Boundary conditions
    psi_1 = lambda t: 0.67
    psi_2 = lambda t: 0.0

    # Initialize function - use initial conditions for the log-moneyness Dupire PDE
    u_0 = fem.Function(V)
    u_0.interpolate(lambda x: np.maximum(1.0 - np.exp(x[0]), 0.0))

    # Forward propagate to get f
    dt = (t_0 - 0.0) / M_time
    f, _ = compute_Af(a_true, u_0, dt, M_time, V, r, y_0, y_1, 0.0, psi_1, psi_2)

    # Forward propagate to get g
    dt = (t_1 - t_0) / M_time
    g, _ = compute_Af(a_true, f.copy(), dt, M_time, V, r, y_0, y_1, t_0, psi_1, psi_2)

    # Add noise to f and g
    f = add_noise(f, 0.0005)
    g = add_noise(g, 0.0005)

    # First guess for a
    a_k = fem.Function(V)
    # a_k.interpolate(lambda x: 0.08 + 0.01*x[0]) # Flat initial guess
    s = 0.5
    a_k.interpolate(lambda x: 0.06 + 0.02 / (s * np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * np.power(x[0] / s, 2.0))) # Gaussian initial guess
    # a_k.interpolate(lambda x: 0.5 * sigma_true(x)**2)
    a_k = Vol(V, a_k, t_0, t_1, N=M_time)
    a_vec_init = np.concatenate([func.x.array for func in a_k.a])

    # Optimization
    print("Starting Optimization...")
    # bounds = [(1e-6, None) for _ in range(len(a_k.x.array))]
    bounds = [(1e-8, None) for _ in range(len(a_vec_init))]
    res = scipy.optimize.minimize(
        fun=cost_function_and_grad,
        x0=a_vec_init,
        args=(a_k, f, g, V, r, y_0, y_1, t_0, t_1, M_time, psi_1, psi_2),
        method='L-BFGS-B',
        jac=True,
        bounds=bounds,
        options={'disp': True, 'maxiter': 20}
    )
    print(f"Optimization Complete. Final Cost: {res.fun:.6f}")

    # Plot
    a_k.update(V, res.x)
    plot_calibration_result_3d(a_k, V, t_0, t_1, title=f"Calibrated Volatility")
