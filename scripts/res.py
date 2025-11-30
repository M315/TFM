from mpi4py import MPI

import numpy as np
import scipy.optimize

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Required for 3D plotting

from dolfinx import fem, mesh, plot

from a_f import propagation, compute_Af
from da_f import gateaux_derivative, compute_dAf
from adj_da_f import adjoint_gateaux, compute_adj_dAf
import utils
# from petsc4py.PETSc import ScalarType


def cost_function_and_grad(a_vec, a_func, u_0, g_target, V, r, y_0, y_1, t_0, t_1, M_time, psi_1, psi_2):
    a_func.x.array[:] = a_vec
    
    # Step 1: 0 -> t_0
    dt_1 = (t_0 - 0.0) / M_time
    f_mid, traj_1 = compute_Af(a_func, u_0, dt_1, M_time, V, r, y_0, y_1, 0.0, psi_1, psi_2)
    
    # Step 2: t_0 -> t_1
    dt_2 = (t_1 - t_0) / M_time
    g_pred, traj_2 = compute_Af(a_func, f_mid, dt_2, M_time, V, r, y_0, y_1, t_0, psi_1, psi_2)
    
    residual = fem.Function(V)
    residual.x.array[:] = g_pred.x.array[:] - g_target.x.array[:]
    cost = 0.5 * np.sum(residual.x.array**2)
    
    grad_2, m_at_t0 = compute_adj_dAf(a_func, residual, traj_2, dt_2, M_time, V, r, y_0, y_1)
    grad_1, m_at_0 = compute_adj_dAf(a_func, m_at_t0, traj_1, dt_1, M_time, V, r, y_0, y_1)
    
    return cost, grad_1.x.array + grad_2.x.array


def plot_calibration_result_3d(a_func, V, t_start, t_end, title="Calibrated Parameter"):
    y_coords = V.mesh.geometry.x[:, 0]
    sort_idx = np.argsort(y_coords)
    y_sorted = y_coords[sort_idx]
    
    # Create grid
    t_grid = np.linspace(t_start, t_end, 20)
    T, Y = np.meshgrid(t_grid, y_sorted, indexing='ij')
    
    # Extrude 'a' along time (assuming constant within this interval)
    a_vals = a_func.x.array.real[sort_idx]
    Z = np.tile(a_vals, (len(t_grid), 1))
    
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(Y, T, Z, cmap='plasma', edgecolor='none', alpha=0.9)
    
    ax.set_title(title)
    ax.set_xlabel('Log-Moneyness (y)')
    ax.set_ylabel('Time (t)')
    ax.set_zlabel('Parameter a(y)')
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
    
    output_file = "calibration_03_06.png"
    plt.savefig(output_file)
    print(f"Plot saved to {output_file}")
    plt.show()


def add_noise(u_func, noise_level=0.01):
    """
    Adds Gaussian noise to a dolfinx Function.
    noise_level: Standard deviation as a percentage of the max value (e.g., 0.01 = 1%)
    """
    # Create a copy to avoid modifying the original
    u_noisy = fem.Function(u_func.function_space)
    vals = u_func.x.array.real.copy()
    
    # Scale noise by the magnitude of the data
    scale = np.max(np.abs(vals))
    noise = np.random.normal(0, noise_level * scale, size=vals.shape)
    
    u_noisy.x.array[:] = vals + noise
    return u_noisy


def L2_norm(u):
    # Compute L2 norm of a dolfinx function
    # Simplified: using vector norm for coefficients if mesh is uniform, 
    # otherwise requires actual integration: sqrt(assemble(u*u*dx))
    # Here we assume simple array norm for the optimization step control
    return np.linalg.norm(u.x.array.real)


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

    # Boundary conditions
    psi_1 = lambda t: 0.67
    psi_2 = lambda t: 0.0

    # Initialize function - use initial conditions for the log-moneyness Dupire PDE
    u_0 = fem.Function(V)
    u_0.interpolate(lambda x: np.maximum(1.0 - np.exp(x[0]), 0.0))

    # Forward propagate to get f
    dt = (t_0 - 0.0) / M_time
    f, _ = compute_Af(a_true, u_0, dt, M_time, V, r, y_0, y_1, 0.0, psi_1, psi_2)
    f = add_noise(f, 0.005)
    utils.plot_function(f)

    # Forward propagate to get g
    dt = (t_1 - t_0) / M_time
    g, _ = compute_Af(a_true, f.copy(), dt, M_time, V, r, y_0, y_1, t_0, psi_1, psi_2)
    g = add_noise(g, 0.005)
    utils.plot_function(g)

    # First guess for a
    a_k = fem.Function(V)
    # a_k.interpolate(lambda x: 0.45 + 0.0*x[0]) # Flat initial guess
    a_k.interpolate(lambda x: 0.5 * sigma_true(x)**2)

    print(f"Initial Cost: {0.5 * np.sum((compute_Af(a_k, f, dt, M_time, V, r, y_0, y_1, t_0, psi_1, psi_2)[0].x.array - g.x.array)**2):.6f}")

    # Optimization
    print("Starting Optimization...")
    bounds = [(1e-6, None) for _ in range(len(a_k.x.array))]
    res = scipy.optimize.minimize(
        fun=cost_function_and_grad,
        x0=a_k.x.array,
        args=(a_k, u_0, g, V, r, y_0, y_1, t_0, t_1, M_time, psi_1, psi_2),
        method='L-BFGS-B',
        jac=True,
        bounds=bounds,
        options={'disp': True, 'maxiter': 20}
    )
    a_k.x.array[:] = res.x
    print(f"Optimization Complete. Final Cost: {res.fun:.6f}")

    # --- [3D Visualization Construction] ---
    print("\n3. Generating 3D Plot of Calibrated Parameter...")
    plot_calibration_result_3d(a_k, V, t_0, t_1, title=f"Calibrated Volatility a(y) for t in [{t_0}, {t_1}]")