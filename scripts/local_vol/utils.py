from mpi4py import MPI

import numpy as np
from scipy.stats import norm

import matplotlib.pyplot as plt

import ufl
from dolfinx import fem, mesh, plot


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


def plot_pde_sol(V, uh):
    try:
        import pyvista
        
        # Create the mesh grid from the FunctionSpace V
        cells, types, x = plot.vtk_mesh(V)
        grid = pyvista.UnstructuredGrid(cells, types, x)
        
        # Assign the solution values from uh to the grid
        grid.point_data["u"] = uh.x.array.real
        grid.set_active_scalars("u")
        
        # Plotting
        plotter = pyvista.Plotter()
        plotter.add_mesh(grid, show_edges=True)
        
        # Optional: Warp by scalar for 3D visualization of magnitude
        warped = grid.warp_by_scalar()
        plotter.add_mesh(warped)
        
        plotter.show()
    except ModuleNotFoundError:
        print("'pyvista' is required to visualise the solution.")
        print("To install pyvista with pip: 'python3 -m pip install pyvista'.")


def plot_function(f):
    # plot function f with matplotlib
    import matplotlib.pyplot as plt

    x = f.function_space.mesh.geometry.x[:, 0]
    y = f.x.array.real
    plt.plot(x, y)
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.show()


def plot_calibration_result_3d(vol_obj, V, t_start, t_end, title="Calibrated Parameter"):
    y_coords = V.mesh.geometry.x[:, 0]
    sort_idx = np.argsort(y_coords)
    y_sorted = y_coords[sort_idx]
    
    # Create grid
    t_grid = np.linspace(t_start, t_end, 20)
    T, Y = np.meshgrid(t_grid, y_sorted, indexing='ij')
    
    # Extrude 'a' along time (assuming constant within this interval)
    Z_rows = []
    for t in t_grid:
        active_func = vol_obj.get(t) # Use your new get() method
        vals = active_func.x.array.real[sort_idx]
        Z_rows.append(vals)
    Z = 2.0 * np.sqrt(np.array(Z_rows))
    # Z = np.tile(a_vals, (len(t_grid), 1))
    # Z = np.tile(a_vals, (len(t_grid), 1))
    
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
    J_form = fem.form(ufl.inner(u, u) * ufl.dx)
    local_val = fem.assemble_scalar(J_form)
    global_val = u.function_space.mesh.comm.allreduce(local_val, op=MPI.SUM)
    return np.sqrt(global_val)


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


def plot_multi_maturity_surface(intervals, V, title="Calibrated Local Volatility",
                                output_file="local_vol_surface.png"):
    """
    Plot σ_loc(y, τ) = sqrt(2 a(y, τ)) assembled from multiple calibration intervals.

    intervals: list of (t_start, t_end, Vol) ordered by t_start.
    """
    y_coords = V.mesh.geometry.x[:, 0]
    sort_idx = np.argsort(y_coords)
    y_sorted = y_coords[sort_idx]

    T_all, Z_all = [], []
    for t_start, t_end, vol_obj in intervals:
        n_t = max(6, int((t_end - t_start) * 80))
        for t in np.linspace(t_start, t_end, n_t):
            a_vals = vol_obj.get(t).x.array.real[sort_idx]
            T_all.append(t)
            Z_all.append(2.0 * np.sqrt(np.maximum(a_vals, 0.0)))

    n_t, n_y = len(T_all), len(y_sorted)
    T_grid = np.tile(np.array(T_all)[:, None], (1, n_y))
    Y_grid = np.tile(y_sorted,                 (n_t, 1))
    Z_grid = np.array(Z_all)

    fig = plt.figure(figsize=(12, 7))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(Y_grid, T_grid, Z_grid, cmap="plasma", edgecolor="none", alpha=0.9)
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label=r"$\sigma_{\mathrm{loc}}$")

    ax.set_title(title)
    ax.set_xlabel("Log-moneyness  $y$")
    ax.set_ylabel("Maturity  $\\tau$")
    ax.set_zlabel("Local vol  $\\sigma_{\\mathrm{loc}}(y,\\tau)$")

    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    print(f"Plot saved to {output_file}")
    plt.show()
