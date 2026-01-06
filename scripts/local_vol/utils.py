from mpi4py import MPI

import numpy as np

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
