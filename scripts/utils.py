from mpi4py import MPI

import numpy as np
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
