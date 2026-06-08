"""
Gradient check for adj_da_f.

Three quantities must agree:
  adj_dot   = ∑_j g_j h_j           (Euclidean DOF product, adjoint gradient)
  gat_dot   = ⟨R, dAf(h)⟩_L2       (L2 inner product, Gâteaux derivative)
  fd_dot    = (J_L2(a+εh) - J_L2(a-εh)) / (2ε)  (central FD of L2 cost)

The adjoint computes the gradient of J = ½‖R‖²_L2, which is the cost used in
calibration.py.  Using the Euclidean norm in the FD would give a result smaller
by a factor of ~h (mesh spacing) due to the mass-matrix scaling.
"""
from mpi4py import MPI
import numpy as np
import ufl
from dolfinx import fem, mesh

from calibration import Vol
from a_f import compute_Af
from adj_da_f import compute_adj_dAf
from da_f import compute_dAf
from utils import L2_norm

Y0, Y1   = -1.0, 1.0
N_ELEMS  = 60
M_TIME   = 20
T0, T1   = 0.0, 0.25
dt       = (T1 - T0) / M_TIME
r, q     = 0.04, 0.02
N_SLICES = 2

msh = mesh.create_interval(MPI.COMM_WORLD, N_ELEMS, points=(Y0, Y1))
V   = fem.functionspace(msh, ("Lagrange", 1))

y_coords = V.mesh.geometry.x[:, 0]

# non-flat a so that a_y != 0 (this is where the old code was wrong)
a_init_fn = fem.Function(V)
a_init_fn.x.array[:] = 0.04 + 0.01 * np.sin(np.pi * y_coords)

a_vol = Vol(V, a_init_fn, T0, T1, N=N_SLICES)

f_fn = fem.Function(V)
f_fn.x.array[:] = np.maximum(1.0 - np.exp(y_coords), 0.0)

g_fn = fem.Function(V)
g_fn.x.array[:] = np.maximum(1.0 - np.exp(y_coords) * 0.95, 0.0)

psi_1 = lambda t: float(np.exp(-q * t) - np.exp(Y0) * np.exp(-r * t))
psi_2 = lambda t: 0.0

# --- forward pass ---
u_T, traj = compute_Af(a_vol, f_fn, dt, M_TIME, V, r, q, Y0, Y1, T0, psi_1, psi_2)

residual = fem.Function(V)
residual.x.array[:] = u_T.x.array - g_fn.x.array

# --- adjoint gradient ---
grads, _ = compute_adj_dAf(a_vol, residual, traj, dt, M_TIME, V, r, q, Y0, Y1, T0)

# --- perturbation direction h (smooth, non-zero a_y contribution) ---
h_fn = fem.Function(V)
h_fn.x.array[:] = 0.001 * np.cos(2 * np.pi * y_coords)

# (1) adjoint: ∑_j g_j h_j  — Euclidean DOF product, same h applied to each slice
adj_dot = sum(np.dot(grads[k].x.array, h_fn.x.array) for k in range(N_SLICES))

# (2) Gâteaux: ⟨R, dAf(h)⟩_L2  — L2 inner product
dAf_h = compute_dAf(a_vol, h_fn, traj, dt, M_TIME, V, r, q, Y0, Y1, T0)
gat_dot = fem.assemble_scalar(fem.form(ufl.inner(residual, dAf_h) * ufl.dx))

# (3) finite differences of J_L2 = ½‖R‖²_L2  — same cost as calibration.py
def J_L2(a_vec):
    av = Vol(V, a_init_fn, T0, T1, N=N_SLICES)
    av.update(V, a_vec)
    u, _ = compute_Af(av, f_fn, dt, M_TIME, V, r, q, Y0, Y1, T0, psi_1, psi_2)
    res = fem.Function(V)
    res.x.array[:] = u.x.array - g_fn.x.array
    return 0.5 * L2_norm(res) ** 2

eps       = 1e-5
a_vec0    = np.concatenate([f.x.array.copy() for f in a_vol.a])
h_stacked = np.tile(h_fn.x.array, N_SLICES)
fd_dot    = (J_L2(a_vec0 + eps * h_stacked) - J_L2(a_vec0 - eps * h_stacked)) / (2 * eps)

print(f"Adjoint  ∑ g_j h_j          = {adj_dot:+.8e}")
print(f"Gâteaux  ⟨R, dAf(h)⟩_L2    = {gat_dot:+.8e}")
print(f"FD  (L2 cost)               = {fd_dot:+.8e}")
print(f"Adj / FD                    = {adj_dot / fd_dot:.6f}  (target: 1.0)")
print(f"Gât / FD                    = {gat_dot / fd_dot:.6f}  (target: ~1.0, O(dt) error ok)")
