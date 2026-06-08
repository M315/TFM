"""
Example 1 — Egger & Engl (2005), section 5.4.

Recover a constant local volatility from exact call-price observations.

Setup
-----
True parameter   : a† = 0.15  (constant) ↔ σ† = √(2·0.15) ≈ 0.548
Domain           : y ∈ [−3, 3],  τ ∈ [0, 1]
Rates            : r = q = 0
Regularisation   : β = 1e-6
Data             : exact (no noise) — case A of the paper

Initial guess and prior (case A)
---------------------------------
    a*_1(y) = 0.15 − 0.05·erf(−y²)

At y = 0 this equals the true value 0.15.  For large |y|, erf(−y²) → −1 so
a*_1 → 0.20.  The paper uses this to demonstrate that the method recovers well
in the informationally rich region (near ATM) and stays close to the prior
elsewhere — a manifestation of the inherent ill-posedness.

Spot price
----------
The paper's text says S = 100, but the reported option values (Tables 1–2,
strikes 600–1800 with a call worth 439 at K=600) and test (D) ("underlying
worth 1000$") only make sense with S₀ = 1000 — the text value is a typo.

S₀ matters here even though the PDE is linear in u: the recovered a(y) is
scale-invariant, but the Tikhonov functional is not.  The data term
‖u(a) − u^δ‖²_{L²} scales as S₀² while β‖a − a*‖²_{H¹} does not, so the
data/regularisation balance — and hence the effective strength of β — depends
on S₀.  Reproducing the paper's β = 1e-6 result therefore requires its S₀.

The initial condition is u(y, 0) = S₀·(1 − e^y)⁺ and the Dirichlet BCs are the
exact Black–Scholes call price evaluated at the boundary strikes K = S₀·e^{±M}
(this is the BC recipe the paper uses in its test example, §5.2).
"""

from mpi4py import MPI

import numpy as np
from scipy.special import erf
from dolfinx import fem, mesh as dolfinx_mesh
import matplotlib.pyplot as plt

from optimization.vol       import Vol
from optimization.tikhonov  import run as run_tikhonov
from pde.forward             import compute_Af
from utils                   import L2_norm, bs_call


# ---------------------------------------------------------------------------
# Experiment parameters
# ---------------------------------------------------------------------------

Y0, Y1     = -3.0, 3.0   # log-moneyness domain  (M = 3)
T_END      = 1.0          # maturity T
R, Q       = 0.0, 0.0    # zero interest / dividend rate
S0         = 1000.0       # spot price (paper's text says 100, but its numerics use 1000)
BETA       = 1e-6         # Tikhonov β (exact data → small β; calibrated for this S0)
N_SPACE    = 199          # spatial elements
M_TIME     = 200          # implicit-Euler time steps
MESH_BETA  = 2.0          # sinh node-clustering toward ATM (y=0): 0 = uniform,
                          # larger = denser near the money (the data-rich region)

A_TRUE     = 0.15
SIGMA_TRUE = np.sqrt(2.0 * A_TRUE)   # ≈ 0.5477

# Prior / initial guess for a*:
#   "erf"  — paper's case A: a*_1(y) = 0.15 − 0.05·erf(−y²)  (→ 0.20 at edges,
#            so the edges revert to 0.20 where data is uninformative)
#   "flat" — a* = 0.15 everywhere: with no edge bias the method recovers the
#            flat true surface across the whole domain (natural dummy-target check)
PRIOR = "erf"


# ---------------------------------------------------------------------------
# Boundary conditions
# ---------------------------------------------------------------------------

# Paper's recipe (§5.2): take the boundary data z0 = u(−M,T), z1 = u(M,T),
# invert Black–Scholes for their implied vols, then BS-price those vols over
# τ ∈ [0, T] to get g0(τ), g1(τ).  For the synthetic case-A data, z0 and z1 are
# *exactly* the BS prices at the true σ, so their implied vols are σ† and the
# recipe reduces to BS-pricing at SIGMA_TRUE — which is what we do below.
# (Literally inverting z0/z1 would be ill-conditioned here anyway: at |y| = 3
# the price is ~intrinsic / ~0, where it carries almost no vol information.)

def _bc_left(tau):
    """Exact BS call price at the deep-ITM boundary y = Y0 (strike K = S0·e^{Y0})."""
    if tau <= 1e-14:
        return float(S0 * max(1.0 - np.exp(Y0), 0.0))
    return float(bs_call(S0, Y0, R, Q, SIGMA_TRUE, tau))


def _bc_right(tau):
    """Exact BS call price at the deep-OTM boundary y = Y1 (strike K = S0·e^{Y1}, ≈ 0)."""
    if tau <= 1e-14:
        return 0.0
    return float(bs_call(S0, Y1, R, Q, SIGMA_TRUE, tau))


# ---------------------------------------------------------------------------
# Problem setup
# ---------------------------------------------------------------------------

def setup(V):
    """
    Build all ingredients for Example 1, case A.

    Returns
    -------
    u_0         : FEM Function — initial condition u(y, 0) = S0·(1 − e^y)^+
    u_obs       : FEM Function — exact call prices at τ = T (generated with true a†)
    a_init      : Vol — initial parameter guess a*_1(y)
    a_prior_vec : np.ndarray — DOF vector of a* = a*_1 (regularisation centre)
    """
    # Initial condition: call payoff scaled by the spot price
    u_0 = fem.Function(V)
    u_0.interpolate(lambda x: S0 * np.maximum(1.0 - np.exp(x[0]), 0.0))

    # Exact observation: forward-solve with the true constant a†
    a_true_fn = fem.Function(V)
    a_true_fn.interpolate(lambda x: np.full_like(x[0], A_TRUE))
    a_true = Vol(V, a_true_fn, 0.0, T_END, N=1)

    dt = T_END / M_TIME
    u_obs, _ = compute_Af(a_true, u_0, dt, M_TIME, V, R, Q, Y0, Y1, 0.0,
                          _bc_left, _bc_right)

    # Initial guess / prior — see the PRIOR flag above
    a_init_fn = fem.Function(V)
    if PRIOR == "flat":
        a_init_fn.interpolate(lambda x: np.full_like(x[0], A_TRUE))
    else:  # "erf" — paper's case A
        a_init_fn.interpolate(lambda x: 0.15 - 0.05 * erf(-x[0] ** 2))
    a_init = Vol(V, a_init_fn, 0.0, T_END, N=1)

    a_prior_vec = a_init_fn.x.array.copy()   # a* = same as initial guess

    return u_0, u_obs, a_init, a_prior_vec


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _make_mesh():
    """
    Interval mesh on [Y0, Y1] with optional sinh clustering toward ATM (y=0).

    The data-informative region is near the money (small |y|); clustering nodes
    there resolves the recovered vol more finely without wasting resolution on
    the deep ITM/OTM tails (where the prices — and hence the recoverable vol —
    carry little information).  The domain endpoints stay at ±M, so the
    asymptotic boundary conditions remain valid.

        y(ξ) = (Y1-Y0)/2 · sinh(β(ξ-½)) / sinh(β/2),   ξ ∈ [0,1] uniform.
    """
    msh = dolfinx_mesh.create_interval(MPI.COMM_WORLD, N_SPACE, points=(Y0, Y1))
    if MESH_BETA > 0:
        xi  = (msh.geometry.x[:, 0] - Y0) / (Y1 - Y0)
        mid = (Y0 + Y1) / 2
        msh.geometry.x[:, 0] = (
            mid + (Y1 - Y0) / 2 * np.sinh(MESH_BETA * (xi - 0.5)) / np.sinh(MESH_BETA / 2)
        )
    return msh


def run():
    msh = _make_mesh()
    V   = fem.functionspace(msh, ("Lagrange", 1))

    u_0, u_obs, a_init, a_prior_vec = setup(V)

    a_cal, _ = run_tikhonov(
        u_0, u_obs, a_init, a_prior_vec, BETA,
        V, R, Q, Y0, Y1, 0.0, T_END, M_TIME, _bc_left, _bc_right,
        max_iter=300, ftol=1e-12, gtol=1e-10,
    )

    _print_table(a_cal, u_0, V)
    _plot(a_cal, a_prior_vec, u_0, u_obs, V)
    _plot_surface(a_cal, u_0, V)


# ---------------------------------------------------------------------------
# Tables 1 & 2 (Egger & Engl 2005, case A)
# ---------------------------------------------------------------------------

def _print_table(a_cal, u_0, V):
    """
    Reproduce Tables 1 & 2 of the paper for case A at T = 1 y.

    Since r = q = 0 and u(y,0) = S0·(1−e^y)^+ = (S0−K)^+, the PDE solution u(y,T)
    is exactly the call price C(K, T).  So:
      - reconstructed option value  = u(a_cal)(y, T)            (Table 1, col A)
      - true option value           = analytic BS call at σ†    (Table 1, "True value")
      - reconstructed parameter a   = a_cal interpolated at y    (Table 2, col A)
        with σ = √(2a).
    """
    y_coords = V.mesh.geometry.x[:, 0]
    idx = np.argsort(y_coords)
    y_s = y_coords[idx]
    a_s = a_cal.a[0].x.array.real[idx]

    dt = T_END / M_TIME
    u_pred, _ = compute_Af(a_cal, u_0, dt, M_TIME, V, R, Q, Y0, Y1, 0.0,
                           _bc_left, _bc_right)
    u_s = u_pred.x.array.real[idx]

    strikes = np.arange(600, 1801, 100)

    print("\n" + "=" * 74)
    print(f"  Example 1, case A — strikes {strikes[0]}…{strikes[-1]}, S0={S0:.0f}, T=1y")
    print("=" * 74)
    print(f"{'Strike':>7} | {'y':>7} | {'C_true':>9} {'C_recon':>9} {'ΔC':>8}"
          f" | {'a_true':>7} {'a_recon':>8} {'σ_recon':>8}")
    print("-" * 74)
    for K in strikes:
        y      = float(np.log(K / S0))
        C_true = float(bs_call(S0, y, R, Q, SIGMA_TRUE, T_END))
        C_rec  = float(np.interp(y, y_s, u_s))
        a_rec  = float(np.interp(y, y_s, a_s))
        sigma  = float(np.sqrt(2.0 * max(a_rec, 0.0)))
        print(f"{K:>7} | {y:>7.3f} | {C_true:>9.2f} {C_rec:>9.2f} {C_rec - C_true:>8.3f}"
              f" | {A_TRUE:>7.4f} {a_rec:>8.4f} {sigma:>8.4f}")
    print("=" * 74)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot(a_cal, a_prior_vec, u_0, u_obs, V):
    y_coords = V.mesh.geometry.x[:, 0]
    idx      = np.argsort(y_coords)
    y        = y_coords[idx]

    a_rec   = a_cal.a[0].x.array.real[idx]
    a_prior = a_prior_vec[idx]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: parameter recovery
    ax = axes[0]
    # Shade the strike range the paper actually tabulates (K = 600..1800):
    #   y = ln(600/1000) ≈ −0.51 … ln(1800/1000) ≈ 0.59
    ax.axvspan(np.log(600 / S0), np.log(1800 / S0), color='gold', alpha=0.15,
               label="Paper's tabulated strikes")
    ax.axhline(A_TRUE,  color='r',    ls='--', lw=1.5, label=r'True $a^\dagger = 0.15$')
    ax.plot(y, a_prior, color='grey', ls=':',  lw=1.5, label=r'Initial guess $a^*_1(y)$')
    ax.plot(y, a_rec,   color='b',    ls='-',  lw=1.5, label=r'Recovered $a(y)$')
    ax.set_xlabel('Log-moneyness $y$')
    ax.set_ylabel(r'$a(y) = \frac{1}{2}\sigma^2$')
    ax.set_title('Parameter recovery — Example 1 (case A)')
    ax.legend()
    ax.grid(True, alpha=0.4)

    # Right: reprice residual at τ = T
    dt = T_END / M_TIME
    u_pred, _ = compute_Af(a_cal, u_0, dt, M_TIME, V, R, Q, Y0, Y1, 0.0,
                            _bc_left, _bc_right)
    err = fem.Function(V)
    err.x.array[:] = u_pred.x.array - u_obs.x.array
    l2_err = L2_norm(err)
    print(f"Reprice L² error: {l2_err:.4e}")

    ax = axes[1]
    ax.plot(y, err.x.array.real[idx], 'g-', lw=1.5)
    ax.axhline(0, color='k', lw=0.8)
    ax.set_xlabel('Log-moneyness $y$')
    ax.set_ylabel(r'$u_{\mathrm{pred}} - u^{\delta}$')
    ax.set_title(f'Reprice residual  (L² = {l2_err:.2e})')
    ax.grid(True, alpha=0.4)

    plt.tight_layout()
    out = 'egger_example1_caseA.png'
    plt.savefig(out, dpi=150)
    print(f"Saved {out}")
    plt.show()


def _plot_surface(a_cal, u_0, V):
    """
    3D surfaces over the full (y, τ) grid:
      left  — recovered local-vol surface σ(y,τ) = √(2 a(y,τ))
      right — option-price solution u(y,τ) from the forward solve

    In Example 1 the parameter a is time-independent (one Vol slice), so the
    σ-surface is a flat extrusion in τ — the left panel confirms that.  The
    right panel shows the genuine time evolution of the price surface, from the
    payoff u(y,0) = S0·(1−e^y)^+ at τ=0 up to the observed prices at τ=T.
    """
    y_coords = V.mesh.geometry.x[:, 0]
    idx = np.argsort(y_coords)
    y   = y_coords[idx]

    dt = T_END / M_TIME
    _, traj = compute_Af(a_cal, u_0, dt, M_TIME, V, R, Q, Y0, Y1, 0.0,
                         _bc_left, _bc_right)
    taus = np.linspace(0.0, T_END, M_TIME + 1)

    U   = np.array([snap.x.array.real[idx] for snap in traj])                 # price surface
    SIG = np.array([np.sqrt(2.0 * a_cal.get(t).x.array.real[idx]) for t in taus])  # vol surface

    Yg, Tg = np.meshgrid(y, taus)

    fig = plt.figure(figsize=(14, 6))

    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1.plot_surface(Yg, Tg, SIG, cmap='viridis', edgecolor='none')
    ax1.set_xlabel('Log-moneyness $y$')
    ax1.set_ylabel(r'Maturity $\tau$')
    ax1.set_zlabel(r'$\sigma_{\mathrm{loc}}(y,\tau)$')
    ax1.set_title(r'Recovered local-vol surface  $\sigma = \sqrt{2a}$')

    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax2.plot_surface(Yg, Tg, U, cmap='plasma', edgecolor='none')
    ax2.set_xlabel('Log-moneyness $y$')
    ax2.set_ylabel(r'Maturity $\tau$')
    ax2.set_zlabel(r'$u(y,\tau)$')
    ax2.set_title('Option-price solution surface')

    plt.tight_layout()
    out = 'egger_example1_caseA_surface.png'
    plt.savefig(out, dpi=150)
    print(f"Saved {out}")
    plt.show()


if __name__ == "__main__":
    run()
