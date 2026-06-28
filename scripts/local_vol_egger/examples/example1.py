"""
Example 1 — Egger & Engl (2005), section 5.4.

Recover a constant local volatility a† = 0.15 (σ† = √(2·0.15) ≈ 0.548) from
European call prices, reproducing the paper's five test runs:

    A   complete, exact data,            prior a*₁ = 0.15 − 0.05·erf(−y²)
    B   incomplete (20 strikes), exact,  prior a*₁
    C   complete, exact data,            BAD prior a* = 0.1   (constant)
    D   complete, 0.1% noisy data,       prior a*₁
    BD  incomplete (20 strikes), noisy,  prior a*₁

run() executes all five cases in series, prints a per-strike table for each,
saves a recovery figure per case, and writes the collected results to two CSV
files (option values and reconstructed parameter a) that reproduce the paper's
Tables 1–2.

Common setup
------------
Domain   : y ∈ [−3, 3] (M = 3),  τ ∈ [0, 1]
Rates    : r = q = 0
Spot     : S₀ = 1000  (the paper's text "100" is a typo — see the note below)

S₀ note
-------
The paper's text says S = 100, but its tabulated option values (a call worth
439 at K=600) and test (D) ("underlying worth 1000$") only make sense with
S₀ = 1000.  S₀ matters even though the PDE is linear in u: the recovered a(y) is
scale-invariant, but the Tikhonov balance is not — the data term scales as S₀²
while β‖a − a*‖²_{H¹} does not — so reproducing the paper's β needs its S₀.
"""

import csv

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
# All five paper test runs — executed in series by run()
# ---------------------------------------------------------------------------
ALL_CASES = ["A", "B", "C", "D", "BD"]
STRIKES   = np.arange(600, 1801, 100)   # paper's tabulated strikes (Tables 1–2)


# ---------------------------------------------------------------------------
# Common experiment parameters
# ---------------------------------------------------------------------------

Y0, Y1     = -3.0, 3.0    # log-moneyness domain  (M = 3)
T_END      = 1.0          # maturity T
R, Q       = 0.0, 0.0     # zero interest / dividend rate
S0         = 1000.0       # spot price (see the S₀ note above)
N_SPACE    = 199          # spatial elements
M_TIME     = 200          # implicit-Euler time steps
MESH_BETA  = 0.0          # sinh node-clustering toward ATM (y=0); 0 = uniform

A_TRUE     = 0.15
SIGMA_TRUE = np.sqrt(2.0 * A_TRUE)   # ≈ 0.5477

# Incomplete-data cases (B, BD): 20 observed strikes, spread over a liquid band
# in log-moneyness; each is snapped to the nearest mesh node to build the mask.
N_OBS         = 20
OBS_Y_RANGE   = (-1.0, 1.0)          # K ≈ 368 … 2718 for S₀ = 1000

# Noisy cases (D, BD): the paper adds "0.1% uniformly distributed noise to the
# data" and notes that for S₀ = 1000$ this is "an uncertainty of ±1$ ... about
# the size of typical bid-ask spreads" (Egger & Engl 2005, §5, test D).  The
# wording is ambiguous — ±1$ is both 0.1%·S₀ AND ~0.1% of the priciest call (a
# call's value is bounded by S₀) — but empirically a flat absolute ±1$ band on
# every node destroys the near-zero OTM tail (negative prices, huge relative
# error) and yields very noisy reconstructions, whereas the paper's tables are
# stable.  We therefore use *relative* noise: ±0.1% of each option's own price,
# u^δ_i = u_i·(1 + U(−ε, ε)), which leaves the cheap tail intact and reproduces
# their stability.  (See memory note egger_noise_model for the full rationale.)
NOISE_REL     = 0.001                # ±0.1% of each option's own price
NOISE_SEED    = 0

# Regularisation: small fixed β for exact data; β ~ 1e-2·δ for noisy data
# (the value at which the paper meets the discrepancy principle, §5.3).
BETA_EXACT    = 1e-6
BETA_NOISE_K  = 1e-2                  # β = BETA_NOISE_K · δ


def _is_incomplete(case):
    return case in ("B", "BD")


def _is_noisy(case):
    return case in ("D", "BD")


# ---------------------------------------------------------------------------
# Boundary conditions  (paper recipe, §5.2)
# ---------------------------------------------------------------------------

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

def _prior_func(case):
    """Initial guess / regularisation centre a*(y) for the given case."""
    if case == "C":
        return lambda y: np.full_like(y, 0.10)        # bad constant prior
    return lambda y: 0.10 - 0.05 * erf(-y ** 2)        # a*₁ (cases A, B, D, BD)


def _build_obs_mask(V):
    """{0,1} DOF mask selecting the N_OBS observed strikes (cases B, BD)."""
    y = V.tabulate_dof_coordinates()[:, 0]
    targets = np.linspace(OBS_Y_RANGE[0], OBS_Y_RANGE[1], N_OBS)
    obs_nodes = np.unique([int(np.argmin(np.abs(y - yt))) for yt in targets])
    mask = np.zeros_like(y)
    mask[obs_nodes] = 1.0
    return mask


def setup(V, case):
    """
    Build the ingredients for the requested case.

    Returns
    -------
    u_0         : FEM Function — initial condition u(y,0) = S0·(1 − e^y)^+
    u_obs       : FEM Function — observed prices at τ = T (noisy for D/BD)
    a_init      : Vol — initial parameter guess a*
    a_prior_vec : np.ndarray — DOF vector of a*
    obs_mask    : np.ndarray or None — observed-strike mask (B/BD) or None
    beta        : float — regularisation parameter
    """
    # Initial condition: call payoff scaled by the spot price
    u_0 = fem.Function(V)
    u_0.interpolate(lambda x: S0 * np.maximum(1.0 - np.exp(x[0]), 0.0))

    # Exact observation: forward-solve with the true constant a†
    a_true_fn = fem.Function(V)
    a_true_fn.interpolate(lambda x: np.full_like(x[0], A_TRUE))
    a_true = Vol(V, a_true_fn, 0.0, T_END, N=1)

    dt = T_END / M_TIME
    u_clean, _ = compute_Af(a_true, u_0, dt, M_TIME, V, R, Q, Y0, Y1, 0.0,
                            _bc_left, _bc_right)

    obs_mask = _build_obs_mask(V) if _is_incomplete(case) else None

    # Observation = exact prices, optionally corrupted by 0.1% relative noise
    u_obs = fem.Function(V)
    u_obs.x.array[:] = u_clean.x.array[:]
    beta = BETA_EXACT
    if _is_noisy(case):
        rng = np.random.default_rng(NOISE_SEED)
        # ±0.1% of each option's own price → cheap OTM calls are not swamped.
        noise = rng.uniform(-NOISE_REL, NOISE_REL, size=u_obs.x.array.shape) \
                * u_clean.x.array
        u_obs.x.array[:] += noise

        # Noise level δ in the data-term norm (masked for incomplete data),
        # used to set β ~ 1e-2·δ.
        noise_fn = fem.Function(V)
        noise_fn.x.array[:] = noise
        if obs_mask is not None:
            noise_fn.x.array[:] *= obs_mask
        delta = L2_norm(noise_fn)
        beta  = BETA_NOISE_K * delta
        print(f"[case {case}] noise level δ = {delta:.4e}  →  β = {beta:.4e}")

    # Initial guess / prior
    prior = _prior_func(case)
    a_init_fn = fem.Function(V)
    a_init_fn.interpolate(lambda x: prior(x[0]))
    a_init = Vol(V, a_init_fn, 0.0, T_END, N=1)
    a_prior_vec = a_init_fn.x.array.copy()

    return u_0, u_obs, a_init, a_prior_vec, obs_mask, beta


# ---------------------------------------------------------------------------
# Mesh
# ---------------------------------------------------------------------------

def _make_mesh():
    """Interval mesh on [Y0, Y1] with sinh node-clustering toward ATM (y=0)."""
    msh = dolfinx_mesh.create_interval(MPI.COMM_WORLD, N_SPACE, points=(Y0, Y1))
    if MESH_BETA > 0:
        xi  = (msh.geometry.x[:, 0] - Y0) / (Y1 - Y0)
        mid = (Y0 + Y1) / 2
        msh.geometry.x[:, 0] = (
            mid + (Y1 - Y0) / 2 * np.sinh(MESH_BETA * (xi - 0.5)) / np.sinh(MESH_BETA / 2)
        )
    return msh


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _forward(a_vol, u_0, V):
    """Forward-solve to the option prices u(·,T) for a given parameter field."""
    dt = T_END / M_TIME
    u_pred, _ = compute_Af(a_vol, u_0, dt, M_TIME, V, R, Q, Y0, Y1, 0.0,
                           _bc_left, _bc_right)
    return u_pred


def _eval_at_strikes(a_field, u_pred, V):
    """
    Interpolate (C_recon, a_recon, σ_recon) at the tabulated strikes.

    Since r = q = 0 and u(y,0) = (S0−K)^+, u(y,T) is exactly the call price C(K,T).
    Returns a list of (K, y, C_recon, a_recon, σ_recon), one per strike.
    """
    y_coords = V.mesh.geometry.x[:, 0]
    idx = np.argsort(y_coords)
    y_s = y_coords[idx]
    a_s = a_field.x.array.real[idx]
    u_s = u_pred.x.array.real[idx]

    rows = []
    for K in STRIKES:
        y     = float(np.log(K / S0))
        C_rec = float(np.interp(y, y_s, u_s))
        a_rec = float(np.interp(y, y_s, a_s))
        rows.append((int(K), y, C_rec, a_rec, float(np.sqrt(2.0 * max(a_rec, 0.0)))))
    return rows


def _solve(V, case):
    """Calibrate one case; return the calibrated field and its setup ingredients."""
    print("\n" + "#" * 74)
    print(f"#  Calibrating case {case}")
    print("#" * 74)
    u_0, u_obs, a_init, a_prior_vec, obs_mask, beta = setup(V, case)
    a_cal, _ = run_tikhonov(
        u_0, u_obs, a_init, a_prior_vec, beta,
        V, R, Q, Y0, Y1, 0.0, T_END, M_TIME, _bc_left, _bc_right,
        max_iter=300, ftol=1e-12, gtol=1e-10, obs_mask=obs_mask,
    )
    return a_cal, u_0, u_obs, a_prior_vec, obs_mask


def run():
    """Run all five cases in series and write the CSVs reproducing Tables 1–2."""
    msh = _make_mesh()
    V   = fem.functionspace(msh, ("Lagrange", 1))

    # Case-independent reference columns of Table 1:
    #   C_true    — analytic Black–Scholes price at the true σ†
    #   C_optimal — numerical price from a forward solve with the true a† on the grid
    u_0 = fem.Function(V)
    u_0.interpolate(lambda x: S0 * np.maximum(1.0 - np.exp(x[0]), 0.0))
    a_true_fn = fem.Function(V)
    a_true_fn.interpolate(lambda x: np.full_like(x[0], A_TRUE))
    opt_rows  = _eval_at_strikes(a_true_fn,
                                 _forward(Vol(V, a_true_fn, 0.0, T_END, 1), u_0, V), V)
    C_optimal = {K: C for (K, _, C, _, _) in opt_rows}
    C_true    = {int(K): float(bs_call(S0, np.log(K / S0), R, Q, SIGMA_TRUE, T_END))
                 for K in STRIKES}

    # Calibrate every case and collect its per-strike table.
    results = {}
    for case in ALL_CASES:
        a_cal, u_0c, u_obs, a_prior_vec, obs_mask = _solve(V, case)
        rows = _eval_at_strikes(a_cal.a[0], _forward(a_cal, u_0c, V), V)
        results[case] = rows
        _print_table(case, rows, C_true)
        _plot(a_cal, a_prior_vec, u_0c, u_obs, V, case, obs_mask)

    _save_results(C_true, C_optimal, results)


# ---------------------------------------------------------------------------
# Tables 1 & 2 (Egger & Engl 2005) — console + CSV
# ---------------------------------------------------------------------------

def _print_table(case, rows, C_true):
    """Print the reconstructed option value / volatility for one case."""
    print("\n" + "=" * 74)
    print(f"  Example 1, case {case} — strikes {STRIKES[0]}…{STRIKES[-1]}, S0={S0:.0f}, T=1y")
    print("=" * 74)
    print(f"{'Strike':>7} | {'y':>7} | {'C_true':>9} {'C_recon':>9} {'ΔC':>8}"
          f" | {'a_true':>7} {'a_recon':>8} {'σ_recon':>8}")
    print("-" * 74)
    for (K, y, C_rec, a_rec, sigma) in rows:
        print(f"{K:>7} | {y:>7.3f} | {C_true[K]:>9.2f} {C_rec:>9.2f} {C_rec - C_true[K]:>8.3f}"
              f" | {A_TRUE:>7.4f} {a_rec:>8.4f} {sigma:>8.4f}")
    print("=" * 74)


def _save_results(C_true, C_optimal, results):
    """
    Write the collected results to two CSVs reproducing the paper's tables:
      egger_example1_option_values.csv  — Table 1 (option prices per case)
      egger_example1_parameter_a.csv    — Table 2 (reconstructed a = ½σ² per case)
    """
    f_val = "egger_example1_option_values.csv"
    with open(f_val, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["Strike", "C_true", "C_optimal"] + [f"C_{c}" for c in ALL_CASES])
        for i, K in enumerate(STRIKES):
            K = int(K)
            w.writerow([K, f"{C_true[K]:.2f}", f"{C_optimal[K]:.2f}"]
                       + [f"{results[c][i][2]:.2f}" for c in ALL_CASES])

    f_par = "egger_example1_parameter_a.csv"
    with open(f_par, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["Strike", "a_true"] + [f"a_{c}" for c in ALL_CASES])
        for i, K in enumerate(STRIKES):
            w.writerow([int(K), f"{A_TRUE:.4f}"]
                       + [f"{results[c][i][3]:.4f}" for c in ALL_CASES])

    print(f"\nSaved {f_val} and {f_par}")


# ---------------------------------------------------------------------------
# Plot — recovered 1-year slice + reprice residual
# ---------------------------------------------------------------------------

def _plot(a_cal, a_prior_vec, u_0, u_obs, V, case, obs_mask):
    y_coords = V.mesh.geometry.x[:, 0]
    idx      = np.argsort(y_coords)
    y        = y_coords[idx]

    a_rec   = a_cal.a[0].x.array.real[idx]
    a_prior = a_prior_vec[idx]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: parameter recovery
    ax = axes[0]
    ax.axvspan(np.log(600 / S0), np.log(1800 / S0), color='gold', alpha=0.15,
               label="Paper's tabulated strikes")
    ax.axhline(A_TRUE,  color='r',    ls='--', lw=1.5, label=r'True $a^\dagger = 0.15$')
    ax.plot(y, a_prior, color='grey', ls=':',  lw=1.5, label=r'Initial guess $a^*$')
    ax.plot(y, a_rec,   color='b',    ls='-',  lw=1.5, label=r'Recovered $a(y)$')
    if obs_mask is not None:
        on = obs_mask[idx] > 0.5
        ax.plot(y[on], a_rec[on], 'k.', ms=8, label='Observed strikes')
    ax.set_xlabel('Log-moneyness $y$')
    ax.set_ylabel(r'$a(y) = \frac{1}{2}\sigma^2$')
    ax.set_title(f'Parameter recovery — Example 1 (case {case})')
    ax.legend(fontsize=9)
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
    out = f'egger_example1_case{case}.png'
    plt.savefig(out, dpi=150)
    print(f"Saved {out}")
    plt.close(fig)


if __name__ == "__main__":
    run()
