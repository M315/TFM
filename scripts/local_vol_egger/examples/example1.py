r"""
Example 1 -- Egger & Engl (2005), section 5.4.

Recover a constant local volatility a^\dagger = 0.15 (\sigma^\dagger = \sqrt(2\cdot0.15) \approx 0.548) from
European call prices, reproducing the paper's five test runs:

    A   complete, exact data,            prior a*_1 = 0.15 - 0.05\cdot erf(-y^2)
    B   incomplete (20 strikes), exact,  prior a*_1
    C   complete, exact data,            BAD prior a* = 0.1   (constant)
    D   complete, 0.1% noisy data,       prior a*_1
    BD  incomplete (20 strikes), noisy,  prior a*_1

run() executes all five cases in series, prints a per-strike table for each,
saves a recovery figure per case, and writes the collected results to two CSV
files (option values and reconstructed parameter a) that reproduce the paper's
Tables 1-2.

Common setup
------------
Domain   : y \in [-3, 3] (M = 3),  \tau \in [0, 1]
Rates    : r = q = 0
Spot     : S_0 = 1000  (the paper's text "100" is a typo - see the note below)

S_0 note
-------
The paper's text says S = 100, but its tabulated option values (a call worth
439 at K=600) and test (D) ("underlying worth 1000$") only make sense with
S_0 = 1000.  S_0 matters even though the PDE is linear in u: the recovered a(y) is
scale-invariant, but the Tikhonov balance is not - the data term scales as S_0^2
while \beta\|a - a*\|^2_{H^1} does not - so reproducing the paper's \beta needs its S_0.
"""

import csv

import numpy as np
from scipy.special import erf
from dolfin import Function, FunctionSpace, IntervalMesh, set_log_level, LogLevel
import matplotlib.pyplot as plt

from optimization.vol       import Vol
from optimization.tikhonov  import run as run_tikhonov
from pde.forward             import compute_Af
from utils                   import (L2_norm, bs_call, interpolate_func,
                                     dof_coordinates, get_array, set_array)

set_log_level(LogLevel.ERROR)   # suppress dolfin solver chatter


# ---------------------------------------------------------------------------
# All five paper test runs - executed in series by run()
# ---------------------------------------------------------------------------
ALL_CASES = ["A", "B", "C", "D", "BD"]
STRIKES   = np.arange(600, 1801, 100)   # paper's tabulated strikes (Tables 1-2)


# ---------------------------------------------------------------------------
# Common experiment parameters
# ---------------------------------------------------------------------------

Y0, Y1     = -3.0, 3.0    # log-moneyness domain  (M = 3)
T_END      = 1.0          # maturity T
R, Q       = 0.0, 0.0     # zero interest / dividend rate
S0         = 1000.0       # spot price (see the S_0 note above)
N_SPACE    = 199          # spatial elements
M_TIME     = 200          # implicit-Euler time steps
MESH_BETA  = 0.0          # sinh node-clustering toward ATM (y=0); 0 = uniform

A_TRUE     = 0.15
SIGMA_TRUE = np.sqrt(2.0 * A_TRUE)   # \approx 0.5477

# Incomplete-data cases (B, BD): 20 observed strikes, spread over a liquid band
# in log-moneyness; each is snapped to the nearest mesh node to build the mask.
N_OBS         = 20
OBS_Y_RANGE   = (-1.0, 1.0)          # K \approx 368 ... 2718 for S_0 = 1000

# Noisy cases (D, BD): the paper adds "0.1% uniformly distributed noise to the
# data" and notes that for S_0 = 1000$ this is "an uncertainty of \pm1$ ... about
# the size of typical bid-ask spreads" (Egger & Engl 2005, \S5, test D).  The
# wording is ambiguous - \pm1$ is both 0.1% \cdot S_0 AND ~0.1% of the priciest call (a
# call's value is bounded by S_0) - but empirically a flat absolute \pm1$ band on
# every node destroys the near-zero OTM tail (negative prices, huge relative
# error) and yields very noisy reconstructions, whereas the paper's tables are
# stable.  We therefore use *relative* noise: \pm0.1% of each option's own price,
# u^\delta_i = u_i\cdot(1 + U(-\varepsilon, \varepsilon)), which leaves the cheap tail intact and reproduces
# their stability.  (See memory note egger_noise_model for the full rationale.)
NOISE_REL     = 0.001                # \pm0.1% of each option's own price
NOISE_SEED    = 0

# Regularisation: small fixed \beta for exact data; \beta ~ 1e-2\cdot\delta for noisy data
# (the value at which the paper meets the discrepancy principle, \S5.3).
BETA_EXACT    = 1e-6
BETA_NOISE_K  = 1e-2                  # \beta = BETA_NOISE_K \cdot \delta


def _is_incomplete(case):
    return case in ("B", "BD")


def _is_noisy(case):
    return case in ("D", "BD")


# ---------------------------------------------------------------------------
# Boundary conditions  (paper recipe, \S5.2)
# ---------------------------------------------------------------------------

def _bc_left(tau):
    r"""Exact BS call price at the deep-ITM boundary y = Y0 (strike K = S0\cdot e^{Y0})."""
    if tau <= 1e-14:
        return float(S0 * max(1.0 - np.exp(Y0), 0.0))
    return float(bs_call(S0, Y0, R, Q, SIGMA_TRUE, tau))


def _bc_right(tau):
    r"""Exact BS call price at the deep-OTM boundary y = Y1 (strike K = S0 \cdot e^{Y1}, \approx 0)."""
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
    return lambda y: 0.15 - 0.05 * erf(-y ** 2)        # a*_1 (cases A, B, D, BD)


def _build_obs_mask(V):
    """{0,1} DOF mask selecting the N_OBS observed strikes (cases B, BD)."""
    y = dof_coordinates(V)
    targets = np.linspace(OBS_Y_RANGE[0], OBS_Y_RANGE[1], N_OBS)
    obs_nodes = np.unique([int(np.argmin(np.abs(y - yt))) for yt in targets])
    mask = np.zeros_like(y)
    mask[obs_nodes] = 1.0
    return mask


def setup(V, case):
    r"""
    Build the ingredients for the requested case.

    Returns
    -------
    u_0         : FEM Function - initial condition u(y,0) = S0\cdot(1 - e^y)^+
    u_obs       : FEM Function - observed prices at \tau = T (noisy for D/BD)
    a_init      : Vol - initial parameter guess a*
    a_prior_vec : np.ndarray - DOF vector of a*
    obs_mask    : np.ndarray or None -- observed-strike mask (B/BD) or None
    beta        : float - regularisation parameter
    """
    # Initial condition: call payoff scaled by the spot price
    u_0 = interpolate_func(V, lambda y: S0 * np.maximum(1.0 - np.exp(y), 0.0))

    # Exact observation: forward-solve with the true constant a^\dagger
    a_true_fn = interpolate_func(V, lambda y: np.full_like(y, A_TRUE))
    a_true = Vol(V, a_true_fn, 0.0, T_END, N=1)

    dt = T_END / M_TIME
    u_clean, _ = compute_Af(a_true, u_0, dt, M_TIME, V, R, Q, Y0, Y1, 0.0,
                            _bc_left, _bc_right)

    obs_mask = _build_obs_mask(V) if _is_incomplete(case) else None

    # Observation = exact prices, optionally corrupted by 0.1% relative noise
    u_obs = Function(V)
    set_array(u_obs, get_array(u_clean))
    beta = BETA_EXACT
    if _is_noisy(case):
        rng = np.random.default_rng(NOISE_SEED)
        # \pm0.1% of each option's own price \to cheap OTM calls are not swamped.
        u_clean_vec = get_array(u_clean)
        noise = rng.uniform(-NOISE_REL, NOISE_REL, size=u_clean_vec.shape) \
                * u_clean_vec
        set_array(u_obs, get_array(u_obs) + noise)

        # Noise level \delta in the data-term norm (masked for incomplete data),
        # used to set \beta ~ 1e-2 \cdot \delta.
        noise_fn = Function(V)
        set_array(noise_fn, noise * obs_mask if obs_mask is not None else noise)
        delta = L2_norm(noise_fn)
        beta  = BETA_NOISE_K * delta
        print(rf"[case {case}] noise level \delta = {delta:.4e}  ->  \beta = {beta:.4e}")

    # Initial guess / prior
    prior = _prior_func(case)
    a_init_fn = interpolate_func(V, prior)
    a_init = Vol(V, a_init_fn, 0.0, T_END, N=1)
    a_prior_vec = get_array(a_init_fn).copy()

    return u_0, u_obs, a_init, a_prior_vec, obs_mask, beta


# ---------------------------------------------------------------------------
# Mesh
# ---------------------------------------------------------------------------

def _make_mesh():
    """Interval mesh on [Y0, Y1] with sinh node-clustering toward ATM (y=0)."""
    msh = IntervalMesh(N_SPACE, Y0, Y1)
    if MESH_BETA > 0:
        coords = msh.coordinates()
        xi  = (coords[:, 0] - Y0) / (Y1 - Y0)
        mid = (Y0 + Y1) / 2
        coords[:, 0] = (
            mid + (Y1 - Y0) / 2 * np.sinh(MESH_BETA * (xi - 0.5)) / np.sinh(MESH_BETA / 2)
        )
    return msh


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _forward(a_vol, u_0, V):
    r"""Forward-solve to the option prices u(\cdot,T) for a given parameter field."""
    dt = T_END / M_TIME
    u_pred, _ = compute_Af(a_vol, u_0, dt, M_TIME, V, R, Q, Y0, Y1, 0.0,
                           _bc_left, _bc_right)
    return u_pred


def _eval_at_strikes(a_field, u_pred, V):
    r"""
    Interpolate (C_recon, a_recon, \sigma_recon) at the tabulated strikes.

    Since r = q = 0 and u(y,0) = (S0 - K)^+, u(y,T) is exactly the call price C(K,T).
    Returns a list of (K, y, C_recon, a_recon, \sigma_recon), one per strike.
    """
    y_coords = dof_coordinates(V)
    idx = np.argsort(y_coords)
    y_s = y_coords[idx]
    a_s = get_array(a_field)[idx]
    u_s = get_array(u_pred)[idx]

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
    """Run all five cases in series and write the CSVs reproducing Tables 1-2."""
    msh = _make_mesh()
    V   = FunctionSpace(msh, "Lagrange", 1)

    # Case-independent reference columns of Table 1:
    #   C_true    - analytic Black-Scholes price at the true \sigma^\dagger
    #   C_optimal - numerical price from a forward solve with the true a^\dagger on the grid
    u_0 = interpolate_func(V, lambda y: S0 * np.maximum(1.0 - np.exp(y), 0.0))
    a_true_fn = interpolate_func(V, lambda y: np.full_like(y, A_TRUE))
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
# Tables 1 & 2 (Egger & Engl 2005) - console + CSV
# ---------------------------------------------------------------------------

def _print_table(case, rows, C_true):
    """Print the reconstructed option value / volatility for one case."""
    print("\n" + "=" * 74)
    print(f"  Example 1, case {case} -- strikes {STRIKES[0]}...{STRIKES[-1]}, S0={S0:.0f}, T=1y")
    print("=" * 74)
    print(f"{'Strike':>7} | {'y':>7} | {'C_true':>9} {'C_recon':>9} {r'\Delta C':>9}"
          f" | {'a_true':>7} {'a_recon':>8} {r'\sigma_rec':>10}")
    print("-" * 74)
    for (K, y, C_rec, a_rec, sigma) in rows:
        print(f"{K:>7} | {y:>7.3f} | {C_true[K]:>9.2f} {C_rec:>9.2f} {C_rec - C_true[K]:>9.3f}"
              f" | {A_TRUE:>7.4f} {a_rec:>8.4f} {sigma:>10.4f}")
    print("=" * 74)


def _save_results(C_true, C_optimal, results):
    r"""
    Write the collected results to two CSVs reproducing the paper's tables:
      egger_example1_option_values.csv  - Table 1 (option prices per case)
      egger_example1_parameter_a.csv    - Table 2 (reconstructed a = \frac{1}{2}\sigma^2 per case)
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
# Plot - recovered 1-year slice + reprice residual
# ---------------------------------------------------------------------------

def _plot(a_cal, a_prior_vec, u_0, u_obs, V, case, obs_mask):
    y_coords = dof_coordinates(V)
    idx      = np.argsort(y_coords)
    y        = y_coords[idx]

    a_rec   = get_array(a_cal.a[0])[idx]
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
    ax.set_title(f'Parameter recovery -- Example 1 (case {case})')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.4)

    # Right: reprice residual at \tau = T
    dt = T_END / M_TIME
    u_pred, _ = compute_Af(a_cal, u_0, dt, M_TIME, V, R, Q, Y0, Y1, 0.0,
                            _bc_left, _bc_right)
    err = Function(V)
    set_array(err, get_array(u_pred) - get_array(u_obs))
    l2_err = L2_norm(err)
    print(f"Reprice L^2 error: {l2_err:.4e}")

    ax = axes[1]
    ax.plot(y, get_array(err)[idx], 'g-', lw=1.5)
    ax.axhline(0, color='k', lw=0.8)
    ax.set_xlabel('Log-moneyness $y$')
    ax.set_ylabel(r'$u_{\mathrm{pred}} - u^{\delta}$')
    ax.set_title(f'Reprice residual  ($L^2$ = {l2_err:.2e})')
    ax.grid(True, alpha=0.4)

    plt.tight_layout()
    out = f'egger_example1_case{case}.png'
    plt.savefig(out, dpi=150)
    print(f"Saved {out}")
    plt.close(fig)


if __name__ == "__main__":
    run()
