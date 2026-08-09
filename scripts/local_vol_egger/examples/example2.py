r"""
Example 2 -- Egger & Engl (2005), section 5.4.

Recover an *oscillatory* local volatility field

    a^\dagger(y) = 0.15 + 0.05 exp[-(y + 0.3)^2] sin(2 pi y) + 0.05 erf(20 y)

(true variance a = 1/2 sigma^2) from European call prices.  Unlike Example 1, the
target now has a sharp feature near the money (the erf(20 y) is nearly a step at
y = 0) and decaying oscillations in the wings -- a much harder reconstruction.

The paper runs the same five cases as Example 1:

    A   complete, exact data,            prior a*_1 = 0.15 + 0.05 erf(2 y)
    B   incomplete (20 strikes), exact,  prior a*_1
    C   complete, exact data,            BAD prior a* = 0.15  (constant)
    D   complete, 0.1% noisy data,       prior a*_1
    BD  incomplete (20 strikes), noisy,  prior a*_1

Avoiding the inverse crime
--------------------------
Egger & Engl generate the synthetic data "on the domain M = (-5, 5) with a fine
discretization and a different time integration scheme as used for the
reconstruction, to avoid inverse crimes."  We follow the same idea: the data are
produced by a forward solve on a WIDER domain [-5, 5] with a FOUR-TIMES-FINER
mesh and time step, then interpolated onto the coarse reconstruction nodes.  The
reconstruction itself runs on [-3, 3] with the standard 199-element / 200-step
implicit-Euler grid, so the data are never produced by the same operator that is
inverted.  (We keep implicit Euler for both rather than switching schemes; the
4x grid refinement already breaks the crime.  See the project note.)

Common setup mirrors Example 1: S_0 = 1000, r = q = 0, tau in [0, 1].
"""

import csv

import numpy as np
from scipy.special import erf
from dolfin import Function, FunctionSpace, IntervalMesh, set_log_level, LogLevel
import matplotlib.pyplot as plt

import os

from optimization.vol       import Vol
from optimization.tikhonov  import run as run_tikhonov
from optimization.tikhonov  import run_discrepancy
from pde.forward             import compute_Af
from utils                   import (L2_norm, bs_call, interpolate_func,
                                     dof_coordinates, get_array, set_array)

set_log_level(LogLevel.ERROR)


# ---------------------------------------------------------------------------
# Paper test runs
# ---------------------------------------------------------------------------
ALL_CASES = ["A", "B", "C", "D", "BD"]
STRIKES   = np.arange(600, 1801, 100)


# ---------------------------------------------------------------------------
# Common experiment parameters
# ---------------------------------------------------------------------------
Y0, Y1     = -3.0, 3.0     # reconstruction log-moneyness domain (M = 3)
T_END      = 1.0
R, Q       = 0.0, 0.0
S0         = 1000.0
N_SPACE    = 199           # reconstruction spatial elements
M_TIME     = 200           # reconstruction implicit-Euler steps

# Data generation grid (wider + finer, to avoid the inverse crime)
Y0_DATA, Y1_DATA = -5.0, 5.0
N_SPACE_DATA     = 4 * N_SPACE
M_TIME_DATA      = 4 * M_TIME


def a_true_field(y):
    r"""True oscillatory variance a^\dagger(y)."""
    return (0.15
            + 0.05 * np.exp(-(y + 0.3) ** 2) * np.sin(2.0 * np.pi * y)
            + 0.05 * erf(20.0 * y))


# Asymptotic boundary variances (erf(+-inf) = +-1, gaussian -> 0):
#   a(-inf) = 0.10  ->  sigma_left  = sqrt(0.20)
#   a(+inf) = 0.20  ->  sigma_right = sqrt(0.40)
SIGMA_LEFT  = np.sqrt(2.0 * 0.10)
SIGMA_RIGHT = np.sqrt(2.0 * 0.20)

# Incomplete-data cases (B, BD)
N_OBS       = 20
OBS_Y_RANGE = (-1.0, 1.0)

# Noisy cases (D, BD): 0.1% relative noise (see project note egger_noise_model)
NOISE_REL   = 0.001
NOISE_SEED  = 0

BETA_EXACT   = 1e-6
BETA_NOISE_K = 1e-2          # beta = BETA_NOISE_K * delta


def _is_incomplete(case):
    return case in ("B", "BD")


def _is_noisy(case):
    return case in ("D", "BD")


# ---------------------------------------------------------------------------
# Boundary conditions  (Black-Scholes values at the boundary strikes)
# ---------------------------------------------------------------------------

def _make_bcs(y_left, y_right):
    r"""Build Dirichlet BC callables psi(tau) for a domain [y_left, y_right]."""
    def left(tau):
        if tau <= 1e-14:
            return float(S0 * max(1.0 - np.exp(y_left), 0.0))
        return float(bs_call(S0, y_left, R, Q, SIGMA_LEFT, tau))

    def right(tau):
        if tau <= 1e-14:
            return 0.0
        return float(bs_call(S0, y_right, R, Q, SIGMA_RIGHT, tau))

    return left, right


# ---------------------------------------------------------------------------
# Priors
# ---------------------------------------------------------------------------

def _prior_func(case):
    """Initial guess / regularisation centre a*(y)."""
    if case == "C":
        return lambda y: np.full_like(y, 0.15)          # bad constant prior
    return lambda y: 0.15 + 0.05 * erf(2.0 * y)         # a*_1 (cases A, B, D, BD)


def _build_obs_mask(V):
    """{0,1} DOF mask selecting the N_OBS observed strikes (cases B, BD)."""
    y = dof_coordinates(V)
    targets = np.linspace(OBS_Y_RANGE[0], OBS_Y_RANGE[1], N_OBS)
    obs_nodes = np.unique([int(np.argmin(np.abs(y - yt))) for yt in targets])
    mask = np.zeros_like(y)
    mask[obs_nodes] = 1.0
    return mask


# ---------------------------------------------------------------------------
# Data generation -- fine, wide-domain forward solve (no inverse crime)
# ---------------------------------------------------------------------------

def _generate_clean_data(V):
    r"""
    Exact option prices u^\dagger(.,T) on the reconstruction nodes of V, produced
    by a forward solve on the finer/wider data grid and interpolated down.
    """
    msh_f = IntervalMesh(N_SPACE_DATA, Y0_DATA, Y1_DATA)
    V_f   = FunctionSpace(msh_f, "Lagrange", 1)

    u0_f   = interpolate_func(V_f, lambda y: S0 * np.maximum(1.0 - np.exp(y), 0.0))
    a_f    = Vol(V_f, interpolate_func(V_f, a_true_field), 0.0, T_END, N=1)
    bcl, bcr = _make_bcs(Y0_DATA, Y1_DATA)

    dt_f = T_END / M_TIME_DATA
    uT_f, _ = compute_Af(a_f, u0_f, dt_f, M_TIME_DATA, V_f, R, Q,
                         Y0_DATA, Y1_DATA, 0.0, bcl, bcr)

    # Interpolate the fine final slice onto the coarse reconstruction nodes.
    yf = dof_coordinates(V_f)
    order = np.argsort(yf)
    yf_s, uf_s = yf[order], get_array(uT_f)[order]

    u_clean = Function(V)
    set_array(u_clean, np.interp(dof_coordinates(V), yf_s, uf_s))
    return u_clean


# ---------------------------------------------------------------------------
# Problem setup
# ---------------------------------------------------------------------------

def setup(V, case):
    """Build (u_0, u_obs, a_init, a_prior_vec, obs_mask, beta, delta) for one case."""
    u_0 = interpolate_func(V, lambda y: S0 * np.maximum(1.0 - np.exp(y), 0.0))

    u_clean  = _generate_clean_data(V)
    obs_mask = _build_obs_mask(V) if _is_incomplete(case) else None

    u_obs = Function(V)
    set_array(u_obs, get_array(u_clean))
    beta, delta = BETA_EXACT, 0.0
    if _is_noisy(case):
        rng = np.random.default_rng(NOISE_SEED)
        u_clean_vec = get_array(u_clean)
        noise = rng.uniform(-NOISE_REL, NOISE_REL, size=u_clean_vec.shape) * u_clean_vec
        set_array(u_obs, get_array(u_obs) + noise)

        noise_fn = Function(V)
        set_array(noise_fn, noise * obs_mask if obs_mask is not None else noise)
        delta = L2_norm(noise_fn)
        beta  = BETA_NOISE_K * delta   # single-beta fallback (unused: schedule below)
        print(f"[case {case}] noise level delta = {delta:.4e}")

    prior = _prior_func(case)
    a_init_fn = interpolate_func(V, prior)
    a_init    = Vol(V, a_init_fn, 0.0, T_END, N=1)
    a_prior_vec = get_array(a_init_fn).copy()

    return u_0, u_obs, a_init, a_prior_vec, obs_mask, beta, delta


# ---------------------------------------------------------------------------
# Evaluation / forward helper
# ---------------------------------------------------------------------------

def _forward(a_vol, u_0, V):
    dt = T_END / M_TIME
    bcl, bcr = _make_bcs(Y0, Y1)
    u_pred, _ = compute_Af(a_vol, u_0, dt, M_TIME, V, R, Q, Y0, Y1, 0.0, bcl, bcr)
    return u_pred


def _eval_at_strikes(a_field, u_pred, V):
    r"""Interpolate (C_recon, a_recon, a_true, sigma_recon) at the tabulated strikes."""
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
        a_tru = float(a_true_field(np.array(y)))
        rows.append((int(K), y, C_rec, a_rec, a_tru,
                     float(np.sqrt(2.0 * max(a_rec, 0.0)))))
    return rows


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _solve(V, case):
    print("\n" + "#" * 74)
    print(f"#  Example 2 -- calibrating case {case}")
    print("#" * 74)
    u_0, u_obs, a_init, a_prior_vec, obs_mask, beta, delta = setup(V, case)
    bcl, bcr = _make_bcs(Y0, Y1)

    if _is_noisy(case):
        # Paper-faithful: decreasing-beta schedule stopped by the Morozov
        # discrepancy principle (Egger & Engl 2005, section 5.3).
        # Warm-started schedule steps converge fast, so 200 iters/step suffices.
        a_cal, _, _ = run_discrepancy(
            u_0, u_obs, a_init, a_prior_vec, delta,
            V, R, Q, Y0, Y1, 0.0, T_END, M_TIME, bcl, bcr,
            n_beta=7, tau_morozov=1.5, max_iter=200, ftol=1e-12, gtol=1e-10,
            obs_mask=obs_mask,
        )
    else:
        # Exact data: a single small fixed beta.
        a_cal, _ = run_tikhonov(
            u_0, u_obs, a_init, a_prior_vec, beta,
            V, R, Q, Y0, Y1, 0.0, T_END, M_TIME, bcl, bcr,
            max_iter=300, ftol=1e-12, gtol=1e-10, obs_mask=obs_mask,
        )
    return a_cal, u_0, u_obs, a_prior_vec, obs_mask


def run(cases=None):
    """Run the requested cases (default: all five) and write the result CSVs."""
    cases = cases or ALL_CASES
    msh = IntervalMesh(N_SPACE, Y0, Y1)
    V   = FunctionSpace(msh, "Lagrange", 1)

    # Reference columns: optimal numerical price with the true field on the grid.
    u_0       = interpolate_func(V, lambda y: S0 * np.maximum(1.0 - np.exp(y), 0.0))
    a_true_fn = interpolate_func(V, a_true_field)
    opt_rows  = _eval_at_strikes(a_true_fn,
                                 _forward(Vol(V, a_true_fn, 0.0, T_END, 1), u_0, V), V)
    C_optimal = {K: C for (K, _, C, _, _, _) in opt_rows}

    results = {}
    recovered = {}          # case -> (a_recon, a_prior, obs_mask) on sorted y, for the combined figure
    y_sorted  = np.sort(dof_coordinates(V))
    for case in cases:
        a_cal, u_0c, u_obs, a_prior_vec, obs_mask = _solve(V, case)
        rows = _eval_at_strikes(a_cal.a[0], _forward(a_cal, u_0c, V), V)
        results[case] = rows
        _print_table(case, rows)
        _plot(a_cal, a_prior_vec, u_0c, u_obs, V, case, obs_mask)

        idx = np.argsort(dof_coordinates(V))
        recovered[case] = (get_array(a_cal.a[0])[idx], a_prior_vec[idx],
                           None if obs_mask is None else obs_mask[idx])

        # Persist incrementally: the CSV merge keeps earlier cases, so a crash or
        # teardown mid-run still leaves valid results for the cases completed so far.
        _save_results(C_optimal, results, [case])
        _save_fields(y_sorted, recovered)

    _plot_combined(y_sorted, recovered)


# ---------------------------------------------------------------------------
# Tables 3 & 4 (Egger & Engl 2005) -- console + CSV
# ---------------------------------------------------------------------------

def _print_table(case, rows):
    print("\n" + "=" * 78)
    print(f"  Example 2, case {case} -- strikes {STRIKES[0]}...{STRIKES[-1]}, S0={S0:.0f}, T=1y")
    print("=" * 78)
    print(f"{'Strike':>7} | {'y':>7} | {'C_recon':>9}"
          f" | {'a_true':>7} {'a_recon':>8} {'sig_recon':>9}")
    print("-" * 78)
    for (K, y, C_rec, a_rec, a_tru, sigma) in rows:
        print(f"{K:>7} | {y:>7.3f} | {C_rec:>9.2f}"
              f" | {a_tru:>7.4f} {a_rec:>8.4f} {sigma:>9.4f}")
    print("=" * 78)


def _save_results(C_optimal, results, cases):
    r"""Merge the per-case Tables 3 (option values) and 4 (parameter a) into CSVs."""
    strikes = [int(K) for K in STRIKES]
    # a_true is case-independent; read it from any computed case.
    a_true_by_K = {int(STRIKES[i]): results[cases[0]][i][4] for i in range(len(STRIKES))}

    _merge_one("egger_example2_option_values.csv",
               ref_name="C_optimal", ref_by_K={K: C_optimal[K] for K in strikes},
               ref_fmt=lambda v: f"{v:.2f}",
               prefix="C", cases=cases,
               case_by_K={c: {int(STRIKES[i]): results[c][i][2]
                              for i in range(len(STRIKES))} for c in cases},
               val_fmt=lambda v: f"{v:.2f}")

    _merge_one("egger_example2_parameter_a.csv",
               ref_name="a_true", ref_by_K=a_true_by_K,
               ref_fmt=lambda v: f"{v:.4f}",
               prefix="a", cases=cases,
               case_by_K={c: {int(STRIKES[i]): results[c][i][3]
                              for i in range(len(STRIKES))} for c in cases},
               val_fmt=lambda v: f"{v:.4f}")

    print("\nSaved/merged egger_example2_option_values.csv and "
          "egger_example2_parameter_a.csv")


def _merge_one(path, ref_name, ref_by_K, ref_fmt, prefix, cases, case_by_K, val_fmt):
    """Write `path`, overwriting only `cases` columns and preserving the rest."""
    strikes = [int(K) for K in STRIKES]

    existing = {}        # colname -> {Strike: formatted str}
    if os.path.exists(path):
        with open(path, newline="") as fh:
            rdr = csv.reader(fh)
            header = next(rdr, [])
            rows = [r for r in rdr if r]
        # Columns after Strike(0) and the reference column(1) are per-case.
        for j, col in enumerate(header):
            if j >= 2:
                existing[col] = {int(r[0]): r[j] for r in rows}

    for c in cases:
        existing[f"{prefix}_{c}"] = {K: val_fmt(case_by_K[c][K]) for K in strikes}

    case_cols = [f"{prefix}_{c}" for c in ALL_CASES if f"{prefix}_{c}" in existing]

    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["Strike", ref_name] + case_cols)
        for K in strikes:
            w.writerow([K, ref_fmt(ref_by_K[K])]
                       + [existing[col][K] for col in case_cols])


def _save_fields(y_sorted, recovered):
    """Save the recovered/prior fields per case for reproducible re-plotting."""
    data = {"y": y_sorted, "a_true": a_true_field(y_sorted)}
    for case, (a_rec, a_prior, mask) in recovered.items():
        data[f"a_{case}"]     = a_rec
        data[f"prior_{case}"] = a_prior
        if mask is not None:
            data[f"mask_{case}"] = mask
    np.savez("egger_example2_fields.npz", **data)
    print("Saved egger_example2_fields.npz")


def _plot_combined(y_sorted, recovered):
    """One multi-panel recovery figure (a(y) per case) for the thesis."""
    cases = [c for c in ALL_CASES if c in recovered]
    a_tru = a_true_field(y_sorted)
    ncol  = 3
    nrow  = int(np.ceil(len(cases) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 3.6 * nrow),
                             sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()

    titles = {"A": "(A) complete, exact", "B": "(B) 20 strikes, exact",
              "C": "(C) bad prior $a^*{=}0.15$", "D": "(D) 0.1% noise",
              "BD": "(BD) 20 strikes + noise"}

    for ax, case in zip(axes, cases):
        a_rec, a_prior, mask = recovered[case]
        ax.axvspan(np.log(600 / S0), np.log(1800 / S0), color='gold', alpha=0.12)
        ax.plot(y_sorted, a_tru,   'r--', lw=1.3, label=r'True $a^\dagger$')
        ax.plot(y_sorted, a_prior, color='grey', ls=':', lw=1.2, label=r'Prior $a^*$')
        ax.plot(y_sorted, a_rec,   'b-',  lw=1.3, label='Recovered')
        if mask is not None:
            on = mask > 0.5
            ax.plot(y_sorted[on], a_rec[on], 'k.', ms=5)
        ax.set_title(titles.get(case, case), fontsize=10)
        ax.grid(True, alpha=0.3)
    for ax in axes[len(cases):]:
        ax.set_visible(False)

    axes[0].legend(fontsize=8, loc='upper left')
    for ax in axes[:len(cases)]:
        ax.set_xlabel('Log-moneyness $y$')
        ax.set_ylabel(r'$a(y)=\frac{1}{2}\sigma^2$')
    fig.suptitle(r'Example 2 -- recovery of the oscillatory $a^\dagger(y)$', fontsize=12)
    fig.tight_layout()
    out = "egger_example2_recovery.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def _plot(a_cal, a_prior_vec, u_0, u_obs, V, case, obs_mask):
    y_coords = dof_coordinates(V)
    idx      = np.argsort(y_coords)
    y        = y_coords[idx]

    a_rec   = get_array(a_cal.a[0])[idx]
    a_prior = a_prior_vec[idx]
    a_tru   = a_true_field(y)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    ax.axvspan(np.log(600 / S0), np.log(1800 / S0), color='gold', alpha=0.15,
               label="Paper's tabulated strikes")
    ax.plot(y, a_tru,   color='r',    ls='--', lw=1.5, label=r'True $a^\dagger(y)$')
    ax.plot(y, a_prior, color='grey', ls=':',  lw=1.5, label=r'Initial guess $a^*$')
    ax.plot(y, a_rec,   color='b',    ls='-',  lw=1.5, label=r'Recovered $a(y)$')
    if obs_mask is not None:
        on = obs_mask[idx] > 0.5
        ax.plot(y[on], a_rec[on], 'k.', ms=8, label='Observed strikes')
    ax.set_xlabel('Log-moneyness $y$')
    ax.set_ylabel(r'$a(y) = \frac{1}{2}\sigma^2$')
    ax.set_title(f'Parameter recovery -- Example 2 (case {case})')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.4)

    dt = T_END / M_TIME
    bcl, bcr = _make_bcs(Y0, Y1)
    u_pred, _ = compute_Af(a_cal, u_0, dt, M_TIME, V, R, Q, Y0, Y1, 0.0, bcl, bcr)
    err = Function(V)
    set_array(err, get_array(u_pred) - get_array(u_obs))
    l2_err = L2_norm(err)
    print(f"Reprice L2 error: {l2_err:.4e}")

    ax = axes[1]
    ax.plot(y, get_array(err)[idx], 'g-', lw=1.5)
    ax.axhline(0, color='k', lw=0.8)
    ax.set_xlabel('Log-moneyness $y$')
    ax.set_ylabel(r'$u_{\mathrm{pred}} - u^{\delta}$')
    ax.set_title(f'Reprice residual  (L2 = {l2_err:.2e})')
    ax.grid(True, alpha=0.4)

    plt.tight_layout()
    out = f'egger_example2_case{case}.png'
    plt.savefig(out, dpi=150)
    print(f"Saved {out}")
    plt.close(fig)


if __name__ == "__main__":
    run()
