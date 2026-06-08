"""
Local vol calibration from the SPY implied volatility surface.

Following Lakhal et al., the calibration is done on each consecutive pair
of liquid maturities selected by the impl_vol pipeline:
  - f: call prices at T_i    (built from actual market quotes)
  - g: call prices at T_{i+1} (built from actual market quotes)
  - Calibrate a(y, τ) on [T_i, T_{i+1}] such that A_f(a)(f) ≈ g

Target prices are built from raw market IVs via 1D linear interpolation in y
at each individual maturity.  No 2D smoothed surface is used, so the residual
is compared only against actual quoted strikes.

Time variation is captured because each calibration interval uses different
market data at both endpoints, constraining a independently per interval.
"""

from mpi4py import MPI
import sys
import os

import numpy as np
from scipy.interpolate import interp1d
from dolfinx import fem, mesh

import matplotlib.pyplot as plt
from scipy.stats import norm as _norm

from calibration import Vol, run_calibration
from a_f import compute_Af
from utils import bs_call, plot_multi_maturity_surface


# --- experiment-specific parameters ---
Y0, Y1           = -1.5, 1.5   # PDE domain (K ≈ 0.22·S0 … 4.5·S0); wide for accurate BCs
LIQUID_Y0, LIQUID_Y1 = -0.3, 0.3   # range of liquid quoted strikes used for evaluation/plots
N_SPACE          = 200          # spatial elements
MESH_BETA        = 4.0          # sinh-clustering strength; 0 = uniform, larger = denser at ATM
M_TIME           = 5            # forward/adjoint time steps per interval
N_CAL_SLICES     = 2            # calibration slices per maturity interval
MIN_QUOTES       = 4            # skip intervals with fewer OTM quotes at either end


# ---------------------------------------------------------------------------
# Helpers: build FEM functions from market quotes at one expiration
# ---------------------------------------------------------------------------

def _atm_iv(df_exp, S0: float) -> float:
    """Return the market IV at the strike closest to ATM."""
    df = df_exp.copy()
    df["_y"] = np.log(df["K"] / S0)
    return float(df.loc[df["_y"].abs().idxmin(), "iv"])


def build_call_fn(df_exp, S0: float, r: float, q: float, T: float, V):
    """
    FEM function of call prices at maturity T from raw market quotes.

    BS prices are computed at each quoted IV.  Outside the quoted range
    the nearest quoted IV is used (flat extrapolation) to fill the wider
    domain.  The Dirichlet BCs at Y0/Y1 are set separately by _make_bcs
    using the asymptotic limits, so the interior values here only need to
    be a reasonable initialisation for the calibration.
    """
    df = df_exp.copy()
    df["_y"] = np.log(df["K"] / S0)
    df = df.sort_values("_y")

    y_mkt   = df["_y"].values
    sig_mkt = df["iv"].values
    C_mkt   = bs_call(S0, y_mkt, r, q, sig_mkt, T)

    mask = (y_mkt >= Y0) & (y_mkt <= Y1)
    y_in, sig_in, C_in = y_mkt[mask], sig_mkt[mask], C_mkt[mask]

    # flat-extrapolate IV to the domain boundaries, then convert to price
    sig_left  = float(sig_in[0])  if len(sig_in) else _atm_iv(df_exp, S0)
    sig_right = float(sig_in[-1]) if len(sig_in) else _atm_iv(df_exp, S0)

    C_left  = float(bs_call(S0, Y0, r, q, sig_left,  T))
    C_right = float(bs_call(S0, Y1, r, q, sig_right, T))

    y_use = np.concatenate([[Y0], y_in, [Y1]])
    C_use = np.concatenate([[C_left], C_in, [C_right]])

    spl = interp1d(y_use, C_use, kind="linear", bounds_error=False,
                   fill_value=(C_left, C_right))

    fn = fem.Function(V)
    fn.interpolate(lambda x: np.maximum(spl(x[0]), 0.0))
    return fn


def build_a0_fn(df_exp, S0: float, T: float, V):
    """
    Initial guess a_0(y) = ½ σ²_impl(y, T) from market IVs at maturity T.

    At a single maturity local vol equals implied vol, so this is the natural
    starting point.  The ATM IV is used as fallback outside the quoted range.
    """
    df = df_exp.copy()
    df["_y"] = np.log(df["K"] / S0)
    df = df.sort_values("_y")

    y_mkt    = df["_y"].values
    sig_mkt  = df["iv"].values
    sig_atm  = _atm_iv(df_exp, S0)

    mask  = (y_mkt >= Y0) & (y_mkt <= Y1)
    y_use = np.concatenate([[Y0], y_mkt[mask], [Y1]])
    s_use = np.concatenate([[sig_atm], sig_mkt[mask], [sig_atm]])

    spl = interp1d(y_use, s_use, kind="linear", bounds_error=False,
                   fill_value=sig_atm)

    fn = fem.Function(V)
    fn.interpolate(lambda x: 0.5 * spl(x[0]) ** 2)
    return fn


def _make_bcs(S0: float, r: float, q: float):
    """
    Build (psi_1, psi_2) using the asymptotic boundary conditions.

    With Y0 = -1.5 (K ≈ 0.22·S0) and Y1 = 1.5 (K ≈ 4.5·S0) the domain
    is wide enough that the ±∞ limits are accurate:

      ψ_1(τ) = S0·e^{-qτ} − K_left·e^{-rτ}   (intrinsic value of deep-ITM call)
      ψ_2(τ) = 0                                (deep-OTM call is worthless)

    The parabolic PDE damps any residual error well before it reaches the
    market-data region ([-0.3, 0.3]), which is ~10–30 diffusion lengths away.
    """
    K_left = S0 * np.exp(Y0)

    def psi_1(t, _S=S0, _K=K_left, _r=r, _q=q):
        return float(_S * np.exp(-_q * t) - _K * np.exp(-_r * t))

    def psi_2(t):
        return 0.0

    return psi_1, psi_2


# ---------------------------------------------------------------------------
# Post-calibration repricing evaluation
# ---------------------------------------------------------------------------

def _bs_iv(C_pred: float, S0: float, y: float, r: float, q: float, T: float,
           option_type: str = "call") -> float:
    """Invert BS for IV; returns NaN if inversion fails or price is sub-intrinsic."""
    from scipy.optimize import brentq
    K = S0 * np.exp(y)
    intrinsic = max(S0 * np.exp(-q * T) - K * np.exp(-r * T), 0.0) if option_type == "call" \
        else max(K * np.exp(-r * T) - S0 * np.exp(-q * T), 0.0)
    if C_pred <= intrinsic + 1e-8:
        return np.nan
    try:
        def _diff(sig):
            sq = sig * np.sqrt(T)
            d1 = (-y + (r - q + 0.5 * sig ** 2) * T) / sq
            d2 = d1 - sq
            if option_type == "call":
                return S0 * np.exp(-q * T) * _norm.cdf(d1) - K * np.exp(-r * T) * _norm.cdf(d2) - C_pred
            else:
                return K * np.exp(-r * T) * _norm.cdf(-d2) - S0 * np.exp(-q * T) * _norm.cdf(-d1) - C_pred
        return brentq(_diff, 1e-4, 5.0, xtol=1e-6)
    except Exception:
        return np.nan


def reprice_and_plot(a_cal, f_fn, df_ip1, S0, r, q,
                     T_i, T_ip1, V, psi_1, psi_2, label=""):
    """
    Run the forward Dupire PDE with the calibrated a and compare predicted
    prices against market quotes at T_{i+1} for both OTM calls and OTM puts.

    OTM calls (y >= 0): predicted call price vs. market call mid.
    OTM puts  (y <  0): predicted put price via PCP:
                        P_pred = C_pred - S0·e^{-qT} + K·e^{-rT}
                        compared against market put mid.

    PCP is model-free (holds in any no-arbitrage model), so this is an
    unambiguous consistency check, not a model-dependent one.

    Also back-calculates implied vols from predicted prices so errors can be
    expressed in vol space (more informative than price space for comparison).
    """
    dt = (T_ip1 - T_i) / M_TIME
    u_pred, _ = compute_Af(a_cal, f_fn, dt, M_TIME, V, r, q, Y0, Y1, T_i, psi_1, psi_2)

    # FEM solution as a sorted array over DOF positions
    y_dof = V.mesh.geometry.x[:, 0]
    u_arr = u_pred.x.array.real
    idx   = np.argsort(y_dof)
    y_fem = y_dof[idx]
    u_fem = u_arr[idx]

    df = df_ip1.copy()
    df["_y"] = np.log(df["K"] / S0)
    # keep only liquid strikes — far-OTM quotes are noisy and illiquid
    df = df[(df["_y"] >= LIQUID_Y0) & (df["_y"] <= LIQUID_Y1)].copy()

    # evaluate predicted call price at each market strike
    df["C_pred"] = np.interp(df["_y"].values, y_fem, u_fem)

    K_arr = S0 * np.exp(df["_y"].values)

    # separate into call and put legs
    mask_call = df["option_type"] == "call"
    mask_put  = df["option_type"] == "put"

    # for OTM calls: predicted call = C_pred
    df.loc[mask_call, "price_pred"] = df.loc[mask_call, "C_pred"]
    df.loc[mask_call, "price_mkt"]  = df.loc[mask_call, "mid"]

    # for OTM puts: invert PCP — P_pred = C_pred - S0·e^{-qT} + K·e^{-rT}
    P_pred = df["C_pred"] - S0 * np.exp(-q * T_ip1) + K_arr * np.exp(-r * T_ip1)
    df.loc[mask_put, "price_pred"] = P_pred[mask_put]
    df.loc[mask_put, "price_mkt"]  = df.loc[mask_put, "mid"]

    df["price_err"] = df["price_pred"] - df["price_mkt"]
    df["price_err_pct"] = df["price_err"] / df["price_mkt"].clip(lower=1e-4) * 100

    # back-calculate predicted IVs and compare in vol space
    df["iv_pred"] = [
        _bs_iv(row["C_pred"], S0, row["_y"], r, q, T_ip1,
               option_type="call")  # always call: both sides use call price
        for _, row in df.iterrows()
    ]
    df["iv_err"] = (df["iv_pred"] - df["iv"]) * 100   # in vol-points (%)

    # --- plot ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Repricing  T={T_ip1:.3f} yr  {label}", fontsize=11)

    # panel 1: market vs predicted price
    ax = axes[0]
    for otype, color, marker in [("call", "steelblue", "s"), ("put", "tomato", "o")]:
        sub = df[df["option_type"] == otype]
        if sub.empty:
            continue
        ax.scatter(sub["_y"], sub["price_mkt"],  color=color, marker=marker,
                   s=20, label=f"Market {otype}s", zorder=3)
        ax.scatter(sub["_y"], sub["price_pred"], color=color, marker="+",
                   s=40, label=f"Predicted {otype}s", zorder=4)
    ax.set_xlabel("Log-moneyness  $y$")
    ax.set_ylabel("Option price")
    ax.axvline(0, color="grey", lw=0.7, ls="--", label="ATM")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # panel 2: IV error in vol-points
    ax = axes[1]
    for otype, color, marker in [("call", "steelblue", "s"), ("put", "tomato", "o")]:
        sub = df[df["option_type"] == otype].dropna(subset=["iv_err"])
        if sub.empty:
            continue
        ax.scatter(sub["_y"], sub["iv_err"], color=color, marker=marker,
                   s=20, label=f"{otype}s")
    ax.axhline(0, color="k", lw=0.8)
    ax.axvline(0, color="grey", lw=0.7, ls="--")
    ax.set_xlabel("Log-moneyness  $y$")
    ax.set_ylabel("IV error  (predicted − market, vol-pts %)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fname = f"reprice_T{T_ip1:.3f}.png".replace(".", "p")
    plt.savefig(fname, dpi=130)
    print(f"  Repricing plot saved: {fname}")
    # plt.show()

    # summary
    print(f"  Calls — mean |price err|: {df.loc[mask_call,'price_err'].abs().mean():.4f}"
          f"  mean |IV err|: {df.loc[mask_call,'iv_err'].dropna().abs().mean():.2f} vp")
    print(f"  Puts  — mean |price err|: {df.loc[mask_put,'price_err'].abs().mean():.4f}"
          f"  mean |IV err|: {df.loc[mask_put,'iv_err'].dropna().abs().mean():.2f} vp")


# ---------------------------------------------------------------------------
# Comparison: calibrated local vol vs market implied vol
# ---------------------------------------------------------------------------

def plot_locvol_vs_implvol(intervals, composite_df, V, S0,
                            output_file="locvol_vs_implvol.png"):
    """
    Two-panel 3D comparison of calibrated local vol and market implied vol,
    both restricted to the liquid log-moneyness range [LIQUID_Y0, LIQUID_Y1].

    Left panel:  σ_loc(y, τ) = sqrt(2 a(y, τ)) assembled from all calibrated intervals.
    Right panel: σ_impl(y, τ) as a scatter of raw market quotes (composite OTM surface).

    This lets you see the structural difference between local vol and implied vol
    (local vol is typically more peaked / has a steeper skew) and check that the
    calibrated surface is consistent with the implied vol exercise.
    """
    y_coords = V.mesh.geometry.x[:, 0]
    sort_idx = np.argsort(y_coords)
    y_all    = y_coords[sort_idx]
    mask_liq = (y_all >= LIQUID_Y0) & (y_all <= LIQUID_Y1)
    y_liq    = y_all[mask_liq]

    # --- local vol surface (calibrated a, liquid range only) ---
    T_loc, Z_loc = [], []
    for T_i, T_ip1, vol_obj in intervals:
        n_t = max(4, int((T_ip1 - T_i) * 50))
        for t in np.linspace(T_i, T_ip1, n_t):
            a_vals = vol_obj.get(t).x.array.real[sort_idx][mask_liq]
            T_loc.append(t)
            Z_loc.append(2.0 * np.sqrt(np.maximum(a_vals, 0.0)))
    T_loc = np.array(T_loc)
    Z_loc = np.array(Z_loc)           # shape (n_t_total, n_y_liq)

    # --- implied vol scatter from raw market data ---
    df_mkt = composite_df.copy()
    df_mkt["_y"] = np.log(df_mkt["K"] / S0)
    df_mkt = df_mkt[
        (df_mkt["_y"] >= LIQUID_Y0) & (df_mkt["_y"] <= LIQUID_Y1) &
        (df_mkt["T"]  >= intervals[0][0]) & (df_mkt["T"] <= intervals[-1][1])
    ]

    fig = plt.figure(figsize=(14, 6))
    fig.suptitle("Local vol vs Implied vol  (liquid range)", fontsize=12)

    # left: local vol surface
    ax1 = fig.add_subplot(121, projection="3d")
    T_grid = np.tile(T_loc[:, None], (1, len(y_liq)))
    Y_grid = np.tile(y_liq,          (len(T_loc), 1))
    surf = ax1.plot_surface(Y_grid, T_grid, Z_loc, cmap="plasma",
                            edgecolor="none", alpha=0.9)
    fig.colorbar(surf, ax=ax1, shrink=0.4, aspect=10,
                 label=r"$\sigma_{\mathrm{loc}}$")
    ax1.set_title("Calibrated local vol")
    ax1.set_xlabel("Log-moneyness  $y$")
    ax1.set_ylabel("Maturity  $\\tau$")
    ax1.set_zlabel(r"$\sigma_{\mathrm{loc}}$")

    # right: implied vol scatter
    ax2 = fig.add_subplot(122, projection="3d")
    sc = ax2.scatter(df_mkt["_y"].values, df_mkt["T"].values, df_mkt["iv"].values,
                     c=df_mkt["iv"].values, cmap="plasma", s=25, depthshade=True)
    fig.colorbar(sc, ax=ax2, shrink=0.4, aspect=10,
                 label=r"$\sigma_{\mathrm{impl}}$")
    ax2.set_title("Market implied vol")
    ax2.set_xlabel("Log-moneyness  $y$")
    ax2.set_ylabel("Maturity  $\\tau$")
    ax2.set_zlabel(r"$\sigma_{\mathrm{impl}}$")

    # match z-axis limits so both panels are directly comparable
    z_min = min(Z_loc.min(), df_mkt["iv"].min())
    z_max = max(Z_loc.max(), df_mkt["iv"].max())
    for ax in (ax1, ax2):
        ax.set_zlim(z_min, z_max)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    print(f"Comparison plot saved: {output_file}")
    plt.show()


# ---------------------------------------------------------------------------
# Data loader (imports from the impl_vol pipeline)
# ---------------------------------------------------------------------------

def load_impl_vol_data(ref_date_str: str = "2026-04-30"):
    """
    Run the impl_vol pipeline for the given reference date.

    Returns (composite_df, term_struct, S0):
      composite_df  — OTM composite implied vols for the liquid expirations
      term_struct   — {expiration: (r, q)} from Treasury + PCP calibration
      S0            — SPY spot from the same market snapshot
    """
    from datetime import date
    import pandas as pd

    _impl_vol_dir = os.path.join(os.path.dirname(__file__), "..", "impl_vol")
    sys.path.insert(0, os.path.abspath(_impl_vol_dir))

    from rates import fetch_treasury_curve
    from data import fetch_spy_options
    from main import (
        select_expirations, calibrate_dividend_yield,
        add_implied_vols, build_composite,
    )

    ref_date = date.fromisoformat(ref_date_str)
    r_curve, _, _, _ = fetch_treasury_curve(ref_date)

    calls_raw = fetch_spy_options(ref_date, r_curve, max_T=2.0, option_type="call")
    puts_raw  = fetch_spy_options(ref_date, r_curve, max_T=2.0, option_type="put")

    all_raw = pd.concat([calls_raw, puts_raw], ignore_index=True)
    exp_sel = select_expirations(all_raw)

    calls_raw = calls_raw[calls_raw["expiration"].isin(exp_sel)].copy()
    puts_raw  = puts_raw[puts_raw["expiration"].isin(exp_sel)].copy()

    term_struct = calibrate_dividend_yield(calls_raw, puts_raw)
    for df in [calls_raw, puts_raw]:
        for exp, (r_e, q_e) in term_struct.items():
            df.loc[df["expiration"] == exp, "r"] = r_e
            df.loc[df["expiration"] == exp, "q"] = q_e

    calls_df     = add_implied_vols(calls_raw)
    puts_df      = add_implied_vols(puts_raw)
    composite_df = build_composite(calls_df, puts_df)

    S0 = float(calls_raw["S"].iloc[0])
    n_exp = composite_df["expiration"].nunique()
    print(f"S0={S0:.2f}  ({len(composite_df)} OTM quotes across {n_exp} expirations)")

    return composite_df, term_struct, S0


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run(ref_date_str: str = "2026-04-30"):
    composite_df, term_struct, S0 = load_impl_vol_data(ref_date_str)

    # sorted list of (expiration, T) pairs present in both composite_df and term_struct
    exp_sorted = sorted(
        [
            (exp, float(composite_df.loc[composite_df["expiration"] == exp, "T"].iloc[0]))
            for exp in composite_df["expiration"].unique()
            if exp in term_struct
        ],
        key=lambda x: x[1],
    )

    msh = mesh.create_interval(MPI.COMM_WORLD, N_SPACE, points=(Y0, Y1))

    # Cluster nodes near ATM (y=0) via sinh transformation so the liquid region
    # [LIQUID_Y0, LIQUID_Y1] gets proportionally more resolution than the edges.
    # y(ξ) = y_mid + (Y1-Y0)/2 · sinh(β(ξ-½)) / sinh(β/2),  ξ ∈ [0,1] uniform.
    if MESH_BETA > 0:
        _xi  = (msh.geometry.x[:, 0] - Y0) / (Y1 - Y0)
        _mid = (Y0 + Y1) / 2
        msh.geometry.x[:, 0] = (
            _mid + (Y1 - Y0) / 2 * np.sinh(MESH_BETA * (_xi - 0.5)) / np.sinh(MESH_BETA / 2)
        )
        _coords = np.sort(msh.geometry.x[:, 0])
        _n_liq  = np.sum((_coords >= LIQUID_Y0) & (_coords <= LIQUID_Y1))
        print(f"Mesh: {N_SPACE} elements, sinh β={MESH_BETA} — "
              f"{_n_liq}/{N_SPACE+1} nodes ({100*_n_liq/(N_SPACE+1):.0f}%) "
              f"in liquid range [{LIQUID_Y0}, {LIQUID_Y1}]")

    V   = fem.functionspace(msh, ("Lagrange", 1))

    intervals = []  # list of (T_i, T_{i+1}, a_calibrated)

    for i in range(len(exp_sorted) - 1):
        exp_i,   T_i   = exp_sorted[i]
        exp_ip1, T_ip1 = exp_sorted[i + 1]

        df_i   = composite_df[composite_df["expiration"] == exp_i]
        df_ip1 = composite_df[composite_df["expiration"] == exp_ip1]

        if len(df_i) < MIN_QUOTES or len(df_ip1) < MIN_QUOTES:
            print(f"  Skipping [{exp_i}, {exp_ip1}]: too few quotes ({len(df_i)}, {len(df_ip1)})")
            continue

        # rates for the initial expiration (used to build f and S_eff)
        r_i, q_i = term_struct[exp_i]
        # terminal rates drive the PDE, BCs, and g
        r, q = term_struct[exp_ip1]

        # Forward price at T_i: the best proxy for the underlying's starting
        # value at the beginning of this interval.  Using the spot S0 for all
        # intervals is wrong because ATM drifts to y=(r-q)*T_i instead of y=0.
        S_eff = S0 * np.exp((r_i - q_i) * T_i)

        print(f"\n--- [{exp_i} T={T_i:.3f}] → [{exp_ip1} T={T_ip1:.3f}]"
              f"  r={r:.4f}  q={q:.4f}  S_eff={S_eff:.2f} ---")

        f_fn   = build_call_fn(df_i,   S_eff, r_i, q_i, T_i,   V)
        g_fn   = build_call_fn(df_ip1, S_eff, r,   q,   T_ip1, V)
        a0_fn  = build_a0_fn(df_i, S_eff, T_i, V)
        a_init = Vol(V, a0_fn, T_i, T_ip1, N=N_CAL_SLICES)
        psi_1, psi_2 = _make_bcs(S_eff, r, q)

        a_cal, _ = run_calibration(
            f_fn, g_fn, a_init, V, r, q, Y0, Y1, T_i, T_ip1, M_TIME,
            psi_1, psi_2, method="LM", max_iter=50,
        )
        reprice_and_plot(a_cal, f_fn, df_ip1, S_eff, r, q,
                         T_i, T_ip1, V, psi_1, psi_2, label=exp_ip1)
        intervals.append((T_i, T_ip1, a_cal))

    if not intervals:
        print("No intervals calibrated — check data quality.")
        return

    # full surface over the wide PDE domain [-1.5, 1.5]
    plot_multi_maturity_surface(
        intervals, V,
        title="SPY local volatility surface (full domain)",
        output_file="local_vol_surface_full.png",
    )

    # side-by-side comparison of local vol vs implied vol in the liquid range
    plot_locvol_vs_implvol(
        intervals, composite_df, V, S0,
        output_file="locvol_vs_implvol.png",
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="2026-04-30", help="Reference date YYYY-MM-DD")
    args = parser.parse_args()
    run(args.date)
