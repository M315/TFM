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

from calibration import Vol, run_calibration
from utils import bs_call, plot_multi_maturity_surface


# --- experiment-specific parameters ---
Y0, Y1       = -0.3, 0.3   # log-moneyness domain (≈ ±26 % from spot)
N_SPACE      = 100          # spatial elements
M_TIME       = 100          # forward/adjoint time steps per interval
N_CAL_SLICES = 5            # calibration slices per maturity interval
MIN_QUOTES   = 4            # skip intervals with fewer OTM quotes at either end


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

    BS prices are computed at each quoted IV (consistent with the Dupire PDE
    since OTM-put IVs give the same call price via no-arbitrage).  Between
    quoted strikes, prices are linearly interpolated in log-moneyness.
    Outside the quoted range the exact limits are used:
      left  (y → -∞):  C → S0·e^{-qT}   (call with K=0)
      right (y → +∞):  C → 0             (deep-OTM call)
    """
    df = df_exp.copy()
    df["_y"] = np.log(df["K"] / S0)
    df = df.sort_values("_y")

    y_mkt = df["_y"].values
    C_mkt = bs_call(S0, y_mkt, r, q, df["iv"].values, T)

    # keep only strikes within the spatial domain, then pad with exact limits
    mask  = (y_mkt >= Y0) & (y_mkt <= Y1)
    y_use = np.concatenate([[Y0], y_mkt[mask], [Y1]])
    C_use = np.concatenate([[S0 * np.exp(-q * T)], C_mkt[mask], [0.0]])

    spl = interp1d(y_use, C_use, kind="linear", bounds_error=False,
                   fill_value=(S0 * np.exp(-q * T), 0.0))

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


def _make_bcs(S0: float, q: float):
    """Return (psi_1, psi_2) with S0 and q captured by value (closure-safe)."""
    return (lambda t, _S=S0, _q=q: _S * np.exp(-_q * t)), (lambda t: 0.0)


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

        # use the terminal maturity's r, q for the PDE and BC consistency
        r, q = term_struct[exp_ip1]
        print(f"\n--- [{exp_i} T={T_i:.3f}] → [{exp_ip1} T={T_ip1:.3f}]  r={r:.4f}  q={q:.4f} ---")

        f_fn   = build_call_fn(df_i,   S0, r, q, T_i,   V)
        g_fn   = build_call_fn(df_ip1, S0, r, q, T_ip1, V)
        a0_fn  = build_a0_fn(df_i, S0, T_i, V)
        a_init = Vol(V, a0_fn, T_i, T_ip1, N=N_CAL_SLICES)
        psi_1, psi_2 = _make_bcs(S0, q)

        a_cal, _ = run_calibration(
            f_fn, g_fn, a_init, V, r, q, Y0, Y1, T_i, T_ip1, M_TIME,
            psi_1, psi_2, method="L-BFGS-B", max_iter=100,
        )
        intervals.append((T_i, T_ip1, a_cal))

    if not intervals:
        print("No intervals calibrated — check data quality.")
        return

    plot_multi_maturity_surface(
        intervals, V,
        title="SPY local volatility surface",
        output_file="local_vol_surface.png",
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="2026-04-30", help="Reference date YYYY-MM-DD")
    args = parser.parse_args()
    run(args.date)
