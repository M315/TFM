"""
Compute and plot the SPY implied volatility surface and a mid-maturity smile slice.

Three surface views are produced to illustrate why OTM-only quotes are used:
  iv_calls.png      — calls across the full strike range (deep-ITM noise visible)
  iv_puts.png       — puts across the full strike range
  iv_composite.png  — OTM calls (K >= S) + OTM puts (K < S), clean composite
  iv_smile.png      — smile slice of the composite at the mid-range maturity

Usage:
    python main.py                        # uses today's date
    python main.py --date 2026-05-01      # fixed reference date (reproducible)
"""

import argparse
from datetime import date

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy.interpolate import LinearNDInterpolator

from rates import fetch_treasury_curve
from data import fetch_spy_options
from implied_vol import implied_vol as compute_iv


MONEYNESS_MIN = 0.75
MONEYNESS_MAX = 1.25
GRID_SIZE = 60
MAX_T = 1.0   # maximum maturity in years
N_BINS = 10   # number of log-spaced maturity buckets to sample


def add_implied_vols(df: pd.DataFrame) -> pd.DataFrame:
    """Compute and append an 'iv' column; drop points where inversion fails."""
    ivs = [
        compute_iv(row["mid"], row["S"], row["K"], row["T"], row["r"],
                   option_type=row["option_type"], q=row["q"])
        for _, row in df.iterrows()
    ]
    df = df.copy()
    df["iv"] = ivs
    return df[df["iv"].notna() & (df["iv"] > 0.01) & (df["iv"] < 2.0)]


def select_expirations(df: pd.DataFrame, n_bins: int = N_BINS) -> list:
    """
    Return ~n_bins representative expiration strings.

    The T axis is split into log-spaced bins; within each bin the expiration
    with the highest combined open interest is kept.  Applied to the merged
    calls+puts frame so liquidity is measured across both option types.
    """
    exp_stats = (
        df.groupby("expiration")
        .agg(T=("T", "first"), total_oi=("openInterest", "sum"))
        .reset_index()
        .sort_values("T")
    )

    T_min, T_max = exp_stats["T"].min(), exp_stats["T"].max()
    bin_edges = np.logspace(np.log10(T_min), np.log10(T_max), n_bins + 1)

    selected = []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        bucket = exp_stats[(exp_stats["T"] >= lo) & (exp_stats["T"] < hi)]
        if bucket.empty:
            continue
        selected.append(bucket.loc[bucket["total_oi"].idxmax(), "expiration"])

    last = exp_stats.iloc[-1]["expiration"]
    if last not in selected:
        selected.append(last)

    return selected


def build_composite(calls_df: pd.DataFrame, puts_df: pd.DataFrame) -> pd.DataFrame:
    """
    Combine OTM options: puts for K < S, calls for K >= S.
    With properly calibrated r and q the two legs meet smoothly at ATM.
    """
    otm_puts  = puts_df[puts_df["K"] <  puts_df["S"]]
    otm_calls = calls_df[calls_df["K"] >= calls_df["S"]]
    return pd.concat([otm_puts, otm_calls], ignore_index=True)


PCP_MONEYNESS_BAND = 0.05   # use only pairs within ±5 % of spot for q calibration


def calibrate_dividend_yield(
    calls_raw: pd.DataFrame, puts_raw: pd.DataFrame
) -> dict:
    """
    Infer the implied dividend yield q at each expiration from put-call parity,
    keeping r fixed from the Treasury curve.

    Put-call parity:  C - P = S·e^{-qT} - K·e^{-rT}
    Rearranging per matched pair:  F_i = (C_i - P_i)·e^{rT} + K_i

    where F = S·e^{(r-q)T} is the cost-of-carry forward.  The median over
    near-ATM pairs (|K/S - 1| ≤ PCP_MONEYNESS_BAND) is used to estimate F.

    Restricting to near-ATM strikes ensures both legs of each pair are liquid:
    deep ITM calls and deep ITM puts have wide bid-ask spreads and their prices
    are dominated by intrinsic value, making them unreliable for PCP calibration.
    OTM puts (K slightly below S) and OTM calls (K slightly above S) are the
    most actively traded and give the tightest forward estimates.

    If fewer than 2 near-ATM pairs are available, falls back to all matched
    pairs.  Returns {expiration: (r, q_calibrated)}.
    """
    pairs = pd.merge(
        calls_raw[["K", "expiration", "T", "S", "r", "q", "mid"]].rename(columns={"mid": "C"}),
        puts_raw[["K", "expiration", "mid"]].rename(columns={"mid": "P"}),
        on=["K", "expiration"],
    )
    pairs["moneyness"] = pairs["K"] / pairs["S"]

    results = {}
    for exp, grp in pairs.groupby("expiration"):
        T  = grp["T"].iloc[0]
        S  = grp["S"].iloc[0]
        r  = grp["r"].iloc[0]
        q0 = grp["q"].iloc[0]

        # prefer near-ATM pairs; fall back to all pairs if too few
        near_atm = grp[
            (grp["moneyness"] >= 1.0 - PCP_MONEYNESS_BAND)
            & (grp["moneyness"] <= 1.0 + PCP_MONEYNESS_BAND)
        ]
        active = near_atm if len(near_atm) >= 2 else grp

        if len(active) < 2:
            continue

        F_implied = (active["C"] - active["P"]) * np.exp(r * T) + active["K"]
        F = float(F_implied.median())

        if F <= 0:
            results[exp] = (r, q0)
            continue

        q = r - np.log(F / S) / T

        if not (-0.02 <= q <= 0.10):
            results[exp] = (r, q0)
            continue

        results[exp] = (r, q)

    return results


def plot_surface(df: pd.DataFrame, title: str, output_path: str) -> None:
    """Plot an implied volatility surface as a function of moneyness and maturity."""
    df = df.copy()
    df["moneyness"] = df["K"] / df["S"]
    df = df[(df["moneyness"] > MONEYNESS_MIN) & (df["moneyness"] < MONEYNESS_MAX)]

    if len(df) < 4:
        print(f"  Skipping {output_path}: not enough data points.")
        return

    points = df[["moneyness", "T"]].values
    values = df["iv"].values

    m_grid = np.linspace(df["moneyness"].min(), df["moneyness"].max(), GRID_SIZE)
    T_grid = np.linspace(df["T"].min(), df["T"].max(), GRID_SIZE)
    TT, MM = np.meshgrid(T_grid, m_grid)

    interp = LinearNDInterpolator(points, values)
    IV = interp(MM, TT)

    fig = plt.figure(figsize=(11, 7))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot_surface(TT, MM, IV, cmap="viridis", alpha=0.80, linewidth=0)
    ax.scatter(
        df["T"], df["moneyness"], df["iv"],
        color="crimson", s=8, zorder=5, label="Market quotes",
    )

    ax.set_xlabel("Time to maturity  $T$ (years)")
    ax.set_ylabel("Moneyness  $K/S$")
    ax.set_zlabel("Implied volatility  $\\hat{\\sigma}$")
    ax.set_ylim(df["moneyness"].max(), df["moneyness"].min())
    ax.set_title(title)
    ax.view_init(elev=20, azim=225)
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Saved {output_path}")


def plot_yield_curve(
    Ts: np.ndarray,
    rs: np.ndarray,
    curve_date,
    output_path: str = "yield_curve.png",
) -> None:
    """Plot the US Treasury par yield curve used for discounting."""
    r_pct = (np.exp(rs) - 1) * 100  # continuously-compounded → annually-compounded %

    T_fine = np.logspace(np.log10(Ts[0]), np.log10(Ts[-1]), 300)
    from rates import fetch_treasury_curve as _ftc  # reuse interpolator via closure
    interp_fine = np.interp(np.log(T_fine), np.log(Ts), rs)
    r_fine_pct = (np.exp(interp_fine) - 1) * 100

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(T_fine, r_fine_pct, color="steelblue", linewidth=1.8, label="Interpolated curve")
    ax.scatter(Ts, r_pct, color="crimson", s=30, zorder=5, label="Treasury par yields")

    ax.set_xscale("log")
    ax.set_xlabel("Maturity  $T$ (years)")
    ax.set_ylabel("Rate  (% p.a.)")
    ax.set_title(f"US Treasury Par Yield Curve — {curve_date}")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:g}"))
    ax.legend()
    ax.grid(True, which="both", linestyle=":", alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Saved {output_path}")


def plot_smile(df: pd.DataFrame, output_path: str = "iv_smile.png") -> None:
    """
    Plot the composite volatility smile at the expiration closest to the median T.

    OTM puts (K < S) are drawn in one colour and OTM calls (K >= S) in another
    so the two legs of the composite are visually distinct.
    """
    df = df.copy()
    df["moneyness"] = df["K"] / df["S"]
    df = df[(df["moneyness"] > MONEYNESS_MIN) & (df["moneyness"] < MONEYNESS_MAX)]

    median_T = df["T"].median()
    exps = df.groupby("expiration")["T"].first()
    mid_exp = (exps - median_T).abs().idxmin()
    mid_T = exps[mid_exp]

    slice_df = df[df["expiration"] == mid_exp].sort_values("moneyness")
    puts_slice  = slice_df[slice_df["option_type"] == "put"]
    calls_slice = slice_df[slice_df["option_type"] == "call"]

    fig, ax = plt.subplots(figsize=(8, 5))

    if not puts_slice.empty:
        ax.plot(puts_slice["moneyness"], puts_slice["iv"], "o-",
                color="tomato", markersize=5, linewidth=1.5,
                label="OTM puts  ($K < S$)")
    if not calls_slice.empty:
        ax.plot(calls_slice["moneyness"], calls_slice["iv"], "s-",
                color="steelblue", markersize=5, linewidth=1.5,
                label="OTM calls  ($K \\geq S$)")

    ax.axvline(1.0, color="grey", linestyle="--", linewidth=0.8,
               label="ATM  ($K = S$)")

    ax.set_xlabel("Moneyness  $K/S$")
    ax.set_ylabel("Implied volatility  $\\hat{\\sigma}$")
    ax.set_title(
        f"SPY Implied Volatility Smile — Composite, "
        f"$T = {mid_T:.3f}$ yr  ({mid_exp})"
    )
    ax.legend()
    ax.grid(True, linestyle=":", alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Saved {output_path}  (expiration {mid_exp}, T={mid_T:.3f} yr)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SPY implied volatility surface")
    parser.add_argument(
        "--date", default="2026-04-30",
        help="Reference date YYYY-MM-DD (default: 2026-04-30). Results are cached per date.",
    )
    args = parser.parse_args()
    ref_date = date.fromisoformat(args.date)

    print(f"Reference date: {ref_date}")
    print("Fetching US Treasury yield curve...")
    r_curve, curve_date, curve_Ts, curve_rs = fetch_treasury_curve(ref_date)
    print(f"  Using curve as of {curve_date}")
    for T_label, T_val in [("1M", 1/12), ("3M", 3/12), ("6M", 6/12), ("1Y", 1.0)]:
        print(f"    r({T_label}) = {r_curve(T_val):.4f}")

    print("Fetching SPY option chains (T up to 1 yr)...")
    calls_raw = fetch_spy_options(ref_date, r_curve, max_T=MAX_T, option_type="call")
    puts_raw  = fetch_spy_options(ref_date, r_curve, max_T=MAX_T, option_type="put")
    print(f"  Calls: {len(calls_raw)} quotes, {calls_raw['expiration'].nunique()} expirations")
    print(f"  Puts:  {len(puts_raw)} quotes, {puts_raw['expiration'].nunique()} expirations")

    print(f"Selecting {N_BINS} representative expirations by combined liquidity...")
    all_raw = pd.concat([calls_raw, puts_raw], ignore_index=True)
    exp_selection = select_expirations(all_raw, n_bins=N_BINS)
    print(f"  Kept: {exp_selection}")

    calls_raw = calls_raw[calls_raw["expiration"].isin(exp_selection)].copy()
    puts_raw  = puts_raw[puts_raw["expiration"].isin(exp_selection)].copy()

    print("Calibrating q from put-call parity (r fixed from Treasury)...")
    term_struct = calibrate_dividend_yield(calls_raw, puts_raw)
    for exp, (r, q) in sorted(term_struct.items()):
        print(f"  {exp}  T={calls_raw.loc[calls_raw['expiration']==exp,'T'].iloc[0]:.3f}"
              f"  r={r:.4f}  q={q:.4f}")
    for df in [calls_raw, puts_raw]:
        for exp, (r, q) in term_struct.items():
            mask = df["expiration"] == exp
            df.loc[mask, "r"] = r
            df.loc[mask, "q"] = q

    print("Computing implied volatilities...")
    calls_df = add_implied_vols(calls_raw)
    puts_df  = add_implied_vols(puts_raw)
    composite_df = build_composite(calls_df, puts_df)
    print(f"  Calls: {len(calls_df)} valid IVs")
    print(f"  Puts:  {len(puts_df)} valid IVs")
    print(f"  Composite (OTM only): {len(composite_df)} valid IVs")

    print("Plotting yield curve...")
    plot_yield_curve(curve_Ts, curve_rs, curve_date, output_path="yield_curve.png")

    print("Plotting surfaces...")
    plot_surface(calls_df,    "SPY Implied Volatility — Calls",          "iv_calls.png")
    plot_surface(puts_df,     "SPY Implied Volatility — Puts",           "iv_puts.png")
    plot_surface(composite_df,"SPY Implied Volatility Surface (OTM)",    "iv_composite.png")

    print("Plotting smile slice...")
    plot_smile(composite_df, output_path="iv_smile.png")
