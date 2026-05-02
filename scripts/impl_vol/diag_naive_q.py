"""
Diagnostic: ATM discontinuity across three dividend-yield assumptions.

Generates annex_atm_gap.png — three-panel figure at a mid-range maturity
(T ≈ 0.63 yr, largest absolute q divergence after T*Δq weighting).

Each panel shows both put-implied and call-implied volatility at EVERY
available strike (not just OTM), so that the consistency (or lack thereof)
between the two option types is immediately visible:

  Panel 1 — q = 0       (no dividend correction)
  Panel 2 — q = fixed annual yield from yfinance (~1.14 %)
  Panel 3 — q calibrated per maturity from put-call parity

With the wrong q the two curves diverge away from ATM; with the calibrated q
they sit on top of each other — a clean composite is then constructed by
taking OTM puts for K < S and OTM calls for K ≥ S.

Uses cached data for 2026-04-30; no network access required.
Run from scripts/impl_vol/:
    conda run -n TFM_stable python diag_naive_q.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datetime import date

from rates import fetch_treasury_curve
from data import fetch_spy_options
from implied_vol import implied_vol as compute_iv
from main import calibrate_dividend_yield, select_expirations, N_BINS

REF_DATE  = date(2026, 4, 30)
MAX_T     = 1.0
TARGET_EXP = "2026-12-18"   # T ≈ 0.63 yr — large T*Δq product

MONEYNESS_MIN = 0.78
MONEYNESS_MAX = 1.22


def add_ivs(df: pd.DataFrame) -> pd.DataFrame:
    ivs = [
        compute_iv(
            row["mid"], row["S"], row["K"], row["T"], row["r"],
            option_type=row["option_type"], q=row["q"],
        )
        for _, row in df.iterrows()
    ]
    df = df.copy()
    df["iv"] = ivs
    return df[df["iv"].notna() & (df["iv"] > 0.01) & (df["iv"] < 2.0)]


def smile_at_exp(
    calls_raw: pd.DataFrame,
    puts_raw: pd.DataFrame,
    q_override,
    target_exp: str,
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """Compute full-strike IV smile for calls and puts at target_exp.

    q_override: float → applied to all rows; dict → {exp: (r, q)} from PCP.
    Returns (calls_slice, puts_slice, T).
    """
    calls = calls_raw.copy()
    puts  = puts_raw.copy()
    if isinstance(q_override, dict):
        for df in (calls, puts):
            for exp, (r, q) in q_override.items():
                mask = df["expiration"] == exp
                df.loc[mask, "r"] = r
                df.loc[mask, "q"] = q
    else:
        for df in (calls, puts):
            df["q"] = q_override

    calls_iv = add_ivs(calls)
    puts_iv  = add_ivs(puts)

    def slice_exp(df):
        df = df[df["expiration"] == target_exp].copy()
        df["moneyness"] = df["K"] / df["S"]
        df = df[(df["moneyness"] > MONEYNESS_MIN) & (df["moneyness"] < MONEYNESS_MAX)]
        return df.sort_values("moneyness")

    c_sl = slice_exp(calls_iv)
    p_sl = slice_exp(puts_iv)
    T = float(c_sl["T"].iloc[0]) if not c_sl.empty else float("nan")
    return c_sl, p_sl, T


def main() -> None:
    print(f"Reference date: {REF_DATE}")

    r_curve, *_ = fetch_treasury_curve(REF_DATE)

    calls_raw = fetch_spy_options(REF_DATE, r_curve, max_T=MAX_T, option_type="call")
    puts_raw  = fetch_spy_options(REF_DATE, r_curve, max_T=MAX_T, option_type="put")

    all_raw = pd.concat([calls_raw, puts_raw], ignore_index=True)
    exp_sel = select_expirations(all_raw, n_bins=N_BINS)
    calls_raw = calls_raw[calls_raw["expiration"].isin(exp_sel)].copy()
    puts_raw  = puts_raw[puts_raw["expiration"].isin(exp_sel)].copy()

    q_annual  = float(calls_raw["q"].iloc[0])
    term_struct = calibrate_dividend_yield(calls_raw, puts_raw)
    q_cal = term_struct[TARGET_EXP][1]
    print(f"  {TARGET_EXP}: q_annual={q_annual:.4f}  q_cal={q_cal:.4f}")

    c0, p0, T = smile_at_exp(calls_raw, puts_raw, 0.0,         TARGET_EXP)
    c1, p1, _ = smile_at_exp(calls_raw, puts_raw, q_annual,    TARGET_EXP)
    c2, p2, _ = smile_at_exp(calls_raw, puts_raw, term_struct, TARGET_EXP)

    # ---- Figure ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

    stages = [
        (axes[0], c0, p0, f"Stage 1 — $q = 0$"),
        (axes[1], c1, p1, f"Stage 2 — fixed $q = {q_annual:.3f}$"),
        (axes[2], c2, p2,  "Stage 3 — PCP-calibrated $q$"),
    ]

    for ax, c_sl, p_sl, title in stages:
        if not c_sl.empty:
            ax.plot(c_sl["moneyness"], c_sl["iv"], "-",
                    color="steelblue", linewidth=1.5, label="Calls (all strikes)")
        if not p_sl.empty:
            ax.plot(p_sl["moneyness"], p_sl["iv"], "--",
                    color="tomato", linewidth=1.5, label="Puts (all strikes)")
        ax.axvline(1.0, color="grey", linestyle=":", linewidth=0.9, label="ATM ($K=S$)")
        ax.set_xlabel("Moneyness  $K/S$")
        ax.set_title(f"$T={T:.2f}$ yr — {title}", fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, linestyle=":", alpha=0.5)

    axes[0].set_ylabel("Implied volatility  $\\hat{\\sigma}$")
    fig.suptitle(
        f"SPY implied vol — call vs. put consistency across dividend yield assumptions  ({TARGET_EXP})",
        fontsize=10, y=1.01,
    )

    plt.tight_layout()
    out = "annex_atm_gap.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
