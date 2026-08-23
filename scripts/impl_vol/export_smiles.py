"""
Export per-expiration CALL-price smiles for local-volatility calibration.

This bridges the implied-vol pipeline (this package, conda env TFM_stable) and the
Egger & Engl Dupire calibration (scripts/local_vol_egger, conda env fenics-legacy,
which has no pandas).  It reuses exactly the same r / q / IV "trickery" as the IV
surface:

  * r  : US Treasury par curve, per maturity              (rates.py)
  * q  : per-expiration dividend yield from put-call parity (main.calibrate_dividend_yield)
  * IV : Black-Scholes-Merton inversion                    (implied_vol.py)

For each expiration it builds a clean OTM *call-price* smile C(K):

  * K >= S : market call mid  (OTM calls)
  * K <  S : OTM put mid converted to a call by put-call parity
             C = P + S e^{-qT} - K e^{-rT}

so every price comes from the liquid leg while still describing call prices, which
is what the Dupire forward PDE solves for.  The result is written as a numpy-only
.npz that the calibration example loads with `np.load(..., allow_pickle=True)`.

Usage:
    python export_smiles.py                     # cached 2026-04-30 snapshot
    python export_smiles.py --date 2026-04-30
"""

import argparse
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from rates import fetch_treasury_curve
from data import fetch_spy_options
from main import (
    calibrate_dividend_yield,
    add_implied_vols,
    select_expirations,
    N_BINS,
    MAX_T,
)

# Written into scripts/local_vol_egger/ so the calibration example can load it
# with a relative path.
OUT_DIR = Path(__file__).resolve().parents[1] / "local_vol_egger" / "market_data"


def _assign_term_structure(df, term_struct):
    """Overwrite r, q per expiration from the PCP calibration (in place)."""
    for exp, (r, q) in term_struct.items():
        mask = df["expiration"] == exp
        df.loc[mask, "r"] = r
        df.loc[mask, "q"] = q


def _call_smile(calls_g, puts_g, S0):
    """
    Build one OTM-composite call-price smile for a single expiration.

    Returns sorted arrays (K, C, iv).  OTM calls (K >= S) use the market call;
    OTM puts (K < S) are converted to calls via put-call parity.
    """
    T = calls_g["T"].iloc[0]
    r = calls_g["r"].iloc[0]
    q = calls_g["q"].iloc[0]

    otm_calls = calls_g[calls_g["K"] >= S0][["K", "mid", "iv"]].copy()
    otm_calls = otm_calls.rename(columns={"mid": "C"})

    otm_puts = puts_g[puts_g["K"] < S0][["K", "mid", "iv"]].copy()
    # Put -> call via put-call parity  C = P + S e^{-qT} - K e^{-rT}
    otm_puts["C"] = (
        otm_puts["mid"]
        + S0 * np.exp(-q * T)
        - otm_puts["K"] * np.exp(-r * T)
    )
    otm_puts = otm_puts[["K", "C", "iv"]]

    smile = (
        pd.concat([otm_puts, otm_calls], ignore_index=True)
        .drop_duplicates(subset="K")
        .sort_values("K")
    )
    # Keep only economically sensible positive call prices.
    smile = smile[smile["C"] > 0]
    return smile["K"].to_numpy(), smile["C"].to_numpy(), smile["iv"].to_numpy()


def export(ref_date: date) -> Path:
    """Run the impl-vol pipeline and write the per-expiration smile .npz."""
    print(f"Reference date: {ref_date}")
    r_curve, curve_date, _, _ = fetch_treasury_curve(ref_date)
    print(f"  Treasury curve as of {curve_date}")

    calls_raw = fetch_spy_options(ref_date, r_curve, max_T=MAX_T, option_type="call")
    puts_raw = fetch_spy_options(ref_date, r_curve, max_T=MAX_T, option_type="put")

    all_raw = pd.concat([calls_raw, puts_raw], ignore_index=True)
    exp_selection = select_expirations(all_raw, n_bins=N_BINS)
    calls_raw = calls_raw[calls_raw["expiration"].isin(exp_selection)].copy()
    puts_raw = puts_raw[puts_raw["expiration"].isin(exp_selection)].copy()

    term_struct = calibrate_dividend_yield(calls_raw, puts_raw)
    _assign_term_structure(calls_raw, term_struct)
    _assign_term_structure(puts_raw, term_struct)

    calls_df = add_implied_vols(calls_raw)
    puts_df = add_implied_vols(puts_raw)

    S0 = float(calls_df["S"].iloc[0])

    # Order expirations by maturity.
    exps = (
        calls_df.groupby("expiration")["T"].first().sort_values().index.tolist()
    )

    out = {"S0": np.array(S0), "expirations": np.array(exps, dtype=object)}
    T_list, r_list, q_list = [], [], []
    K_list, C_list, iv_list = [], [], []

    print(f"\n{'expiration':>12} {'T':>6} {'r':>7} {'q':>7} {'nK':>4} "
          f"{'K range':>14}")
    for exp in exps:
        cg = calls_df[calls_df["expiration"] == exp]
        pg = puts_df[puts_df["expiration"] == exp]
        if cg.empty or pg.empty:
            continue
        K, C, iv = _call_smile(cg, pg, S0)
        if len(K) < 5:
            continue
        T = float(cg["T"].iloc[0])
        r = float(cg["r"].iloc[0])
        q = float(cg["q"].iloc[0])
        T_list.append(T); r_list.append(r); q_list.append(q)
        K_list.append(K); C_list.append(C); iv_list.append(iv)
        print(f"{exp:>12} {T:>6.3f} {r:>7.4f} {q:>7.4f} {len(K):>4} "
              f"{K.min():>6.0f}-{K.max():<6.0f}")

    out["T"] = np.array(T_list)
    out["r"] = np.array(r_list)
    out["q"] = np.array(q_list)
    out["K"] = np.array(K_list, dtype=object)
    out["C"] = np.array(C_list, dtype=object)
    out["iv"] = np.array(iv_list, dtype=object)
    out["kept_expirations"] = np.array(
        [exps[i] for i in range(len(T_list))], dtype=object
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"spy_{ref_date}_smiles.npz"
    np.savez(out_path, **out)
    print(f"\nSaved {out_path}  (S0={S0:.2f}, {len(T_list)} expirations)")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export call-price smiles")
    parser.add_argument("--date", default="2026-04-30",
                        help="Reference date YYYY-MM-DD (default: 2026-04-30)")
    args = parser.parse_args()
    export(date.fromisoformat(args.date))
