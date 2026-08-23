r"""
Example 6 -- out of sample pricing: local volatility vs implied volatility interpolation.

The two descriptions of the smile disagree on how to price a contract that falls between the
quoted maturities. This is the test that separates them. We hold one interior expiration out of
the calibration, then price its quotes two ways:

  * Local volatility: calibrate the surface on the remaining maturities and propagate the Dupire
    equation through the gap. The surface interpolates in time by construction, so sampling the
    forward solution at the held-out maturity is a genuine model price, not an interpolation of
    the smile.
  * Implied volatility: interpolate the total implied variance in maturity at fixed
    log-moneyness (the arbitrage-reasonable interpolation) and read the price off Black-Scholes.

Both are compared to the held-out market prices. The local-vol surface is reused from the
Example 4 machinery; only the maturity set changes.

Run from the package root (scripts/local_vol_egger):

    conda run -n fenics-legacy python -m examples.example6_holdout --holdout 2026-12-18
"""

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from dolfin import Function, set_log_level, LogLevel

from examples.example4_surface import (load_surface, build_problem, DATA_PATH,
                                       M_TIME, OBS_MONEY_MIN, OBS_MONEY_MAX)
from optimization.surface import run as run_surface
from optimization.surface import compute_Af_surface
from utils import bs_call, get_array

set_log_level(LogLevel.ERROR)

DEFAULT_HOLDOUT = "2026-12-18"   # interior expiration in the widest gap (tau ~ 0.63)


def implied_vol(price, S0, y, r, q, tau):
    """Black-Scholes implied volatility of a call price, by bracketed root finding."""
    f = lambda s: bs_call(S0, y, r, q, s, tau) - price
    try:
        return brentq(f, 1e-4, 5.0)
    except ValueError:
        return np.nan


def iv_interp(S0, mats_cal, hold, y_h):
    """
    Price the held-out strikes by interpolating implied variance in maturity.

    For each held-out log-moneyness y, read the implied vol off the two bracketing calibration
    smiles, linearly interpolate the total variance w = sigma^2 tau in maturity, and convert
    back to a Black-Scholes price with the held-out maturity's own rates.
    """
    Ts = np.array([m["T"] for m in mats_cal])
    lo = np.where(Ts < hold["T"])[0][-1]
    hi = np.where(Ts > hold["T"])[0][0]
    mlo, mhi = mats_cal[lo], mats_cal[hi]

    def smile(m):
        y = np.log(m["K"] / S0); o = np.argsort(y)
        return np.interp(y_h, y[o], m["iv"][o])

    w_lo, w_hi = smile(mlo) ** 2 * mlo["T"], smile(mhi) ** 2 * mhi["T"]
    frac = (hold["T"] - mlo["T"]) / (mhi["T"] - mlo["T"])
    iv_h = np.sqrt((w_lo + frac * (w_hi - w_lo)) / hold["T"])
    price = bs_call(S0, y_h, hold["r"], hold["q"], iv_h, hold["T"])
    return iv_h, price, (mlo["T"], mhi["T"])


def run(holdout=DEFAULT_HOLDOUT, beta_y=1.0, beta_tau=1.0, max_iter=200,
        out="egger_ex6_holdout.png", replot=False):
    S0, mats = load_surface(DATA_PATH)

    hold = next(m for m in mats if m["expiration"] == holdout)
    mats_cal = [m for m in mats if m["expiration"] != holdout]
    print("#" * 74)
    print(f"#  Example 6 -- holdout {holdout} (T={hold['T']:.3f}) priced out of sample")
    print("#" * 74)

    p = build_problem(S0, mats_cal)

    cache_npz = os.path.splitext(out)[0] + ".npz"
    if replot:
        a_cal = p["a_init"]
        a_cal.update(p["V"], np.load(cache_npz)["a_vec"])
        print("Replotting from cached slices (no recalibration).")
    else:
        a_cal, _ = run_surface(
            p["u_0"], p["obs"], p["a_init"], p["a_prior_mat"], beta_y, beta_tau,
            p["V"], p["r_of_t"], p["q_of_t"], p["y_0"], p["y_1"],
            p["dt"], M_TIME, p["bcl"], p["bcr"], max_iter=max_iter,
        )
        np.savez(cache_npz, a_vec=a_cal.to_vec())

    _report_and_plot(p, a_cal, S0, mats_cal, hold, out)


def _report_and_plot(p, a_cal, S0, mats_cal, hold, out):
    V = p["V"]
    order = np.argsort(p["y_n"])
    y_s = p["y_n"][order]

    # Held-out strikes, band and liquidity restricted.
    m = hold["K"] / S0
    band = (m >= OBS_MONEY_MIN) & (m <= OBS_MONEY_MAX) & (hold["C"] >= 0.005 * S0)
    y_h = np.log(hold["K"][band] / S0)
    C_mkt = hold["C"][band]
    iv_mkt = hold["iv"][band]
    o = np.argsort(y_h)
    y_h, C_mkt, iv_mkt = y_h[o], C_mkt[o], iv_mkt[o]

    # Local-vol price: one forward solve on the calibration maturities, sampled at the gap.
    _, traj = compute_Af_surface(a_cal, p["u_0"], p["dt"], M_TIME, V,
                                 p["r_of_t"], p["q_of_t"], p["y_0"], p["y_1"],
                                 p["bcl"], p["bcr"])
    n_h = int(round(hold["T"] / p["dt"]))
    u_h = get_array(traj[n_h])[order]
    C_lv = np.interp(y_h, y_s, u_h)
    iv_lv = np.array([implied_vol(c, S0, yy, hold["r"], hold["q"], hold["T"])
                      for c, yy in zip(C_lv, y_h)])

    # Implied-vol interpolation price.
    iv_ip, C_ip, bracket = iv_interp(S0, mats_cal, hold, y_h)

    rel_lv = np.abs(C_lv - C_mkt) / C_mkt
    rel_ip = np.abs(C_ip - C_mkt) / C_mkt

    # Near the money the options carry real value, so the relative error is meaningful there;
    # the deep upside wing divides by near-zero prices and inflates both methods.
    near = np.abs(y_h) <= 0.10

    print(f"\nHeld-out T={hold['T']:.3f} bracketed by "
          f"T={bracket[0]:.3f} and T={bracket[1]:.3f}, {band.sum()} liquid strikes.")
    print(f"{'method':>18} {'med%':>7} {'max%':>7} {'med% |y|<.1':>12} {'max% |y|<.1':>12}")
    print(f"{'local volatility':>18} {np.median(rel_lv):>6.2%} {np.max(rel_lv):>6.2%} "
          f"{np.median(rel_lv[near]):>11.2%} {np.max(rel_lv[near]):>11.2%}")
    print(f"{'IV interpolation':>18} {np.median(rel_ip):>6.2%} {np.max(rel_ip):>6.2%} "
          f"{np.median(rel_ip[near]):>11.2%} {np.max(rel_ip[near]):>11.2%}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ax = axes[0]
    ax.plot(y_h, iv_mkt, "ko", ms=4, label="market IV (held out)")
    ax.plot(y_h, iv_lv, "-", color="crimson", lw=1.8, label="local-vol price -> IV")
    ax.plot(y_h, iv_ip, "--", color="tab:blue", lw=1.8, label="IV interpolation")
    ax.set_xlabel(r"Log-moneyness  $y=\log(K/S_0)$")
    ax.set_ylabel(r"Implied volatility  $\sigma_{iv}$")
    ax.set_title(rf"Held-out smile  $\tau$={hold['T']:.2f}")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.4)

    ax = axes[1]
    ax.plot(y_h, 100 * rel_lv, "-", color="crimson", lw=1.8, label="local volatility")
    ax.plot(y_h, 100 * rel_ip, "--", color="tab:blue", lw=1.8, label="IV interpolation")
    ax.set_xlabel(r"Log-moneyness  $y=\log(K/S_0)$")
    ax.set_ylabel("Relative price error (%)")
    ax.set_title("Out-of-sample reprice error")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.4)

    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Example 6 -- out-of-sample holdout pricing")
    parser.add_argument("--holdout", default=DEFAULT_HOLDOUT)
    parser.add_argument("--beta-y", type=float, default=1.0)
    parser.add_argument("--beta-tau", type=float, default=1.0)
    parser.add_argument("--max-iter", type=int, default=200)
    parser.add_argument("--out", default="egger_ex6_holdout.png")
    parser.add_argument("--replot", action="store_true")
    args = parser.parse_args()
    run(holdout=args.holdout, beta_y=args.beta_y, beta_tau=args.beta_tau,
        max_iter=args.max_iter, out=args.out, replot=args.replot)
