r"""
Example 7 -- no-arbitrage check of the calibrated local volatility surface.

The direct Tikhonov calibration reproduces the quotes by solving the forward Dupire equation
with a strictly positive local variance a = 1/2 sigma^2 (the optimizer bounds a >= 1e-8). A
price surface generated this way is arbitrage-free by construction: it is the law of a genuine
diffusion, so unlike the implied-volatility-to-Dupire-formula route it cannot produce the
negative or imaginary local variance that a wiggly input surface would. This script verifies
that the discrete surface really satisfies Bergomi's two conditions:

  * strike (butterfly): the risk-neutral density p(K) = d2C/dK2 stays non-negative;
  * time (calendar):    the total implied variance w = sigma_iv^2 tau is non-decreasing in tau.

It reuses the surface calibrated in Example 4 (cached in egger_ex4_surface.npz).

Run from the package root (scripts/local_vol_egger):

    conda run -n fenics-legacy python -m examples.example7_arbitrage
"""

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
from dolfin import set_log_level, LogLevel

from examples.example4_surface import (load_surface, build_problem, DATA_PATH,
                                       M_TIME, OBS_MONEY_MIN, OBS_MONEY_MAX)
from examples.example6_holdout import implied_vol
from optimization.surface import compute_Af_surface
from utils import get_array

set_log_level(LogLevel.ERROR)

MONEY_LINES = [-0.15, -0.05, 0.0, 0.05, 0.15]   # log-moneyness lines for the calendar check


def density(C, K):
    """Risk-neutral density p(K) = d2C/dK2 on a uniform strike grid."""
    dK = K[1] - K[0]
    return (C[:-2] - 2.0 * C[1:-1] + C[2:]) / dK ** 2


def main(out="egger_ex7_arbitrage.png"):
    S0, mats = load_surface(DATA_PATH)
    p = build_problem(S0, mats)
    rq = {m["expiration"]: (m["r"], m["q"]) for m in mats}

    cache = np.load(os.path.splitext(out.replace("ex7_arbitrage", "ex4_surface"))[0] + ".npz")
    a = p["a_init"]
    a.update(p["V"], cache["a_vec"])

    # (0) Positivity of the local variance.
    a_min = min(get_array(ak).min() for ak in a.a)
    print(f"(0) local variance: min a = {a_min:.3e}  ->  min sigma_loc = {np.sqrt(2*max(a_min,0)):.4f}")

    # Model call surface: one forward solve, whole trajectory.
    _, traj = compute_Af_surface(a, p["u_0"], p["dt"], M_TIME, p["V"],
                                 p["r_of_t"], p["q_of_t"], p["y_0"], p["y_1"],
                                 p["bcl"], p["bcr"])
    order = np.argsort(p["y_n"])
    ys = p["y_n"][order]
    K = S0 * np.exp(ys)

    # Uniform strike grid over the observed band, where the quotes live.
    yb = np.log(np.array([OBS_MONEY_MIN, OBS_MONEY_MAX]))
    Ku = np.linspace(S0 * np.exp(yb[0]), S0 * np.exp(yb[1]), 200)

    print(f"\n(1) butterfly: risk-neutral density on the band")
    print(f"{'expiration':>12}{'T':>7}{'min density':>13}{'#negative':>10}")
    dens_all, w_all, Tj = [], [], []
    for ob in p["obs"]:
        C = get_array(traj[ob["idx"]])[order]
        Cu = np.interp(Ku, K, C)
        d = density(Cu, Ku)
        n_neg = int((d < -1e-9).sum())
        print(f"{ob['exp']:>12}{ob['T']:>7.3f}{d.min():>13.2e}{n_neg:>10}")
        dens_all.append(d)
        Tj.append(ob["T"])

        r, q = rq[ob["exp"]]
        w_all.append([implied_vol(float(np.interp(yv, ys, C)), S0, yv, r, q, ob["T"]) ** 2 * ob["T"]
                      for yv in MONEY_LINES])

    Tj = np.array(Tj)
    w_all = np.array(w_all)

    print(f"\n(2) calendar: total implied variance w = sigma_iv^2 tau vs tau")
    print(f"{'y':>7}{'min dW/dtau':>13}")
    for i, yv in enumerate(MONEY_LINES):
        dW = np.diff(w_all[:, i])
        flag = "" if np.nanmin(dW) >= -1e-6 else "   <-- VIOLATION"
        print(f"{yv:>7.2f}{np.nanmin(dW):>13.2e}{flag}")

    _plot(Ku / S0, dens_all, Tj, w_all, out)


def _plot(mu, dens_all, Tj, w_all, out):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    cmap = plt.cm.viridis(np.linspace(0, 1, len(Tj)))

    ax = axes[0]
    for k, d in enumerate(dens_all):
        ax.plot(mu[1:-1], d, color=cmap[k], lw=1.2, label=rf"$\tau$={Tj[k]:.2f}")
    ax.axhline(0.0, color="k", lw=0.8, ls=":")
    ax.set_xlabel(r"Moneyness  $K/S_0$")
    ax.set_ylabel(r"Risk-neutral density  $\partial^2 C/\partial K^2$")
    ax.set_title("Strike (butterfly): density stays non-negative")
    ax.legend(fontsize=7, ncol=2); ax.grid(True, alpha=0.4)

    ax = axes[1]
    for i, yv in enumerate(MONEY_LINES):
        ax.plot(Tj, w_all[:, i], "o-", lw=1.4, label=rf"$y$={yv:+.2f}")
    ax.set_xlabel(r"Maturity  $\tau$ (years)")
    ax.set_ylabel(r"Total implied variance  $\sigma_{iv}^2\,\tau$")
    ax.set_title("Time (calendar): total variance increases in maturity")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.4)

    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Example 7 -- no-arbitrage check")
    parser.add_argument("--out", default="egger_ex7_arbitrage.png")
    args = parser.parse_args()
    main(out=args.out)
