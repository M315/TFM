r"""
Example 5 -- the Derman/Bergomi factor of two, checked on the calibrated surface.

Near the money the short-maturity implied volatility is the harmonic average of local
volatility over the log-strike interval from spot to strike (Berestycki, Busca and Florent
2002). Expanding that average gives Derman's rule: the local-vol skew is about twice the
implied-vol skew at the money,

    d sigma_loc / dy |_{y=0}  ~  2 * d sigma_iv / dy |_{y=0}.

This script reuses the surface calibrated in Example 4 (cached in egger_ex4_surface.npz). For
each maturity it estimates both at-the-money skews by a local quadratic fit in a small window
around y = 0, one on the recovered local-vol slice and one on the market implied-vol smile, and
plots the two skews and their ratio against maturity.

Run from the package root (scripts/local_vol_egger):

    conda run -n fenics-legacy python -m examples.example5_skew
"""

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
from dolfin import set_log_level, LogLevel

from examples.example4_surface import load_surface, build_problem, DATA_PATH
from utils import get_array

set_log_level(LogLevel.ERROR)

ATM_WINDOW = 0.05   # fit sigma(y) over |y| <= ATM_WINDOW to read the slope at the money


def atm_skew(y, sigma, window=ATM_WINDOW):
    """
    At-the-money slope d sigma / dy at y = 0 from a local quadratic fit.

    A quadratic sigma ~ c0 + c1 y + c2 y^2 over the near-ATM window is robust to the curvature
    of the smile and to quote noise; the skew is the linear coefficient c1.
    """
    m = np.abs(y) <= window
    if m.sum() < 3:
        m = np.argsort(np.abs(y))[:5]
    c2, c1, c0 = np.polyfit(y[m], sigma[m], 2)
    return c1


def main(out="egger_ex5_skew.png"):
    S0, mats = load_surface(DATA_PATH)
    p = build_problem(S0, mats)

    cache = np.load(os.path.splitext(out.replace("ex5_skew", "ex4_surface"))[0] + ".npz")
    a_cal = p["a_init"]
    a_cal.update(p["V"], cache["a_vec"])

    order = np.argsort(p["y_n"])
    y_s = p["y_n"][order]

    print(f"{'T':>6} {'iv skew':>9} {'loc skew':>9} {'ratio':>7}")
    T, s_iv, s_loc = [], [], []
    for k, ob in enumerate(p["obs"]):
        sig_k = np.sqrt(2.0 * np.maximum(get_array(a_cal.a[k])[order], 0.0))
        loc = atm_skew(y_s, sig_k)
        iv = atm_skew(ob["y_q"], ob["iv"])
        T.append(ob["T"]); s_iv.append(iv); s_loc.append(loc)
        print(f"{ob['T']:>6.3f} {iv:>9.4f} {loc:>9.4f} {loc / iv:>7.2f}")

    T, s_iv, s_loc = np.array(T), np.array(s_iv), np.array(s_loc)
    ratio = s_loc / s_iv

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ax = axes[0]
    ax.plot(T, s_iv, "o-", color="tab:blue", label=r"implied skew  $\partial_y\sigma_{iv}$")
    ax.plot(T, s_loc, "s-", color="crimson", label=r"local skew  $\partial_y\sigma_{loc}$")
    ax.plot(T, 2.0 * s_iv, "--", color="gray",
            label=r"$2\times$ implied skew")
    ax.set_xlabel(r"Maturity  $\tau$ (years)")
    ax.set_ylabel(r"ATM skew  $\partial_y\sigma$")
    ax.set_title("At-the-money skews")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.4)

    ax = axes[1]
    ax.plot(T, ratio, "o-", color="tab:purple")
    ax.axhline(2.0, ls="--", color="gray", label="Derman factor 2")
    ax.set_xlabel(r"Maturity  $\tau$ (years)")
    ax.set_ylabel(r"$\partial_y\sigma_{loc} / \partial_y\sigma_{iv}$")
    ax.set_title("Skew ratio vs the factor of two")
    ax.set_ylim(0, 3)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.4)

    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print(f"\nShort end ratio (tau < 0.3): mean {np.mean(ratio[T < 0.3]):.2f}")
    print(f"Saved {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Example 5 -- Derman/Bergomi factor of two")
    parser.add_argument("--out", default="egger_ex5_skew.png")
    args = parser.parse_args()
    main(out=args.out)
