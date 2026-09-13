r"""
Example 5 -- the Derman/Bergomi factor of two, checked on the calibrated surface.

Near the money the short-maturity implied volatility is the harmonic average of local
volatility over the log-strike interval from spot to strike (Berestycki, Busca and Florent
2002). Expanding that average gives Derman's rule: the local-vol skew is about twice the
implied-vol skew at the money,

    d sigma_loc / dy |_{y=0}  ~  2 * d sigma_iv / dy |_{y=0}.

This script reuses the surface calibrated in Example 4 (cached in egger_ex4_surface.npz). For
each maturity it estimates both at-the-money skews by a local quadratic fit in a small window
around y = 0, one on the recovered local-vol slice and one on the market implied-vol smile. It
prints the skews and their ratio for every maturity (also as LaTeX table rows) and draws the
rule on a couple of representative maturities: the smile, the local-vol slice, their ATM
tangents, and a line of twice the implied slope.

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

ATM_WINDOW  = 0.05   # fit sigma(y) over |y| <= ATM_WINDOW to read the slope at the money
PLOT_WINDOW = 0.10   # log-moneyness range shown in the slice figure
SLICE_PICKS = (0, 4)  # maturities drawn in the slice figure (index into the kept maturities)


def atm_fit(y, sigma, window=ATM_WINDOW):
    """
    Quadratic fit sigma ~ c0 + c1 y + c2 y^2 over the near-ATM window |y| <= window.

    Returns (c0, c1): the at-the-money level and the at-the-money skew d sigma / dy at y = 0.
    The quadratic is robust to the curvature of the smile and to quote noise.
    """
    m = np.abs(y) <= window
    if m.sum() < 3:
        m = np.argsort(np.abs(y))[:5]
    c2, c1, c0 = np.polyfit(y[m], sigma[m], 2)
    return c0, c1


def plot_slices(p, sig, y_s, fits, out, picks=SLICE_PICKS):
    """
    Draw the rule of two on a couple of maturities: for each one the market smile and the
    recovered local-vol slice near the money, with their ATM tangents and, from the local
    level, a line of twice the implied slope. The rule holds when that line sits on the local
    tangent.
    """
    fig, axes = plt.subplots(1, len(picks), figsize=(11, 4.5))
    # Tangents are drawn only a little past the window they were fitted on.
    yy = np.linspace(-1.4 * ATM_WINDOW, 1.4 * ATM_WINDOW, 50)
    w = np.abs(y_s) <= PLOT_WINDOW

    for ax, k in zip(np.atleast_1d(axes), picks):
        ob = p["obs"][k]
        (iv0, s_iv), (loc0, s_loc) = fits[k]
        band = np.abs(ob["y_q"]) <= PLOT_WINDOW

        ax.plot(ob["y_q"][band], ob["iv"][band], "o", ms=3.5, color="tab:blue", alpha=0.6,
                label=r"market $\sigma_{iv}$")
        ax.plot(y_s[w], sig[k][w], "-", color="crimson", lw=1.8,
                label=r"recovered $\sigma_{loc}$")
        ax.plot(yy, iv0 + s_iv * yy, "--", color="tab:blue", lw=1.0,
                label=rf"implied tangent, slope {s_iv:.2f}")
        ax.plot(yy, loc0 + s_loc * yy, "--", color="crimson", lw=1.0,
                label=rf"local tangent, slope {s_loc:.2f}")
        ax.plot(yy, loc0 + 2.0 * s_iv * yy, ":", color="k", lw=1.6,
                label=r"slope $2\,\partial_y\sigma_{iv}$")

        ax.axvspan(-ATM_WINDOW, ATM_WINDOW, color="gray", alpha=0.12)
        ax.axvline(0.0, color="gray", lw=0.6)
        ax.set_xlim(-PLOT_WINDOW, PLOT_WINDOW)
        lo = min(sig[k][w].min(), ob["iv"][band].min())
        hi = max(sig[k][w].max(), ob["iv"][band].max())
        ax.set_ylim(lo - 0.05 * (hi - lo), hi + 0.05 * (hi - lo))
        ax.set_xlabel(r"Log-moneyness  $y=\log(K/S_0)$")
        ax.set_ylabel(r"Volatility  $\sigma$")
        ax.set_title(rf"$\tau$ = {ob['T']:.3f} ({ob['exp']}),  ratio {s_loc / s_iv:.2f}")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.4)

    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close(fig)


def main(out="egger_ex5_skew.png"):
    S0, mats = load_surface(DATA_PATH)
    p = build_problem(S0, mats)

    cache = np.load(os.path.splitext(out.replace("ex5_skew", "ex4_surface"))[0] + ".npz")
    a_cal = p["a_init"]
    a_cal.update(p["V"], cache["a_vec"])

    order = np.argsort(p["y_n"])
    y_s = p["y_n"][order]

    # Per maturity: the recovered slice, its ATM fit and the market smile's ATM fit.
    sig, fits, T, s_iv, s_loc = [], [], [], [], []
    for k, ob in enumerate(p["obs"]):
        sig_k = np.sqrt(2.0 * np.maximum(get_array(a_cal.a[k])[order], 0.0))
        loc_fit = atm_fit(y_s, sig_k)
        iv_fit = atm_fit(ob["y_q"], ob["iv"])
        sig.append(sig_k); fits.append((iv_fit, loc_fit))
        T.append(ob["T"]); s_iv.append(iv_fit[1]); s_loc.append(loc_fit[1])

    T, s_iv, s_loc = np.array(T), np.array(s_iv), np.array(s_loc)
    ratio = s_loc / s_iv

    print(f"\n{'expiration':>12} {'T':>6} {'iv skew':>9} {'loc skew':>9} {'ratio':>7}")
    for k, ob in enumerate(p["obs"]):
        print(f"{ob['exp']:>12} {T[k]:>6.3f} {s_iv[k]:>9.4f} {s_loc[k]:>9.4f} {ratio[k]:>7.2f}")

    # LaTeX rows for the table in the thesis.
    print("\n% LaTeX table rows")
    for k, ob in enumerate(p["obs"]):
        print(f"    {ob['exp']} & {T[k]:.3f} & {s_iv[k]:.3f} & {s_loc[k]:.3f} "
              f"& {ratio[k]:.2f} \\\\")

    plot_slices(p, sig, y_s, fits, out)
    print(f"\nShort end ratio (tau < 0.3): mean {np.mean(ratio[T < 0.3]):.2f}")
    print(f"Saved {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Example 5 -- Derman/Bergomi factor of two")
    parser.add_argument("--out", default="egger_ex5_skew.png")
    args = parser.parse_args()
    main(out=args.out)
