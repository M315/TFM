r"""
Example 3 -- calibrating local volatility from REAL market data.

Unlike Examples 1 & 2 (synthetic true fields), this recovers the local variance
a(y) = 1/2 sigma^2(y) from an observed SPY call-price smile, using the same
Tikhonov / adjoint machinery.  This is a first, single-maturity step: we fit a
time-independent a(y) to ONE expiration's smile.  (The full-surface, multi-
maturity extension is the planned next step.)

Data hand-off
-------------
The market data are prepared by `scripts/impl_vol/export_smiles.py` (conda env
TFM_stable) and dropped in `market_data/spy_<date>_smiles.npz`.  That step reuses
the implied-vol pipeline's r (Treasury curve), q (put-call-parity term structure)
and IV inversion -- the "same trickery" as the IV surface.  Here we only need
numpy to read it, so it runs in the fenics-legacy env alongside dolfin.

Method (mirrors Egger & Engl 2005)
----------------------------------
  * Work in log-moneyness y = log(K/S0) at the ACTUAL snapshot spot S0 (the SPY
    close, ~720, from the impl-vol data) with prices in real dollars.  Beta must
    therefore be tuned for this scale (the L^2 data term scales as S0^2; the toy
    Examples 1 & 2 used S0 = 1000 -- see project note).
  * Initial guess AND H^1 regularisation centre: a*(y) = 1/2 sigma_iv(y)^2, the
    Black-Scholes implied-vol smile interpolated onto the mesh.  Local vol differs
    from implied vol, so the calibration corrects this warm start.
  * Solve the forward PDE on a WIDE y-domain for boundary stability, but only
    trust / fit the NARROW band actually covered by liquid strikes (obs_mask) --
    exactly Egger's "restrict to the observed region" setting.
  * Boundary values: Black-Scholes call price at the domain edges using the wing
    implied vol.
  * Noisy real quotes -> decreasing-beta schedule stopped by the Morozov
    discrepancy principle, with delta from a relative price-noise estimate.
"""

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
from dolfin import (Function, FunctionSpace, IntervalMesh,
                    set_log_level, LogLevel)

from optimization.vol      import Vol
from optimization.tikhonov import run as run_tikhonov
from optimization.tikhonov import run_discrepancy
from pde.forward           import compute_Af
from utils                 import (L2_norm, bs_call, interpolate_func,
                                   dof_coordinates, get_array, set_array)

set_log_level(LogLevel.ERROR)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_DATE = "2026-04-30"
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "market_data",
                         f"spy_{DATA_DATE}_smiles.npz")

T_TARGET  = 0.384

N_SPACE   = 199
M_TIME    = 200

OBS_MONEY_MIN = 0.80
OBS_MONEY_MAX = 1.20
DOMAIN_PAD    = 0.10

NOISE_REL = 0.005

USE_DISCREPANCY = False
FIXED_BETA = 1e-4


# ---------------------------------------------------------------------------
# Data loading (numpy only)
# ---------------------------------------------------------------------------

def load_smile(path, t_target):
    """Load the expiration whose maturity is closest to `t_target`."""
    d = np.load(path, allow_pickle=True)
    S0 = float(d["S0"])
    T_arr = d["T"]
    i = int(np.argmin(np.abs(T_arr - t_target)))
    return {
        "S0":  S0,
        "T":   float(T_arr[i]),
        "r":   float(d["r"][i]),
        "q":   float(d["q"][i]),
        "K":   np.asarray(d["K"][i], dtype=float),
        "C":   np.asarray(d["C"][i], dtype=float),
        "iv":  np.asarray(d["iv"][i], dtype=float),
        "expiration": str(d["kept_expirations"][i]),
    }


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def build_problem(smile):
    r"""
    Assemble everything the calibration needs from one market smile.

    Returns a dict with the FEM space, IC, observations, IV-seeded prior/initial
    a, the observed-band mask, boundary callables and the noise level delta.
    Prices are kept in real dollars at the actual snapshot spot S0.
    """
    S0 = smile["S0"]                              # actual SPY spot at the snapshot
    r, q, T = smile["r"], smile["q"], smile["T"]
    K, C, iv = smile["K"], smile["C"], smile["iv"]

    # Keep only the reliable moneyness band (drop deep, illiquid wings).
    band = (K / S0 >= OBS_MONEY_MIN) & (K / S0 <= OBS_MONEY_MAX)
    K, C, iv = K[band], C[band], iv[band]

    y_q = np.log(K / S0)                          # observed-quote log-moneyness
    order = np.argsort(y_q)
    y_q, C, iv = y_q[order], C[order], iv[order]
    print(f"S0 = {S0:.2f} (snapshot spot).  Observed band: moneyness "
          f"[{OBS_MONEY_MIN}, {OBS_MONEY_MAX}] -> y in [{y_q[0]:.3f}, "
          f"{y_q[-1]:.3f}]  ({len(y_q)} strikes)")

    # Computational domain: genuinely wider than the observed band (BCs far away).
    y_0, y_1 = y_q[0] - DOMAIN_PAD, y_q[-1] + DOMAIN_PAD
    print(f"Computational domain: y in [{y_0:.3f}, {y_1:.3f}]  "
          f"(moneyness [{np.exp(y_0):.2f}, {np.exp(y_1):.2f}])")
    msh = IntervalMesh(N_SPACE, y_0, y_1)
    V   = FunctionSpace(msh, "Lagrange", 1)
    y_n = dof_coordinates(V)

    # Initial condition: call payoff  S0 (1 - e^y)^+   in real dollars.
    u_0 = interpolate_func(V, lambda y: S0 * np.maximum(1.0 - np.exp(y), 0.0))

    # Observations: market call prices interpolated onto the nodes (flat wings).
    u_obs = Function(V)
    set_array(u_obs, np.interp(y_n, y_q, C))

    # Observe only the trusted band; outside it a(y) is prior-driven.
    obs_mask = np.where((y_n >= y_q[0]) & (y_n <= y_q[-1]), 1.0, 0.0)

    # IV-seeded prior / initial guess:  a*(y) = 1/2 sigma_iv(y)^2  (flat wings).
    sigma_seed = np.interp(y_n, y_q, iv)
    a_prior_vec = 0.5 * sigma_seed ** 2
    a_init = Vol(V, interpolate_func(V, lambda y: np.interp(y, y_q, 0.5 * iv ** 2)),
                 0.0, T, N=1)

    # Boundary values: BS call price at the edges with the wing implied vol.
    sig_L, sig_R = float(iv[0]), float(iv[-1])

    def bcl(tau):
        if tau <= 1e-14:
            return float(S0 * max(1.0 - np.exp(y_0), 0.0))
        return float(bs_call(S0, y_0, r, q, sig_L, tau))

    def bcr(tau):
        if tau <= 1e-14:
            return float(S0 * max(1.0 - np.exp(y_1), 0.0))
        return float(bs_call(S0, y_1, r, q, sig_R, tau))

    # Noise level delta from a relative price-noise model, over the observed band.
    noise_fn = Function(V)
    set_array(noise_fn, NOISE_REL * get_array(u_obs) * obs_mask)
    delta = L2_norm(noise_fn)

    return {
        "V": V, "y_n": y_n, "y_0": y_0, "y_1": y_1,
        "u_0": u_0, "u_obs": u_obs, "obs_mask": obs_mask,
        "a_init": a_init, "a_prior_vec": a_prior_vec,
        "bcl": bcl, "bcr": bcr, "delta": delta,
        "r": r, "q": q, "T": T,
        "y_q": y_q, "C_mkt": C, "iv": iv, "S0": S0,
    }


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def run(t_target=T_TARGET, beta=FIXED_BETA, use_discrepancy=USE_DISCREPANCY,
        max_iter=400, out=None):
    smile = load_smile(DATA_PATH, t_target)
    print("#" * 74)
    print(f"#  Example 3 -- market calibration  {smile['expiration']}  "
          f"T={smile['T']:.3f}y  r={smile['r']:.4f}  q={smile['q']:.4f}")
    print(f"#  S0={smile['S0']:.2f}  {len(smile['K'])} strikes  "
          f"K={smile['K'].min():.0f}-{smile['K'].max():.0f}")
    print("#" * 74)

    p = build_problem(smile)
    V, r, q, T = p["V"], p["r"], p["q"], p["T"]
    y_0, y_1 = p["y_0"], p["y_1"]

    if use_discrepancy:
        print(f"delta estimate = {p['delta']:.4e}  (relative noise {NOISE_REL:.1%})")
        a_cal, _, _ = run_discrepancy(
            p["u_0"], p["u_obs"], p["a_init"], p["a_prior_vec"], p["delta"],
            V, r, q, y_0, y_1, 0.0, T, M_TIME, p["bcl"], p["bcr"],
            n_beta=7, tau_morozov=1.5, max_iter=200, ftol=1e-12, gtol=1e-10,
            obs_mask=p["obs_mask"],
        )
    else:
        a_cal, _ = run_tikhonov(
            p["u_0"], p["u_obs"], p["a_init"], p["a_prior_vec"], beta,
            V, r, q, y_0, y_1, 0.0, T, M_TIME, p["bcl"], p["bcr"],
            max_iter=max_iter, ftol=1e-12, gtol=1e-10, obs_mask=p["obs_mask"],
        )

    if out is None:
        out = f"egger_example3_market_beta{beta:.0e}.png"
    _report_and_plot(smile, p, a_cal, beta, out)


def _report_and_plot(smile, p, a_cal, beta, out):
    V, T = p["V"], p["T"]
    dt = T / M_TIME
    u_pred, _ = compute_Af(a_cal, p["u_0"], dt, M_TIME, V, p["r"], p["q"],
                           p["y_0"], p["y_1"], 0.0, p["bcl"], p["bcr"])

    y_n = p["y_n"]
    idx = np.argsort(y_n)
    y_s = y_n[idx]
    a_rec = get_array(a_cal.a[0])[idx]
    sig_rec = np.sqrt(2.0 * np.maximum(a_rec, 0.0))
    sig_seed = np.sqrt(2.0 * p["a_prior_vec"][idx])

    # In-band reprice error (residual masked to the observed band), in dollars.
    res = Function(V)
    set_array(res, (get_array(u_pred) - get_array(p["u_obs"])) * p["obs_mask"])
    l2_band = L2_norm(res)
    print(f"\nIn-band reprice L2 error: {l2_band:.4e}  (price units, $)")

    # Model vs market call prices at the quoted strikes.
    C_model = np.interp(p["y_q"], y_s, get_array(u_pred)[idx])
    C_mkt   = p["C_mkt"]
    rel = np.abs(C_model - C_mkt) / np.maximum(C_mkt, 1e-8)
    print(f"Median | max relative price error at strikes: "
          f"{np.median(rel):.2%} / {np.max(rel):.2%}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    ax.axvspan(p["y_q"][0], p["y_q"][-1], color="gold", alpha=0.15,
               label="Observed strike band")
    ax.plot(y_s, sig_seed, color="grey", ls=":", lw=1.5,
            label=r"IV seed $\sigma_{\mathrm{iv}}(y)$")
    ax.plot(y_s, sig_rec, color="b", ls="-", lw=1.5,
            label=r"Recovered local vol $\sigma(y)$")
    ax.set_xlabel("Log-moneyness  $y = \\log(K/S_0)$")
    ax.set_ylabel(r"Volatility  $\sigma$")
    ax.set_title(f"Local vol vs implied vol -- {smile['expiration']} "
                 rf"(T={p['T']:.2f}y, $\beta$={beta:.0e})")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.4)

    ax = axes[1]
    ax.plot(p["y_q"], C_mkt, "o", color="crimson", ms=4, label="Market call")
    ax.plot(p["y_q"], C_model, "-", color="b", lw=1.5, label="Calibrated model")
    ax.set_xlabel("Log-moneyness  $y = \\log(K/S_0)$")
    ax.set_ylabel(r"Call price  ($\$$)")
    ax.set_title(f"Reprice at strikes  (L2 = {l2_band:.2e} \\$)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.4)

    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Example 3 -- market calibration")
    parser.add_argument("--beta", type=float, default=FIXED_BETA,
                        help="H^1 regularisation weight (fixed-beta mode)")
    parser.add_argument("--t-target", type=float, default=T_TARGET,
                        help="Calibrate the expiration with T closest to this")
    parser.add_argument("--max-iter", type=int, default=400)
    parser.add_argument("--discrepancy", action="store_true",
                        help="Use the Morozov beta-schedule instead of fixed beta")
    parser.add_argument("--out", default=None, help="Output PNG path")
    args = parser.parse_args()
    run(t_target=args.t_target, beta=args.beta,
        use_discrepancy=args.discrepancy, max_iter=args.max_iter, out=args.out)
