r"""
Example 4 -- SURFACE calibration of local volatility from real market data.

Extends Example 3 from one expiration to the whole term structure, recovering a time
dependent a(y,tau) = 1/2 sigma^2(y,tau). The surface is held as one spatial slice per
observed maturity (optimization/vol.py) and calibrated with the space-time Tikhonov
functional of optimization/surface.py: an H1 penalty in y per slice and an L2 penalty on the
jump between consecutive slices. This goes beyond Egger & Engl's proven (time-independent)
setting, so it is a numerical extension rather than a covered case.

Data hand-off matches Example 3: scripts/impl_vol/export_smiles.py (conda env TFM_stable)
writes market_data/spy_<date>_smiles.npz with r (Treasury curve), q (put-call parity) and the
inverted IV. Calibration runs in fenics-legacy alongside dolfin.

Run from the package root (scripts/local_vol_egger):

    conda run -n fenics-legacy python -m examples.example4_surface --beta-y 1e-1 --beta-tau 1e0

By default the solve keeps the explicit zero-order term of the Dupire equation. `--discount`
switches to Egger & Engl's discounted variable u = e^{\int_0^tau q} C, which cancels that term
and leaves -u_tau + a(u_yy - u_y) + (q-r)u_y = 0; data and boundary values are scaled by the
same factor and the scaling is undone when repricing, so the two settings differ only in
discretization error. Examples 1-2 have q = 0, where the change of variable is the identity.
Run examples/example8_discounting.py to reproduce the comparison of the two.
"""

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers the 3d projection)
from dolfin import (Function, FunctionSpace, IntervalMesh,
                    set_log_level, LogLevel)

from optimization.vol     import Vol
from optimization.surface import run as run_surface
from optimization.surface import compute_Af_surface
from utils                import (L2_norm, bs_call, interpolate_func,
                                  dof_coordinates, get_array, set_array)

set_log_level(LogLevel.ERROR)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_DATE = "2026-04-30"
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "market_data",
                         f"spy_{DATA_DATE}_smiles.npz")

N_SPACE = 199
M_TIME  = 300

OBS_MONEY_MIN = 0.80
OBS_MONEY_MAX = 1.20
DOMAIN_PAD    = 0.25

# Drop expirations with too few liquid strikes: they are under-determined and, when they sit a
# few days from a well-populated expiration, add a redundant thin slice that just relaxes to a
# noisy prior. On this data set it removes one sparse near-duplicate quarterly at the long end.
MIN_BAND_STRIKES = 25

FIXED_BETA_Y   = 1e-1
FIXED_BETA_TAU = 1e0

DISCOUNTED = False   # True solves for u = e^{\int_0^tau q} C instead of C


# ---------------------------------------------------------------------------
# Data loading (numpy only)
# ---------------------------------------------------------------------------

def load_surface(path):
    """Load every exported expiration, sorted by maturity."""
    d = np.load(path, allow_pickle=True)
    S0 = float(d["S0"])
    T = np.asarray(d["T"], dtype=float)
    order = np.argsort(T)
    mats = []
    for i in order:
        mats.append({
            "T":  float(T[i]),
            "r":  float(d["r"][i]),
            "q":  float(d["q"][i]),
            "K":  np.asarray(d["K"][i], dtype=float),
            "C":  np.asarray(d["C"][i], dtype=float),
            "iv": np.asarray(d["iv"][i], dtype=float),
            "expiration": str(d["kept_expirations"][i]),
        })
    return S0, mats


# ---------------------------------------------------------------------------
# Term structure: piecewise-constant forward rates from the zero rates
# ---------------------------------------------------------------------------

def _forward_rate_fn(T_arr, z_arr):
    r"""
    Build a piecewise-constant instantaneous forward-rate callable f(tau) from the zero
    rates z_j at maturities T_j. The forward rate on (T_{j-1}, T_j] is
    (z_j T_j - z_{j-1} T_{j-1}) / (T_j - T_{j-1}), with T_0 = 0, so integrating f up to any
    T_j gives back the zero rate z_j used to price that maturity's smile. This is the
    forward-rate analogue of Derman's local-vol construction, and matches Egger's
    time-dependent b(tau) = q(tau) - r(tau) (eq. 2.6).
    """
    T_prev = np.concatenate([[0.0], T_arr[:-1]])
    z_prev = np.concatenate([[0.0], z_arr[:-1]])
    fwd = np.where(T_arr > T_prev,
                   (z_arr * T_arr - z_prev * T_prev) / np.maximum(T_arr - T_prev, 1e-12),
                   z_arr)

    def f(tau):
        j = int(np.searchsorted(T_arr, tau, side="left"))
        return float(fwd[min(j, len(fwd) - 1)])

    return f


def _integrated_rate_fn(T_arr, z_arr):
    r"""
    Cumulative rate Q(tau) = \int_0^tau f(s) ds for the forward curve of _forward_rate_fn.

    By construction Q(T_j) = z_j T_j, so e^{Q(tau)} is the reciprocal of the discount factor
    used to price maturity T_j. Applied to q it is the factor of the change of variable
    u = e^{\int_0^tau q} C.
    """
    f = _forward_rate_fn(T_arr, z_arr)
    T_prev = np.concatenate([[0.0], T_arr[:-1]])
    Z_prev = np.concatenate([[0.0], (z_arr * T_arr)[:-1]])

    def Q(tau):
        j = min(int(np.searchsorted(T_arr, tau, side="left")), len(T_arr) - 1)
        return float(Z_prev[j] + f(tau) * (tau - T_prev[j]))

    return Q


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

# Short-time (Berestycki, Busca, Florent 2002) inversion of the implied smile into a local-vol
# prior. The implied vol is the harmonic mean of local vol from spot to strike, so to leading
# order sigma_loc(y) = sigma_iv(y) / (1 - y sigma_iv'(y)/sigma_iv(y)), which reproduces Derman's
# factor of two at the money (sigma_loc' = 2 sigma_iv'). Seeding the Tikhonov prior with this,
# rather than the raw 1/2 sigma_iv^2, stops the H1 penalty from pulling the recovered skew back
# toward the twice-too-flat implied skew. See the factor-of-two subsection (sec:factor_two).
PRIOR_DEN_FLOOR = 0.35   # guard the inversion denominator against the steep deep-downside wing


def _local_vol_prior(y_q, iv):
    r"""
    Factor-of-two-consistent local-vol seed a*(y) = 1/2 sigma_loc^2 from an implied smile.

    A quadratic fit smooths the band smile and supplies an analytic slope; the BBF inversion
    then maps it to a local-vol shape whose at-the-money skew is twice the implied skew.
    Returns a* sampled on the input log-moneyness y_q.
    """
    c2, c1, c0 = np.polyfit(y_q, iv, 2)
    iv_fit = c0 + c1 * y_q + c2 * y_q ** 2
    ivp    = c1 + 2.0 * c2 * y_q
    den = np.maximum(1.0 - y_q * ivp / iv_fit, PRIOR_DEN_FLOOR)
    sig_loc = iv_fit / den
    return 0.5 * sig_loc ** 2


def build_problem(S0, mats, discounted=DISCOUNTED):
    r"""
    Assemble the surface calibration problem from all maturities.

    One time slice per maturity, with slice boundaries snapped to the maturities so that
    maturity tau_j sits on the boundary between slice j-1 and slice j: its data then constrain
    the local vol on (tau_{j-1}, tau_j], the causal bootstrap. Returns the FEM space, the
    per-maturity observation list, the per-slice prior a* (that maturity's IV seed), the
    Vol initialised on the aligned slices, and the term-structure and boundary callables.

    With `discounted` the data and the boundary values are multiplied by e^{Q_q(tau)} so the
    solver works with u = e^{\int_0^tau q} C; each observation carries its own 'scale' so the
    repricing can undo it exactly.
    """
    # Keep only expirations with enough liquid strikes in the band (see MIN_BAND_STRIKES); the
    # rest of the setup, including the term structure, uses this kept set.
    kept = []
    for m in mats:
        n_band = int(((m["K"] / S0 >= OBS_MONEY_MIN) & (m["K"] / S0 <= OBS_MONEY_MAX)).sum())
        if n_band < MIN_BAND_STRIKES:
            print(f"  skip {m['expiration']} (T={m['T']:.3f}): "
                  f"{n_band} band strikes < {MIN_BAND_STRIKES}")
            continue
        kept.append(m)
    mats = kept

    T_arr = np.array([m["T"] for m in mats])
    r_arr = np.array([m["r"] for m in mats])
    q_arr = np.array([m["q"] for m in mats])
    T_max = T_arr[-1]

    # Change of variable u = e^{Q_q(tau)} C (identity when not discounted).
    Q_q = _integrated_rate_fn(T_arr, q_arr) if discounted else (lambda tau: 0.0)

    # Restrict every kept maturity to its trusted band; collect log-moneyness extents.
    per = []
    for m in mats:
        band = (m["K"] / S0 >= OBS_MONEY_MIN) & (m["K"] / S0 <= OBS_MONEY_MAX)
        y_q = np.log(m["K"][band] / S0)
        o = np.argsort(y_q)
        per.append({"y_q": y_q[o], "C": m["C"][band][o], "iv": m["iv"][band][o],
                    "T": m["T"], "exp": m["expiration"]})

    y_lo = min(p["y_q"][0] for p in per)
    y_hi = max(p["y_q"][-1] for p in per)
    y_0, y_1 = y_lo - DOMAIN_PAD, y_hi + DOMAIN_PAD
    print(f"S0 = {S0:.2f}.  {len(per)} maturities T in "
          f"[{T_arr[0]:.3f}, {T_max:.3f}].  Band union y in [{y_lo:.3f}, {y_hi:.3f}]")
    print(f"Computational domain: y in [{y_0:.3f}, {y_1:.3f}]  "
          f"(moneyness [{np.exp(y_0):.2f}, {np.exp(y_1):.2f}]),  M_time={M_TIME}")

    msh = IntervalMesh(N_SPACE, y_0, y_1)
    V = FunctionSpace(msh, "Lagrange", 1)
    y_n = dof_coordinates(V)

    # Initial condition: call payoff  S0 (1 - e^y)^+.
    u_0 = interpolate_func(V, lambda y: S0 * np.maximum(1.0 - np.exp(y), 0.0))

    # Time grid; snap each maturity to a distinct trajectory step.
    dt = T_max / M_TIME
    obs, seeds, snapped = [], [], []
    used_idx = set()
    for p in per:
        idx = max(1, min(M_TIME, int(round(p["T"] / dt))))
        while idx in used_idx:
            idx += 1
        used_idx.add(idx)
        snapped.append(idx * dt)

        mask = np.where((y_n >= p["y_q"][0]) & (y_n <= p["y_q"][-1]), 1.0, 0.0)
        # Scale at the SNAPPED maturity, the time level the solver actually reports, so the
        # factor cancels exactly in the residual and in the reprice.
        scale = np.exp(Q_q(idx * dt))
        g = Function(V)
        set_array(g, np.interp(y_n, p["y_q"], p["C"]) * scale)
        obs.append({"idx": idx, "mask": mask, "g": g, "T": p["T"], "scale": scale,
                    "exp": p["exp"], "y_q": p["y_q"], "C": p["C"], "iv": p["iv"]})

        # Per-maturity prior seed: BBF factor-of-two-consistent local vol on the nodes (flat
        # wings). See _local_vol_prior; this replaces the naive a* = 1/2 sigma_iv^2.
        seeds.append(np.interp(y_n, p["y_q"], _local_vol_prior(p["y_q"], p["iv"])))

    # Slices aligned to the snapped maturities; each slice's prior is that maturity's seed.
    edges = np.concatenate([[0.0], np.asarray(snapped)])
    a_prior_mat = np.vstack(seeds)
    seed0 = Function(V)
    set_array(seed0, a_prior_mat[0])
    a_init = Vol(V, seed0, 0.0, T_max, edges=edges)
    a_init.update(V, a_prior_mat.ravel())

    # Term structure: forward rates for the PDE, zero rates for the BS boundaries.
    r_of_t = _forward_rate_fn(T_arr, r_arr)
    q_of_t = _forward_rate_fn(T_arr, q_arr)

    # Boundary values: BS call at the (far) domain edges, with maturity tau's zero rate and a
    # wing IV interpolated across the term structure.
    ivL = np.array([p["iv"][0] for p in per])
    ivR = np.array([p["iv"][-1] for p in per])

    def _R(tau):  return float(np.interp(tau, T_arr, r_arr, left=r_arr[0]))
    def _Q(tau):  return float(np.interp(tau, T_arr, q_arr, left=q_arr[0]))
    def _sL(tau): return float(np.interp(tau, T_arr, ivL, left=ivL[0]))
    def _sR(tau): return float(np.interp(tau, T_arr, ivR, left=ivR[0]))

    def bcl(tau):
        if tau <= 1e-14:
            return float(S0 * max(1.0 - np.exp(y_0), 0.0))
        return float(bs_call(S0, y_0, _R(tau), _Q(tau), _sL(tau), tau) * np.exp(Q_q(tau)))

    def bcr(tau):
        if tau <= 1e-14:
            return float(S0 * max(1.0 - np.exp(y_1), 0.0))
        return float(bs_call(S0, y_1, _R(tau), _Q(tau), _sR(tau), tau) * np.exp(Q_q(tau)))

    return {
        "V": V, "y_n": y_n, "y_0": y_0, "y_1": y_1, "u_0": u_0, "obs": obs,
        "a_init": a_init, "a_prior_mat": a_prior_mat, "edges": edges,
        "r_of_t": r_of_t, "q_of_t": q_of_t, "bcl": bcl, "bcr": bcr,
        "dt": dt, "T_max": T_max, "S0": S0, "discounted": discounted,
    }


# ---------------------------------------------------------------------------
# Run + report
# ---------------------------------------------------------------------------

def run(beta_y=FIXED_BETA_Y, beta_tau=FIXED_BETA_TAU, max_iter=300,
        out="egger_ex4_surface.png", replot=False, discounted=DISCOUNTED):
    S0, mats = load_surface(DATA_PATH)
    print("#" * 74)
    print(f"#  Example 4 -- SURFACE calibration  {DATA_DATE}  S0={S0:.2f}  "
          f"{len(mats)} maturities  discounted={discounted}")
    print("#" * 74)

    p = build_problem(S0, mats, discounted)

    if replot:
        # Reuse the cached calibrated slices; skip the L-BFGS solve, just redraw.
        cache = np.load(os.path.splitext(out)[0] + ".npz")
        a_cal = p["a_init"]
        a_cal.update(p["V"], cache["a_vec"])
        print("Replotting from cached slices (no recalibration).")
    else:
        a_cal, _ = run_surface(
            p["u_0"], p["obs"], p["a_init"], p["a_prior_mat"], beta_y, beta_tau,
            p["V"], p["r_of_t"], p["q_of_t"], p["y_0"], p["y_1"],
            p["dt"], M_TIME, p["bcl"], p["bcr"], max_iter=max_iter,
            discounted=discounted,
        )
        # Persist the calibrated slices so the figure can be rebuilt without recalibrating.
        np.savez(os.path.splitext(out)[0] + ".npz",
                 a_vec=a_cal.to_vec(), edges=p["edges"], y_n=p["y_n"])

    _report_and_plot(p, a_cal, beta_y, beta_tau, out)


def _reprice_errors(p, a_cal):
    """
    Reprice at the calibrated surface; return (T, median%, max%) per maturity. The relative
    error is measured over liquid strikes only (market price >= 0.5% of S0): a relative price
    error is not meaningful where the option is worth a few cents, and those deep-OTM strikes
    would otherwise dominate the maximum.
    """
    V = p["V"]
    _, traj = compute_Af_surface(a_cal, p["u_0"], p["dt"], M_TIME, V,
                                 p["r_of_t"], p["q_of_t"], p["y_0"], p["y_1"],
                                 p["bcl"], p["bcr"], p["discounted"])
    order = np.argsort(p["y_n"])
    y_s = p["y_n"][order]
    floor = 0.005 * p["S0"]

    print(f"\n{'expiration':>12} {'T':>6} {'nK':>4} {'L2($)':>9} {'med%':>7} {'max%':>7}")
    Ts, meds, maxs = [], [], []
    for ob in p["obs"]:
        # Undo the change of variable: C = e^{-Q_q(tau)} u (scale is 1 when not discounted).
        u_pred = get_array(traj[ob["idx"]]) / ob["scale"]
        res = Function(V)
        set_array(res, (u_pred - get_array(ob["g"]) / ob["scale"]) * ob["mask"])
        liq = ob["C"] >= floor
        C_model = np.interp(ob["y_q"][liq], y_s, u_pred[order])
        rel = np.abs(C_model - ob["C"][liq]) / ob["C"][liq]
        Ts.append(ob["T"]); meds.append(np.median(rel)); maxs.append(np.max(rel))
        print(f"{ob['exp']:>12} {ob['T']:>6.3f} {int(liq.sum()):>4} "
              f"{L2_norm(res):>9.4f} {np.median(rel):>6.2%} {np.max(rel):>6.2%}")
    print(f"\nOverall median reprice error: {np.median(meds):.2%}")
    return np.array(Ts), np.array(meds), np.array(maxs)


def _report_and_plot(p, a_cal, beta_y, beta_tau, out):
    order = np.argsort(p["y_n"])
    y_s = p["y_n"][order]

    # Recovered local vol per slice: sigma_k(y) = sqrt(2 a_k(y)); slice k <-> maturity obs[k].
    sig = np.vstack([np.sqrt(2.0 * np.maximum(get_array(a_k)[order], 0.0))
                     for a_k in a_cal.a])
    T_obs = np.array([ob["T"] for ob in p["obs"]])

    # Show only the informative band (a small margin past the observed strikes); the far
    # wings just relax to the prior and add nothing but noise to the picture.
    y_lo = min(ob["y_q"][0]  for ob in p["obs"]) - 0.02
    y_hi = max(ob["y_q"][-1] for ob in p["obs"]) + 0.02
    win = (y_s >= y_lo) & (y_s <= y_hi)
    yw, sigw = y_s[win], sig[:, win]

    Ts, meds, maxs = _reprice_errors(p, a_cal)

    fig = plt.figure(figsize=(18, 5))
    ax0 = fig.add_subplot(1, 3, 1, projection="3d")
    ax1 = fig.add_subplot(1, 3, 2)
    ax2 = fig.add_subplot(1, 3, 3)

    # (1) 3D surface sigma(y, tau). The slices are refined in tau by linear interpolation
    # so the mesh reads as a smooth surface rather than a handful of ribbons.
    tau_f = np.linspace(T_obs[0], T_obs[-1], 60)
    sig_f = np.column_stack([np.interp(tau_f, T_obs, sigw[:, j])
                             for j in range(sigw.shape[1])])
    Yg, Tg = np.meshgrid(yw, tau_f)
    surf = ax0.plot_surface(Yg, Tg, sig_f, cmap="viridis",
                            rstride=1, cstride=2, linewidth=0, antialiased=True)
    ax0.set_xlabel(r"$y=\log(K/S_0)$", labelpad=6)
    ax0.set_ylabel(r"Maturity  $\tau$", labelpad=6)
    ax0.set_zlabel(r"$\sigma$", labelpad=2)
    ax0.set_title(r"Recovered local-vol surface  $\sigma(y,\tau)$")
    ax0.view_init(elev=26, azim=-58)
    fig.colorbar(surf, ax=ax0, shrink=0.6, pad=0.08, label=r"$\sigma$")

    # (2) Recovered slices against each maturity's market IV smile.
    ax = ax1
    cmap = plt.cm.viridis(np.linspace(0, 1, len(p["obs"])))
    for k, ob in enumerate(p["obs"]):
        ax.plot(ob["y_q"], ob["iv"], color=cmap[k], lw=0.8, alpha=0.5)
        ax.plot(yw, sigw[k], color=cmap[k], lw=1.8, label=rf"$\tau$={ob['T']:.2f}")
    ax.set_xlim(y_lo, y_hi)
    ax.set_xlabel(r"Log-moneyness  $y=\log(K/S_0)$")
    ax.set_ylabel(r"Volatility  $\sigma$")
    ax.set_title(r"Local-vol slices (solid) vs market IV (faint)")
    ax.legend(fontsize=7, ncol=2); ax.grid(True, alpha=0.4)

    # (3) Per-maturity reprice error.
    ax = ax2
    ax.plot(Ts, 100 * meds, "o-", color="b", label="median")
    ax.plot(Ts, 100 * maxs, "s--", color="crimson", label="max (liquid)")
    ax.set_xlabel(r"Maturity  $\tau$ (years)")
    ax.set_ylabel("Relative reprice error (%)")
    ax.set_title(rf"Reprice error ($\beta_y$={beta_y:.0e}, $\beta_\tau$={beta_tau:.0e})")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.4)

    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Example 4 -- surface calibration")
    parser.add_argument("--beta-y", type=float, default=FIXED_BETA_Y)
    parser.add_argument("--beta-tau", type=float, default=FIXED_BETA_TAU)
    parser.add_argument("--max-iter", type=int, default=300)
    parser.add_argument("--out", default="egger_ex4_surface.png")
    parser.add_argument("--replot", action="store_true",
                        help="redraw from cached slices, skip recalibration")
    parser.add_argument("--discount", dest="discounted", action="store_true",
                        default=DISCOUNTED,
                        help="solve for u = e^{int q} C instead of C")
    args = parser.parse_args()
    run(beta_y=args.beta_y, beta_tau=args.beta_tau, max_iter=args.max_iter,
        out=args.out, replot=args.replot, discounted=args.discounted)
