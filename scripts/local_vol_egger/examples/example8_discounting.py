r"""
Example 8 -- comparing the two normalizations of the Dupire forward equation.

Section 6 keeps the explicit zero-order term of the log-moneyness equation

    u_tau = a(u_yy - u_y) - (r-q)u_y - q u,

while Egger & Engl (2005) absorb the dividend factor into the unknown by setting
u = e^{\int_0^tau q} C, which cancels that term and leaves

    -u_tau + a(u_yy - u_y) + (q-r)u_y = 0.

The two describe the same continuous problem and coincide when q = 0, so they can only differ
through the discretization. This script measures that difference and reproduces the two tables
of the annex subsection on the discounted normalization:

  * TABLE 1 (--pde, always run): time-discretization error of the forward solve at the
    production grid against a refined reference, for the calibrated q term structure and for
    artificially inflated flat q. Answers which normalization is closer to the exact solution.
  * TABLE 2 (--cross): reprices each calibrated surface with BOTH solvers. If swapping the
    solver moves the error far less than swapping the surface, the gap between the two
    calibrations is optimizer path rather than normalization.

Run from the package root (scripts/local_vol_egger):

    conda run -n fenics-legacy python -m examples.example8_discounting

Table 2 needs the two calibrated surfaces, produced beforehand by

    conda run -n fenics-legacy python -m examples.example4_surface --out plain.png
    conda run -n fenics-legacy python -m examples.example4_surface --discount --out disc.png

then

    conda run -n fenics-legacy python -m examples.example8_discounting --cross plain.npz disc.npz
"""

import argparse
import contextlib
import io

import numpy as np
from dolfin import Function, set_log_level, LogLevel

import examples.example4_surface as ex4
from optimization.surface import compute_Af_surface
from utils import L2_norm, get_array, set_array

set_log_level(LogLevel.ERROR)

M_PROD  = 300    # production time grid, matches ex4.M_TIME
REF_MUL = 8      # the reference solve uses REF_MUL times as many steps
Q_FLAT  = [None, 0.02, 0.05, 0.10, 0.20]   # None keeps the calibrated q term structure


def _solve(S0, mats, M, discounted):
    """
    Forward solve at the IV prior on an M-step grid, returned as CALL PRICES per maturity.

    ex4.build_problem reads the time grid from the module-level M_TIME, so we set it here to
    sweep the refinement. Because T_max is fixed, step 8*n of the refined grid sits at exactly
    the same time as step n of the production grid, which is what makes the two comparable.
    """
    ex4.M_TIME = M
    with contextlib.redirect_stdout(io.StringIO()):     # build_problem reports the setup
        p = ex4.build_problem(S0, mats, discounted=discounted)
        _, traj = compute_Af_surface(p["a_init"], p["u_0"], p["dt"], M, p["V"],
                                     p["r_of_t"], p["q_of_t"], p["y_0"], p["y_1"],
                                     p["bcl"], p["bcr"], discounted)
    # Undo the change of variable so both branches are compared in dollars.
    return p, [get_array(traj[ob["idx"]]) / ob["scale"] for ob in p["obs"]]


def table_pde(S0, mats0):
    """TABLE 1: which normalization discretizes better, as a function of q."""
    print("\nTime-discretization error of the forward solve, "
          f"M={M_PROD} vs M={M_PROD * REF_MUL} reference.")
    print("Max |C - C_ref| over the observed band, averaged over maturities, in dollars.\n")
    print(f"{'q':>8} {'explicit -qu':>13} {'discounted':>12} {'ratio':>7}")

    for q_flat in Q_FLAT:
        mats = [dict(m) for m in mats0]
        if q_flat is not None:                       # override the calibrated term structure
            for m in mats:
                m["q"] = q_flat

        p, C_plain = _solve(S0, mats, M_PROD, False)
        _, C_disc  = _solve(S0, mats, M_PROD, True)
        _, R_disc  = _solve(S0, mats, M_PROD * REF_MUL, True)
        _, R_plain = _solve(S0, mats, M_PROD * REF_MUL, False)

        e_p = e_d = 0.0
        for k, ob in enumerate(p["obs"]):
            band = ob["mask"] > 0
            ref = 0.5 * (R_disc[k] + R_plain[k])     # the two references agree to ~1e-3 $
            e_p += np.abs(C_plain[k] - ref)[band].max()
            e_d += np.abs(C_disc[k] - ref)[band].max()
        n = len(p["obs"])
        label = "market" if q_flat is None else f"{q_flat:.0%}"
        print(f"{label:>8} {e_p / n:>13.2e} {e_d / n:>12.2e} {e_p / max(e_d, 1e-30):>6.2f}x")


def _reprice(S0, mats, a_vec, discounted):
    """Median relative error over the liquid strikes, and the unweighted data misfit."""
    with contextlib.redirect_stdout(io.StringIO()):
        ex4.M_TIME = M_PROD
        p = ex4.build_problem(S0, mats, discounted=discounted)
        p["a_init"].update(p["V"], a_vec)
        _, traj = compute_Af_surface(p["a_init"], p["u_0"], p["dt"], M_PROD, p["V"],
                                     p["r_of_t"], p["q_of_t"], p["y_0"], p["y_1"],
                                     p["bcl"], p["bcr"], discounted)
    order = np.argsort(p["y_n"])
    y_s = p["y_n"][order]
    floor = 0.005 * S0                               # skip strikes worth a few cents
    meds, cost = [], 0.0
    for ob in p["obs"]:
        C = get_array(traj[ob["idx"]]) / ob["scale"]
        res = Function(p["V"])
        set_array(res, (C - get_array(ob["g"]) / ob["scale"]) * ob["mask"])
        cost += L2_norm(res) ** 2
        liq = ob["C"] >= floor
        C_model = np.interp(ob["y_q"][liq], y_s, C[order])
        meds.append(np.median(np.abs(C_model - ob["C"][liq]) / ob["C"][liq]))
    return np.median(meds), cost


def table_cross(S0, mats, plain_npz, disc_npz):
    """TABLE 2: each calibrated surface repriced with both solvers."""
    a_vec = {"explicit -qu": np.load(plain_npz)["a_vec"],
             "discounted":   np.load(disc_npz)["a_vec"]}

    print("\nCross evaluation of the two calibrated surfaces.\n")
    print(f"{'calibrated with':>16} {'repriced with':>15} {'median err':>11} {'misfit($^2)':>12}")
    for cal_name, vec in a_vec.items():
        for rep_disc in (True, False):
            med, cost = _reprice(S0, mats, vec, rep_disc)
            rep_name = "discounted" if rep_disc else "explicit -qu"
            print(f"{cal_name:>16} {rep_name:>15} {med:>10.3%} {cost:>12.5f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Example 8 -- explicit vs discounted normalization")
    parser.add_argument("--cross", nargs=2, metavar=("PLAIN_NPZ", "DISC_NPZ"),
                        help="also run TABLE 2 from two example4 surface caches")
    parser.add_argument("--skip-pde", action="store_true", help="skip TABLE 1")
    args = parser.parse_args()

    S0, mats = ex4.load_surface(ex4.DATA_PATH)
    print("#" * 74)
    print(f"#  Example 8 -- explicit -qu vs discounted u = e^(int q) C   S0={S0:.2f}")
    print("#" * 74)

    if not args.skip_pde:
        table_pde(S0, mats)
    if args.cross:
        table_cross(S0, mats, *args.cross)
    else:
        print("\n(TABLE 2 skipped: pass --cross PLAIN_NPZ DISC_NPZ, see the module docstring.)")
