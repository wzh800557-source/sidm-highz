#!/usr/bin/env python3
"""
Profile likelihood optimizer for the f_esc x f_star product degeneracy — v2.

Changes vs. the original fesc_degeneracy/scripts/reionization_optimizer.py:

1. UNIT FIX (critical). The original solved
       dQ/dt = ndot / nH_proper(z) - C alpha_B nH_proper(z) Q
   with ndot in s^-1 Mpc^-3 (comoving) but nH in cm^-3 (proper), and an
   n_H0 normalisation that double-counted (h^2, Ob) factors. The net
   effect made the source term ~10^20 too large: Q -> 1 instantly for
   ANY f_esc, so chi^2 was independent of all parameters and
   "delta chi^2 identically zero" was an artefact of the bug, not physics.
   v2 uses the standard comoving Madau form:
       dQ/dt = ndot_com / <nH>_com - C(z) alpha_B nH_proper(z) Q .

2. CALIBRATION. With correct units, ndot_ref is recalibrated so the CDM
   baseline (f_star0 = 0.019, f_esc = 0.13) reproduces xHI(z=7) ~ 0.30
   (tau then comes out 0.060, within 1 sigma of Planck 0.054 +/- 0.007):
   log10 ndot_ref = 51.509 (was 50.85).

3. DATA VECTOR. XHI_DATA replaced by the exact published compilation
   cited in the manuscript appendix (values as published):
     Mason+25   (arXiv:2501.11702): 0.33 +0.18-0.27 (z~6.5),
                                    0.64 +0.17-0.23 (z~9.3)
     Greig+17   (MNRAS 466, 4239):  0.40 +0.21-0.19 (z=7.08)
     Umeda+24   (ApJ 971, 124, Table 5):
                 0.53 +0.18-0.47 (z=7.12), 0.65 +0.27-0.34 (z=7.44),
                 0.91 +0.09-0.22 (z=8.28), 0.92 +0.08-0.10 (z=9.91)
     Mason+19   (MNRAS 485, 3947):  > 0.76 (68%) at z~7.9, one-sided
   Two-sided errors are symmetrised, sigma = (up + down)/2; the Mason+19
   lower limit enters as a one-sided Gaussian penalty (sigma = 0.30).

4. CDM BASELINE PROFILED. delta chi^2 is now (min over f_esc of SIDM)
   minus (min over f_esc of CDM), instead of CDM fixed at f_esc = 0.13.

Verification run (2026-07-15, 3x3 sigma/m x eta grid):
   legacy vector    : max |dchi2| = 0.015
   published vector : max |dchi2| = 0.0045
=> swapping the approximate legacy list for the exact published
   compilation changes no conclusion (both far below 0.1).

Usage:
    python reionization_optimizer_v2.py --scan
    python reionization_optimizer_v2.py --scan --legacy   # old data vector
"""

import argparse
import json
import os

import numpy as np

# ---------------------------------------------------------------- cosmology
H0 = 67.74
Om = 0.3089
Ob = 0.0486
h = H0 / 100.0
rho_crit = 1.87834e-29 * h * h          # g cm^-3
m_p = 1.6726e-24                        # g
nH0 = 0.76 * Ob * rho_crit / m_p        # cm^-3 (comoving / z=0 proper)
Mpc_cm = 3.0857e24
nH0_Mpc = nH0 * Mpc_cm ** 3             # comoving Mpc^-3
alpha_B = 2.6e-13 * 2.0 ** (-0.7)       # cm^3 s^-1 at T = 2e4 K
sigma_T = 6.652e-25
c_cgs = 2.998e10

FSTAR_CDM = 0.019
FESC_CDM = 0.13
LOG_NDOT_REF = 51.509                   # recalibrated, see docstring
NDOT_REF = 10 ** LOG_NDOT_REF

TAU_OBS = 0.054
TAU_ERR = 0.007

# (z, xHI, sigma, kind)
XHI_PUBLISHED = [
    (6.5,  0.33, 0.225, 'two'),    # Mason+25
    (7.08, 0.40, 0.200, 'two'),    # Greig+17 (ULAS J1120+0641)
    (7.12, 0.53, 0.325, 'two'),    # Umeda+24
    (7.44, 0.65, 0.305, 'two'),    # Umeda+24
    (7.9,  0.76, 0.300, 'lower'),  # Mason+19 (>0.76 at 68%)
    (8.28, 0.91, 0.155, 'two'),    # Umeda+24
    (9.3,  0.64, 0.200, 'two'),    # Mason+25
    (9.91, 0.92, 0.090, 'two'),    # Umeda+24
]

XHI_LEGACY = [
    (6.5, 0.10, 0.10, 'two'), (7.0, 0.30, 0.10, 'two'),
    (7.09, 0.40, 0.20, 'two'), (7.54, 0.55, 0.15, 'two'),
    (7.6, 0.60, 0.15, 'two'), (8.0, 0.60, 0.15, 'two'),
    (9.3, 0.80, 0.10, 'two'), (9.5, 0.85, 0.10, 'two'),
]

# ------------------------------------------------------------- ODE machinery
ZS = np.linspace(25.0, 5.0, 600)
_Hz = H0 * np.sqrt(Om * (1 + ZS) ** 3 + 1 - Om)
_dtdz = -1.0 / (_Hz * 1e5 / Mpc_cm * (1 + ZS))
_dts = _dtdz[1:] * np.diff(ZS)
_rec = np.maximum(1.0, 21.0 / (1 + ZS[1:])) * alpha_B * nH0 * (1 + ZS[1:]) ** 3
_src = ((1 + ZS[1:]) / 9.0) ** (-1.5) / nH0_Mpc * NDOT_REF
_tau_w = 1.08 * nH0 * (1 + ZS[1:]) ** 3 * sigma_T * c_cgs * np.abs(_dts)


def xhi_batch(f_eff):
    """Neutral-fraction histories for a vector of f_esc*fstar_ratio."""
    f_eff = np.atleast_1d(np.asarray(f_eff, dtype=float))
    Q = np.zeros(f_eff.shape)
    X = np.empty((len(ZS) - 1, len(f_eff)))
    for i in range(len(ZS) - 1):
        Q = np.clip(Q + (f_eff * _src[i] - _rec[i] * Q) * _dts[i], 0.0, 1.0)
        X[i] = 1.0 - Q
    tau = 0.018 + _tau_w @ (1.0 - X)
    return X, tau


def chi2_vec(f_eff, data):
    X, tau = xhi_batch(f_eff)
    c2 = (tau - TAU_OBS) ** 2 / TAU_ERR ** 2
    z_asc = ZS[1:][::-1]
    X_asc = X[::-1]
    for z_obs, x_obs, sig, kind in data:
        j = min(max(np.searchsorted(z_asc, z_obs), 1), len(z_asc) - 1)
        w = (z_obs - z_asc[j - 1]) / (z_asc[j] - z_asc[j - 1])
        xp = X_asc[j - 1] * (1 - w) + X_asc[j] * w
        pen = (xp - x_obs) ** 2 / sig ** 2
        if kind == 'lower':
            pen = np.where(xp >= x_obs, 0.0, pen)
        c2 = c2 + pen
    return c2


# ------------------------------------------------------- SIDM suppression
def load_grid(json_path=None):
    if json_path is None:
        json_path = os.path.join(os.path.dirname(__file__), '..', '..',
                                 'data', 'dbind_table.json')
        if not os.path.exists(json_path):
            json_path = '/tmp/sidm-highz/data/dbind_table.json'
    with open(json_path) as fh:
        return json.load(fh)


def get_mass_weighted_suppression(grid, sigma_str, eta, z_str='z7'):
    sigma_codes = {'0.5': '0005', '1': '0010', '2': '0020',
                   '5': '0050', '10': '0100', '20': '0200'}
    mass_codes = ['090', '095', '100', '105', '110']
    logM_grid = np.array([9.0, 9.5, 10.0, 10.5, 11.0])
    sc = sigma_codes[sigma_str]
    dbs = np.array([grid.get(f'M{mc}_{z_str}_s{sc}_const',
                             {'delta_bind': [0, 0]})['delta_bind'][1]
                    for mc in mass_codes])
    lmb = np.linspace(8.5, 11.5, 100)
    db = np.interp(lmb, logM_grid, dbs)
    M = 10 ** lmb
    Ms = 10 ** 10.5
    w = (M / Ms) ** (-0.5) * np.exp(-M / Ms) * M
    w /= w.sum()
    return float(np.sum(w * np.clip(1 - eta * db, 0.01, None)))


def profile_dchi2(sigma_str, eta, grid, data, fgrid=None):
    """Profiled dchi2 (SIDM best - CDM best), both sides profiled."""
    if fgrid is None:
        fgrid = np.linspace(0.005, 0.50, 300)
    supp = get_mass_weighted_suppression(grid, sigma_str, eta)
    fstar_ratio = 1.0 / supp
    cs_cdm = chi2_vec(fgrid, data)
    k0 = int(np.argmin(cs_cdm))
    cs = chi2_vec(fgrid * fstar_ratio, data)
    k = int(np.argmin(cs))
    prod_ratio = (fgrid[k] * fstar_ratio) / fgrid[k0]
    return cs[k] - cs_cdm[k0], fgrid[k], FSTAR_CDM * fstar_ratio, prod_ratio


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scan', action='store_true')
    ap.add_argument('--legacy', action='store_true',
                    help='use the pre-referee hard-coded data vector')
    ap.add_argument('--sigma', type=str)
    ap.add_argument('--eta', type=float)
    ap.add_argument('--data', default=None)
    args = ap.parse_args()

    grid = load_grid(args.data)
    data = XHI_LEGACY if args.legacy else XHI_PUBLISHED
    label = 'legacy' if args.legacy else 'published'

    if args.scan:
        print(f"data vector: {label}")
        print(f"{'s/m':>6s} {'eta':>6s} {'dchi2':>9s} {'fesc_prof':>10s} "
              f"{'fstar_prof':>11s} {'prod_ratio':>11s}")
        print('-' * 58)
        worst = 0.0
        for s in ['1', '5', '10']:
            for e in [0.10, 0.25, 0.50]:
                d, fp, fs, pr = profile_dchi2(s, e, grid, data)
                worst = max(worst, abs(d))
                print(f"{s:>6s} {e:6.2f} {d:9.4f} {fp:10.4f} {fs:11.5f} {pr:11.3f}")
        print(f"\nmax |dchi2| = {worst:.4f}")
    else:
        if not args.sigma or args.eta is None:
            ap.error('--sigma and --eta required unless --scan')
        d, fp, fs, pr = profile_dchi2(args.sigma, args.eta, grid, data)
        print(f"dchi2 = {d:.4f}, fesc_prof = {fp:.4f}, "
              f"fstar_prof = {fs:.5f}, product ratio = {pr:.3f}")


if __name__ == '__main__':
    main()
