#!/usr/bin/env python3
"""
Profile likelihood optimizer for the f_esc x f_star product degeneracy.

Computes Δχ² from τ + x_HI(z) for SIDM parameter points, with optional
f_esc prior and blowout correlation.

Usage:
    python reionization_optimizer.py --sigma 10 --eta 0.5
    python reionization_optimizer.py --sigma 10 --eta 0.5 --alpha_esc 0.3 --sigma_fesc 0.03
    python reionization_optimizer.py --scan  # scan full grid
"""

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.interpolate import interp1d
import json
import argparse
import os

# ----------------------------------------------------------------
# Cosmological parameters
# ----------------------------------------------------------------
H0 = 67.74  # km/s/Mpc
Om = 0.3089
Ob = 0.0486
fb = Ob / Om  # 0.157
nH0 = 1.88e-7 * (H0/100)**2 * Ob * 0.76  # cm^{-3}
alpha_B = 2.6e-13 * 2.0**(-0.7)  # cm^3/s at T=2e4 K
sigma_T = 6.652e-25  # cm^2
c_cgs = 2.998e10  # cm/s
Mpc_cm = 3.0857e24

# CDM baseline
FSTAR_CDM = 0.019
FESC_CDM = 0.13
NDOT_REF = 10**50.85  # s^{-1} Mpc^{-3}

# Observational data
TAU_OBS = 0.054
TAU_ERR = 0.007
XHI_DATA = [
    (6.5, 0.10, 0.10), (7.0, 0.30, 0.10), (7.09, 0.40, 0.20),
    (7.54, 0.55, 0.15), (7.6, 0.60, 0.15), (8.0, 0.60, 0.15),
    (9.3, 0.80, 0.10), (9.5, 0.85, 0.10),
]


def Hz(z):
    return H0 * np.sqrt(Om * (1+z)**3 + (1-Om))

def dtdz(z):
    return -1.0 / (Hz(z) * 1e5 / Mpc_cm * (1+z))

def clumping(z):
    return max(1.0, 3.0 * 7.0 / (1+z))


def solve_reionization(fesc_val, fstar_ratio=1.0):
    """Solve the reionization ODE for given f_esc and f_star ratio."""
    zs = np.linspace(25, 5.5, 500)
    Q = 0.0
    z_out, xHI_out = [], []

    for i in range(1, len(zs)):
        z = zs[i]
        dz = zs[i] - zs[i-1]
        dt = dtdz(z) * dz
        nH_z = nH0 * (1+z)**3
        C = clumping(z)

        # Photon rate with redshift evolution
        ndot = fesc_val * NDOT_REF * fstar_ratio * ((1+z)/9)**(-1.5)
        dQdt = ndot / nH_z - C * alpha_B * nH_z * Q
        Q = min(max(Q + dQdt * dt, 0), 1.0)
        z_out.append(z)
        xHI_out.append(1 - Q)

    return np.array(z_out), np.array(xHI_out)


def compute_tau(z_arr, xHI_arr):
    """Compute Thomson optical depth."""
    Q_arr = 1 - xHI_arr
    fe = 1.08
    tau_early = 0.018

    tau = 0.0
    for i in range(1, len(z_arr)):
        z = z_arr[i]
        dz = z_arr[i] - z_arr[i-1]
        nH_z = nH0 * (1+z)**3
        tau += fe * nH_z * Q_arr[i] * sigma_T * c_cgs * abs(dtdz(z) * dz)

    return tau + tau_early


def compute_chi2(fesc_val, fstar_ratio, fesc_prior_mean=None,
                  sigma_fesc=None):
    """Compute χ² from τ + x_HI(z) + optional f_esc prior."""
    z_arr, xHI_arr = solve_reionization(fesc_val, fstar_ratio)

    # τ contribution
    tau_pred = compute_tau(z_arr, xHI_arr)
    chi2 = (tau_pred - TAU_OBS)**2 / TAU_ERR**2

    # x_HI(z) contributions
    xHI_interp = interp1d(z_arr, xHI_arr, fill_value='extrapolate')
    for z_obs, xHI_obs, sig in XHI_DATA:
        xHI_pred = xHI_interp(z_obs)
        chi2 += (xHI_pred - xHI_obs)**2 / sig**2

    # f_esc prior
    if fesc_prior_mean is not None and sigma_fesc is not None:
        chi2 += (fesc_val - fesc_prior_mean)**2 / sigma_fesc**2

    return chi2


def load_grid(json_path=None):
    """Load the Δ_bind grid."""
    if json_path is None:
        json_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'dbind_table.json')
    with open(json_path) as f:
        return json.load(f)


def get_mass_weighted_suppression(grid, sigma_str, eta, z_str='z7'):
    """Compute luminosity-weighted effective suppression."""
    sigma_codes = {'0.5': '0005', '1': '0010', '2': '0020',
                   '5': '0050', '10': '0100', '20': '0200'}
    mass_codes = ['090', '095', '100', '105', '110']
    logM_grid = np.array([9.0, 9.5, 10.0, 10.5, 11.0])

    sc = sigma_codes[sigma_str]
    dbs = []
    for mc in mass_codes:
        tag = f'M{mc}_{z_str}_s{sc}_const'
        if tag in grid:
            dbs.append(grid[tag]['delta_bind'][1])  # R<0.5 kpc
        else:
            dbs.append(0.0)

    # Interpolate to fine mass grid
    db_interp = interp1d(logM_grid, dbs, fill_value=(dbs[0], dbs[-1]),
                          bounds_error=False)

    # HMF-weighted average
    logM_bins = np.linspace(8.5, 11.5, 100)
    def hmf_w(lm):
        M = 10**lm
        Ms = 10**10.5
        return (M/Ms)**(-0.5) * np.exp(-M/Ms) * M

    weights = np.array([hmf_w(lm) for lm in logM_bins])
    weights /= weights.sum()

    supp = np.array([max(1 - eta * db_interp(lm), 0.01) for lm in logM_bins])
    return np.sum(weights * supp)


def profile_dchi2(sigma_str, eta, alpha_esc=0.0, sigma_fesc=None,
                   fesc_prior_mean=0.15, grid=None):
    """
    Compute profiled Δχ² for a given (σ/m, η) point.

    Parameters
    ----------
    sigma_str : str
        Cross-section value as string (e.g., '10')
    eta : float
        SFE coupling parameter
    alpha_esc : float
        Blowout correlation exponent (0 = no correlation)
    sigma_fesc : float or None
        Width of f_esc Gaussian prior (None = no prior)
    fesc_prior_mean : float
        Central value of f_esc prior
    grid : dict
        Loaded Δ_bind grid

    Returns
    -------
    dchi2 : float
        Profiled Δχ² relative to CDM
    fesc_prof : float
        Profiled f_esc value
    fstar_prof : float
        Profiled f_star value
    """
    if grid is None:
        grid = load_grid()

    # Mass-weighted suppression
    supp_eff = get_mass_weighted_suppression(grid, sigma_str, eta)
    fstar_ratio = 1.0 / supp_eff

    # SIDM f_esc enhancement from blowout
    sigma_codes = {'0.5': '0005', '1': '0010', '2': '0020',
                   '5': '0050', '10': '0100', '20': '0200'}
    sc = sigma_codes[sigma_str]
    tag = f'M100_z7_s{sc}_const'
    db_M10 = grid[tag]['delta_bind'][1] if tag in grid else 0.0

    if alpha_esc > 0 and db_M10 > 0 and db_M10 < 0.99:
        fesc_enhancement = (1.0 / (1.0 - db_M10))**alpha_esc
    else:
        fesc_enhancement = 1.0

    # CDM baseline χ²
    chi2_cdm = compute_chi2(FESC_CDM, 1.0,
                             fesc_prior_mean if sigma_fesc else None,
                             sigma_fesc)

    # Profile over f_esc
    def neg_chi2(fesc_val):
        fesc_eff = fesc_val * fesc_enhancement
        return compute_chi2(fesc_eff, fstar_ratio,
                             fesc_prior_mean if sigma_fesc else None,
                             sigma_fesc)

    result = minimize_scalar(neg_chi2, bounds=(0.001, 0.50), method='bounded')
    chi2_sidm = result.fun
    fesc_prof = result.x

    return chi2_sidm - chi2_cdm, fesc_prof, FSTAR_CDM * fstar_ratio


def main():
    parser = argparse.ArgumentParser(description='Reionization profile likelihood optimizer')
    parser.add_argument('--sigma', type=str, help='σ/m value (e.g., 10)')
    parser.add_argument('--eta', type=float, help='η value')
    parser.add_argument('--alpha_esc', type=float, default=0.0, help='Blowout correlation exponent')
    parser.add_argument('--sigma_fesc', type=float, default=None, help='f_esc prior width')
    parser.add_argument('--scan', action='store_true', help='Scan full 3x3 grid')
    parser.add_argument('--data', default=None, help='Path to dbind_table.json')
    args = parser.parse_args()

    grid = load_grid(args.data)

    if args.scan:
        sigmas = ['1', '5', '10']
        etas = [0.10, 0.25, 0.50]
        print(f"{'σ/m':>6s} {'η':>6s} {'Δχ²':>8s} {'f_esc_prof':>12s} {'f★_prof':>10s}")
        print("-" * 48)
        for s in sigmas:
            for e in etas:
                dchi2, fesc_p, fstar_p = profile_dchi2(
                    s, e, args.alpha_esc, args.sigma_fesc, grid=grid)
                print(f"{s:>6s} {e:6.2f} {dchi2:8.3f} {fesc_p:12.5f} {fstar_p:10.5f}")
    else:
        if args.sigma is None or args.eta is None:
            parser.error("--sigma and --eta required unless --scan is used")

        dchi2, fesc_p, fstar_p = profile_dchi2(
            args.sigma, args.eta, args.alpha_esc, args.sigma_fesc, grid=grid)

        print(f"σ/m = {args.sigma} cm²/g, η = {args.eta}")
        print(f"α_esc = {args.alpha_esc}, σ_fesc = {args.sigma_fesc}")
        print(f"Profiled f_esc = {fesc_p:.5f}")
        print(f"Profiled f_star = {fstar_p:.5f}")
        print(f"Δχ² = {dchi2:.4f}")

if __name__ == '__main__':
    main()
