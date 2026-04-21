#!/usr/bin/env python3
"""
Joint UVLF + topology constraint analysis.

Reads profile_scan_results.json from the cluster scan and computes
the joint (sigma/m, eta) constraint by combining:
  - UVLF+SMF exclusion (from profile likelihood + Weibel+2024 prior)
  - Topology detection (from calibrated Paper 1 SKA forecast)

Usage:
  python joint_constraint.py [path_to_profile_scan_results.json]
"""

import numpy as np
import json
import sys

# ================================================================
# TOPOLOGY MODEL (calibrated to Paper 1)
# ================================================================

def p_sidm_blowout(sigma_m, alpha_blowout=0.7):
    """Duty cycle from blowout model (Paper 1 Eq. 6)."""
    r1 = 0.8 * (sigma_m / 1.0)**0.5  # kpc at M=10^10.5, z=7
    r_inner = 0.3  # kpc
    if r1 < r_inner:
        return 0.10
    delta_W = min(0.26 * sigma_m**0.6, 0.95)
    W_ratio = 1.0 / (1.0 - delta_W)
    return min(0.10 * W_ratio**alpha_blowout, 0.50)


def ska_snr(sigma_m, sys_floor=0.05):
    """
    Cumulative SKA1-Low SNR for SIDM detection.
    Calibrated: SIDM10 -> 15 sigma at 5% floor (Paper 1).
    """
    ps = p_sidm_blowout(sigma_m)
    if ps <= 0.10:
        return 0.0
    signal = 1.0 - 0.10 / ps
    signal_10 = 1.0 - 0.10 / p_sidm_blowout(10.0)
    snr_10 = min(15.0 * (0.05 / max(sys_floor, 0.01)), 25.0)
    return snr_10 * (signal / signal_10)


# ================================================================
# MAIN
# ================================================================

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else 'data/profile_scan_results.json'
    with open(path) as f:
        d = json.load(f)

    b = d['cdm_baseline']
    g = d['grids']
    sm_grid = np.array(g['sigma_m'])
    eta_grid = np.array(g['eta'])
    f0_cdm = b['f_star0']

    print("=" * 70)
    print("CDM BASELINE")
    print("=" * 70)
    print(f"  f_star0  = {b['f_star0']:.6f}")
    print(f"  alpha_lo = {b['alpha_lo']:.4f}")
    print(f"  sigma_UV = {b['sigma_UV']:.4f} mag")
    print(f"  z_evol   = {b['z_evol']:.4f}")
    print(f"  chi2/dof = {b['chi2']:.2f}/{31-4} = {b['chi2_per_dof']:.4f}")

    # UVLF+SMF limits
    f0_2sig = 0.023  # Weibel+2024 2-sigma upper limit
    print(f"\n{'='*70}")
    print(f"UVLF+SMF LIMITS (f_star0 > {f0_2sig} threshold)")
    print(f"{'='*70}")
    print(f"{'eta':>6} | {'sigma/m limit':>14} | {'f0 at limit':>12}")
    print("-" * 40)
    smf_limits = d['limits_95CL'].get('uvlf_smf', {})
    for key in sorted(smf_limits.keys(), key=float):
        print(f"{key:>6} | {smf_limits[key]:>14.3f} | ...")

    # Topology thresholds
    print(f"\n{'='*70}")
    print(f"TOPOLOGY DETECTION THRESHOLDS (3-sigma)")
    print(f"{'='*70}")
    sm_fine = np.linspace(0.01, 20, 5000)
    for floor, label in [(0.05, '5%'), (0.10, '10%'), (0.20, '20%')]:
        snr_fine = np.array([ska_snr(s, floor) for s in sm_fine])
        idx = np.where(snr_fine >= 3)[0]
        thresh = sm_fine[idx[0]] if len(idx) > 0 else float('inf')
        print(f"  {label} floor: sigma/m >= {thresh:.2f} cm^2/g")

    # Joint constraint table
    print(f"\n{'='*70}")
    print(f"JOINT CONSTRAINT")
    print(f"{'='*70}")
    print(f"{'eta':>6} | {'UVLF+SMF':>14} | {'Topo (5%)':>10} | {'Topo (10%)':>11} | {'Status':>25}")
    print("-" * 80)
    for eta_str in sorted(smf_limits.keys(), key=float):
        eta_val = float(eta_str)
        sm_lim = smf_limits[eta_str]
        snr5 = ska_snr(sm_lim, 0.05)
        snr10 = ska_snr(sm_lim, 0.10)

        if sm_lim < 0.76:
            status = "Both probes reject (5%)"
        elif sm_lim < 2.26:
            status = "UVLF excl, topo detects (5%)"
        else:
            status = "UVLF excl, topo needs <10%"

        print(f"{eta_val:>6.2f} | sm < {sm_lim:>8.3f} | {snr5:>10.1f}s | {snr10:>10.1f}s | {status:>25}")


if __name__ == '__main__':
    main()
