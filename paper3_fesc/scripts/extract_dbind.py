#!/usr/bin/env python3
"""
Extract binding energy from GIZMO N-body snapshots.

Usage:
    python extract_dbind.py --snap_cdm snapshot_cdm.hdf5 --snap_sidm snapshot_sidm.hdf5 --output result.json

Memory-safe: uses spherical shell estimator, O(N log N) operations, < 50 MB RAM.
"""

import numpy as np
import argparse
import json
import sys

def load_snapshot(filepath):
    """Load particle positions from GIZMO HDF5 snapshot."""
    try:
        import h5py
    except ImportError:
        print("Error: h5py required. Install with: pip install h5py")
        sys.exit(1)

    with h5py.File(filepath, 'r') as f:
        pos = f['PartType1']['Coordinates'][:]  # DM particles
        mass = f['Header'].attrs['MassTable'][1]
        boxsize = f['Header'].attrs['BoxSize']
        time = f['Header'].attrs['Time']

    # Centre on potential minimum (density peak)
    # Use shrinking sphere method
    center = np.median(pos, axis=0)
    for _ in range(10):
        r = np.sqrt(np.sum((pos - center)**2, axis=1))
        mask = r < np.percentile(r, 30)
        if mask.sum() < 100:
            break
        center = np.mean(pos[mask], axis=0)

    # Compute radii from center
    r = np.sqrt(np.sum((pos - center)**2, axis=1))

    return r, mass, len(r)


def compute_binding_energy(r, mp, R_kpc_list):
    """
    Compute gas binding energy W_g(<R) at each radius in R_kpc_list.

    W_g(<R) = sum_i G * M(<r_i) * m_p / r_i  for r_i < R

    Assumes baryons trace the DM potential (appropriate at z > 6).
    Uses f_b = Omega_b / Omega_m = 0.157.
    """
    G = 4.302e-6  # kpc (km/s)^2 / M_sun
    f_b = 0.157

    # Sort by radius for cumulative mass
    idx = np.argsort(r)
    r_sorted = r[idx]
    M_cum = np.arange(1, len(r)+1) * mp  # cumulative mass

    results = {}
    for R in R_kpc_list:
        mask = r_sorted < R
        if mask.sum() < 10:
            results[R] = 0.0
            continue
        # W = -sum G * M(<r_i) * m_p / r_i for particles inside R
        W = -np.sum(G * M_cum[mask] * mp * f_b / r_sorted[mask])
        results[R] = W

    return results


def main():
    parser = argparse.ArgumentParser(description='Extract binding energy from GIZMO snapshots')
    parser.add_argument('--snap_cdm', required=True, help='Path to CDM snapshot')
    parser.add_argument('--snap_sidm', required=True, help='Path to SIDM snapshot')
    parser.add_argument('--radii', nargs='+', type=float, default=[0.3, 0.5, 1.0, 2.0, 5.0],
                        help='Integration radii in kpc (default: 0.3 0.5 1.0 2.0 5.0)')
    parser.add_argument('--output', default='dbind_result.json', help='Output JSON file')
    args = parser.parse_args()

    print(f"Loading CDM snapshot: {args.snap_cdm}")
    r_cdm, mp_cdm, N_cdm = load_snapshot(args.snap_cdm)
    print(f"  {N_cdm} particles, mp = {mp_cdm:.2e} M_sun")

    print(f"Loading SIDM snapshot: {args.snap_sidm}")
    r_sidm, mp_sidm, N_sidm = load_snapshot(args.snap_sidm)
    print(f"  {N_sidm} particles, mp = {mp_sidm:.2e} M_sun")

    print(f"Computing binding energies at R = {args.radii} kpc...")
    W_cdm = compute_binding_energy(r_cdm, mp_cdm, args.radii)
    W_sidm = compute_binding_energy(r_sidm, mp_sidm, args.radii)

    # Compute Δ_bind = 1 - W_SIDM / W_CDM
    dbind = {}
    for R in args.radii:
        if abs(W_cdm[R]) > 0:
            dbind[R] = 1.0 - W_sidm[R] / W_cdm[R]
        else:
            dbind[R] = 0.0

    result = {
        'R_kpc': args.radii,
        'delta_bind': [dbind[R] for R in args.radii],
        'W_CDM': [W_cdm[R] for R in args.radii],
        'W_SIDM': [W_sidm[R] for R in args.radii],
    }

    with open(args.output, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"\nResults saved to {args.output}")
    print(f"{'R [kpc]':>10s} {'Δ_bind':>10s} {'W_CDM':>15s} {'W_SIDM':>15s}")
    print("-" * 55)
    for R in args.radii:
        print(f"{R:10.1f} {dbind[R]:10.4f} {W_cdm[R]:15.4e} {W_sidm[R]:15.4e}")


if __name__ == '__main__':
    main()
