#!/usr/bin/env python3
"""
Generate initial conditions for isolated NFW halos using galpy Eddington inversion.

Usage:
    python generate_ics.py --mass 1e10 --redshift 7 --npart 500000 --output ic_M10_z7.hdf5

Concentrations from Dutton & Maccio (2014). Velocities from the isotropic
NFW distribution function via Eddington inversion (Widrow 2000).
"""

import numpy as np
import argparse
import sys

def dutton_maccio_concentration(M, z):
    """Dutton & Maccio (2014) c-M relation."""
    a = 0.537 + (1.025 - 0.537) * np.exp(-0.718 * z**1.08)
    b = -0.097 + 0.024 * z
    log_c = a + b * (np.log10(M) - 12.0)
    return 10**log_c

def nfw_parameters(M, z):
    """Compute NFW halo parameters."""
    H0 = 67.74  # km/s/Mpc
    Om = 0.3089
    Hz = H0 * np.sqrt(Om * (1 + z)**3 + (1 - Om))
    rho_crit = 3 * (Hz * 1e5 / 3.0857e24)**2 / (8 * np.pi * 6.674e-8)
    rho_crit *= (3.0857e21)**3 / 1.989e33  # convert to M_sun / kpc^3

    c = dutton_maccio_concentration(M, z)
    r_vir = (3 * M / (4 * np.pi * 200 * rho_crit))**(1./3.)
    r_s = r_vir / c
    rho_s = (200. / 3.) * rho_crit * c**3 / (np.log(1 + c) - c / (1 + c))

    return r_vir, r_s, c, rho_s, rho_crit

def generate_nfw_ic(M, z, N, r_max_factor=3.0):
    """
    Generate IC positions and velocities for an NFW halo.

    Uses galpy's isotropicNFWdf for Eddington-inverted velocities.
    Falls back to Jeans equilibrium if galpy not available.
    """
    r_vir, r_s, c, rho_s, rho_crit = nfw_parameters(M, z)
    r_max = r_max_factor * r_vir
    mp = M / N

    print(f"  M_vir = {M:.2e} M_sun")
    print(f"  r_vir = {r_vir:.2f} kpc, r_s = {r_s:.3f} kpc, c = {c:.2f}")
    print(f"  N = {N}, m_p = {mp:.2e} M_sun")

    try:
        from galpy.df import isotropicNFWdf
        from galpy.potential import NFWPotential
        import galpy.util.conversion as conv

        # galpy uses natural units; set ro, vo
        ro = 8.0  # kpc
        vo = 220.0  # km/s

        pot = NFWPotential(conc=c, mvir=M / 1e12, H=67.74, overdens=200,
                           wrtcrit=True, ro=ro, vo=vo)
        df = isotropicNFWdf(pot=pot, rmax=r_max / ro, ro=ro, vo=vo)

        print("  Sampling from galpy isotropicNFWdf...")
        samples = df.sample(n=N)
        R = samples.R(use_physical=True)  # kpc
        z_cyl = samples.z(use_physical=True)  # kpc
        phi = samples.phi()
        vR = samples.vR(use_physical=True)  # km/s
        vT = samples.vT(use_physical=True)
        vz = samples.vz(use_physical=True)

        # Convert cylindrical to Cartesian
        x = R * np.cos(phi)
        y = R * np.sin(phi)
        z_pos = z_cyl
        vx = vR * np.cos(phi) - vT * np.sin(phi)
        vy = vR * np.sin(phi) + vT * np.cos(phi)
        vz_vel = vz

        pos = np.column_stack([x, y, z_pos])
        vel = np.column_stack([vx, vy, vz_vel])

        # Truncate beyond r_max
        r = np.sqrt(np.sum(pos**2, axis=1))
        mask = r < r_max
        pos = pos[mask]
        vel = vel[mask]
        print(f"  After truncation at {r_max:.1f} kpc: {len(pos)} particles")

    except ImportError:
        print("  galpy not found, using rejection sampling + Jeans velocities")
        # Rejection sampling for NFW density profile
        r = _sample_nfw_radii(N * 2, r_s, r_max)[:N]
        theta = np.arccos(2 * np.random.random(N) - 1)
        phi = 2 * np.pi * np.random.random(N)

        pos = np.column_stack([
            r * np.sin(theta) * np.cos(phi),
            r * np.sin(theta) * np.sin(phi),
            r * np.cos(theta)
        ])

        # Isotropic Jeans velocities (approximate)
        G = 4.302e-6
        x_arr = r / r_s
        M_enc = 4 * np.pi * rho_s * r_s**3 * (np.log(1 + x_arr) - x_arr / (1 + x_arr))
        v_circ = np.sqrt(G * M_enc / r)
        sigma_v = v_circ / np.sqrt(2)

        vel = np.random.randn(N, 3) * sigma_v[:, None]

    return pos, vel, mp

def _sample_nfw_radii(N, r_s, r_max):
    """Sample radii from NFW profile using inverse CDF."""
    x_max = r_max / r_s
    M_max = np.log(1 + x_max) - x_max / (1 + x_max)
    u = np.random.random(N) * M_max

    # Newton's method to invert M(x) = ln(1+x) - x/(1+x)
    x = np.ones(N) * x_max / 2
    for _ in range(20):
        Mx = np.log(1 + x) - x / (1 + x)
        dMx = 1 / (1 + x) - 1 / (1 + x) + x / (1 + x)**2
        dMx = x / (1 + x)**2
        x = x - (Mx - u) / dMx
        x = np.clip(x, 1e-6, x_max)

    return x * r_s

def write_hdf5(filepath, pos, vel, mp, boxsize=1000.0):
    """Write IC to GIZMO-compatible HDF5."""
    try:
        import h5py
    except ImportError:
        print("Error: h5py required for HDF5 output")
        sys.exit(1)

    N = len(pos)
    with h5py.File(filepath, 'w') as f:
        header = f.create_group('Header')
        header.attrs['NumPart_ThisFile'] = [0, N, 0, 0, 0, 0]
        header.attrs['NumPart_Total'] = [0, N, 0, 0, 0, 0]
        header.attrs['MassTable'] = [0, mp, 0, 0, 0, 0]
        header.attrs['Time'] = 0.0
        header.attrs['BoxSize'] = boxsize
        header.attrs['NumFilesPerSnapshot'] = 1

        pt1 = f.create_group('PartType1')
        pt1.create_dataset('Coordinates', data=pos.astype(np.float64))
        pt1.create_dataset('Velocities', data=vel.astype(np.float64))
        pt1.create_dataset('ParticleIDs', data=np.arange(1, N+1, dtype=np.uint32))

    print(f"  Written {N} particles to {filepath}")

def main():
    parser = argparse.ArgumentParser(description='Generate NFW halo ICs')
    parser.add_argument('--mass', type=float, required=True, help='Virial mass in M_sun')
    parser.add_argument('--redshift', type=float, required=True, help='Redshift')
    parser.add_argument('--npart', type=int, default=500000, help='Number of particles')
    parser.add_argument('--output', default='ic.hdf5', help='Output HDF5 file')
    parser.add_argument('--rmax_factor', type=float, default=3.0, help='Truncation radius in units of r_vir')
    args = parser.parse_args()

    print(f"Generating NFW IC: M={args.mass:.2e}, z={args.redshift}, N={args.npart}")
    pos, vel, mp = generate_nfw_ic(args.mass, args.redshift, args.npart, args.rmax_factor)
    write_hdf5(args.output, pos, vel, mp)
    print("Done.")

if __name__ == '__main__':
    main()
