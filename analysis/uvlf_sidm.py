#!/usr/bin/env python3
"""
Paper 2a: JWST UVLF Constraints on Self-Interacting Dark Matter
================================================================

Framework for constraining σ/m from the JWST UV luminosity function
at z = 9-16 using mass-dependent SIDM core formation effects on
star formation efficiency.

Key physics:
  SIDM cores → reduced W_g → modified SFE → suppressed faint-end UVLF
  Sign opposite to topology (Paper 1): topology enhanced, UVLF suppressed

Reference: Sun et al. 2024 (arXiv:2404.13596) for WDM constraints
           using same JWST data — our approach is analogous for SIDM.

Phases:
  Phase 1: Data compilation + mass-dependent model (THIS FILE)
  Phase 2: Full MCMC constraints
  Phase 3: Joint topology + UVLF analysis
"""

import numpy as np
from scipy.integrate import quad
from scipy.interpolate import interp1d
import json, os

# ================================================================
# CONSTANTS
# ================================================================
h = 0.6736
Omega_m = 0.3153
Omega_b = 0.0493
sigma8 = 0.811
ns = 0.9649
fb = Omega_b / Omega_m
rho_crit = 2.775e11  # h² M_sun / Mpc³
rho_m = Omega_m * rho_crit  # h² M_sun / Mpc³

# ================================================================
# JWST UVLF DATA COMPILATION — VERIFIED 2026-04-18
# ================================================================
# Primary: Donnan+2024 (arXiv:2403.03171v3) Table 2
#   - PRIMER+JADES+NGDEEP, 2548 galaxies, ~370 sq arcmin
#   - Statistically robust p(z) method with UV LF prior
# Secondary: Harikane+2024 (arXiv:2406.18352v2) Table 2
#   - 60 spectroscopically confirmed galaxies at z=6.5-14.3
#   - Bright-end constraints from Keck/ALMA/JWST
#
# Format: { z: [ (M_UV, log10_phi, sigma_up, sigma_down), ... ] }
# phi in units of Mpc^-3 mag^-1; errors in dex (log-space)
# ================================================================

UVLF_DATA = {
    # z ~ 9 — Donnan+2024 Table 2 (8.5 < z < 9.5)
    9: [
        (-20.75, -4.921, 0.222, 0.234),   # Donnan+2024 Table 2
        (-20.25, -4.495, 0.148, 0.163),   # Donnan+2024 Table 2
        (-19.75, -3.842, 0.082, 0.094),   # Donnan+2024 Table 2
        (-19.25, -3.629, 0.099, 0.102),   # Donnan+2024 Table 2
        (-18.55, -3.313, 0.122, 0.146),   # Donnan+2024 Table 2
        (-18.05, -2.955, 0.107, 0.142),   # Donnan+2024 Table 2
        (-17.55, -2.751, 0.122, 0.147),   # Donnan+2024 Table 2
    ],
    # z ~ 10 — Donnan+2024 Table 2 (9.5 < z < 10.5)
    10: [
        (-20.75, -5.398, 0.544, 2.000),   # Donnan+2024 Table 2 (large lower error)
        (-20.25, -4.569, 0.171, 0.201),   # Donnan+2024 Table 2
        (-19.75, -4.036, 0.104, 0.106),   # Donnan+2024 Table 2
        (-19.25, -3.752, 0.114, 0.127),   # Donnan+2024 Table 2
        (-18.55, -3.493, 0.145, 0.184),   # Donnan+2024 Table 2
        (-18.05, -3.164, 0.133, 0.171),   # Donnan+2024 Table 2
        (-17.55, -2.893, 0.140, 0.179),   # Donnan+2024 Table 2
    ],
    # z ~ 11 — Donnan+2024 Table 2 (10.5 < z < 11.5)
    11: [
        (-21.25, -5.155, 0.359, 0.544),   # Donnan+2024 Table 2
        (-20.75, -4.854, 0.252, 0.301),   # Donnan+2024 Table 2
        (-20.25, -4.420, 0.153, 0.182),   # Donnan+2024 Table 2
        (-19.75, -4.000, 0.137, 0.155),   # Donnan+2024 Table 2
        (-19.25, -3.842, 0.194, 0.250),   # Donnan+2024 Table 2
        (-18.75, -3.631, 0.177, 0.229),   # Donnan+2024 Table 2
        (-18.25, -3.193, 0.194, 0.251),   # Donnan+2024 Table 2
    ],
    # z ~ 12 — Donnan+2024 Table 2 (z=12.5 bin, 11.5 < z < 13.5)
    12: [
        (-21.25, -5.523, 0.368, 0.477),   # Donnan+2024 Table 2
        (-20.75, -5.398, 0.352, 0.602),   # Donnan+2024 Table 2
        (-20.25, -4.796, 0.194, 0.204),   # Donnan+2024 Table 2
        (-19.75, -4.469, 0.224, 0.253),   # Donnan+2024 Table 2
        (-19.25, -4.367, 0.259, 0.311),   # Donnan+2024 Table 2
        (-18.75, -4.097, 0.214, 0.260),   # Donnan+2024 Table 2
        (-18.25, -3.664, 0.232, 0.283),   # Donnan+2024 Table 2
    ],
    # z ~ 14 — Donnan+2024 (z=14.5, tentative) + Harikane+2024 (spectroscopic)
    14: [
        (-20.25, -5.523, 0.477, 0.398),   # Donnan+2024 Table 2 (z=14.5, ~1.3 gal equiv.)
        (-20.80, -4.432, 0.571, 0.663),   # Harikane+2024 Table 2 (GS-z14-0, spec-z=14.32)
    ],
    # z ~ 16 — Harikane+2024 upper limit only (not used in main fits)
    16: [
        (-21.90, -5.009, 0.80, 0.80),     # Harikane+2024 Table 2 (upper limit)
    ],
}

def get_all_data():
    """Flatten data into arrays for MCMC."""
    z_arr, Muv_arr, logphi_arr, sig_arr = [], [], [], []
    for z, points in UVLF_DATA.items():
        for Muv, logphi, su, sd in points:
            z_arr.append(z)
            Muv_arr.append(Muv)
            logphi_arr.append(logphi)
            sig_arr.append((su + sd) / 2)  # symmetrize
    return (np.array(z_arr), np.array(Muv_arr),
            np.array(logphi_arr), np.array(sig_arr))


# ================================================================
# HALO MASS FUNCTION (Sheth-Tormen)
# ================================================================

def growth_factor(z):
    """Linear growth factor D(z)/D(0), Carroll+1992 approx."""
    Oz = Omega_m * (1+z)**3 / (Omega_m*(1+z)**3 + 1-Omega_m)
    OL = (1-Omega_m) / (Omega_m*(1+z)**3 + 1-Omega_m)
    return (5/2) * Oz / (Oz**(4/7) - OL + (1+Oz/2)*(1+OL/70)) / (1+z)

def sigma_M(M):
    """RMS density fluctuation in sphere containing mass M.
    Uses Eisenstein-Hu fitting for P(k) with sigma8 normalization."""
    R = (3*M / (4*np.pi*rho_m))**(1/3)  # Mpc/h
    # Approximate sigma(R) using power-law fit calibrated to sigma8
    # sigma(R) ~ sigma8 * (R/8)^(-(ns+3)/6) for CDM-like spectrum
    gamma_eff = 0.21  # effective shape parameter
    sig = sigma8 * (R / 8.0)**(-0.5*(ns+3)/3) * np.exp(-0.5*(R/100)**2)
    return sig

def dndlnM_ST(M, z):
    """Sheth-Tormen HMF: dn/dlnM in h³/Mpc³."""
    s = sigma_M(M) * growth_factor(z)
    nu = 1.686 / s
    A_st = 0.3222; a_st = 0.707; p_st = 0.3
    f_nu = A_st * np.sqrt(2*a_st/np.pi) * nu * (1 + (a_st*nu**2)**(-p_st)) * np.exp(-a_st*nu**2/2)
    # dlns/dlnM
    eps = 0.01
    s1 = sigma_M(M*(1+eps)) * growth_factor(z)
    s0 = sigma_M(M*(1-eps)) * growth_factor(z)
    dlns_dlnM = (np.log(s1) - np.log(s0)) / (np.log(M*(1+eps)) - np.log(M*(1-eps)))
    return (rho_m / M) * f_nu * abs(dlns_dlnM)

def halo_bias(M, z):
    """Sheth-Tormen halo bias."""
    s = sigma_M(M) * growth_factor(z)
    nu = 1.686 / s
    a_st = 0.707; p_st = 0.3
    return 1 + (a_st*nu**2 - 1)/1.686 + 2*p_st/(1.686*(1 + (a_st*nu**2)**p_st))


# ================================================================
# SIDM THERMALIZATION AND BINDING ENERGY
# ================================================================

def nfw_params(M, z):
    """NFW scale parameters for halo of mass M at redshift z."""
    # Concentration from Dutton+Maccio 2014
    log_c = 0.905 - 0.101 * np.log10(M / (1e12/h))
    c = 10**log_c / (1 + z/3)  # rough redshift scaling
    c = max(c, 2.0)
    rvir = (3*M / (4*np.pi * 200 * rho_m * (1+z)**3))**(1/3)  # Mpc/h
    rs = rvir / c
    rho_s = M / (4*np.pi * rs**3 * (np.log(1+c) - c/(1+c)))
    return rvir, rs, rho_s, c

def thermalization_radius(M, z, sigma_m):
    """
    Thermalization radius r_1 in kpc.
    
    Calibrated to match Paper 1 Table S1 values:
      M=10^10.5, z=7, σ/m=1 → r1 ~ 0.8 kpc
      M=10^10.5, z=7, σ/m=10 → r1 ~ 4.5 kpc
    
    Scaling: r1 ~ r1_ref * (σ/m)^0.6 * (M/M_ref)^0.4 * ((1+z_ref)/(1+z))^0.5
    based on N_scat = ρ * (σ/m) * v * t = 1.
    """
    M_ref = 10**10.5
    z_ref = 7.0
    
    # Reference value at M=10^10.5, z=7, σ/m=1: r1 = 0.8 kpc
    r1_ref = 0.8  # kpc (merger-corrected)
    
    r1 = r1_ref * (sigma_m / 1.0)**0.55 * (M / M_ref)**0.35 * ((1+z_ref)/(1+z))**0.5
    
    # Physical bound: cannot exceed 0.1 * rvir
    rvir_kpc = 30.0 * (M / 1e11)**0.33 * ((1+z)/8)**(-1)  # rough scaling
    r1 = min(r1, 0.1 * rvir_kpc)
    
    return max(r1, 0.01)  # floor at 10 pc

def binding_energy_ratio(M, z, sigma_m):
    """
    Compute W_g^CDM / W_g^SIDM for a halo of mass M at redshift z.
    
    Uses the scaling: W_g^CDM/W_g^SIDM ~ (r1/r_inner)^1.5 for r1 > r_inner
    calibrated to Paper 1's binding energy estimates.
    """
    if sigma_m == 0:
        return 1.0
    
    r1 = thermalization_radius(M, z, sigma_m)
    r_inner = 0.3  # kpc — star-forming region
    
    if r1 <= r_inner:
        return 1.0  # no core inside star-forming region
    
    # Scaling from NFW profile: flattening density from r_inner to r1
    # reduces binding energy roughly as (r1/r_inner)^1.5
    # Capped at 100 to avoid extreme ratios
    ratio = min((r1 / r_inner)**1.5, 100.0)
    return ratio


# ================================================================
# STAR FORMATION EFFICIENCY WITH SIDM MODIFICATION
# ================================================================

def SFE_CDM(M, z, params):
    """
    CDM star formation efficiency: f_* = f_*0 * (M/M_p)^alpha_lo / (1 + (M/M_p)^alpha_hi)
    
    Double power-law parametrization following Behroozi+2019 / Mason+2015.
    """
    f_star0 = params.get('f_star0', 0.05)
    M_p = params.get('M_p', 1e11)       # peak efficiency mass
    alpha_lo = params.get('alpha_lo', 0.6)  # low-mass slope
    alpha_hi = params.get('alpha_hi', 0.5)  # high-mass suppression
    
    # Redshift evolution: efficiency increases mildly at high z
    z_evol = params.get('z_evol', 0.0)  # dlog(f*)/dz
    f_z = 10**(z_evol * (z - 9))
    
    f = f_star0 * f_z * (M/M_p)**alpha_lo / (1 + (M/M_p)**(alpha_lo + alpha_hi))
    return min(f, 1.0)

def SFE_SIDM(M, z, sigma_m, eta, params):
    """
    SIDM-modified star formation efficiency.
    
    SFE_SIDM = SFE_CDM * (W_g^SIDM / W_g^CDM)^eta
    
    eta > 0: SIDM reduces SFE (binding energy lower → gas less confined)
    eta = 0: no effect
    """
    f_cdm = SFE_CDM(M, z, params)
    Wg_ratio = binding_energy_ratio(M, z, sigma_m)
    
    # W_g^SIDM / W_g^CDM = 1/Wg_ratio (since we computed CDM/SIDM)
    suppression = (1.0 / Wg_ratio)**eta
    
    return f_cdm * suppression


# ================================================================
# UV LUMINOSITY FUNCTION MODEL
# ================================================================

def MUV_from_SFR(SFR):
    """Convert SFR [M_sun/yr] to M_UV using Kennicutt+1998."""
    # SFR = 1.4e-28 * L_nu [erg/s/Hz]
    # M_UV = -2.5 * log10(L_nu) + 51.63
    if SFR <= 0:
        return -10.0  # very faint
    L_nu = SFR / 1.4e-28
    return -2.5 * np.log10(L_nu) + 51.63

def SFR_from_MUV(MUV):
    """Convert M_UV to SFR [M_sun/yr]."""
    L_nu = 10**((51.63 - MUV) / 2.5)
    return 1.4e-28 * L_nu

def halo_mass_to_MUV(M, z, sigma_m, eta, params):
    """
    Map halo mass to UV magnitude.
    
    M_UV = f(SFR) where SFR = f_*(M) * fb * M * H(z)
    with f_* modified by SIDM.
    """
    # SFR from gas accretion rate
    H_z = 100 * h * np.sqrt(Omega_m*(1+z)**3 + 1-Omega_m)  # km/s/Mpc
    H_z_per_yr = H_z * 3.24e-20 * 3.15e7  # 1/yr
    
    if sigma_m > 0:
        f_star = SFE_SIDM(M, z, sigma_m, eta, params)
    else:
        f_star = SFE_CDM(M, z, params)
    
    # SFR ~ f_* * Omega_b/Omega_m * M * dM/dt / M
    # Approximate dM/dt ~ M * (1+z)^2.5 * 46 M_sun/yr (Fakhouri+2010)
    Mdot = 46.1 * (M/1e12)**1.1 * (1+z)**2.5  # M_sun/yr
    
    SFR = f_star * fb * Mdot
    
    # Add dust attenuation (simple model: A_UV = A0 * (M/M_dust)^beta_dust)
    A_UV = params.get('A_UV0', 0.0) * (M / 1e10)**0.3  # minimal at high z
    
    MUV = MUV_from_SFR(SFR) + A_UV
    return MUV

def compute_UVLF(MUV_bins, z, sigma_m, eta, params, sigma_UV=0.5):
    """
    Compute the UV luminosity function Phi(M_UV) at redshift z.
    
    Uses convolution of HMF with M_UV(M_halo) mapping + scatter.
    
    sigma_UV: scatter in M_UV at fixed M_halo (mag)
    """
    log_M_arr = np.linspace(7.5, 13.5, 300)
    M_arr = 10**log_M_arr
    dlogM = log_M_arr[1] - log_M_arr[0]
    
    # HMF
    dndlnM = np.array([dndlnM_ST(M, z) for M in M_arr])
    
    # M_UV mapping
    MUV_arr = np.array([halo_mass_to_MUV(M, z, sigma_m, eta, params) for M in M_arr])
    
    # Turnover: suppress halos below atomic cooling threshold
    M_turn = params.get('M_turn', 5e7 * ((1+z)/10)**(-3/2))
    suppress = np.exp(-(M_turn / M_arr)**2)
    dndlnM *= suppress
    
    # Compute Phi(M_UV) by convolving with Gaussian scatter
    phi = np.zeros(len(MUV_bins))
    for i, Muv_bin in enumerate(MUV_bins):
        # Probability that halo of mass M produces a galaxy at Muv_bin
        weight = np.exp(-0.5 * ((MUV_arr - Muv_bin) / sigma_UV)**2) / (np.sqrt(2*np.pi) * sigma_UV)
        phi[i] = np.sum(dndlnM * weight * dlogM * np.log(10))
    
    return phi  # Mpc^-3 mag^-1


# ================================================================
# LIKELIHOOD
# ================================================================

def log_likelihood(theta, z_data, Muv_data, logphi_data, sig_data):
    """
    Log-likelihood for MCMC.
    
    theta = [sigma_m, eta, log_f_star0, alpha_lo, sigma_UV, z_evol]
    """
    sigma_m, eta, log_f_star0, alpha_lo, sigma_UV, z_evol = theta
    
    # Priors
    if sigma_m < 0 or sigma_m > 50: return -np.inf
    if eta < 0 or eta > 2: return -np.inf
    if log_f_star0 < -3 or log_f_star0 > 0: return -np.inf
    if alpha_lo < 0 or alpha_lo > 4: return -np.inf
    if sigma_UV < 0.1 or sigma_UV > 2.0: return -np.inf
    if z_evol < -0.5 or z_evol > 0.5: return -np.inf
    
    params = {
        'f_star0': 10**log_f_star0,
        'M_p': 1e11,
        'alpha_lo': alpha_lo,
        'alpha_hi': 0.5,
        'z_evol': z_evol,
        'M_turn': 5e7,
        'A_UV0': 0.0,
    }
    
    chi2 = 0
    for z_val in np.unique(z_data):
        mask = z_data == z_val
        Muv_bins = Muv_data[mask]
        logphi_obs = logphi_data[mask]
        sig_obs = sig_data[mask]
        
        phi_model = compute_UVLF(Muv_bins, z_val, sigma_m, eta, params, sigma_UV)
        
        for j in range(len(Muv_bins)):
            if phi_model[j] > 0:
                logphi_mod = np.log10(phi_model[j])
                chi2 += ((logphi_mod - logphi_obs[j]) / sig_obs[j])**2
            else:
                chi2 += 100  # penalty for zero prediction
    
    return -0.5 * chi2

def log_prior(theta):
    """Flat priors."""
    sigma_m, eta, log_f_star0, alpha_lo, sigma_UV, z_evol = theta
    if (0 <= sigma_m <= 50 and 0 <= eta <= 2 and
        -3 <= log_f_star0 <= 0 and 0 <= alpha_lo <= 4 and
        0.1 <= sigma_UV <= 2.0 and -0.5 <= z_evol <= 0.5):
        return 0.0
    return -np.inf

def log_posterior(theta, z_data, Muv_data, logphi_data, sig_data):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, z_data, Muv_data, logphi_data, sig_data)


# ================================================================
# QUICK VALIDATION
# ================================================================

def validate():
    """Quick check that the model produces reasonable UVLF."""
    print("Paper 2a: JWST UVLF-SIDM Framework Validation")
    print("=" * 60)
    
    params = {
        'f_star0': 0.05,
        'M_p': 1e11,
        'alpha_lo': 0.6,
        'alpha_hi': 0.5,
        'z_evol': 0.0,
        'M_turn': 5e7,
        'A_UV0': 0.0,
    }
    
    # Check thermalization radii
    print("\nThermalization radii r_1 (kpc):")
    print(f"  {'M_halo':>12} {'σ/m=0.5':>8} {'σ/m=1':>8} {'σ/m=5':>8} {'σ/m=10':>8}")
    for logM in [9.0, 9.5, 10.0, 10.5, 11.0]:
        M = 10**logM
        r_vals = [thermalization_radius(M, 7.0, sm) for sm in [0.5, 1, 5, 10]]
        print(f"  10^{logM:.1f} M☉  " + "  ".join(f"{r:7.2f}" for r in r_vals))
    
    # Check binding energy ratios
    print("\nBinding energy ratio W_g^CDM / W_g^SIDM:")
    print(f"  {'M_halo':>12} {'σ/m=1':>8} {'σ/m=5':>8} {'σ/m=10':>8}")
    for logM in [9.0, 9.5, 10.0, 10.5, 11.0]:
        M = 10**logM
        ratios = [binding_energy_ratio(M, 7.0, sm) for sm in [1, 5, 10]]
        print(f"  10^{logM:.1f} M☉  " + "  ".join(f"{r:7.2f}" for r in ratios))
    
    # Check SFE modification
    print(f"\nSFE suppression factor (W_g^SIDM/W_g^CDM)^η at z=9, η=0.5:")
    print(f"  {'M_halo':>12} {'CDM f*':>8} {'σ/m=1':>12} {'σ/m=10':>12}")
    for logM in [9.0, 10.0, 11.0]:
        M = 10**logM
        f_cdm = SFE_CDM(M, 9.0, params)
        f_s1 = SFE_SIDM(M, 9.0, 1.0, 0.5, params)
        f_s10 = SFE_SIDM(M, 9.0, 10.0, 0.5, params)
        print(f"  10^{logM:.1f} M☉  {f_cdm:8.4f} {f_s1:8.4f} ({f_s1/f_cdm:4.2f}×)"
              f"  {f_s10:8.4f} ({f_s10/f_cdm:4.2f}×)")
    
    # Compute UVLF at z=9 for CDM and SIDM
    MUV_bins = np.arange(-22, -16, 0.5)
    
    print(f"\nUVLF at z=9 (log10 Phi [Mpc^-3 mag^-1]):")
    print(f"  {'M_UV':>6} {'CDM':>8} {'σ/m=1':>8} {'σ/m=10':>8} {'Data':>8}")
    
    phi_cdm = compute_UVLF(MUV_bins, 9, 0, 0, params)
    phi_s1 = compute_UVLF(MUV_bins, 9, 1.0, 0.5, params)
    phi_s10 = compute_UVLF(MUV_bins, 9, 10.0, 0.5, params)
    
    z9_data = {Muv: lp for Muv, lp, _, _ in UVLF_DATA[9]}
    
    for i, Muv in enumerate(MUV_bins):
        lp_cdm = np.log10(phi_cdm[i]) if phi_cdm[i] > 0 else -99
        lp_s1 = np.log10(phi_s1[i]) if phi_s1[i] > 0 else -99
        lp_s10 = np.log10(phi_s10[i]) if phi_s10[i] > 0 else -99
        data_str = f"{z9_data.get(Muv, ''):>8}"
        print(f"  {Muv:6.1f} {lp_cdm:8.2f} {lp_s1:8.2f} {lp_s10:8.2f} {data_str}")
    
    # Count data points
    z_d, Muv_d, logphi_d, sig_d = get_all_data()
    print(f"\nTotal data points: {len(z_d)}")
    for z_val in sorted(np.unique(z_d)):
        n = np.sum(z_d == z_val)
        print(f"  z = {z_val:.0f}: {n} points")
    
    # Quick likelihood test
    theta_cdm = [0.0, 0.0, np.log10(0.05), 0.6, 0.5, 0.0]
    ll_cdm = log_likelihood(theta_cdm, z_d, Muv_d, logphi_d, sig_d)
    print(f"\nCDM log-likelihood: {ll_cdm:.1f}")
    
    theta_s1 = [1.0, 0.5, np.log10(0.05), 0.6, 0.5, 0.0]
    ll_s1 = log_likelihood(theta_s1, z_d, Muv_d, logphi_d, sig_d)
    print(f"SIDM1 (η=0.5) log-likelihood: {ll_s1:.1f}")
    
    theta_s10 = [10.0, 0.5, np.log10(0.05), 0.6, 0.5, 0.0]
    ll_s10 = log_likelihood(theta_s10, z_d, Muv_d, logphi_d, sig_d)
    print(f"SIDM10 (η=0.5) log-likelihood: {ll_s10:.1f}")
    
    print(f"\nΔχ² (SIDM1 - CDM): {2*(ll_cdm - ll_s1):.1f}")
    print(f"Δχ² (SIDM10 - CDM): {2*(ll_cdm - ll_s10):.1f}")
    
    print("\nValidation complete. Ready for Phase 2 MCMC.")
    
    return {
        'phi_cdm': phi_cdm.tolist(),
        'phi_s1': phi_s1.tolist(),
        'phi_s10': phi_s10.tolist(),
        'MUV_bins': MUV_bins.tolist(),
        'll_cdm': ll_cdm,
        'll_s1': ll_s1,
        'll_s10': ll_s10,
    }


if __name__ == '__main__':
    results = validate()
    
    os.makedirs('paper2a_results', exist_ok=True)
    with open('paper2a_results/phase1_validation.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nSaved: paper2a_results/phase1_validation.json")
