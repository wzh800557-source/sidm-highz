#!/usr/bin/env python3
"""
Paper 2a/merged: Complete profile likelihood scan on cluster.

Self-contained — no imports from uvlf_sidm.py needed.
Outputs: profile_scan_results.json with all numbers for the paper.

Usage:
  python3 run_profile_scan.py              # full scan (~2-4 hours)
  python3 run_profile_scan.py --quick      # reduced grid (~20 min, for testing)

On MIT Engaging:
  sbatch submit_profile.sh
"""

import numpy as np
from scipy.optimize import minimize
import json, os, sys, time
from datetime import datetime

QUICK = '--quick' in sys.argv

# ================================================================
# COSMOLOGY
# ================================================================
h = 0.6736; Omega_m = 0.3153; Omega_b = 0.0493
sigma8 = 0.811; ns = 0.9649
fb = Omega_b / Omega_m
rho_crit = 2.775e11; rho_m = Omega_m * rho_crit

# ================================================================
# VERIFIED JWST DATA (Donnan+2024 Table 2 + Harikane+2024 Table 2)
# 31 points: z = 9(7), 10(7), 11(7), 12(7), 14(2), 16(1)
# ================================================================
UVLF_DATA = {
    9: [(-20.75,-4.921,0.222,0.234),(-20.25,-4.495,0.148,0.163),
        (-19.75,-3.842,0.082,0.094),(-19.25,-3.629,0.099,0.102),
        (-18.55,-3.313,0.122,0.146),(-18.05,-2.955,0.107,0.142),
        (-17.55,-2.751,0.122,0.147)],
    10:[(-20.75,-5.398,0.544,2.000),(-20.25,-4.569,0.171,0.201),
        (-19.75,-4.036,0.104,0.106),(-19.25,-3.752,0.114,0.127),
        (-18.55,-3.493,0.145,0.184),(-18.05,-3.164,0.133,0.171),
        (-17.55,-2.893,0.140,0.179)],
    11:[(-21.25,-5.155,0.359,0.544),(-20.75,-4.854,0.252,0.301),
        (-20.25,-4.420,0.153,0.182),(-19.75,-4.000,0.137,0.155),
        (-19.25,-3.842,0.194,0.250),(-18.75,-3.631,0.177,0.229),
        (-18.25,-3.193,0.194,0.251)],
    12:[(-21.25,-5.523,0.368,0.477),(-20.75,-5.398,0.352,0.602),
        (-20.25,-4.796,0.194,0.204),(-19.75,-4.469,0.224,0.253),
        (-19.25,-4.367,0.259,0.311),(-18.75,-4.097,0.214,0.260),
        (-18.25,-3.664,0.232,0.283)],
    14:[(-20.25,-5.523,0.477,0.398),(-20.80,-4.432,0.571,0.663)],
    16:[(-21.90,-5.009,0.80,0.80)],
}

def get_all_data():
    z_arr, Muv_arr, logphi_arr, sig_arr = [], [], [], []
    for z, points in UVLF_DATA.items():
        for Muv, logphi, su, sd in points:
            z_arr.append(z); Muv_arr.append(Muv)
            logphi_arr.append(logphi); sig_arr.append((su+sd)/2)
    return np.array(z_arr), np.array(Muv_arr), np.array(logphi_arr), np.array(sig_arr)

# ================================================================
# HMF (Sheth-Tormen)
# ================================================================
def growth_factor(z):
    Oz = Omega_m*(1+z)**3 / (Omega_m*(1+z)**3 + 1-Omega_m)
    OL = (1-Omega_m) / (Omega_m*(1+z)**3 + 1-Omega_m)
    return (5/2)*Oz / (Oz**(4/7) - OL + (1+Oz/2)*(1+OL/70)) / (1+z)

def sigma_M(M):
    R = (3*M / (4*np.pi*rho_m))**(1/3)
    return sigma8 * (R/8.0)**(-0.5*(ns+3)/3) * np.exp(-0.5*(R/100)**2)

def dndlnM_ST(M, z):
    s = sigma_M(M) * growth_factor(z)
    nu = 1.686 / s
    A_st=0.3222; a_st=0.707; p_st=0.3
    f_nu = A_st*np.sqrt(2*a_st/np.pi)*nu*(1+(a_st*nu**2)**(-p_st))*np.exp(-a_st*nu**2/2)
    eps=0.01
    s1 = sigma_M(M*(1+eps))*growth_factor(z)
    s0 = sigma_M(M*(1-eps))*growth_factor(z)
    dlns = (np.log(s1)-np.log(s0)) / (np.log(M*(1+eps))-np.log(M*(1-eps)))
    return (rho_m/M) * f_nu * abs(dlns)

# ================================================================
# SIDM MODEL
# ================================================================
def thermalization_radius(M, z, sigma_m):
    r1 = 0.8 * (sigma_m/1.0)**0.55 * (M/10**10.5)**0.35 * (8.0/(1+z))**0.5
    rvir_kpc = 30.0 * (M/1e11)**0.33 * ((1+z)/8)**(-1)
    return max(min(r1, 0.1*rvir_kpc), 0.01)

def binding_energy_ratio(M, z, sigma_m):
    if sigma_m == 0: return 1.0
    r1 = thermalization_radius(M, z, sigma_m)
    if r1 <= 0.3: return 1.0
    return min((r1/0.3)**1.5, 100.0)

# ================================================================
# SFE + UVLF
# ================================================================
def SFE(M, z, sigma_m, eta, params):
    f0 = params['f_star0']; Mp = params['M_p']
    alo = params['alpha_lo']; ahi = params['alpha_hi']
    fz = 10**(params['z_evol'] * (z-9))
    f = f0 * fz * (M/Mp)**alo / (1 + (M/Mp)**(alo+ahi))
    f = min(f, 1.0)
    if sigma_m > 0 and eta > 0:
        Wr = binding_energy_ratio(M, z, sigma_m)
        f *= (1.0/Wr)**eta
    return f

def halo_to_MUV(M, z, sigma_m, eta, params):
    f = SFE(M, z, sigma_m, eta, params)
    Mdot = 46.1 * (M/1e12)**1.1 * (1+z)**2.5
    SFR = f * fb * Mdot
    if SFR <= 0: return -10.0
    L_nu = SFR / 1.4e-28
    return -2.5*np.log10(L_nu) + 51.63

def compute_UVLF(MUV_bins, z, sigma_m, eta, params, sigma_UV):
    logM = np.linspace(7.5, 13.5, 300)
    M_arr = 10**logM; dlogM = logM[1]-logM[0]
    dndlnM = np.array([dndlnM_ST(M, z) for M in M_arr])
    MUV_arr = np.array([halo_to_MUV(M, z, sigma_m, eta, params) for M in M_arr])
    M_turn = params.get('M_turn', 5e7*((1+z)/10)**(-1.5))
    dndlnM *= np.exp(-(M_turn/M_arr)**2)
    phi = np.zeros(len(MUV_bins))
    for i, Mb in enumerate(MUV_bins):
        w = np.exp(-0.5*((MUV_arr-Mb)/sigma_UV)**2) / (np.sqrt(2*np.pi)*sigma_UV)
        phi[i] = np.sum(dndlnM * w * dlogM * np.log(10))
    return phi

# ================================================================
# LIKELIHOOD
# ================================================================
def log_likelihood(theta, z_d, Muv_d, logphi_d, sig_d):
    sm, eta, lf0, alo, suv, zev = theta
    if (sm<0 or sm>50 or eta<0 or eta>2 or lf0<-3 or lf0>0
        or alo<0 or alo>4 or suv<0.1 or suv>2.5 or zev<-1 or zev>1):
        return -np.inf
    params = {'f_star0':10**lf0, 'M_p':1e11, 'alpha_lo':alo,
              'alpha_hi':0.5, 'z_evol':zev, 'M_turn':5e7, 'A_UV0':0.0}
    chi2 = 0
    for zv in np.unique(z_d):
        mask = z_d == zv
        phi_mod = compute_UVLF(Muv_d[mask], zv, sm, eta, params, suv)
        for j in range(np.sum(mask)):
            if phi_mod[j] > 0:
                chi2 += ((np.log10(phi_mod[j]) - logphi_d[mask][j]) / sig_d[mask][j])**2
            else:
                chi2 += 100
    return -0.5 * chi2

# ================================================================
# TOPOLOGY LIKELIHOOD (calibrated to Paper 1)
# ================================================================
def p_sidm_blowout(sigma_m, alpha_blowout=0.7):
    r1 = 0.8 * (sigma_m/1.0)**0.5
    if r1 < 0.3: return 0.10
    dW = min(0.26 * sigma_m**0.6, 0.95)
    return min(0.10 * (1.0/(1.0-dW))**alpha_blowout, 0.50)

def ska_snr(sigma_m, sys_floor=0.05):
    """SKA cumulative SNR, calibrated: SIDM10 → 15σ at 5% floor."""
    ps = p_sidm_blowout(sigma_m)
    if ps <= 0.10: return 0.0
    sig = 1.0 - 0.10/ps
    sig10 = 1.0 - 0.10/p_sidm_blowout(10.0)
    snr10 = min(15.0 * (0.05/max(sys_floor, 0.01)), 25.0)
    return snr10 * (sig/sig10)

# ================================================================
# MAIN SCAN
# ================================================================
def main():
    t0 = time.time()
    z_d, Muv_d, logphi_d, sig_d = get_all_data()
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Loaded {len(z_d)} data points")

    # --- CDM baseline ---
    def neg_ll(p, sm=0, eta=0):
        val = log_likelihood((sm, eta, *p), z_d, Muv_d, logphi_d, sig_d)
        return -val if np.isfinite(val) else 1e10

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Finding CDM baseline...")
    best_nll, best_p = 1e10, None
    for lf0 in np.arange(-2.2, -1.2, 0.15):
        for alo in np.arange(0.8, 3.0, 0.3):
            for suv in np.arange(0.4, 1.4, 0.2):
                for zev in [-0.1, 0.0, 0.1, 0.2]:
                    val = neg_ll([lf0, alo, suv, zev])
                    if val < best_nll:
                        best_nll, best_p = val, [lf0, alo, suv, zev]

    res_cdm = minimize(neg_ll, best_p, method='Nelder-Mead',
                       options={'maxiter':30000, 'xatol':1e-8, 'fatol':1e-8})
    p_cdm = res_cdm.x
    ll_cdm = -res_cdm.fun
    chi2_cdm = -2*ll_cdm

    print(f"[{datetime.now().strftime('%H:%M:%S')}] CDM baseline:")
    print(f"  f★₀={10**p_cdm[0]:.5f}  α_lo={p_cdm[1]:.4f}  "
          f"σ_UV={p_cdm[2]:.4f}  z_evol={p_cdm[3]:.4f}")
    print(f"  χ²={chi2_cdm:.2f}  χ²/dof={chi2_cdm/(len(z_d)-4):.3f}")

    # --- Grid definition ---
    if QUICK:
        sm_grid = np.array([0.0, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0])
        eta_grid = np.array([0.05, 0.10, 0.25, 0.50, 1.00])
    else:
        sm_grid = np.array([0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5,
                            0.7, 1.0, 1.5, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0])
        eta_grid = np.array([0.01, 0.03, 0.05, 0.07, 0.10, 0.12, 0.15, 0.20,
                             0.25, 0.30, 0.40, 0.50, 0.70, 1.00])

    n_total = len(sm_grid) * len(eta_grid)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Scanning {len(sm_grid)}×{len(eta_grid)} = {n_total} grid points")

    # --- Profile scan ---
    results = {}
    dchi2_cond = np.full((len(sm_grid), len(eta_grid)), np.nan)
    dchi2_prof = np.full((len(sm_grid), len(eta_grid)), np.nan)
    f0_shift = np.full((len(sm_grid), len(eta_grid)), np.nan)
    suv_shift = np.full((len(sm_grid), len(eta_grid)), np.nan)

    done = 0
    for i, sm in enumerate(sm_grid):
        for j, eta in enumerate(eta_grid):
            done += 1
            # Conditional
            ll_cond = log_likelihood((sm, eta, *p_cdm), z_d, Muv_d, logphi_d, sig_d)
            dc2_c = -2*(ll_cond - ll_cdm)
            dchi2_cond[i,j] = dc2_c

            # Profiled: re-optimize nuisance params
            res_p = minimize(neg_ll, p_cdm, args=(sm, eta), method='Nelder-Mead',
                            options={'maxiter':10000, 'xatol':1e-6, 'fatol':1e-6})
            ll_p = -res_p.fun
            dc2_p = -2*(ll_p - ll_cdm)
            dchi2_prof[i,j] = dc2_p

            f0_new = 10**res_p.x[0]
            f0_cdm = 10**p_cdm[0]
            df0 = (f0_new/f0_cdm - 1) * 100  # percent
            ds = res_p.x[2] - p_cdm[2]
            f0_shift[i,j] = df0
            suv_shift[i,j] = ds

            results[f"sm{sm:.2f}_eta{eta:.2f}"] = {
                'sigma_m': float(sm), 'eta': float(eta),
                'dchi2_cond': round(float(dc2_c), 2),
                'dchi2_prof': round(float(dc2_p), 3),
                'best_fit': [round(float(x), 5) for x in res_p.x],
                'f0_new': round(float(f0_new), 5),
                'f0_shift_pct': round(float(df0), 1),
                'suv_shift': round(float(ds), 4),
                'chi2_total': round(float(-2*ll_p), 2),
            }

            if done % 10 == 0 or done == n_total:
                elapsed = time.time() - t0
                rate = done / elapsed
                eta_time = (n_total - done) / rate if rate > 0 else 0
                print(f"  [{datetime.now().strftime('%H:%M:%S')}] {done}/{n_total} "
                      f"({elapsed:.0f}s, ETA {eta_time:.0f}s) | "
                      f"σ/m={sm:.1f} η={eta:.2f}: Δχ²_c={dc2_c:.1f} Δχ²_p={dc2_p:.2f} "
                      f"Δf₀={df0:+.1f}% Δσ_UV={ds:+.3f}")

    # --- Extract 95% CL limits ---
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Extracting limits...")

    limits_cond = {}  # conditional
    limits_prof = {}  # profiled
    limits_smf = {}   # with SMF prior (f0 < 0.023)
    f0_2sig = 0.023   # Weibel+2024

    for j, eta in enumerate(eta_grid):
        # Conditional 95% CL
        col = dchi2_cond[:, j]
        for k in range(len(sm_grid)-1):
            if not np.isnan(col[k]) and not np.isnan(col[k+1]):
                if col[k] < 3.84 and col[k+1] >= 3.84:
                    f = (3.84-col[k])/(col[k+1]-col[k])
                    limits_cond[f"{eta:.2f}"] = round(float(sm_grid[k] + f*(sm_grid[k+1]-sm_grid[k])), 3)
                    break

        # Profiled 95% CL
        col_p = dchi2_prof[:, j]
        for k in range(len(sm_grid)-1):
            if not np.isnan(col_p[k]) and not np.isnan(col_p[k+1]):
                if col_p[k] < 3.84 and col_p[k+1] >= 3.84:
                    f = (3.84-col_p[k])/(col_p[k+1]-col_p[k])
                    limits_prof[f"{eta:.2f}"] = round(float(sm_grid[k] + f*(sm_grid[k+1]-sm_grid[k])), 3)
                    break

        # SMF-based: find σ/m where f0_new first exceeds 0.023
        col_f0 = f0_shift[:, j]
        for k in range(len(sm_grid)):
            f0_at_k = 10**p_cdm[0] * (1 + col_f0[k]/100) if not np.isnan(col_f0[k]) else 0
            if f0_at_k > f0_2sig:
                if k > 0:
                    f0_prev = 10**p_cdm[0] * (1 + f0_shift[k-1, j]/100)
                    f = (f0_2sig - f0_prev) / (f0_at_k - f0_prev)
                    limits_smf[f"{eta:.2f}"] = round(float(sm_grid[k-1] + f*(sm_grid[k]-sm_grid[k-1])), 3)
                else:
                    limits_smf[f"{eta:.2f}"] = round(float(sm_grid[k]), 3)
                break

    # --- Topology ---
    topo = {}
    for sm in sm_grid:
        snr5 = ska_snr(sm, 0.05)
        snr10 = ska_snr(sm, 0.10)
        snr20 = ska_snr(sm, 0.20)
        topo[f"sm{sm:.2f}"] = {
            'sigma_m': float(sm), 'p_sidm': round(float(p_sidm_blowout(sm)), 4),
            'snr_5pct': round(float(snr5), 2),
            'snr_10pct': round(float(snr10), 2),
            'snr_20pct': round(float(snr20), 2),
        }

    # --- Print summary ---
    print(f"\n{'='*70}")
    print(f"RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"\nCDM baseline: f★₀={10**p_cdm[0]:.5f} α_lo={p_cdm[1]:.3f} "
          f"σ_UV={p_cdm[2]:.3f} χ²/dof={chi2_cdm/(len(z_d)-4):.3f}")

    print(f"\n95% CL upper limits on σ/m:")
    print(f"  {'η':>6} | {'Conditional':>12} | {'Profiled':>12} | {'UVLF+SMF':>12}")
    print(f"  {'-'*6}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}")
    for eta in eta_grid:
        key = f"{eta:.2f}"
        lc = limits_cond.get(key, 'none')
        lp = limits_prof.get(key, 'none')
        ls = limits_smf.get(key, 'none')
        print(f"  {eta:>6.2f} | {str(lc):>12} | {str(lp):>12} | {str(ls):>12}")

    print(f"\nTopology detection thresholds:")
    for floor, label in [(0.05,'5%'),(0.10,'10%'),(0.20,'20%')]:
        sm_fine = np.linspace(0, 20, 2000)
        snr_fine = np.array([ska_snr(s, floor) for s in sm_fine])
        idx = np.where(snr_fine >= 3)[0]
        thresh = sm_fine[idx[0]] if len(idx) > 0 else '>20'
        print(f"  {label} floor: σ/m ≈ {thresh}")

    # --- Save ---
    output = {
        'metadata': {
            'date': datetime.now().isoformat(),
            'n_data': int(len(z_d)),
            'grid_size': f"{len(sm_grid)}x{len(eta_grid)}",
            'runtime_seconds': round(time.time()-t0, 1),
            'mode': 'quick' if QUICK else 'full',
        },
        'cdm_baseline': {
            'log_f_star0': round(float(p_cdm[0]), 5),
            'f_star0': round(float(10**p_cdm[0]), 6),
            'alpha_lo': round(float(p_cdm[1]), 4),
            'sigma_UV': round(float(p_cdm[2]), 4),
            'z_evol': round(float(p_cdm[3]), 4),
            'chi2': round(float(chi2_cdm), 2),
            'chi2_per_dof': round(float(chi2_cdm/(len(z_d)-4)), 4),
        },
        'grids': {
            'sigma_m': sm_grid.tolist(),
            'eta': eta_grid.tolist(),
            'dchi2_conditional': [[round(float(x),2) if not np.isnan(x) else None
                                   for x in row] for row in dchi2_cond],
            'dchi2_profiled': [[round(float(x),3) if not np.isnan(x) else None
                                for x in row] for row in dchi2_prof],
            'f0_shift_pct': [[round(float(x),1) if not np.isnan(x) else None
                              for x in row] for row in f0_shift],
            'suv_shift_mag': [[round(float(x),4) if not np.isnan(x) else None
                               for x in row] for row in suv_shift],
        },
        'limits_95CL': {
            'conditional': limits_cond,
            'profiled': limits_prof,
            'uvlf_smf': limits_smf,
        },
        'topology': topo,
        'scan_points': results,
    }

    outdir = os.path.dirname(os.path.abspath(__file__))
    outpath = os.path.join(outdir, 'profile_scan_results.json')
    with open(outpath, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Saved to {outpath}")
    print(f"Total time: {time.time()-t0:.0f}s")


if __name__ == '__main__':
    main()
