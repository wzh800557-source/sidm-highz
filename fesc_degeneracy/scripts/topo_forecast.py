#!/usr/bin/env python3
"""Field-level SKA1-Low forecast for the ionized-region count beta0.

Actual topology forecast (option 2 for the claim-method mismatch):
  1. delta T_b mocks from the rebuilt semi-numerical pipeline; zeta
     calibrated to x_ion = 0.5 at z = 7 per model/seed (fc-stack method,
     identical to v2_formscan.py; Gaussian source-suppression form,
     SUPP_AMP = 0.35, production dbind values).
  2. Instrument transfer: only uv-sampled modes kept (synthetic
     224-station Gaussian core, same as Appendix G), k_par > k_par_min,
     foreground wedge (optimistic: k_par > 0.05; moderate: horizon
     wedge + 0.1 buffer), mean removed, Gaussian smoothing R_s.
  3. Anisotropic thermal-noise realizations with P_N(k_perp) from the
     same uv model, scaled to t_obs.
  4. Estimator: number of connected below-median regions of the
     smoothed observed field (median split = natural threshold at
     x_ion = 0.5); applied identically to CDM and SIDM mocks.
  5. Covariance from Monte Carlo over noise realizations and density
     seeds. Significance = |<b0_S> - <b0_C>| / sqrt(var_S + var_C).

Env: GRID(128) SEEDS(42,43,44) NNOISE(8) HOURS(1000,5000) RS(4.0 Mpc/h)
     CORE_DIR OUT CKDIR. Resumable: per-model results checkpointed.
Thermal noise + wedge only; the eps_sys floor of the P21 forecast has no
field-level analogue (stated as a caveat in the text).
"""
import sys, os, json, time
import numpy as np
from scipy.ndimage import label

CORE_DIR = os.environ.get('CORE_DIR', os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CORE_DIR)
os.environ.setdefault('SUPP_AMP', '0.35')
import core

N = int(os.environ.get('GRID', '128'))
SEEDS = [int(s) for s in os.environ.get('SEEDS', '42,43,44').split(',')]
NNOISE = int(os.environ.get('NNOISE', '8'))
HOURS = [int(x) for x in os.environ.get('HOURS', '1000,5000').split(',')]
RS = float(os.environ.get('RS', '4.0'))
OUT = os.environ.get('OUT', 'topo_forecast_results.json')
CKDIR = os.environ.get('CKDIR', '/tmp')
L, Z = 200.0, 7.0
TARGET = 0.5
AMP, R0 = 0.35, 4.0
DB = {'cdm': 0.0, 'sidm1': 0.241, 'sidm5': 0.620, 'sidm10': 0.720}

# ---------------- instrument (identical numbers to Appendix G) -------------
nu = 1420.4 / (1 + Z); lam = 299.79 / nu
DC = 5936.0
TSYS = (60 * (nu / 300) ** -2.55 * 1.1 + 40) * 1e3          # mK
AE = 500.0
Y_HZ = 10.6 / 1e6
KPAR_MIN = 0.074
Ez = np.sqrt(0.3153 * (1 + Z) ** 3 + 1 - 0.3153)
HORIZON_SLOPE = DC * 100 * Ez / (299790.0 * (1 + Z))
BUFFER = 0.1

rng0 = np.random.default_rng(4)
pos = rng0.normal(0, 300.0, size=(224, 2))
bl = []
for i in range(224):
    bl.append(np.hypot(pos[i + 1:, 0] - pos[i, 0], pos[i + 1:, 1] - pos[i, 1]))
u_bl = np.concatenate(bl) / lam
ub = np.linspace(5, 2500, 120)
rho = np.zeros(len(ub) - 1)
for i in range(len(ub) - 1):
    n = np.sum((u_bl >= ub[i]) & (u_bl < ub[i + 1]))
    rho[i] = n / (np.pi * (ub[i + 1] ** 2 - ub[i] ** 2))
uc = 0.5 * (ub[1:] + ub[:-1])

def PN_kperp(kperp, t_hours):
    u = kperp * DC / (2 * np.pi)
    r = np.interp(u, uc, rho, left=rho[0], right=0.0)
    t_u = np.maximum(2 * r * (AE / lam ** 2) * t_hours * 3600.0, 1e-30)
    Omp = lam ** 2 / AE
    return np.where(r > 0, DC ** 2 * Y_HZ * Omp * TSYS ** 2 / (2 * t_u), np.inf)

kx = np.fft.fftfreq(N, d=L / N) * 2 * np.pi
KXX, KYY, KZZ = np.meshgrid(kx, kx, kx, indexing='ij')
KPERP = np.hypot(KXX, KYY)
KPAR = np.abs(KZZ)
K2 = KPERP ** 2 + KPAR ** 2
SMOOTH = np.exp(-0.5 * K2 * RS ** 2)
PN_GRID = {t: PN_kperp(KPERP, t) for t in HOURS}
SAMPLED = np.isfinite(PN_GRID[HOURS[0]]) & (KPAR >= KPAR_MIN)
SAMPLED.flat[0] = False
MASKS = {'optimistic': SAMPLED & (KPAR > 0.05),
         'moderate': SAMPLED & (KPAR > BUFFER + HORIZON_SLOPE * KPERP)}
VCELL = (L / N) ** 3

def observe(tb, mask, t_hours, rng):
    sk = np.fft.fftn(tb) * mask
    wn = rng.normal(size=(N, N, N))
    nk = np.fft.fftn(wn) * np.sqrt(np.where(mask, PN_GRID[t_hours], 0.0) / VCELL)
    return np.fft.ifftn((sk + nk) * SMOOTH).real

PCTS = [30, 40, 50, 60, 70]

def beta0_of(field):
    _, n = label(field < np.median(field))
    return int(n)

def beta0_vec(field):
    thr = np.percentile(field, PCTS)
    return [int(label(field < t)[1]) for t in thr]

# ---------------- fc-stack calibration (as v2_formscan) --------------------
def fc_stack(seed):
    ck = f'{CKDIR}/fcstack_N{N}_{seed}.npz'
    if os.path.exists(ck):
        d = np.load(ck); return d['fc'], d['Rs'], d['delta']
    delta = core.make_density(N, L, Z, seed=seed)
    s8 = core._sigma_R_tophat(8.0, core.linear_pk)
    A = (core.sigma8 / s8) ** 2
    smin = core.sigma_min_z(Z, A)
    cell = L / N
    Rs = np.logspace(np.log10(cell), np.log10(30.0), 20)
    fc = np.empty((len(Rs), N, N, N), dtype=np.float32)
    for i, R in enumerate(Rs):
        f, _ = core.fcoll_field(delta, L, R, Z, A, smin)
        fc[i] = f.astype(np.float32)
    np.savez(ck, fc=fc, Rs=Rs, delta=delta.astype(np.float32))
    return fc, Rs, delta

def xi_from_stack(fc, Rs, model):
    w = np.array([max(1.0 - AMP * DB[model] * np.exp(-(R / R0) ** 2), 0.05)
                  for R in Rs])
    lo, hi = 0.5, 2000.0
    for _ in range(14):
        mid = np.sqrt(lo * hi)
        ion = np.zeros((N, N, N), dtype=bool)
        for i in range(len(Rs)):
            ion |= (mid * w[i] * fc[i] >= 1.0)
        xi = np.maximum(ion.astype(np.float32), np.clip(mid * w[0] * fc[0], 0, 1))
        xb = float(xi.mean())
        if xb < TARGET: lo = mid
        else: hi = mid
        if abs(xb - TARGET) < 0.004: break
    return xi, xb

# ---------------- run ------------------------------------------------------
res = json.load(open(OUT)) if os.path.exists(OUT) else {}
res['config'] = dict(grid=N, seeds=SEEDS, nnoise=NNOISE, hours=HOURS, RS=RS,
                     estimator='connected below-median regions, smoothed',
                     noise='thermal (App G uv model) + wedge; no eps_sys')
t0 = time.time()
for seed in SEEDS:
    fc, Rsv, delta = fc_stack(seed)
    print(f'seed {seed} stack ready ({time.time()-t0:.0f}s)', flush=True)
    for model in DB:
        key = f'{model}_s{seed}'
        if key in res and f'{list(MASKS)[-1]}_{HOURS[-1]}h' in res.get(key, {}):
            continue
        xi, xbar = xi_from_stack(fc, Rsv, model)
        tb = core.brightness_temp(xi, delta, Z)
        res[key] = {'xbar': xbar, 'b0_true': beta0_of(-xi)}
        rng = np.random.default_rng(1000 + seed)
        for scen, mask in MASKS.items():
            for t in HOURS:
                res[key][f'{scen}_{t}h'] = [beta0_vec(observe(tb, mask, t, rng))
                                            for _ in range(NNOISE)]
        print(f'{key} done ({time.time()-t0:.0f}s)', flush=True)
        json.dump(res, open(OUT, 'w'), indent=1)

print('\n=== detection significance (vs CDM) ===')
sig = {}
for scen in MASKS:
    for t in HOURS:
        tag = f'{scen}_{t}h'
        cdm = np.vstack([np.atleast_2d(res[f'cdm_s{s}'][tag]) for s in SEEDS])
        line = f'{tag:>16s}: '
        for model in ['sidm1', 'sidm5', 'sidm10']:
            sm = np.vstack([np.atleast_2d(res[f'{model}_s{s}'][tag]) for s in SEEDS])
            dmu = sm.mean(0) - cdm.mean(0)
            var = sm.var(0) + cdm.var(0) + 1e-12
            zz = float(np.sqrt(np.sum(dmu ** 2 / var)))
            sig[f'{model}_{tag}'] = round(float(zz), 2)
            line += f'{model}={zz:5.1f}sigma  '
        print(line, flush=True)
res['significance'] = sig
json.dump(res, open(OUT, 'w'), indent=1)
print('saved', OUT)
