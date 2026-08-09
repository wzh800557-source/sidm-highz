"""Five-functional-form V2 (Euler) robustness scan for the topology claim.

Replaces the withdrawn +58--94% range from the deleted pipeline.
Results (production core.py, SUPP_AMP=0.35, sigma/m=10, dbind=0.72,
seeds 42/43/44, dV2 = EC_SIDM/EC_CDM - 1 at x_ion=0.5):
  256^3 (SLURM job 18242482, sched_mit_mvogelsb, 2026-07-16):
    gauss +18.2%, tanh +18.1%, lorentz +16.4%, linear +15.0%, exp +14.7%
    (seed means; full range +14.5% to +18.6%)  -> data/v2_formscan_256_results.json
  128^3 (sandbox): range +11% to +27%          -> data/v2_formscan_results.json
Fiducial (gauss) at 256^3 reproduces the manuscript's +18%+-1%.

Usage: GRID=256 SEED=42 python3 v2_formscan.py
Set CORE_DIR to the directory containing the rebuilt core.py
(default: ~/sidm-highz/referee_rev on the cluster).
"""
import sys, os, json, time
import numpy as np
CORE_DIR = os.environ.get('CORE_DIR', os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CORE_DIR)
os.environ.setdefault('SUPP_AMP', '0.35')
import core

N = int(os.environ.get('GRID', '128'))
L, Z = 200.0, 7.0
SEED = int(os.environ.get('SEED','42'))
TARGET = 0.5          # ionized fraction
AMP, DB, R0 = 0.35, 0.72, 4.0    # sigma/m=10 production config
CK = f'/tmp/fcstack_N{N}_{SEED}.npz'

t0=time.time()
if os.path.exists(CK):
    dat = np.load(CK)
    fc = dat['fc']; Rs = dat['Rs']
else:
    delta = core.make_density(N, L, Z, seed=SEED)
    s8 = core._sigma_R_tophat(8.0, core.linear_pk)
    A = (core.sigma8/s8)**2
    smin = core.sigma_min_z(Z, A)
    cell = L/N
    Rs = np.logspace(np.log10(cell), np.log10(30.0), 20)
    fc = np.empty((len(Rs), N, N, N), dtype=np.float32)
    for i, R in enumerate(Rs):
        f, _ = core.fcoll_field(delta, L, R, Z, A, smin)
        fc[i] = f.astype(np.float32)
    np.savez_compressed(CK, fc=fc, Rs=Rs)
print('fc stack ready %.1fs' % (time.time()-t0))

FORMS = {
 'gauss':   lambda R: np.exp(-(R/R0)**2),
 'exp':     lambda R: np.exp(-R/R0),
 'lorentz': lambda R: 1.0/(1.0+(R/R0)**2),
 'tanh':    lambda R: 0.5*(1.0-np.tanh((R-R0)/(R0/2.0))),
 'linear':  lambda R: np.clip(1.0-R/(2.0*R0), 0.0, 1.0),
}

def xi_of(zeta, w):
    ion = np.zeros((N,N,N), dtype=bool)
    for i in range(len(Rs)):
        ion |= (zeta*w[i]*fc[i] >= 1.0)
    part = np.clip(zeta*w[0]*fc[0], 0, 1)
    xi = np.maximum(ion.astype(np.float32), part)
    return xi

def calib(w):
    lo, hi = 0.5, 2000.0
    for _ in range(14):
        mid = np.sqrt(lo*hi)
        xi = xi_of(mid, w)
        xb = float(xi.mean())
        if xb < TARGET: lo = mid
        else: hi = mid
        if abs(xb-TARGET) < 0.004: break
    return mid, xi, xb

res = {}
w_cdm = np.ones(len(Rs))
z_c, xi_c, xb_c = calib(w_cdm)
ec_c = core.euler_characteristic(xi_c)
b_c = core.euler_characteristic(xi_c)  # same call; keep EC
lab_c = None
res['cdm'] = dict(zeta=z_c, xion=xb_c, EC=ec_c)
print('CDM: zeta=%.1f x_ion=%.3f EC=%.3f  (%.1fs)'%(z_c,xb_c,ec_c,time.time()-t0))
for name, f in FORMS.items():
    w = np.array([max(1.0-AMP*DB*f(R), 0.05) for R in Rs])
    z_s, xi_s, xb_s = calib(w)
    ec_s = core.euler_characteristic(xi_s)
    dv2 = (ec_s/ec_c - 1.0)*100 if abs(ec_c)>1e-9 else float('nan')
    res[name] = dict(zeta=z_s, xion=xb_s, EC=ec_s, dV2_pct=dv2)
    print('%-8s zeta=%.1f x_ion=%.3f EC=%.3f dV2=%+.1f%%  (%.1fs)'%(name,z_s,xb_s,ec_s,dv2,time.time()-t0))
json.dump(res, open(f'/tmp/v2scan_seed{SEED}.json','w'), indent=1)
print('done %.1fs'%(time.time()-t0))
