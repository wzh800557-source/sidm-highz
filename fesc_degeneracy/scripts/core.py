#!/usr/bin/env python3
"""
core.py -- Semi-numerical excursion-set reionization pipeline (RECONSTRUCTION).

Rebuilt from the design documented in session "Research proposal review" [19]:
  - Gaussian random-field density on an N^3 grid, box L (Mpc/h), at redshift z.
  - FZH04 excursion-set ionization: a cell is ionized if, on ANY smoothing
    scale R, zeta_eff(R) * f_coll(delta_R, R) >= 1.
  - x_HI is fixed to a target by a binary search on the zeta normalization.
  - SIDM enters through a mass(scale)-dependent emissivity/duty-cycle modulation
    (pure density-based excursion-set gives identical fields for all models;
     the source model is what differentiates them). SIDM suppresses small-scale
     (low-mass) sources by a factor tied to the dbind delta_bind value, shifting
     ionization toward larger halos -> changes morphology/topology while the
     globally calibrated x_HI is held fixed.
  - Observables: 21-cm brightness-temperature power spectrum Delta^2_21(k),
    plus the Euler characteristic / genus of the ionization field.

This is a faithful re-implementation of the documented algorithm. It will NOT
reproduce the paper's exact numbers (different code, calibration, RNG path).

Usage:
  python3 core.py --grid 128 --box 200 --model cdm    --xhi 0.5 --z 7 --out p21_cdm_z7.npz
  python3 core.py --grid 128 --box 200 --model sidm10  --xhi 0.5 --z 7 --out p21_sidm10_z7.npz
"""
import numpy as np
import argparse, json, os
from scipy.ndimage import uniform_filter
from scipy.special import erfc as _erfc

# ---------------- cosmology ----------------
H0 = 67.74; Om = 0.3089; Ob = 0.0486; ns = 0.965; sigma8 = 0.8159
h = H0/100.0
delta_c = 1.686

def eh_transfer(k):
    """Eisenstein & Hu 1998 (no-baryon, zero-baryon shape) transfer function.
    k in h/Mpc."""
    kk = k * h                       # 1/Mpc
    theta = 2.728/2.7
    Omh2 = Om*h*h
    s = 44.5*np.log(9.83/Omh2)/np.sqrt(1+10*(Ob*h*h)**0.75)   # Mpc
    alpha = 1 - 0.328*np.log(431*Omh2)*Ob/Om + 0.38*np.log(22.3*Omh2)*(Ob/Om)**2
    Gamma = Om*h*(alpha + (1-alpha)/(1+(0.43*kk*s)**4))
    q = k*theta*theta/Gamma          # k in h/Mpc with Gamma in h/Mpc units
    L0 = np.log(2*np.e + 1.8*q)
    C0 = 14.2 + 731.0/(1+62.5*q)
    return L0/(L0 + C0*q*q)

def linear_pk(k):
    """Linear matter P(k) at z=0, (Mpc/h)^3, normalised to sigma8."""
    T = eh_transfer(k)
    pk = k**ns * T*T
    return pk

def _sigma_R_tophat(R, pk_func, kmin=1e-3, kmax=5e2, n=2000):
    k = np.logspace(np.log10(kmin), np.log10(kmax), n)
    x = k*R
    w = 3*(np.sin(x) - x*np.cos(x))/x**3
    integ = pk_func(k) * w*w * k*k
    return np.sqrt(np.trapezoid(integ, k)/(2*np.pi**2))

def growth_D(z):
    """Linear growth factor normalised to D(0)=1 (Carroll-Press-Turner)."""
    def g(zz):
        Omz = Om*(1+zz)**3
        Ez2 = Omz + (1-Om)
        Omega = Omz/Ez2; OmL = (1-Om)/Ez2
        return 2.5*Omega/(Omega**(4/7.) - OmL + (1+Omega/2)*(1+OmL/70.))
    return (g(z)/(1+z))/g(0.0)

# ---------------- density field ----------------
def make_density(N, L, z, seed=42):
    """Gaussian random field delta on N^3, box L (Mpc/h), at redshift z.
    Built so that the field's power spectrum is the linear P(k) normalised to
    sigma8 and scaled by the growth factor D(z)."""
    rng = np.random.default_rng(seed)
    kx = np.fft.fftfreq(N, d=L/N)*2*np.pi          # h/Mpc
    KX, KY, KZ = np.meshgrid(kx, kx, kx, indexing='ij')
    kmag = np.sqrt(KX**2+KY**2+KZ**2)
    kmag.flat[0] = 1.0                              # avoid div0; zeroed below
    s8 = _sigma_R_tophat(8.0, linear_pk)
    A = (sigma8/s8)**2
    Pk = A*linear_pk(kmag) * growth_D(z)**2         # (Mpc/h)^3
    Pk.flat[0] = 0.0
    # white noise with unit variance per cell -> colour by sqrt(P(k)/Vcell)
    wn = rng.normal(size=(N, N, N))
    wk = np.fft.fftn(wn)
    Vcell = (L/N)**3
    dk = wk * np.sqrt(Pk/Vcell)
    delta = np.fft.ifftn(dk).real
    return delta

# ---------------- excursion-set ionization ----------------
def R_of_M(M):
    """Lagrangian radius (Mpc/h) of halo mass M (Msun/h)."""
    rho_m0 = Om*2.775e11           # (Msun/h)/(Mpc/h)^3
    return (3*M/(4*np.pi*rho_m0))**(1/3.)

# minimum source-halo mass (atomic-cooling): set sigma_min once
M_MIN = 1e8
def sigma_min_z(z, A):
    Rm = R_of_M(M_MIN)
    return _sigma_R_tophat(Rm, lambda k: A*linear_pk(k))*growth_D(z)

def fcoll_field(delta, L, R, z, A, smin):
    """Collapsed fraction (>M_min) smoothed on scale R via FZH04 erfc barrier."""
    N = delta.shape[0]; cell = L/N
    width = max(int(round(2*R/cell)), 1)
    dR = uniform_filter(delta, size=width, mode='wrap')
    sigR = _sigma_R_tophat(max(R, cell*0.5), lambda k: A*linear_pk(k))*growth_D(z)
    denom = np.sqrt(np.maximum(2.0*(smin**2 - sigR**2), 1e-6))
    return 0.5*_erfc((delta_c - dR)/denom), sigR

def source_weight(R, model, dbind):
    """Model-dependent emissivity/duty-cycle weight on scale R (Mpc/h).
    SIDM suppresses small-R (low-mass) sources by up to db; CDM = 1."""
    if model == 'cdm' or dbind <= 0:
        return 1.0
    # SIDM suppresses small-scale (low-mass) ionizing sources; the Eulerian
    # imprint scale R0 sets where the duty-cycle suppression turns on. Weight
    # rises from ~(1-db) on small bubbles to 1 on large ones -> after zeta
    # recalibration to fixed x_HI, this changes morphology (scale-dependent
    # P21 ratio + topology) without changing the global ionized fraction.
    R0 = float(os.environ.get('SUPP_R0', 4.0))   # Mpc/h Eulerian transition scale
    AMP = float(os.environ.get('SUPP_AMP', 0.5))  # duty-cycle suppression amplitude
    supp = 1.0 - AMP*dbind*np.exp(-(R/R0)**2)
    return max(supp, 0.05)

def ionization_field(delta, L, z, A, zeta, model, dbind, Rmax=30.0, nR=20, smin=None):
    N = delta.shape[0]; cell = L/N
    if smin is None: smin = sigma_min_z(z, A)
    Rs = np.logspace(np.log10(cell), np.log10(Rmax), nR)
    ion = np.zeros_like(delta)
    cell_fcoll = None
    for R in Rs:
        fc, _ = fcoll_field(delta, L, R, z, A, smin)
        w = source_weight(R, model, dbind)
        crit = zeta*w*fc
        ion = np.maximum(ion, (crit >= 1.0).astype(float))
        if abs(R-cell) < cell*0.01 or cell_fcoll is None:
            cell_fcoll = (fc, w)
    # partial ionization at cell scale for not-yet-ionized cells
    fc, w = cell_fcoll
    partial = np.clip(zeta*w*fc, 0, 1)
    xi = np.maximum(ion, partial)
    return np.clip(xi, 0, 1)

def mean_xi(delta, L, z, A, zeta, model, dbind, **kw):
    xi = ionization_field(delta, L, z, A, zeta, model, dbind, **kw)
    return xi.mean(), xi

def find_zeta(delta, L, z, A, target_xi, model, dbind, max_iter=18):
    """Binary search zeta to reach target ionized fraction."""
    lo, hi = 0.5, 2000.0
    xi = None; smin = sigma_min_z(z, A)
    for _ in range(max_iter):
        mid = np.sqrt(lo*hi)
        xbar, xi = mean_xi(delta, L, z, A, mid, model, dbind, smin=smin)
        if xbar < target_xi:
            lo = mid
        else:
            hi = mid
        if abs(xbar-target_xi) < 0.005:
            break
    return mid, xi, xbar

# ---------------- observables ----------------
def power_spectrum(field, L, nbins=18):
    N = field.shape[0]
    f = field - field.mean()
    fk = np.fft.fftn(f)
    Pk3 = (np.abs(fk)**2) * (L**3) / (N**6)
    kx = np.fft.fftfreq(N, d=L/N)*2*np.pi
    KX,KY,KZ = np.meshgrid(kx,kx,kx, indexing='ij')
    kmag = np.sqrt(KX**2+KY**2+KZ**2)
    kmin = 2*np.pi/L; kmax = np.pi*N/L
    bins = np.logspace(np.log10(kmin), np.log10(kmax), nbins+1)
    kc = 0.5*(bins[1:]+bins[:-1])
    P = np.zeros(nbins); cnt = np.zeros(nbins)
    idx = np.digitize(kmag.ravel(), bins)-1
    pr = Pk3.ravel()
    for b in range(nbins):
        sel = idx==b
        if sel.any():
            P[b] = pr[sel].mean(); cnt[b]=sel.sum()
    good = cnt>0
    return kc[good], P[good]

def brightness_temp(xi, delta, z):
    """21-cm differential brightness temperature field (mK), spin-saturated."""
    xHI = 1.0 - xi
    Tb0 = 27.0*np.sqrt((1+z)/10.0 * 0.15/(Om*h*h)) * (Ob*h*h/0.023)
    return Tb0 * xHI * (1.0 + delta)

def euler_characteristic(xi, thresh=0.5):
    """Genus/Euler proxy: net (ionized blobs - tunnels) via thresholded field."""
    from scipy.ndimage import label
    b = xi > thresh
    _, n_ion = label(b)
    _, n_neu = label(~b)
    return (n_ion - n_neu) / b.size * 1e3   # per 1000 cells, sign-aware proxy

# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--grid', type=int, default=128)
    ap.add_argument('--box', type=float, default=200.0)
    ap.add_argument('--model', default='cdm')
    ap.add_argument('--xhi', type=float, default=0.5, help='target IONIZED fraction')
    ap.add_argument('--z', type=float, default=7.0)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--dbind-json', default=None, help='path to dbind_table.json')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()

    # SIDM strength from dbind delta_bind (M100_z7), else built-in defaults
    sm_map = {'cdm':None,'sidm1':'1','sidm5':'5','sidm10':'10'}
    code = {'1':'0010','5':'0050','10':'0100'}
    default_db = {'cdm':0.0,'sidm1':0.241,'sidm5':0.620,'sidm10':0.720}
    dbind = default_db.get(a.model, 0.0)
    if a.dbind_json and sm_map.get(a.model):
        try:
            g = json.load(open(a.dbind_json))
            tag = f"M100_z7_s{code[sm_map[a.model]]}_const"
            if tag in g: dbind = float(g[tag]['delta_bind'][1])
        except Exception as e:
            print('dbind read failed, using default:', e)

    s8 = _sigma_R_tophat(8.0, linear_pk)
    A = (sigma8/s8)**2
    print(f"model={a.model} dbind={dbind:.3f} grid={a.grid} box={a.box} z={a.z}")
    delta = make_density(a.grid, a.box, a.z, seed=a.seed)
    print(f"  density rms={delta.std():.3f}")
    zeta, xi, xbar = find_zeta(delta, a.box, a.z, A, a.xhi, a.model, dbind)
    print(f"  calibrated zeta={zeta:.3f}, achieved x_ion={xbar:.3f}")
    Tb = brightness_temp(xi, delta, a.z)
    k, P = power_spectrum(Tb, a.box)
    D2 = k**3 * P/(2*np.pi**2)        # mK^2
    chi = float(euler_characteristic(xi))
    print(f"  Euler/genus proxy={chi:.4f}")
    out = a.out or f"p21_{a.model}_z{int(a.z)}.npz"
    np.savez(out, k=k, d2=D2, P=P, zeta=zeta, xbar=xbar, euler=chi,
             model=a.model, dbind=dbind, z=a.z, xhi=a.xhi)
    np.savetxt(out.replace('.npz','.txt'), np.c_[k, D2])
    print(f"  wrote {out}")
    for kk, dd in zip(k, D2):
        print(f"    k={kk:7.4f}  D2_21={dd:10.4f}")

if __name__ == '__main__':
    main()
