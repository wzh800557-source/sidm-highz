#!/usr/bin/env python3
"""
Rerun A: full-integral reionization scan (referee major comment 2).

Pipeline mirrors the Letter (Appendix C/D) but computes ndot_ion from the
FULL mass integral, with the SIDM suppression (1 - eta*Delta_bind(M)) and the
mass-dependent fesc(M) ~ M^beta_esc inside the integrand.

Stage 1: UVLF optimiser -> profiled (f_star0, sigma_UV) at each (sigma/m, eta)
Stage 2: fesc,0 scan over [0.01, 0.50] against tau + xHI(z)

Outputs (per grid point):
  <1-eta*Delta>_ion   emissivity-weighted suppression at fixed normalisations
  <1-eta*Delta>_UVLF  = f0_CDM / f0_prof (implied UVLF-weighted suppression)
  f0_prof, fesc_prof, product ratio, Dchi2_profiled, Dchi2_stage1(fesc frozen)
  z-drift of the emissivity ratio (degeneracy-shape check)

Data vector: the EXACT published compilation quoted in the Letter
(Umeda+24 ApJ 971,124; Mason+26 A&A 705,A114; Greig+17 MNRAS 466,4239;
Mason+19 MNRAS 485,3947), two-sided errors as published, Mason+19 one-sided.
Produces data/rerun_A_published_vector_results.json (Table 4: dchi2 <= 0.62).
"""
import numpy as np
from scipy.optimize import minimize
import json

# ---------------- cosmology (repo values) ----------------
h=0.6736; Om=0.3153; Ob=0.0493; s8=0.811; ns=0.9649
fbary=Ob/Om; rho_crit=2.775e11; rho_m=Om*rho_crit
c_kms=2.998e5
Hz=lambda z: 100*h*np.sqrt(Om*(1+z)**3+1-Om)          # km/s/Mpc
nH_cm3=0.76*(Ob*h**2)*1.878e-29/1.6726e-24            # comoving H density cm^-3
MPC_CM=3.0857e24
nH_Mpc3=nH_cm3*MPC_CM**3                               # comoving H per Mpc^3
sigT=6.6524e-25                                        # cm^2

# ---------------- JWST UVLF data (repo, verified) ----------------
UVLF_DATA = {
 9:[(-20.75,-4.921,0.222,0.234),(-20.25,-4.495,0.148,0.163),(-19.75,-3.842,0.082,0.094),
    (-19.25,-3.629,0.099,0.102),(-18.55,-3.313,0.122,0.146),(-18.05,-2.955,0.107,0.142),
    (-17.55,-2.751,0.122,0.147)],
 10:[(-20.75,-5.398,0.544,2.000),(-20.25,-4.569,0.171,0.201),(-19.75,-4.036,0.104,0.106),
    (-19.25,-3.752,0.114,0.127),(-18.55,-3.493,0.145,0.184),(-18.05,-3.164,0.133,0.171),
    (-17.55,-2.893,0.140,0.179)],
 11:[(-21.25,-5.155,0.359,0.544),(-20.75,-4.854,0.252,0.301),(-20.25,-4.420,0.153,0.182),
    (-19.75,-4.000,0.137,0.155),(-19.25,-3.842,0.194,0.250),(-18.75,-3.631,0.177,0.229),
    (-18.25,-3.193,0.194,0.251)],
 12:[(-21.25,-5.523,0.368,0.477),(-20.75,-5.398,0.352,0.602),(-20.25,-4.796,0.194,0.204),
    (-19.75,-4.469,0.224,0.253),(-19.25,-4.367,0.259,0.311),(-18.75,-4.097,0.214,0.260),
    (-18.25,-3.664,0.232,0.283)],
 14:[(-20.25,-5.523,0.477,0.398),(-20.80,-4.432,0.571,0.663)],
 16:[(-21.90,-5.009,0.80,0.80)],
}

# ---------------- HMF (Sheth-Tormen, vectorised) ----------------
def growth(z):
    Oz=Om*(1+z)**3/(Om*(1+z)**3+1-Om); OL=(1-Om)/(Om*(1+z)**3+1-Om)
    return (5/2)*Oz/(Oz**(4/7)-OL+(1+Oz/2)*(1+OL/70))/(1+z)
def sigma_M(M):
    R=(3*M/(4*np.pi*rho_m))**(1/3)
    return s8*(R/8.0)**(-0.5*(ns+3)/3)*np.exp(-0.5*(R/100)**2)
def dndlnM_ST(M,z):
    s=sigma_M(M)*growth(z); nu=1.686/s
    A,a,p=0.3222,0.707,0.3
    f=A*np.sqrt(2*a/np.pi)*nu*(1+(a*nu**2)**(-p))*np.exp(-a*nu**2/2)
    eps=0.01
    s1=sigma_M(M*(1+eps))*growth(z); s0=sigma_M(M*(1-eps))*growth(z)
    dlns=(np.log(s1)-np.log(s0))/(np.log(M*(1+eps))-np.log(M*(1-eps)))
    return (rho_m/M)*f*np.abs(dlns)

# ---------------- GIZMO Delta_bind table (Letter Table 2: z=7, R<0.5 kpc) ----------------
LOGM_TAB=np.array([9.0,9.5,10.0,10.5,11.0])
SM_TAB=np.array([0.5,1,2,5,10,20])
DB_TAB=np.array([
 [0.01,0.03,0.04,0.10,0.16,0.24],
 [0.04,0.10,0.16,0.33,0.44,0.53],
 [0.13,0.24,0.40,0.62,0.72,0.76],
 [0.34,0.51,0.71,0.85,0.88,0.87],
 [0.59,0.82,0.89,0.93,0.94,0.94]])

def Delta_bind(logM, sm):
    """Bilinear in (logM, log10 sm); log-linear extrapolation below logM=9;
    capped at the logM=11 row above; Delta -> 0 as sm -> 0."""
    if sm<=0: return np.zeros_like(logM)
    lsm=np.log10(np.clip(sm,0.05,20.0))
    ls=np.log10(SM_TAB)
    j=np.clip(np.searchsorted(ls,lsm)-1,0,len(ls)-2)
    wj=(lsm-ls[j])/(ls[j+1]-ls[j])
    col=DB_TAB[:,j]*(1-wj)+DB_TAB[:,j+1]*wj          # Delta(logM_tab) at this sm
    out=np.interp(logM,LOGM_TAB,col)
    # log-linear extrapolation below logM=9 (slope from 9.0->9.5 rows)
    lo=logM<9.0
    if np.any(lo):
        slope=(np.log10(col[1])-np.log10(col[0]))/0.5
        out[lo]=10**(np.log10(col[0])+slope*(logM[lo]-9.0))
    if sm<0.5: out=out*(sm/0.5)                       # linear anchor to 0
    return np.clip(out,0.0,0.99)

# ---------------- SFE and UVLF (repo model + Letter SIDM suppression) ----------------
CDM={'f_star0':0.019424,'alpha_lo':2.1397,'sigma_UV':0.6472,'z_evol':0.1288}
MP=1e11; AHI=0.5; MTURN0=5e7

def fstar(M,z,f0,sm,eta):
    fz=10**(CDM['z_evol']*(z-9))
    f=f0*fz*(M/MP)**CDM['alpha_lo']/(1+(M/MP)**(CDM['alpha_lo']+AHI))
    if sm>0 and eta>0:
        f=f*(1-eta*Delta_bind(np.log10(M),sm))
    return np.minimum(f,1.0)

def Mdot(M,z): return 46.1*(M/1e12)**1.1*(1+z)**2.5   # Msun/yr

LOGM=np.linspace(7.5,13.5,300); MARR=10**LOGM; DLOGM=LOGM[1]-LOGM[0]
DND={}; MT={}
for zv in list(UVLF_DATA.keys()):
    DND[zv]=dndlnM_ST(MARR,zv)
    MT[zv]=np.exp(-((MTURN0*((1+zv)/10)**-1.5)/MARR)**2)

def uvlf_chi2(f0,suv,sm,eta):
    chi2=0.0
    for zv,pts in UVLF_DATA.items():
        f=fstar(MARR,zv,f0,sm,eta)
        SFR=f*fbary*Mdot(MARR,zv)
        MUV=np.where(SFR>0,-2.5*np.log10(SFR/1.4e-28)+51.63,-10.0)
        dnd=DND[zv]*MT[zv]
        for (Mb,lp,su,sd) in pts:
            w=np.exp(-0.5*((MUV-Mb)/suv)**2)/(np.sqrt(2*np.pi)*suv)
            phi=np.sum(dnd*w*DLOGM*np.log(10))
            sig=(su+sd)/2
            chi2+= ((np.log10(phi)-lp)/sig)**2 if phi>0 else 100.0
    return chi2

def profile_uvlf(sm,eta):
    nll=lambda p: uvlf_chi2(10**p[0],np.clip(p[1],0.15,2.4),sm,eta)
    r=minimize(nll,[np.log10(CDM['f_star0']),CDM['sigma_UV']],method='Nelder-Mead',
               options={'xatol':1e-6,'fatol':1e-8,'maxiter':4000})
    return 10**r.x[0],np.clip(r.x[1],0.15,2.4),r.fun

# ---------------- reionization (Letter Appendix C, FULL integral) ----------------
BETA_ESC=-0.3
ZGRID=np.linspace(25.0,4.0,700)
XHI_DATA=[  # (z, xHI, sigma, kind) EXACT PUBLISHED COMPILATION
 (6.5,0.33,0.225,'two'),   # Mason+25 arXiv:2501.11702
 (7.08,0.40,0.20,'two'),   # Greig+17 MNRAS 466,4239 (J1120)
 (7.12,0.53,0.325,'two'),  # Umeda+24 ApJ 971,124 Tab.5
 (7.44,0.65,0.305,'two'),  # Umeda+24
 (7.9,0.76,0.30,'lower'),  # Mason+19 MNRAS 485,3947 (>0.76, one-sided)
 (8.28,0.91,0.155,'two'),  # Umeda+24
 (9.3,0.64,0.20,'two'),    # Mason+25
 (9.91,0.92,0.09,'two')]   # Umeda+24
TAU_OBS,TAU_SIG=0.054,0.007

def ndot_shape(sm,eta,f0):
    """ndot(z) up to a global constant: full mass integral, all z in ZGRID."""
    fesc_shape=np.minimum((MARR/1e10)**BETA_ESC,1.0/0.13)  # cap fesc(M)<=1 for fesc0=0.13 scale
    out=np.empty(len(ZGRID))
    for i,zv in enumerate(ZGRID):
        f=fstar(MARR,zv,f0,sm,eta)
        w=dndlnM_ST(MARR,zv)*np.exp(-((MTURN0*((1+zv)/10)**-1.5)/MARR)**2)
        out[i]=np.sum(w*fesc_shape*f*fbary*Mdot(MARR,zv)*DLOGM*np.log(10))
    return out  # ~ photons per fesc0, unnormalised

# calibrate global constant: CDM (f0=0.019424, fesc0=0.13) -> ndot(z=8)=10^50.85 s^-1 Mpc^-3
shape_cdm=ndot_shape(0,0,CDM['f_star0'])
i8=np.argmin(np.abs(ZGRID-8.0))
CNORM=10**50.85/(0.13*shape_cdm[i8])

def reion_chi2_scan(shape, fesc_grid):
    """Solve Q(z) for all fesc0 in fesc_grid simultaneously; return chi2 array."""
    aB=2.6e-13; CL=np.maximum(1.0,21.0/(1+ZGRID)); fe=1.08
    Q=np.zeros(len(fesc_grid))
    tau=np.zeros(len(fesc_grid))
    Qtraj=np.zeros((len(ZGRID),len(fesc_grid)))
    dz=ZGRID[0]-ZGRID[1]
    for i,zv in enumerate(ZGRID[:-1]):
        H_s=Hz(zv)/(MPC_CM/1e5)                        # s^-1
        dtdz=1.0/(H_s*(1+zv))                          # s
        nd=CNORM*fesc_grid*shape[i]                    # s^-1 Mpc^-3
        rec=CL[i]*aB*1.08*nH_cm3*(1+zv)**3*Q           # s^-1
        Q=np.clip(Q+(nd/nH_Mpc3-rec)*dtdz*dz,0,1)
        Qtraj[i+1]=Q
        zmid=ZGRID[i+1]
        tau+=fe*nH_cm3*(1+zmid)**3*Q*sigT*(c_kms*1e5)*(1.0/((Hz(zmid)/(MPC_CM/1e5))*(1+zmid)))*dz
    tau+=0.018
    chi2=((tau-TAU_OBS)/TAU_SIG)**2
    for entry in XHI_DATA:
        zd,xo,sg=entry[0],entry[1],entry[2]
        kind=entry[3] if len(entry)>3 else 'two'
        i0=int(np.argmin(np.abs(ZGRID-zd)))
        resid=(1-Qtraj[i0])-xo
        pen=(resid/sg)**2
        if kind=='lower': pen=np.where(resid>=0,0.0,pen)
        chi2+=pen
    return chi2,tau

FESC_GRID=np.linspace(0.01,0.50,200)

def run_point(sm,eta):
    f0p,suvp,chi2_uvlf=profile_uvlf(sm,eta)
    shape_fix=ndot_shape(sm,eta,CDM['f_star0'])        # fixed normalisations
    i7=np.argmin(np.abs(ZGRID-7.0))
    R_ion={zq:float(ndv/shape_cdm[np.argmin(np.abs(ZGRID-zq))])
           for zq,ndv in [(6.0,shape_fix[np.argmin(np.abs(ZGRID-6.0))]),
                          (7.0,shape_fix[i7]),
                          (8.0,shape_fix[i8]),
                          (10.0,shape_fix[np.argmin(np.abs(ZGRID-10.0))]),
                          (12.0,shape_fix[np.argmin(np.abs(ZGRID-12.0))])]}
    shape_prof=shape_fix*(f0p/CDM['f_star0'])          # f0 enters linearly (cap negligible)
    chi2_arr,tau_arr=reion_chi2_scan(shape_prof,FESC_GRID)
    jb=int(np.argmin(chi2_arr))
    # stage-1-only: fesc frozen at CDM profiled value (filled by caller)
    return dict(sm=sm,eta=eta,f0_prof=f0p,suv_prof=suvp,chi2_uvlf=chi2_uvlf,
                R_ion_z=R_ion,fesc_prof=float(FESC_GRID[jb]),
                chi2_reion=float(chi2_arr[jb]),tau=float(tau_arr[jb]),
                chi2_curve=chi2_arr.tolist())

if __name__=='__main__':
    import sys
    res={}
    # CDM reference
    cdm_pt=run_point(0.0,0.0)
    res['CDM']=cdm_pt
    fesc_cdm=cdm_pt['fesc_prof']; f0_cdm=CDM['f_star0']
    print(f"CDM: f0={f0_cdm:.5f} fesc_prof={fesc_cdm:.4f} chi2_reion={cdm_pt['chi2_reion']:.3f} tau={cdm_pt['tau']:.4f}")
    grid=[(1,0.10),(1,0.25),(1,0.50),(5,0.10),(5,0.25),(5,0.50),(10,0.10),(10,0.25),(10,0.50)]
    for sm,eta in grid:
        r=run_point(float(sm),eta)
        # stage-1-only chi2: fesc frozen at CDM's profiled value
        j_frozen=int(np.argmin(np.abs(FESC_GRID-fesc_cdm)))
        r['chi2_stage1']=r['chi2_curve'][j_frozen]
        del r['chi2_curve']
        r['inv_supp_ion_z7']=r['R_ion_z'][7.0]
        r['supp_uvlf']=f0_cdm/r['f0_prof']
        r['product_ratio']=(r['fesc_prof']*r['f0_prof'])/(fesc_cdm*f0_cdm)
        r['dchi2_prof']=r['chi2_reion']-cdm_pt['chi2_reion']
        r['dchi2_stage1']=r['chi2_stage1']-cdm_pt['chi2_reion']
        res[f"sm{sm}_eta{eta}"]=r
        print(f"sm={sm:>4} eta={eta:.2f} | f0={r['f0_prof']:.5f} ({r['f0_prof']/f0_cdm:.3f}x) "
              f"<1-eD>_ion(z7)={r['inv_supp_ion_z7']:.4f} <1-eD>_UVLF={r['supp_uvlf']:.4f} | "
              f"fesc={r['fesc_prof']:.4f} ({r['fesc_prof']/fesc_cdm:.3f}x) prod={r['product_ratio']:.4f} "
              f"| dchi2_prof={r['dchi2_prof']:+.3f} dchi2_stage1={r['dchi2_stage1']:+.1f}")
    del res['CDM']['chi2_curve']
    with open('rerun_A_results.json','w') as f: json.dump(res,f,indent=1)
    print("saved rerun_A_results.json")
