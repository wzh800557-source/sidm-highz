#!/usr/bin/env python3
"""
Generate all 7 publication figures for the SIDM UVLF paper.

Usage:
  python generate_figures.py                    # uses default data path
  python generate_figures.py path/to/results.json  # custom path

Requires: numpy, scipy, matplotlib
Outputs: fig[1-7]_*.pdf in the current directory
"""

import numpy as np
import json
import sys
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# ================================================================
# STYLE
# ================================================================
STYLE = {
    'font.family': 'serif', 'font.size': 10,
    'axes.labelsize': 12, 'xtick.labelsize': 9, 'ytick.labelsize': 9,
    'xtick.direction': 'in', 'ytick.direction': 'in',
    'xtick.top': True, 'ytick.right': True,
    'xtick.minor.visible': True, 'ytick.minor.visible': True,
    'mathtext.fontset': 'cm', 'figure.dpi': 300,
}
plt.rcParams.update(STYLE)

# ================================================================
# DATA AND MODEL (import from analysis module)
# ================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from uvlf_sidm import (get_all_data, compute_UVLF, log_likelihood,
                        SFE_CDM, SFE_SIDM, binding_energy_ratio)

# Topology model
def p_sidm(sigma_m, alpha=0.7):
    r1 = 0.8 * (sigma_m / 1.0)**0.5
    if r1 < 0.3: return 0.10
    dW = min(0.26 * sigma_m**0.6, 0.95)
    return min(0.10 * (1.0 / (1.0 - dW))**alpha, 0.50)

def ska_snr(sigma_m, floor=0.05):
    ps = p_sidm(sigma_m)
    if ps <= 0.10: return 0.0
    sig = 1.0 - 0.10 / ps
    sig10 = 1.0 - 0.10 / p_sidm(10.0)
    snr10 = min(15.0 * (0.05 / max(floor, 0.01)), 25.0)
    return snr10 * (sig / sig10)

# CDM best-fit params
PARAMS_CDM = {
    'f_star0': 0.019, 'M_p': 1e11, 'alpha_lo': 2.14,
    'alpha_hi': 0.5, 'z_evol': 0.13, 'M_turn': 5e7, 'A_UV0': 0.0,
}
SIGMA_UV_CDM = 0.65


def fig1_cdm_uvlf():
    """CDM baseline UVLF fit (4-panel)."""
    z_d, Muv_d, logphi_d, sig_d = get_all_data()
    Muv_model = np.linspace(-22.5, -16.5, 60)

    fig, axes = plt.subplots(2, 2, figsize=(7.0, 6.5), sharex=True, sharey=True)
    fig.subplots_adjust(hspace=0.08, wspace=0.08, left=0.12, right=0.97,
                        top=0.97, bottom=0.10)

    for (row, col, z_val) in [(0,0,9), (0,1,10), (1,0,11), (1,1,12)]:
        ax = axes[row, col]
        phi_mod = compute_UVLF(Muv_model, z_val, 0, 0, PARAMS_CDM, SIGMA_UV_CDM)
        logphi_mod = np.log10(np.maximum(phi_mod, 1e-10))
        valid = phi_mod > 1e-8
        ax.plot(Muv_model[valid], logphi_mod[valid], '-', color='#333333', lw=1.5, zorder=3)

        mask = z_d == z_val
        if np.any(mask):
            ax.errorbar(Muv_d[mask], logphi_d[mask], yerr=sig_d[mask],
                        fmt='o', ms=5, color='#2166AC', elinewidth=1.0,
                        capsize=2.5, capthick=0.8, mec='#2166AC', mfc='white',
                        mew=1.0, zorder=5)

        ax.text(0.05, 0.06, r'$z = {}$'.format(z_val), transform=ax.transAxes,
                fontsize=13, fontweight='bold', va='bottom')
        ax.set_xlim(-22.5, -17.0); ax.set_ylim(-6.5, -2.2)
        ax.xaxis.set_major_locator(MultipleLocator(1))
        ax.yaxis.set_major_locator(MultipleLocator(1))

    fig.text(0.54, 0.02, r'$M_{\rm UV}$ [mag]', ha='center', fontsize=13)
    fig.text(0.02, 0.54, r'$\log_{10}\,\Phi\;[\mathrm{Mpc}^{-3}\,\mathrm{mag}^{-1}]$',
             ha='center', va='center', rotation='vertical', fontsize=13)
    fig.savefig('fig1_uvlf_cdm.pdf', bbox_inches='tight', dpi=300)
    print("  fig1_uvlf_cdm.pdf")


def fig2_topology():
    """Topology signal and SKA detectability."""
    sm = np.linspace(0.01, 15, 500)
    p_arr = np.array([p_sidm(s) for s in sm])
    snr_5 = np.array([ska_snr(s, 0.05) for s in sm])
    snr_10 = np.array([ska_snr(s, 0.10) for s in sm])
    snr_20 = np.array([ska_snr(s, 0.20) for s in sm])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.2))
    fig.subplots_adjust(wspace=0.35, left=0.10, right=0.97, top=0.92, bottom=0.16)

    ax1.plot(sm, p_arr, '-', color='#333333', lw=1.5)
    ax1.axhline(0.10, ls=':', color='#999999', lw=0.8)
    ax1.plot([1.0, 10.0], [0.116, 0.30], 'D', ms=6, color='#D6604D', mfc='white', mew=1.3, zorder=5)
    ax1.set_xlabel(r'$\sigma/m\;[\mathrm{cm^2\,g^{-1}}]$')
    ax1.set_ylabel(r'Duty cycle $p$')
    ax1.set_xscale('log'); ax1.set_xlim(0.1, 15); ax1.set_ylim(0.05, 0.55)
    ax1.text(0.05, 0.93, '(a)', transform=ax1.transAxes, fontsize=11, fontweight='bold')

    ax2.plot(sm, snr_5, '-', color='#1B7837', lw=1.5)
    ax2.plot(sm, snr_10, '--', color='#1B7837', lw=1.3)
    ax2.plot(sm, snr_20, ':', color='#1B7837', lw=1.3)
    ax2.axhline(3, ls='-', color='#D6604D', lw=0.8, alpha=0.6)
    ax2.set_xlabel(r'$\sigma/m\;[\mathrm{cm^2\,g^{-1}}]$')
    ax2.set_ylabel(r'Cumulative SKA1-Low SNR')
    ax2.set_xscale('log'); ax2.set_xlim(0.1, 15); ax2.set_ylim(0, 20)
    ax2.text(0.05, 0.93, '(b)', transform=ax2.transAxes, fontsize=11, fontweight='bold')

    fig.savefig('fig2_topology.pdf', bbox_inches='tight', dpi=300)
    print("  fig2_topology.pdf")


def fig3_degeneracy(scan_data):
    """Conditional vs profiled Δχ²."""
    g = scan_data['grids']
    sm_grid = np.array(g['sigma_m'])
    eta_grid = np.array(g['eta'])

    eta_sel = {0.05: 2, 0.10: 4, 0.25: 8, 0.50: 11}
    colors = {0.05: '#92C5DE', 0.10: '#4393C3', 0.25: '#D6604D', 0.50: '#762A83'}
    markers = {0.05: '^', 0.10: 'o', 0.25: 's', 0.50: 'D'}
    sm_idx = [i for i, s in enumerate(sm_grid) if s > 0]
    sm_plot = sm_grid[sm_grid > 0]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.2))
    fig.subplots_adjust(wspace=0.35, left=0.10, right=0.97, top=0.92, bottom=0.16)

    for eta_val, j in eta_sel.items():
        dc_c = [g['dchi2_conditional'][i][j] for i in sm_idx]
        dc_c = [x if x is not None else np.nan for x in dc_c]
        v = ~np.isnan(dc_c)
        ax1.semilogy(np.array(sm_plot)[v], np.array(dc_c)[v], '-',
                     marker=markers[eta_val], color=colors[eta_val], ms=4, mfc='white', mew=1.0, lw=1.2)

        dc_p = [g['dchi2_profiled'][i][j] for i in sm_idx]
        dc_p = [x if x is not None else np.nan for x in dc_p]
        v2 = ~np.isnan(dc_p)
        ax2.plot(np.array(sm_plot)[v2], np.array(dc_p)[v2], '-',
                 marker=markers[eta_val], color=colors[eta_val], ms=4, mfc='white', mew=1.0, lw=1.2)

    ax1.axhline(3.84, ls='--', color='#999999', lw=0.8); ax1.set_xscale('log')
    ax1.set_xlim(0.03, 25); ax1.set_ylim(0.5, 10000)
    ax1.set_xlabel(r'$\sigma/m\;[\mathrm{cm^2\,g^{-1}}]$'); ax1.set_ylabel(r'$\Delta\chi^2_{\rm cond}$')
    ax1.text(0.05, 0.93, r'(a) Fixed astrophysics', transform=ax1.transAxes, fontsize=9, fontweight='bold')

    ax2.axhline(3.84, ls='--', color='#999999', lw=0.8); ax2.set_xscale('log')
    ax2.set_xlim(0.03, 25); ax2.set_ylim(-0.05, 5.5)
    ax2.set_xlabel(r'$\sigma/m\;[\mathrm{cm^2\,g^{-1}}]$'); ax2.set_ylabel(r'$\Delta\chi^2_{\rm prof}$')
    ax2.text(0.05, 0.93, r'(b) Re-optimized astrophysics', transform=ax2.transAxes, fontsize=9, fontweight='bold')

    fig.savefig('fig3_degeneracy.pdf', bbox_inches='tight', dpi=300)
    print("  fig3_degeneracy.pdf")


def fig4_astro_cost(scan_data):
    """f_star0 required vs sigma/m with SMF band."""
    g = scan_data['grids']
    sm_grid = np.array(g['sigma_m'])
    f0_cdm = scan_data['cdm_baseline']['f_star0']

    eta_sel = {0.10: 4, 0.25: 8, 0.50: 11}
    colors = {0.10: '#4393C3', 0.25: '#D6604D', 0.50: '#762A83'}
    markers = {0.10: 'o', 0.25: 's', 0.50: 'D'}

    fig, ax = plt.subplots(1, 1, figsize=(4.5, 3.8))
    fig.subplots_adjust(left=0.17, right=0.95, top=0.95, bottom=0.14)
    ax.axhspan(0.014, 0.023, color='#B8D4E3', alpha=0.35, zorder=0)
    ax.axhline(f0_cdm, ls=':', color='#999999', lw=0.8, zorder=1)

    for eta_val, j in eta_sel.items():
        f0_vals, sm_vals = [], []
        for i, s in enumerate(sm_grid):
            shift = g['f0_shift_pct'][i][j]
            if shift is not None and s >= 0:
                f0_vals.append(f0_cdm * (1 + shift / 100))
                sm_vals.append(s)
        ax.plot(sm_vals, f0_vals, '-', marker=markers[eta_val], color=colors[eta_val],
                ms=4, mfc='white', mew=1.0, lw=1.2, zorder=5)

    ax.set_xlabel(r'$\sigma/m\;[\mathrm{cm^2\,g^{-1}}]$')
    ax.set_ylabel(r'$f_{\star,0}$ (profiled best fit)')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlim(0.03, 25); ax.set_ylim(0.012, 0.15)
    fig.savefig('fig4_astro_cost.pdf', bbox_inches='tight', dpi=300)
    print("  fig4_astro_cost.pdf")


def fig5_joint(scan_data):
    """Joint constraint in (sigma/m, eta) plane."""
    smf_limits = scan_data['limits_95CL']['uvlf_smf']
    eta_smf = np.array([float(k) for k in sorted(smf_limits.keys(), key=float)])
    sm_smf = np.array([smf_limits[f"{e:.2f}"] for e in eta_smf])

    fig, ax = plt.subplots(1, 1, figsize=(5.5, 5.0))
    fig.subplots_adjust(left=0.14, right=0.95, top=0.95, bottom=0.12)

    ax.fill_betweenx(eta_smf, sm_smf, 20, color='#FDDBC7', alpha=0.6, zorder=1)
    ax.fill_betweenx([eta_smf[-1], 1.0], [sm_smf[-1]]*2, [20]*2, color='#FDDBC7', alpha=0.6, zorder=1)
    ax.axvspan(2.26, 20, color='#D9F0D3', alpha=0.4, zorder=0)
    ax.axvspan(0.76, 2.26, color='#E8F5E2', alpha=0.3, zorder=0)

    ax.plot(sm_smf, eta_smf, '-', color='#B2182B', lw=2.0, zorder=5)
    for sm_t, ls in [(0.76, '-'), (2.26, '--'), (6.09, ':')]:
        ax.axvline(sm_t, ls=ls, color='#1B7837', lw=1.5, zorder=4)

    ax.set_xlabel(r'$\sigma/m\;[\mathrm{cm^2\,g^{-1}}]$')
    ax.set_ylabel(r'$\eta$ (SFE coupling)')
    ax.set_xscale('log'); ax.set_xlim(0.1, 20); ax.set_ylim(0, 1.0)
    ax.set_xticks([0.1, 0.3, 1, 3, 10])
    ax.set_xticklabels(['0.1', '0.3', '1', '3', '10'])
    fig.savefig('fig5_joint_constraint.pdf', bbox_inches='tight', dpi=300)
    print("  fig5_joint_constraint.pdf")


def fig6_uvlf_sidm():
    """CDM vs SIDM UVLF at z=10."""
    z_d, Muv_d, logphi_d, sig_d = get_all_data()
    mask = z_d == 10
    Muv_model = np.linspace(-22.0, -17.0, 50)

    phi_cdm = compute_UVLF(Muv_model, 10, 0, 0, PARAMS_CDM, 0.65)
    phi_sidm = compute_UVLF(Muv_model, 10, 2.0, 0.25, PARAMS_CDM, 0.65)
    p_adj = dict(PARAMS_CDM); p_adj['f_star0'] = 0.035
    phi_adj = compute_UVLF(Muv_model, 10, 2.0, 0.25, p_adj, 0.79)

    fig, ax = plt.subplots(1, 1, figsize=(4.5, 4.0))
    fig.subplots_adjust(left=0.16, right=0.95, top=0.95, bottom=0.14)
    ax.errorbar(Muv_d[mask], logphi_d[mask], yerr=sig_d[mask], fmt='o', ms=5,
                color='#2166AC', elinewidth=1.0, capsize=2.5, mfc='white', mew=1.0, zorder=10)
    for phi, style, c in [(phi_cdm, '-', '#333333'), (phi_sidm, '--', '#D6604D'), (phi_adj, '-.', '#762A83')]:
        v = phi > 1e-8
        ax.plot(Muv_model[v], np.log10(phi[v]), style, color=c, lw=1.5, zorder=5)
    ax.set_xlabel(r'$M_{\rm UV}$ [mag]'); ax.set_ylabel(r'$\log_{10}\,\Phi\;[\mathrm{Mpc}^{-3}\,\mathrm{mag}^{-1}]$')
    ax.set_xlim(-22.2, -17.2); ax.set_ylim(-6.0, -2.2)
    ax.text(0.05, 0.06, r'$z = 10$', transform=ax.transAxes, fontsize=13, fontweight='bold')
    fig.savefig('fig6_uvlf_sidm.pdf', bbox_inches='tight', dpi=300)
    print("  fig6_uvlf_sidm.pdf")


def fig7_vdsidm():
    """Velocity-dependent SIDM escape."""
    def sigma_v(v, s0, w):
        return s0 / (1 + (v / w)**2)**2
    v = np.logspace(0.5, 3.5, 300)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.2))
    fig.subplots_adjust(wspace=0.35, left=0.10, right=0.97, top=0.92, bottom=0.16)

    ax1.loglog(v, [sigma_v(vi, 50, 30) for vi in v], '-', color='#762A83', lw=1.5)
    ax1.loglog(v, [sigma_v(vi, 20, 50) for vi in v], '--', color='#4393C3', lw=1.5)
    ax1.axhspan(0.1, 1.0, color='#FEE08B', alpha=0.3, zorder=0)
    for vc in [30, 100, 1000]:
        ax1.axvline(vc, ls=':', color='#BBBBBB', lw=0.7)
    ax1.set_xlabel(r'$v\;[\mathrm{km\,s^{-1}}]$'); ax1.set_ylabel(r'$\sigma/m\;[\mathrm{cm^2\,g^{-1}}]$')
    ax1.set_xlim(3, 2000); ax1.set_ylim(0.01, 100)
    ax1.text(0.05, 0.93, '(a)', transform=ax1.transAxes, fontsize=11, fontweight='bold')

    probes = [('21 cm topology', 40, 70, '#1B7837'), ('UVLF (z>9)', 50, 150, '#D6604D'), ('Clusters', 800, 1500, '#B8860B')]
    for label, vlo, vhi, c in probes:
        eff = np.mean([sigma_v(vi, 50, 30) for vi in np.linspace(vlo, vhi, 50)])
        ax2.barh(label, eff, height=0.3, color=c, alpha=0.6, edgecolor=c, lw=0.8)
    ax2.set_xlabel(r'Effective $\sigma/m\;[\mathrm{cm^2\,g^{-1}}]$')
    ax2.set_xscale('log'); ax2.set_xlim(0.01, 100)
    ax2.text(0.05, 0.93, '(b)', transform=ax2.transAxes, fontsize=11, fontweight='bold')

    fig.savefig('fig7_vdsidm.pdf', bbox_inches='tight', dpi=300)
    print("  fig7_vdsidm.pdf")


if __name__ == '__main__':
    data_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(SCRIPT_DIR, '..', 'data', 'profile_scan_results.json')

    print("Loading scan results...")
    with open(data_path) as f:
        scan = json.load(f)

    print(f"Grid: {scan['metadata']['grid_size']}, Mode: {scan['metadata']['mode']}")
    print("Generating figures:")
    fig1_cdm_uvlf()
    fig2_topology()
    fig3_degeneracy(scan)
    fig4_astro_cost(scan)
    fig5_joint(scan)
    fig6_uvlf_sidm()
    fig7_vdsidm()
    print("Done.")
