#!/usr/bin/env python3
"""Regenerate Figure 3 (per-probe sensitivity) consistent with the reruns.

Panels (a)-(e): 0/9 exclusions (full-integral rerun, published data vector;
blowout enhancement reabsorbed by profiling).
Panel (f): 21 cm POWER SPECTRUM forecast (Appendix G): 3.9/14.8/18.3 sigma
for sigma/m = 1/5/10 at 1000 h (optimistic) -> 6/9 above 5 sigma, 95% CL
line at the 2 cm^2/g threshold. The field-level beta0 forecast is noted.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SM = [1, 5, 10]
ETA = [0.10, 0.25, 0.50]
GREEN = '#2e8b2e'
RED = '#b22222'

panels = [
    ('(a)', 'UV luminosity function\nJWST, z = 8 to 14',
     r'$f_{\star,0},\ \sigma_{\rm UV}$ absorb the signal', 'none'),
    ('(b)', 'Thomson optical depth\nPlanck, $\\tau = 0.054 \\pm 0.007$',
     r'profiled $f_{\rm esc}$ tunes $\dot n_{\rm ion}$ to match $\tau$', 'none'),
    ('(c)', r'$\bar{x}_{\rm HI}(z)$ damping wings' + '\nJWST + QSO, z = 6.5 to 10',
     r'profiled $f_{\rm esc}$ absorbs the timeline shift', 'none'),
    ('(d)', r'Ly$\alpha$ luminosity function' + '\nLAEs, z = 6 to 8 (schematic)',
     'inherits the emissivity dependence;\nnot independently fitted', 'schematic'),
    ('(e)', r'Direct $f_{\rm esc}$ measurements' + '\nLyC surveys, $\\sigma_{f_{\\rm esc}} = 0.03$',
     'blowout enhancement reabsorbed\nby profiling', 'none'),
    ('(f)', r'21 cm power spectrum $P_{21}(k)$' + '\nSKA1-Low, 1000 hr (optimistic)',
     r'$\beta_0$ field-level: $2.0\sigma$ at $\sigma/m{=}10$ (1000 h)', 'p21'),
]

fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.2))
fig.suptitle('Each individual probe and its constraining power on SIDM',
             fontsize=13, fontweight='bold')

for ax, (tag, title, note, mode) in zip(axes.ravel(), panels):
    excluded = (lambda s, e: s >= 5) if mode == 'p21' else (lambda s, e: False)
    n_exc = 0
    for s in SM:
        for e in ETA:
            if excluded(s, e):
                ax.plot(s, e, 'X', color=RED, ms=13, mec='darkred', zorder=5)
                n_exc += 1
            else:
                ax.plot(s, e, 'o', color=GREEN, ms=11, mec='darkgreen', zorder=5)
    shade = '#fdecea' if mode == 'p21' else '#eaf6ea'
    ax.set_facecolor(shade)
    if mode == 'p21':
        ax.axvline(2.0, color=RED, lw=2)
        ax.text(2.0, 0.34, '95% CL', color=RED, rotation=90, fontsize=8,
                ha='right', va='center')
    if mode != 'schematic':
        cnt = f'{n_exc}/9'
        ax.text(0.955, 0.945, cnt, transform=ax.transAxes, fontsize=10,
                fontweight='bold', ha='right', va='top',
                color=(RED if n_exc else GREEN),
                bbox=dict(fc='white', ec=(RED if n_exc else GREEN), lw=1.2,
                          boxstyle='round,pad=0.25'))
    ax.text(0.045, 0.945, tag, transform=ax.transAxes, fontsize=11,
            fontweight='bold', va='top',
            bbox=dict(fc='white', ec='0.4', boxstyle='round,pad=0.25'))
    ax.text(0.97, 0.05, note, transform=ax.transAxes, fontsize=7.5,
            style='italic', color='0.35', ha='right',
            bbox=dict(fc='white', ec='0.75', boxstyle='round,pad=0.3'))
    ax.set_title(title, fontsize=10)
    ax.set_xscale('log')
    ax.set_xlim(0.55, 20)
    ax.set_ylim(0.05, 0.56)
    ax.set_xlabel(r'$\sigma/m$ [cm$^2$ g$^{-1}$]')
    ax.set_ylabel(r'$\eta$')

h = [plt.Line2D([], [], marker='o', color=GREEN, mec='darkgreen', ls='', ms=10,
                label='Allowed'),
     plt.Line2D([], [], marker='X', color=RED, mec='darkred', ls='', ms=11,
                label='Excluded (95% CL)')]
fig.legend(handles=h, loc='upper center', bbox_to_anchor=(0.5, 0.955),
           ncol=2, fontsize=10, frameon=True)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig('fig2_per_probe_v2.pdf')
fig.savefig('fig2_per_probe_v2.png', dpi=110)
print('saved fig2_per_probe_v2')
