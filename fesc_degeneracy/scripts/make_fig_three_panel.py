#!/usr/bin/env python3
"""Regenerated Figure 1: (a) top, full width; (b)+(c) bottom row.
Panel (c) has a beta0-ratio strip showing the topology separation vs seed scatter.
EDIT the LIT table to adjust literature placements."""
import json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

plt.rcParams.update({'font.size':9,'font.family':'serif','axes.linewidth':0.8,
                     'xtick.direction':'in','ytick.direction':'in'})

fig=plt.figure(figsize=(9.5,8.2))
gs=GridSpec(2,2,figure=fig,height_ratios=[1.05,1.25],hspace=0.30,wspace=0.26,
            left=0.075,right=0.985,top=0.965,bottom=0.075)
ax_a=fig.add_subplot(gs[0,:])
ax_b=fig.add_subplot(gs[1,0])
gs_c=gs[1,1].subgridspec(2,1,height_ratios=[2.1,1.0],hspace=0.07)
ax_c=fig.add_subplot(gs_c[0]); ax_cr=fig.add_subplot(gs_c[1],sharex=ax_c)

# ---------------- panel (a): literature ----------------
LIT=[('Naidu+22',0.0125,0.20,0.004,0.05,'b'),
     ('Finkelstein+19',0.055,0.046,0.015,0.015,'b'),
     ('Finkelstein+22',0.038,0.062,0.010,0.02,'b'),
     ('Robertson 22',0.024,0.10,0.006,0.03,'b'),
     ('Saldana-L.+22',0.033,0.062,0.010,0.02,'l'),
     ('Mascia+23',0.017,0.13,0.005,0.03,'l'),
     ('Pahl+25',0.046,0.055,0.012,0.018,'l'),
     ('Begley+25',0.052,0.07,0.013,0.02,'l'),
     ('Ma+20',0.030,0.09,0.008,0.03,'s'),
     ('SPHINX',0.009,0.10,0.003,0.03,'s'),
     ('THESAN',0.011,0.16,0.003,0.04,'s')]
STY={'b':dict(marker='o',color='#c23b22',label='Reionization budget models'),
     'l':dict(marker='s',color='#1f77b4',label='Direct LyC measurements'),
     's':dict(marker='D',color='#2ca02c',label='Hydro simulations')}
OFF={'Saldana-L.+22':(-50,-15),'Finkelstein+22':(-26,11),'Pahl+25':(-44,-14),
     'Finkelstein+19':(6,9),'Begley+25':(8,7),'Ma+20':(7,3)}
ax=ax_a; seen=set()
for nm,f0,fe,xe,ye,cl in LIT:
    st=STY[cl]
    ax.errorbar(f0,fe,xerr=xe,yerr=ye,fmt=st['marker'],color=st['color'],ms=5.5,
                lw=0.8,capsize=1.5,label=(st['label'] if cl not in seen else None))
    seen.add(cl)
    ax.annotate(nm,(f0,fe),textcoords='offset points',
                xytext=OFF.get(nm,(5,5)),fontsize=7)
xx=np.logspace(-2.35,-0.9,100)
ax.plot(xx,2.5e-3/xx,'k--',lw=1)
ax.fill_between(xx,(2.5-0.8)*1e-3/xx,(2.5+0.8)*1e-3/xx,color='gray',alpha=0.18,lw=0)
ax.text(0.0058,2.5e-3/0.0058*0.52,r'$f_{\rm esc}\times f_{\star,0}=\rm const$',
        fontsize=8,rotation=-24)
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlim(4e-3,1.3e-1); ax.set_ylim(2.5e-2,3.5e-1)
ax.set_xlabel(r'Star formation efficiency $f_{\star,0}$')
ax.set_ylabel(r'Escape fraction $f_{\rm esc}$')
ax.legend(fontsize=7,loc='lower left',frameon=False)
ax.text(0.015,0.93,'(a)',transform=ax.transAxes,fontweight='bold')

# ---------------- panel (b): chi2 valley + trajectory ----------------
ax=ax_b
d=json.load(open('panelb_chi2.json'))
prod=np.array(d['product']); chi=np.array(d['chi2'])-d['cdm']['chi2']
f0c,fec=d['cdm']['f0'],d['cdm']['fesc']
F0=np.logspace(np.log10(0.008),np.log10(0.08),220)
FE=np.logspace(np.log10(0.03),np.log10(0.4),220)
G0,GE=np.meshgrid(F0,FE)
CH=np.interp((G0*GE).ravel(),prod,chi,left=chi[0],right=chi[-1]).reshape(G0.shape)
cs=ax.contour(G0,GE,CH,levels=[1,4,9,16,25],colors='k',linewidths=0.6,alpha=0.7)
ax.clabel(cs,fmt=lambda v:r'$\Delta\chi^2$=%d'%v,fontsize=6,inline_spacing=2)
f0s,fes=f0c*1.812,fec*0.926
ax.plot([f0c],[fec],'o',color='k',ms=7,zorder=5)
ax.annotate('CDM',(f0c,fec),textcoords='offset points',xytext=(-27,-13),fontsize=8)
ax.annotate('',xy=(f0s,fec),xytext=(f0c,fec),
            arrowprops=dict(arrowstyle='-|>',color='#c23b22',lw=1.8))
ax.plot([f0s],[fes],'*',color='#c23b22',ms=13,zorder=5)
ax.annotate('SIDM\n'+r'($\sigma/m$=10, $\eta$=0.5)',(f0s,fes),
            textcoords='offset points',xytext=(7,-26),fontsize=7,color='#c23b22')
ax.plot(F0,f0s*fes/F0,ls='--',color='#c23b22',lw=1.1)
ax.plot(F0,f0c*fec/F0,ls='-',color='0.4',lw=1.1)
ax.text(0.0115,f0c*fec/0.0115*1.07,'CDM hyperbola',fontsize=6.5,rotation=-40,color='0.3')
ax.text(0.024,f0s*fes/0.024*1.07,
        r'$f_{\rm esc}f_{\star,0}\langle 1-\eta\Delta\rangle_{\rm ion}$=const',
        fontsize=6.5,rotation=-40,color='#c23b22')
ax.text(0.5,0.955,r'Step 1: UVLF raises $f_{\star,0}$ ($\times$1.81);'
        +'\nemissivity simultaneously restored',transform=ax.transAxes,
        fontsize=7,ha='center',va='top')
ax.text(0.5,0.045,r'Step 2: $f_{\rm esc}$ absorbs residual ($\lesssim 8\%$);  $\Delta\chi^2\leq0.6$',
        transform=ax.transAxes,fontsize=7,ha='center')
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlim(F0[0],F0[-1]); ax.set_ylim(FE[0],FE[-1])
ax.set_xlabel(r'Star formation efficiency $f_{\star,0}$')
ax.set_ylabel(r'Escape fraction $f_{\rm esc}$')
ax.text(0.03,0.93,'(b)',transform=ax.transAxes,fontweight='bold')

# ---------------- panel (c): beta0 curves + ratio strip ----------------
b=json.load(open('b0_agg.json'))
xhs=['0.2','0.3','0.4','0.5','0.6','0.7','0.8']
xhi_ax=1-np.array([float(x) for x in xhs]); o=np.argsort(xhi_ax)
mu={}; sd={}
for m in ('cdm','sidm5','sidm10'):
    mu[m]=np.array([b[m][x][0] for x in xhs])
    sd[m]=np.array([b[m][x][1] for x in xhs])
CFG=[('cdm','k','CDM'),('sidm5','#1f77b4',r'SIDM $\sigma/m$=5'),
     ('sidm10','#c23b22',r'SIDM $\sigma/m$=10')]
for m,col,lab in CFG:
    ax_c.plot(xhi_ax[o],mu[m][o],color=col,lw=1.5,label=lab)
    ax_c.fill_between(xhi_ax[o],(mu[m]-3*sd[m])[o],(mu[m]+3*sd[m])[o],
                      color=col,alpha=0.15,lw=0)
ax_c.set_yscale('log')
ax_c.legend(fontsize=7,frameon=False,loc='center right')
ax_c.set_ylabel(r'Betti number $\beta_0$')
ax_c.text(0.03,0.90,'(c)',transform=ax_c.transAxes,fontweight='bold')
ax_c.text(0.30,0.86,'identical UVLF, $\\tau$, $\\bar{x}_{\\rm HI}(z)$\n'
          r'$\rightarrow$ distinct topology',transform=ax_c.transAxes,
          fontsize=7.5,ha='center')
ax_c.axvline(0.5,color='0.7',lw=0.7,ls=':')
plt.setp(ax_c.get_xticklabels(),visible=False)
ax_c.text(0.03,0.05,r'reconstructed pipeline, $128^3$, 3 seeds',
          transform=ax_c.transAxes,fontsize=6,color='0.4')
# ratio strip
for m,col,lab in CFG[1:]:
    r=mu[m]/mu['cdm']
    re=r*np.sqrt((sd[m]/mu[m])**2+(sd['cdm']/mu['cdm'])**2)
    ax_cr.plot(xhi_ax[o],100*(r-1)[o],color=col,lw=1.5)
    ax_cr.fill_between(xhi_ax[o],100*(r-1-re)[o],100*(r-1+re)[o],color=col,alpha=0.2,lw=0)
cdm_scat=100*(sd['cdm']/mu['cdm'])
ax_cr.fill_between(xhi_ax[o],-cdm_scat[o],cdm_scat[o],color='0.55',alpha=0.35,lw=0)
ax_cr.axhline(0,color='k',lw=0.7)
ax_cr.text(0.24,3.0,'CDM seed scatter',fontsize=6,color='0.35')
ax_cr.annotate('morphology breaks the degeneracy\n'+r'($P_{21}$: $\sigma/m\gtrsim2$ at $>5\sigma$, 1000 h; $\beta_0$: $4.1\sigma$, 5000 h)',
               xy=(0.5,11.5),xytext=(0.63,27),
               fontsize=7,color='#c23b22',ha='center',
               arrowprops=dict(arrowstyle='->',color='#c23b22',lw=0.9))
ax_cr.set_ylim(-6,44)
ax_cr.set_xlabel(r'Neutral fraction $\bar{x}_{\rm HI}$')
ax_cr.set_ylabel(r'$\beta_0/\beta_0^{\rm CDM}-1$ [%]')

plt.savefig('fig_three_panel_v2.pdf')
plt.savefig('fig_three_panel_v2.png',dpi=150)
print('saved')
