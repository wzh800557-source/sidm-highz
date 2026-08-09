# SIDM at High Redshift: Simulation Grid and Reionization Analysis

Code and data for a series of studies connecting self-interacting dark matter (SIDM) microphysics to reionization-era observables:

- *Reionization Topology as a Probe of Self-Interacting Dark Matter* (Wang 2026) [[arXiv:2604.10726]](https://arxiv.org/abs/2604.10726)
- *Breaking the UV Luminosity Function Degeneracy: SIDM Constraints from Reionization Topology* (Wang & Shan 2026) [[arXiv:2604.19726]](https://arxiv.org/abs/2604.19726)
- *A Structural Degeneracy Explains Reionization Tensions and Limits Dark Matter Constraints* (Wang & Shan 2026, **accepted, ApJL**) [[arXiv:2605.01380]](https://arxiv.org/abs/2605.01380)

All studies use the same 230-run GIZMO N-body simulation grid spanning 5 halo masses, 7 cross-sections, 4 redshifts, plus 90 velocity-dependent Yukawa runs.

## Repository structure

    sidm-highz/
    ├── analysis/                     UVLF profile likelihood
    │   ├── uvlf_sidm.py             Core UVLF model, HMF, SFE, likelihood
    │   ├── generate_figures.py       Figure generation
    │   └── joint_constraint.py       Joint UVLF + topology analysis
    ├── cluster/                      Cluster job scripts
    │   ├── run_profile_scan.py       252-point profile scan
    │   ├── submit_profile.sh         SLURM script (MIT Engaging)
    │   └── quick_test.sh             Reduced grid for local testing
    ├── fesc_degeneracy/              Escape fraction degeneracy analysis
    │   ├── scripts/
    │   │   ├── reionization_optimizer.py     Original optimizer (SUPERSEDED: unit bug, see v2)
    │   │   ├── reionization_optimizer_v2.py  Corrected profile likelihood (published data vector)
    │   │   ├── rerun_A_published.py          Full-integral reionization scan (Table 4 / dchi2 <= 0.62)
    │   │   ├── rerun_A_asym.py               Symmetrised-error robustness check
    │   │   ├── core.py                       Semi-numerical reionization pipeline (density, fcoll, xi, dTb, V2)
    │   │   ├── v2_formscan.py                V2 topology functional-form scan (5 forms, 128^3/256^3)
    │   │   ├── topo_forecast.py              Field-level beta0 SKA1-Low forecast (mock obs + noise MC)
    │   │   ├── make_fig_per_probe_v3.py      Figure 3 (per-probe sensitivity)
    │   │   ├── make_fig_joint_v3.py          Figure 4 (joint constraints + forecast)
    │   │   ├── make_fig_three_panel.py       Figure 1 (degeneracy panels)
    │   │   ├── extract_dbind.py              Binding energy from GIZMO snapshots
    │   │   └── generate_ics.py               NFW IC generator (galpy)
    │   └── figures/
    ├── data/                         Shared data
    │   ├── dbind_table.json          Delta_bind grid (210 entries, 5 radii)
    │   ├── dbind_table.csv           Same in CSV format
    │   ├── profile_scan_results.json UVLF scan output (252 points)
    │   ├── uvlf_data_verified.json   31 JWST UVLF data points
    │   ├── rerun_A_published_vector_results.json  Reionization scan vs published xHI + tau
    │   ├── v2_formscan_results.json               V2 form scan, 128^3 (3 seeds x 5 forms)
    │   ├── v2_formscan_256_results.json           V2 form scan, 256^3 production
    │   ├── topo_forecast_128_results.json         beta0 forecast, 128^3
    │   ├── topo_forecast_256_results.json         beta0 forecast, 256^3 production
    │   └── panelb_chi2.json                       Fig 1b chi2 surface
    ├── figures/                      UVLF analysis figures
    ├── docs/
    │   └── eta_calibration.md        eta calibration literature survey
    ├── README.md
    ├── LICENSE
    └── requirements.txt

## Quick start

```bash
pip install numpy scipy matplotlib

# UVLF profile scan
python analysis/uvlf_sidm.py

# Product-degeneracy scan (f_esc and f_star both profiled): prints
# near-exact cancellation, max |Delta_chi2| ~ 0.005
python fesc_degeneracy/scripts/reionization_optimizer_v2.py --scan --data data/dbind_table.json

# Full-integral reionization scan (UVLF-profiled f_star, f_esc profiled
# against the published tau + xHI(z) vector): Delta_chi2 <= 0.62 at all
# 9 grid points (paper Table 4; ~minutes)
python fesc_degeneracy/scripts/rerun_A_published.py
```

Note: `reionization_optimizer.py` (v1) is retained for provenance but is
superseded — it contained a unit error in the ionization ODE that made
Delta_chi2 exactly zero by construction. `reionization_optimizer_v2.py`
fixes the units, recalibrates the emissivity, and uses the exact published
data vector (Umeda+24; Mason+26; Greig+17; Mason+19) quoted in the paper.

## Simulation grid

| Parameter | Values | Count |
|-----------|--------|-------|
| Halo mass | 10^9, 10^9.5, 10^10, 10^10.5, 10^11 M_sun | 5 |
| sigma/m (constant) | 0, 0.5, 1, 2, 5, 10, 20 cm^2/g | 7 |
| Redshift | 0, 4, 7, 10 | 4 |
| **Constant-sigma total** | | **140** |
| sigma_0/m (Yukawa) | 1, 3, 10 cm^2/g | 3 |
| w (Yukawa scale) | 50, 100, 200 km/s | 3 |
| Redshift (Yukawa) | 4, 7 | 2 |
| **Yukawa total** | | **90** |
| **Grand total** | | **230** |

All simulations use GIZMO with N = 500,000 particles per halo, evolved for 0.5 Gyr. Concentrations follow Dutton & Maccio (2014). Initial conditions from galpy Eddington inversion.

## Data format

Key naming: `M{MMM}_z{Z}_s{SSSS}_{type}`

- `MMM`: Mass code (090 = 10^9.0, 095 = 10^9.5, 100 = 10^10.0, 105 = 10^10.5, 110 = 10^11.0)
- `Z`: Redshift (0, 4, 7, 10)
- `SSSS`: Cross-section code (0005 = 0.5, 0010 = 1, 0020 = 2, 0050 = 5, 0100 = 10, 0200 = 20)
- `type`: `const` for constant cross-section; for Yukawa add `_w{WWW}_vd`

```python
import json
with open('data/dbind_table.json') as f:
    grid = json.load(f)

entry = grid['M100_z7_s0100_const']
# entry['R_kpc']      = [0.3, 0.5, 1.0, 2.0, 5.0]
# entry['delta_bind'] = [0.90, 0.72, 0.28, -0.06, ...]
# Delta_bind = 1 - W_g^SIDM / W_g^CDM
```

## Key results

### UVLF degeneracy

The UVLF alone cannot constrain SIDM: Delta_chi2 drops from ~3000 to < 0.21 across 252 grid points when astrophysical parameters are profiled.

### Joint dark-matter–galaxy degeneracy (accepted paper)

At fixed dark matter parameters the ionizing photon rate is separable,
n_ion ∝ f_esc × f_star,0, so emissivity-based probes constrain only the
product. Once the dark matter parameters vary, re-matching the observed
UVLF also restores the ionizing emissivity to within a few per cent
(the same haloes dominate both), producing a joint
dark-matter–f_star,0–f_esc degeneracy. Confronted with the published
tau + xHI(z) compilation, the profiled Delta_chi2 stays <= 0.62 at every
grid point:

| Probe (current data, f_esc profiled) | Excluded (of 9) |
|---------------------------------------|-----------------|
| UVLF alone | 0/9 |
| + tau + x_HI (published compilation) | 0/9 |
| + f_esc prior + blowout correlation | 0/9 (reabsorbed by profiling) |

### SKA1-Low forecasts (two-pointing mosaic, times per pointing)

| Channel | 1000 h | 5000 h |
|---------|--------|--------|
| P21(k), 5-sigma threshold (optimistic foregrounds) | sigma/m ≈ 2 cm²/g | sigma/m ≈ 1.2 cm²/g |
| beta0 field-level topology (sigma/m = 10) | 2.0 sigma | 4.1 sigma |
| beta0 (moderate wedge, any sigma/m) | < 0.8 sigma | < 0.8 sigma |

The V2 (Euler characteristic) contrast is +18% ± 1% at 256^3 (three
seeds), stable at +15% to +18% across five alternative functional forms
of the emissivity modification (v2_formscan_256_results.json).




## Citation

If you use this code or data, please cite:

```bibtex
@article{WangShan2026,
  author  = {{Wang}, Zihan and {Shan}, Huanyuan},
  title   = {A Structural Degeneracy Explains Reionization Tensions and
             Limits Dark Matter Constraints},
  journal = {The Astrophysical Journal Letters},
  year    = {2026},
  note    = {accepted; arXiv:2605.01380}
}
```

## License

MIT License. See [LICENSE](LICENSE).

## Contact

Zihan Wang — zihan.wang@queens.ox.ac.uk




