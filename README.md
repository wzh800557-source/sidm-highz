# SIDM at High Redshift: Simulation Grid and Reionization Analysis

Code and data for a series of studies connecting self-interacting dark matter (SIDM) microphysics to reionization-era observables:

- *Reionization Topology as a Probe of Self-Interacting Dark Matter* (Wang 2026) [[arXiv:2604.10726]](https://arxiv.org/abs/2604.10726)
- *Breaking the UV Luminosity Function Degeneracy: SIDM Constraints from Reionization Topology* (Wang & Shan 2026) [[arXiv:2604.19726]](https://arxiv.org/abs/2604.19726)
- *The escape fraction degeneracy: a fundamental barrier to constraining dark matter from the epoch of reionization* 
[[arXiv:2605.01380]](https://arxiv.org/abs/2605.01380)
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
    │   │   ├── reionization_optimizer.py   Profile likelihood for f_esc x f_star
    │   │   ├── extract_dbind.py            Binding energy from GIZMO snapshots
    │   │   └── generate_ics.py             NFW IC generator (galpy)
    │   └── figures/
    ├── data/                         Shared data
    │   ├── dbind_table.json          Delta_bind grid (210 entries, 5 radii)
    │   ├── dbind_table.csv           Same in CSV format
    │   ├── profile_scan_results.json UVLF scan output (252 points)
    │   └── uvlf_data_verified.json   31 JWST UVLF data points
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

# Escape fraction degeneracy (should print Delta_chi2 = 0 at all 9 points)
python fesc_degeneracy/scripts/reionization_optimizer.py --scan --data data/dbind_table.json

# With blowout correlation
python fesc_degeneracy/scripts/reionization_optimizer.py --scan --data data/dbind_table.json --alpha_esc 0.3 --sigma_fesc 0.03
```

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

### Escape fraction degeneracy

The product f_esc x f_star renders all reionization-history probes (tau, x_HI, Lya LF, QSO proximity zones) collectively blind to SIDM. Only the 21 cm topology, which depends on the duty cycle per halo at fixed x_HI, breaks this degeneracy.

| Configuration | Excluded (of 9) |
|---------------|-----------------|
| UVLF alone | 0/9 |
| + tau + x_HI (product degeneracy) | 0/9 |
| + f_esc prior + blowout correlation | 5/9 |
| 21 cm topology | 9/9 |




## License

MIT License. See [LICENSE](LICENSE).

## Contact

Zihan Wang — zihan.wang@queens.ox.ac.uk




