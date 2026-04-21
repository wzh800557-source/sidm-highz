# SIDM constraints from reionization topology and the JWST UV luminosity function

Code and data accompanying:

**"Complementary constraints on self-interacting dark matter from reionization topology and the z > 9 UV luminosity function"**
Wang & Shan (2026)

## Summary

Self-interacting dark matter (SIDM) core formation reduces the gas binding energy in high-redshift halos, producing two observable effects through independent physical channels:

1. **Increased duty cycle** of ionizing-photon escape → enhanced 21 cm reionization topology
2. **Decreased star formation efficiency** → suppressed UV luminosity function at z > 9

We show that the UVLF alone **cannot** constrain SIDM (astrophysical parameters absorb the signal in a profile likelihood), but the topology provides an independent constraint that is immune to this degeneracy. Together, the two probes constrain SIDM across the full (σ/m, η) parameter space.

## Repository structure

```
sidm_uvlf_code/
├── analysis/
│   ├── uvlf_sidm.py          # Core UVLF model: HMF, SFE, SIDM modification, likelihood
│   ├── generate_figures.py    # Reproduce all 7 paper figures
│   └── joint_constraint.py    # Joint UVLF + topology analysis
├── cluster/
│   ├── run_profile_scan.py    # Self-contained profile likelihood scan (252 grid points)
│   ├── submit_profile.sh      # SLURM submission script (MIT Engaging)
│   └── quick_test.sh          # Reduced grid for local testing
├── data/
│   ├── profile_scan_results.json   # Full scan output (18×14 grid)
│   └── uvlf_data_verified.json    # 31 JWST data points (Donnan+2024, Harikane+2024)
├── figures/
│   └── fig[1-7]_*.pdf         # Publication figures
├── docs/
│   └── eta_calibration.md     # η calibration literature survey
├── README.md
├── LICENSE
└── requirements.txt
```

## Quick start

```bash
# Install dependencies
pip install numpy scipy matplotlib

# Run the CDM baseline fit
python analysis/uvlf_sidm.py

# Run a quick profile scan (35 points, ~20 min)
cd cluster && bash quick_test.sh

# Generate all figures from existing results
python analysis/generate_figures.py
```

## Key results

| η range | UVLF+SMF limit | Topology (SKA 10% floor) | Joint |
|---------|---------------|-------------------------|-------|
| ≥ 0.25  | σ/m < 0.3     | σ/m > 2.3 detectable    | Both reject σ/m ≥ 2.3 |
| 0.07–0.25 | σ/m < 0.3–2.4 | σ/m > 2.3 detectable | SKA target window |
| ≲ 0.07  | None          | σ/m > 2.3 detectable    | Topology only |

## Data sources

- **UVLF:** Donnan et al. (2024, arXiv:2403.03171) Table 2; Harikane et al. (2024, arXiv:2406.18352) Table 2
- **SMF prior:** Weibel et al. (2024), JWST PRIMER
- **Topology:** Wang & Vogelsberger (2026, arXiv:2604.10726), Paper I

## Requirements

- Python ≥ 3.8
- numpy ≥ 1.20
- scipy ≥ 1.7
- matplotlib ≥ 3.5



## License

MIT License. See LICENSE file.
