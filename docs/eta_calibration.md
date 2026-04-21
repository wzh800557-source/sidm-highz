# η Calibration from SIDM Simulations: Literature Survey

## Summary for Paper Discussion Section

The coupling parameter η — which sets how strongly SIDM core formation
suppresses star formation — is the least constrained parameter in our
model. No existing simulation directly measures η at z > 6. This
document compiles all available simulation results and translates them
into approximate η values to establish physically motivated priors.

**Bottom line:** z = 0 simulations consistently find η ≲ 0.1 for
dwarf galaxies and η ≈ 0 (or negative) for Milky Way-mass systems.
But these results may not apply at z > 6 where halos are less
concentrated and baryonic potential wells are shallower. η at high
redshift is genuinely uncalibrated and could plausibly range from
0 to ~0.3.

---

## Compilation Table

| Reference | Code/Model | M_halo | σ/m | z | ΔM★/M★ | Implied η | Notes |
|-----------|-----------|--------|-----|---|--------|-----------|-------|
| Robles+2017 | FIRE-2 | 10^10 M☉ | 1 | 0 | ~0% ("very similar") | ≲ 0.05 | 4 dwarfs; M★ = 10^5.7–7.0; density profiles differ but stellar masses don't |
| Vogelsberger+2014 | AREPO/Illustris | ~10^10 M☉ | 1, 10 | 0 | ~0% | ≲ 0.05 | 2 dwarfs; M★ ~ 10^8; first hydro+SIDM simulations |
| Fry+2015 | GASOLINE | ~10^10 M☉ | 0.5, 5 | 0 | ~0% | ≲ 0.05 | M★ ~ 10^8; consistent with Vogelsberger+2014 |
| TangoSIDM (Correa+2024) | SWIFT-EAGLE | 10^10–10^12 M☉ | v-dep | 0 | ~0% (global) | ~0 | 25 Mpc box; "does not modify global galaxy properties such as stellar masses and SFRs" |
| Sameie+2021 | FIRE-2 | ~10^12 M☉ | 1, 10 | 0 | HIGHER M★ | η < 0 | MW-mass; baryon contraction REVERSES core formation; SIDM halos denser than CDM |
| Gutcke+2025 | AREPO | LG dwarf | v-dep | 0 | −25% | ~0.10–0.15 | Most direct ΔM★ measurement; "prolonged quiescent phase in star formation" |
| Shen+2022 | FIRE-2 (dSIDM) | 10^10 M☉ | dissipative | 0 | ~0% | ~0 | Dissipative SIDM; stellar masses "do not show appreciable difference" |

---

## Detailed Notes

### Robles et al. (2017) — FIRE-2, dwarfs, σ/m = 1

4 cosmological zoom-in simulations of isolated dwarf galaxies in
M_halo = 10^10 M☉ halos. Stellar masses in SIDM with σ/m = 1
"very similar" to CDM, spanning M★ ≈ 10^5.7–7.0 M☉. SIDM produces
flat cores (α > −0.4) while CDM produces cusps (α < −0.8) in the
DM density, but this does NOT translate into a measurable stellar
mass difference. At M_halo = 10^10, σ/m = 1, our model gives
Δ_bind ~ 0.26. If ΔM★ < 10%, then η < 0.1/0.26 ≈ 0.4 (weak limit).
If ΔM★ ~ 0% (as stated), η ≲ 0.05.

### Sameie et al. (2021) — MW-mass, baryon contraction

3 MW-mass halos (M_halo ~ 10^12 M☉), FIRE-2. SIDM galaxies have
HIGHER SFR at z ≤ 1, producing MORE massive galaxies. Mechanism:
in massive halos where baryons dominate the central potential, SIDM
thermalization helps DM contract more efficiently → deeper potential
→ more star formation. This OPPOSITE sign means η < 0 at MW masses.
Our UVLF constraint targets M_halo ~ 10^9–10^11, below this threshold,
so the reversal likely doesn't apply. But it shows η is mass-dependent.

### Gutcke et al. (2025) — most direct measurement

Local Group dwarf analogue, v-dependent σ(v). "SIDM model leads to a
25% reduction in stellar mass and retains more gas within the stellar
half-mass radius due to a prolonged quiescent phase in star formation."
This is the strongest evidence that SIDM CAN suppress star formation.
For v-dep σ/m (effective σ/m ~ 10–50 at dwarf velocities), Δ_bind ~ 0.8–0.95,
giving η ~ 0.25/0.9 ~ 0.10–0.15.

### TangoSIDM (Correa et al. 2024) — cosmological volume

First 25 Mpc cosmological hydro volume with SIDM. "SIDM does not modify
global galaxy properties such as stellar masses and star formation rates,
it does make the galaxies more extended." But the Tully-Fisher relation
IS affected in MW-mass halos — SIDM enhances central DM density via
thermalization, increasing circular velocity. This rules out some
v-dependent models.

---

## Conversion to η

Our model: f★_SIDM = f★_CDM × (1 − η × Δ_bind)

Therefore: η = (1 − M★_SIDM/M★_CDM) / Δ_bind

| Source | ΔM★/M★ | σ/m | Δ_bind | Implied η |
|--------|--------|-----|--------|-----------|
| Robles+2017 | ~0% | 1 | 0.26 | ≲ 0.05 |
| Vogelsberger+2014 | ~0% | 1 | 0.26 | ≲ 0.05 |
| Vogelsberger+2014 | ~0% | 10 | 0.90 | ≲ 0.02 |
| TangoSIDM 2024 | ~0% | v-dep | ~0.3–0.5 | ≲ 0.03 |
| Gutcke+2025 | −25% | v-dep (~10–50) | ~0.8–0.95 | ~0.10–0.15 |
| Sameie+2021 (MW) | +positive | 1 | ~0.1 | < 0 (reversed) |

---

## Why z > 6 Could Be Different

All results above are z = 0 simulations. η could be LARGER at z > 6:

1. **Less concentrated halos.** c(M,z) decreases with z. At z = 7,
   c ~ 3–5 vs c ~ 10–15 at z = 0. Lower concentration → SIDM core
   removes a larger fraction of binding energy → larger Δ_bind.

2. **Shallower baryonic potential.** At z > 6, M★/M_halo is ~10× lower
   than z = 0. Baryon-contraction reversal (Sameie+2021) is less likely
   — the baryons haven't built up enough to dominate the center.

3. **Different feedback regime.** SN feedback prescriptions are calibrated
   to z = 0. At z > 6, whether CDM feedback can flatten cusps as
   efficiently is unclear. If CDM cores are weaker at high z, SIDM
   cores stand out more → effectively larger η.

4. **Higher gas fractions.** f_gas > 0.5 at z > 6. Gas responds more
   sensitively to reduced binding energy than a stellar population.

---

## Recommended Priors

| Range | η | Justification |
|-------|---|---------------|
| Conservative | 0–0.10 | All z = 0 results; η > 0.10 inconsistent with Robles+2017 at σ/m = 1 |
| Moderate | 0–0.25 | Allows high-z amplification; upper end consistent with Gutcke+2025 |
| Agnostic | 0–1.00 | For completeness; η = 1 implausible but shows constraint scaling |

**For the paper:** present results across all three ranges but emphasize
the moderate prior [0, 0.25]. State that η at z > 6 is uncalibrated.

---

## Impact on Paper Constraints

At the conservative η ≤ 0.10:
- UVLF+SMF can only exclude σ/m ≥ 2 (f★₀ = 0.024, marginal)
- The topology constraint becomes the dominant probe
- SKA target window is wide: σ/m ~ 1–10

At the moderate η = 0.25:
- UVLF+SMF excludes σ/m ≥ 0.5 (f★₀ > 0.026, exceeds 2σ SMF)
- Topology overlaps with UVLF exclusion at σ/m ≥ 1
- Constant-σ SIDM at σ/m ≥ 1 is jointly excluded/detectable

The paper should present BOTH scenarios and let the reader choose
based on their belief about high-z SIDM physics.

---

## Draft Text for §7 (Discussion)

"The parameter η remains the dominant theoretical uncertainty in our
analysis. Existing SIDM hydrodynamical simulations at z = 0 consistently
find that stellar masses in SIDM halos are very similar to CDM at dwarf
scales (Vogelsberger et al. 2014; Fry et al. 2015; Robles et al. 2017),
implying η ≲ 0.05. The TangoSIDM cosmological volume confirms this
for global properties across 10^10–10^12 M☉ (Correa et al. 2024).
The most direct measurement of stellar mass suppression comes from
Gutcke et al. (2025), who find a 25% reduction in a Local Group dwarf
with velocity-dependent SIDM, implying η ≈ 0.10–0.15.

At Milky Way masses, the picture reverses: Sameie et al. (2021) find
that baryon contraction in SIDM halos produces HIGHER central
densities and star formation rates than CDM, implying η < 0. This
mass-dependent behavior means η is not a universal constant — our
parametrization as a single value is necessarily approximate and
should be understood as an effective coupling averaged over the halo
mass range 10^9–10^11 M☉ relevant for the z > 9 UVLF.

Critically, all existing simulations target z = 0. At z > 6, several
factors could amplify the SIDM–SFE coupling: lower halo concentrations
increase the fractional binding energy reduction (§2.3); shallower
baryonic potentials make the baryon-contraction reversal less likely;
and higher gas fractions may render star formation more sensitive to
changes in the gravitational potential. We therefore present our
constraints for η ∈ [0, 0.25] as a physically motivated range,
noting that dedicated SIDM zoom-in simulations at z > 6 with
radiative transfer are needed to pin down η in the regime relevant
for our analysis."
