# CiBolus truth interpolation reference audit

This audit records an independent reference for the endpoint-interpolation
workflow in Thall et al., “Optimizing the Concentration and Bolus of a Drug
Delivered by Continuous Infusion,” *Biometrics* 67 (2011), 1638–1646,
doi:10.1111/j.1541-0420.2011.01580.x. The cached source used here is
`research/raw/CiBolus/paper.pdf` and its extracted text
`research/raw/CiBolus/paper.txt`.

The paper defines response-time probability from cumulative response
probabilities at zero and at standardized time one. Its interpolation
families are linear, below-linear (`s²`), above-linear (`sqrt(s)`), and
S-shaped: `(2s)²/2` for `s <= 1/2`, then `1/2 + sqrt(2s-1)/2`. Given the
interpolated response CDF `FE`, the response categories are the bolus atom
`FE(0)`, interval masses `FE(e[k])-FE(e[k-1])` over `(e[k-1],e[k]]`, and
failure mass `1-FE(1)`. Eq. (8) assigns each interval's conditional toxicity
at that interval's right endpoint; bolus toxicity uses the zero endpoint and
failure toxicity is separately specified. Each category's two joint cells
are its response mass times `(1-pT, pT)`. Expected regimen utility is the sum
of these joint cells times the corresponding utility table entries (Eq. 9).

The generator `tools/reference_cibolus_scenarios.R` implements these formulas
directly in base R and emits small synthetic, irregular-grid fixtures for all
16 ordered pairs of response/toxicity curve families. They exercise positive
concentrations, bolus fractions 0, an interior value, and 1, and include an
endpoint at the S-curve join `s=0.5`. The endpoint-probability arrays are
explicit test inputs, not inferred clinical scenarios. The paper's reported
scenario summaries do not fully specify all scenario endpoint probabilities,
so these fixtures do not claim to reproduce its six simulation scenarios.

The independent comparisons cover every joint cell, reconstructed response
marginals, category boundaries (bolus, open-left/closed-right intervals, and
failure), and expected utility for every regimen. `tests/test_cibolus_scenario_reference.py`
compares Python outputs against the checked-in base-R CSVs with zero relative
tolerance and tight absolute tolerances. It also checks that returned arrays
are read-only as required by the Python API; this ownership property is not a
claim about the paper's implementation.

Reference artifacts:

- `tools/reference_cibolus_scenarios.R`
- `tests/fixtures/cibolus-scenario-settings.csv`
- `tests/fixtures/cibolus-scenario-cells.csv`
- `tests/fixtures/cibolus-scenario-marginals.csv`
- `tests/fixtures/cibolus-scenario-utility.csv`
- `tests/fixtures/cibolus-scenario-expected-utility.csv`
- `tests/test_cibolus_scenario_reference.py`
