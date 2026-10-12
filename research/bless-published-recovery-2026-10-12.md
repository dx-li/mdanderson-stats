# BLESS published-model recovery (parent research handoff)

The parent recovered the actual47-page mmc1.pdf supplement to the
[primary BLESS paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC8449006/).
e-Appendix1 supplies the reference-centered Cox prediction equation; e-Tables1–8
supply all four published coefficient tables and baseline survivors at days
0/30/60/90/120/150/180. The artifact, hash and structured extraction are awaiting
transfer and independent executor inspection. No coefficients have yet been
transcribed into package code and no status promotion is claimed.

This is potentially sufficient for a useful published-precision grid model;
absence of a full-precision deployed fit does not justify leaving these published
methods unimplemented. Exact arbitrary-day deployed parity is a separate claim.
Do not invent interpolation or replace unknown daily baselines silently.

Source inconsistencies reported by the parent must remain explicit:

- Lung prose centers LDH/100 at7.31 and protein at3.8, while the equation and
  e-Tables1–2 use7.19 and3.7. The latter reproduce example LP2.99852.
- Breast equation/table NLR centering is5.398; the baseline footnote says5.389.
- Hematologic rounded coefficients yield LP−0.502 versus printed−0.49662.
- GCMM's example omits laterality but requires bilateral1 to yield LP1.52905.

Next: inspect and hash the actual supplement, preserve a structured source-backed
specification and explicit ambiguity choices, implement bounded model-specific
inputs and source-grid predictions, independently verify all published examples
and limits, then assess partial versus complete declared workflow status. No
outreach, licensing changes or fitted-parameter guesses are authorized.
