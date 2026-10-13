# WFMM remaining contracts: evidence boundary

## Numerical file/report interchange

The guide documents headerless doubles with coordinates fastest, components
next and samples last. Three pinned WFMM1/2/3 cases independently supply doubles
with Python's struct library, reconstruct with native WFMM3, and verify Haar
synthesis, report means and sample SDs. One/two effects and one/two chains are
covered. The fixture regenerates byte-identically. Production codecs require
exact dimensions and explicit endian profiles. Scientific MAT report exports
use selected native-style field names with explicit non-identity metadata.
Neither a headerless codec nor report fields imply full native Init-file import.
Version-1 Python saved studies already preserve the implemented workflow.

## Imputed inputs

Twelve probes in wfmm-imputed-audit-2026-10-12.json regenerate identically.
With missing_data=0, the original reads Y and transforms it. With missing_data=1,
ReadInput reads Vstar and Mstar; complete processed basis metadata is required.
At ANOVA_MOM entry, Mstar is copied exactly into the coefficient data argument
for the three complete-basis probes. It is not an observed/missing indicator mask.
Native MAT numeric scalar fields must be supplied as doubles: rewriting uint8
metadata as-is produced unrelated invalid-dimension errors, excluded from the
accepted probes.

The tested complete imputed inputs still fail full initialization with an
out-of-range column error, even for Vstar constants 0, 0.1 and 1. These are
qualified failures of a simple no-random-effect design, not proof that all native
imputed inputs fail. Vstar's statistical meaning, required covariance structure
and updates are not specified by the guide's single flag description. No
arbitrary-NaN imputation or guessed likelihood is implemented. A passing native
example or public equations specifying this input law are genuine requirements.
Native WFMM1 ReadInput at 0x4058a0 and ReadImputedData at 0x4267b0 corroborate
selection; only numerical observations/error messages are retained, not binary
or disassembly bytes.

## Interval and reduced boundaries

The earlier 28-case interval audit remains unresolved. Thirty-six additional
reduced-mode cases cover db1/db2/db3, lengths 32/33, levels 1/2 and periodic,
zero/reflection boundaries. All reach a native inverse assertion failure; no
successful reduced workflow is inferred. The independently supported Python
orthogonal periodic API and extended transforms retain their declared contracts.
Failures here are not a mathematical rejection of reduced or interval wavelets.

Public recovery reconfirmed the CDV 1993 publication record and publisher entry;
the HAL document still returned access denial. Jameson's public NASA follow-up
separates sampled values from scaling coefficients and does not by itself supply
XWaveLib's discrete conventions. An arbitrary replacement boundary basis would
be a different model. A verified discrete convention or independently specified
new scientific scope is still needed. No author contact was made.

Sources: official WFMM guide and pinned archive already recorded in previous
audits; https://hal.science/hal-01311753/document (access denial);
https://scholars.duke.edu/publication/767482;
https://ntrs.nasa.gov/api/citations/19940019818/downloads/19940019818.pdf.
The older official archive https://bioinformatics.mdanderson.org/public-software/archive/wfmm/
is version 1.0, not evidence that its omitted fields resolve version 3.1 gates.

WFMM remains partial. Exact RNG identity and clinical validity are separate
claims, and no missing prediction-file interface has been invented as a gate.
After acceptance of supported interchange, the next public-paper recovery target
is BFMonitor143's documented ESS-to-iMOM-shape calibration gap.

LOCAL ONLY: WFMM numerical file interchange accepted with 77,187 full tests / 496.2s, 1,055 focused and 1,083 installed-wheel checks; six examples. New implementation/tests are byte-identical from the pre-test snapshot through validation, not unchanged from the preceding commit. Both artifacts match 790 package files and 29 retained notices, with originals excluded. See wfmm-files-validation-2026-10-12.json and wfmm-remaining-gates-2026-10-12.md. WFMM stays partial at a qualified interval/reduced/imputed-likelihood evidence boundary. BFMonitor143 calibration recovery still lacks equations; work advances to CondiS157 saved workflows. No remote authentication request or push.
