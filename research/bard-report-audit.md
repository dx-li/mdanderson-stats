# BARD saved-study report source and scope audit

The cached BARD guide describes entering dose-specific true DLT and population
response probabilities, either manually or through a scenario upload (Guide,
Section 2(a), printed pages 12–13). It then asks for prognostic-factor level
distributions and response odds ratios relative to level 1 (Remarks 1, printed
pages 12–13), and says simulation output appears in two operating-characteristic
tables (Section 2(b), printed page 14). After simulation, the app offers HTML
and Word protocol templates (Section 3, printed page 15). The guide text does
not establish the native input-file schema or the full native report-column
formulae.

The Python saved specification captures the dose truth, explicit categorical
profile model, joint endpoint table if provided, BF-BOIN stage-one settings,
stage-two eligibility/prior/cutoff/utility/allocation settings, separate
caller-provided noninferiority and utility true-OBD labels, timing controls,
simulation count, and seed. The HTML includes that versioned JSON verbatim in
an escaped details section, alongside readable scalar settings and the
stage-one boundary table.

The OC quantities follow the streaming simulator's defined summaries:
all-trial dose selection/no-selection probabilities and correct-OBD
probabilities, selected-only correct-OBD accuracy with its own denominator,
total enrollment and duration moments, absolute two-arm count gap, and
per-factor absolute level-1 proportion gaps. The report keeps no-pair,
safety-rejected-pair, and empty-arm cases out of undefined allocation metrics
and displays the corresponding denominators. It does not infer either true OBD
from a utility curve or from a table position.

## Python conventions and limits

Scenario seeds initialize independent serial streams, so one scenario's result
does not depend on its position in the report request. This is a reproducible
Python convention, not a claim about native app stream ordering. Scenario
schemas and HTML are bounded; the aggregate patient-work limit and all input
validation are checked before the first simulation starts.

The report documents that conditional independence is used when no joint
toxicity-response table is supplied, while an explicit table is validated
against marginal Frechet bounds. Timing laws and stage-two scheduling are the
Python settings captured by each specification; the guide does not define
complete native timing or quota rules. The report does not claim native file
format, defaults, or presentation parity. It covers BF-BOIN OCs only; the
paper's separate stochastic BF-BLRM operating-characteristic experiment is
not simulated here.
