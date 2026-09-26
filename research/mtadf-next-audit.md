# MTADF implementation target

Historical source audit before implementation. The isotonic core and serial
simulation are now integrated; see mtadf-audit.md for results and remaining scope.

Catalog entry 114 remains pending. Its current application was read on
2026-09-26:
https://biostatistics.mdanderson.org/shinyapps/MTADF/

The page identifies version 1.0.5.0, updated 2026-01-06. It explicitly states
that its calculations use double-sided isotonic regression, and cites Zang,
Lee and Yuan (2014), *Adaptive Designs for Identifying Optimal Biological Dose
for Molecularly Targeted Agents*. Do not assume the app implements every
logistic variant in the paper.

The authors' institutional code index provides both references:
https://odin.mdacc.tmc.edu/~yyuan/index_code.html

- Paper: https://odin.mdacc.tmc.edu/~yyuan/Software/TargetAgent/OBD_rev.pdf
- R reference: https://odin.mdacc.tmc.edu/~yyuan/Software/TargetAgent/targetAgentDF.r

The paper's isotonic section enumerates candidate peak locations, fits an
increasing segment and a decreasing segment, and selects a fit by squared
error. Toxicity determines an admissible set. Interim movement is one dose at
a time, with an exploration rule at the highest tried dose; final selection
favors the lowest dose attaining maximum estimated efficacy among admissible
tried doses. The paper supplies an unequal-count PAVA example.

Before implementation, reconcile the paper and reference program: their tie
directions appear to differ, the program uses `Iso::ufit`, and its safety
helper appears to retain at least the lowest dose. Establish explicit behavior
for all-unsafe doses, untried levels, plateau ties and the weighting used in
peak fitting. The web text extraction truncates R expressions containing
less-than signs; inspect the raw file rather than reconstructing code from
that rendering. Read-only source is reference material, not permission to
redistribute the original code.

A bounded direct download of the R reference was attempted, but the local
shell could not resolve the hostname under its network restrictions. No raw
file was saved and no permission escalation or retry was attempted. The web
reader can access the paper and rendered source; use the paper for an
independent implementation until full raw reference access is available.

Paper safety formula to preserve: pool posterior overdose probabilities from
independent Beta(a+x, b+n-x) models, then admit only values strictly below CT.
Prior elicitation uses a+b=0.5 and BetaCDF(phi; a,b)=1-CT+delta. The reference
program's fixed 0.3/0.22 calibration and inclusive cutoff need separate treatment.

Existing reusable primitives include weighted PAVA in
`keyboard_combination._pava` and unweighted pooling in `merit._isotonic`.
Repository search found no standalone double-sided/unimodal fit. Check `bard`
and the existing OBD APIs before introducing another safety implementation.
Delegate implementation to one Luna child, preserving serial bounded work.
Use a small independent isotonic/decision reference, then add serial trial
simulation; avoid a large simulation replication or CI expansion.
