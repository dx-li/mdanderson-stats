# BARD saved-study input audit

The cached BARD user guide advertises saved study setup and scenario input.
`research/raw/BARD/Guide.txt` lines 13–17 says users can enter or upload the
trial settings and save them with “Save Input”; lines 201–205 says the current
inputs can be saved after generating the BF-BOIN decision table. For simulation,
lines 224–240 describes entering or uploading dose-wise true DLT and
population-response probabilities, saving scenarios, and uploading a scenario
CSV template. Those passages establish the user workflow, but do not provide
the native serialized format or its complete field contract in the recovered
text.

`BARDStudySpecification` provides a strict versioned Python schema for one
named simulation scenario. It captures the exact arguments consumed by
`simulate_bard_bf_boin`, including the primitive inputs used to reconstruct the
categorical response model, both true OBD labels, all BF-BOIN and stage-two
settings, calendar/titration/grade-2/joint-outcome options, a caller-visible
64-bit seed, and the work bound. Its `run` method delegates to the established
simulator. It does not repeat the simulator's statistical or calendar logic.

The stored model inputs deliberately retain the population response margins,
factor profiles, joint profile probabilities, and conditional odds ratios.
These are the actual inputs to the Python calibration factory; fitted
intercepts and full repeated-trial ledgers are not duplicated in a scenario
file. The JSON schema is limited to 1 MiB and 100,000 serialized numeric cells.
The work bound follows the simulator's own conservative per-trial allowance
and can be summed across independently seeded named scenarios before running
any of them.

The community JSON schema is portable and intentionally strict, but it is not
claimed to match the native Save Input JSON or Save Scenarios CSV template.
The cached guide describes upload/download actions without defining a text
schema in the recovered copy; inventing a native parser would therefore be
unsupported. The new community interface records actual Python settings and
replays them without claiming native defaults or random-stream parity.
