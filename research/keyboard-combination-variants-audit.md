# Keyboard combination movement variants audit

## Primary-source rules

Pan, Lin and Yuan, *Keyboard design for phase I drug-combination trials*,
Contemporary Clinical Trials 92 (2020), Section 3, pp. 9–11, defines five
candidate movement algorithms. The cached primary manuscript is
`research/raw/KeyboardComb/keyboard-combination-paper.pdf` (text extraction:
`research/raw/KeyboardComb/keyboard-combination-paper.txt`); its source record
and SHA-256 metadata are in `docs/keyboard-combination-source.md`.

For a current dose pair `(j,k)`, the paper defines
`AE1={(j+1,k),(j,k+1)}`, `AD1={(j-1,k),(j,k-1)}`,
`AE2=AE1 union {(j+1,k+1)}`, and
`AD2=AD1 union {(j-1,k-1)}`. Candidate values are the posterior probability
that that candidate's toxicity lies in the target interval. Section 2 uses a
uniform Beta(1,1) prior for these binary toxicity probabilities.

The implemented options are:

| option | escalation candidates | de-escalation candidates | rule |
| --- | --- | --- | --- |
| `key1` | AE1 | AD1 | Existing audited CRAN behavior, unchanged by this addition. |
| `key2` | AE1 | AD2 | Choose the largest raw Beta(1,1) target-key mass; uniform tie among maxima. |
| `key3` | AE2 | AD2 | Choose the largest raw Beta(1,1) target-key mass; uniform tie among maxima. |
| `key4` | AE1 | AD1 | Randomize proportional to raw Beta(1,1) target-key masses. |

The default `key1` remains the package implementation's Beta(.5,.5) neighbor
score plus `.0005 * n` adjustment, with its existing tie-resolution and random
number calls. The new paper variants do not reuse or relabel that adjusted
score. Their masses are computed as a stable difference of Beta tails and key4
normalizes after dividing by the largest mass. An all-zero or non-finite
normalization raises rather than inventing equal weights.

In all options, this Python port filters out-of-grid and eliminated candidates,
retains the established conduct safety and early-stop precedence, and leaves
final matrix-isotonic MTD selection unchanged. Those shared edge conventions
are documented Python/package choices and do not establish complete trial or
native-app parity. `KeyboardCombDesign.movement_algorithm` drives both direct
`next_dose` calls and the existing `simulate_keyboard_combination` workflow.
Decision results expose the eligible candidate order and their transition
probabilities for key2-key4, allowing callers to audit randomized weights.

## Unimplemented source contracts

Key5 is deliberately omitted. The paper defines AE2 and AD2 with three
elements, but its key5 paragraph says to randomize among “any one of two”
components in each set. The cached source does not resolve whether diagonal
members are included in that randomized choice.

The paper's Section 4.1 random-matrix scenario generator is also not reproduced
as a source-faithful utility. It specifies `pmax` as a Beta-distributed bound
with mean `mu` and variance `mu*(1-mu)`, which is not attainable by any proper
Beta distribution. Choosing a concentration or a different distribution would
add an unsupported assumption. Supplied-scenario simulation remains available
through `simulate_keyboard_combination`.

## Validation scope

Focused tests distinguish diagonal-allowed and diagonal-prohibited movements,
check key4's normalized weights against direct Beta CDF differences, exercise
candidate exclusion and matrix boundaries, and compare implicit and explicit
key1 seeded decisions. Five independent base-R fixture configurations compare
candidate order, raw posterior masses and transition probabilities for all
three variants. In the interior escalation reference, key2 selects axial
`(2,1)` while key3 selects diagonal `(2,2)`; key4 assigns the axial candidates
probabilities `0.742104777634259` and `0.257895222365741`. In the interior
de-escalation reference, key2/key3 select diagonal `(1,1)`, while key4 excludes
that diagonal and gives the two axial predecessors probability `0.5` each.
Masked-candidate, one-neighbor boundary and exact uniform-tie cases also match.
The independent generator and fixture record are
`tools/reference_keyboard_combination_variants.R` and
`research/keyboard-combination-variants-reference.md`. Broader integration and
native RNG parity are outside this change.
