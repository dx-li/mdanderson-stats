# Independent KeyboardComb movement-variant reference

The base-R script `tools/reference_keyboard_combination_variants.R` generates
three compact fixtures for the published key2, key3, and key4 movement
algorithms. It uses only the equations in the cached primary paper,
`research/raw/KeyboardComb/keyboard-combination-paper.txt`:

- Section 2 (printed pp. 4–5) states the local model
  `p | (n,y) ~ Beta(y+1,n-y+1)` under a uniform prior and defines the target
  key by posterior probability mass.
- Section 3 (printed pp. 9–10) defines axial sets AE1/AD1 and diagonal-enabled
  AE2/AD2. Key2 uses AE1 for escalation and AD2 for de-escalation; key3 uses
  AE2 and AD2; key4 uses AE1 and AD1. Fixed variants choose a maximum-mass
  candidate and split exact ties uniformly. Randomized key4 allocates in
  proportion to the candidate posterior target-key masses.

Each candidate mass is calculated independently as
`pbeta(target + margin_right, y + 1, n - y + 1) -
pbeta(target - margin_left, y + 1, n - y + 1)`. The reference deliberately
does not apply the key1-only Jeffreys-prior-plus-`0.0005*n` scoring rule.
Candidates outside the 3×3 grid and candidates marked excluded in the
configuration are omitted before choosing or normalizing. The fixture set
includes an interior escalation where a diagonal candidate changes the
key3 choice, an interior de-escalation where the diagonal changes the choice
relative to key4, a nonuniform key4 axial allocation, one masked candidate, a
single in-grid boundary candidate, and an exact uniform tie.

The generator writes three CSVs: `keyboard-combination-variants-config.csv`
records row-major patient/toxicity/exclusion matrices and the movement
direction; `...-masses.csv` contains posterior shapes, raw key masses, and
selection probabilities for eligible candidates; `...-decisions.csv` records
the ordered candidate set and the candidates with nonzero allocation
probability. Invoke it from the repository root with:

```sh
Rscript tools/reference_keyboard_combination_variants.R tests/fixtures
```

The fixture reference takes the movement direction as an input. It validates
the distinct candidate-set and allocation rules, not the upstream current-dose
strongest-key cutoff, overdose-control behavior, trial simulation, or native
random-number stream. Python's selection among the reported randomization
probabilities is validated separately by deterministic seeds in focused tests.
