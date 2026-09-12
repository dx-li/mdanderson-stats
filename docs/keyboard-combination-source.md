# Keyboard combination reference audit

This audit records the public sources used to build the bounded combination
fixtures in `tests/fixtures/keyboard-combination-*`.  The fixtures are generated
by [`tools/reference_keyboard_combination.R`](../tools/reference_keyboard_combination.R)
with the CRAN `Keyboard` package 0.1.3 in a task-local R library.  The downloaded
archive and paper text remain under ignored `research/raw/`; original package
source is not copied into the repository.

## Pinned sources

| source | observed version/date | SHA-256 |
| --- | --- | --- |
| [KeyboardComb app](https://biostatistics.mdanderson.org/shinyapps/KeyboardComb/) | V1.2.3.0, last updated 2025-12-15 | `888344ac52958af3a3d669898f43187a3a6e7df24da12379272aaf2d77fcce` |
| [KeyboardComb Guide.pdf](https://biostatistics.mdanderson.org/shinyapps/KeyboardComb/Guide.pdf) | app guide | `4843e015dc45cf7cfbe2c03039ffc1c010f16c78d36a293d9f1a40a8c6d2513c` |
| [KeyboardComb Doses.pdf](https://biostatistics.mdanderson.org/shinyapps/KeyboardComb/Doses.pdf) | app parameter help | `6c579666ebb55aadbd471214acdf0a822e627fa5fd40680bbe78432600ba197f` |
| [KeyboardComb Probability.pdf](https://biostatistics.mdanderson.org/shinyapps/KeyboardComb/Probability.pdf) | app parameter help | `32301b9cad97d1135ecb9c997e3dfd7d959e502c502bd0a7939f07c397beb089` |
| [KeyboardComb Safety.pdf](https://biostatistics.mdanderson.org/shinyapps/KeyboardComb/Safety.pdf) | app parameter help | `1eef79de40991a06c86a056b70b30bb664ead4410f9f922183eaed1e0b573521` |
| [Keyboard 0.1.3 source archive](https://cran.r-project.org/src/contrib/Keyboard_0.1.3.tar.gz) | packaged 2022-08-10, published 2022-08-11 | `909775234ead042c707f8af78a030ed718ab90beb8ab22a90decd90f72b0ccf9` |
| [Pan, Lin, Zhou & Yuan (2020) paper](https://arxiv.org/abs/1712.06718) | *Statistical Properties of the Keyboard Design with Extension to Drug-Combination Trials* | `c352121f6eeb3d95c6d6111ee609ff4ead13c2c34d2862a90d2fa7d69f8bc8c2` |

The CRAN archive declares `License: GPL-2` in `DESCRIPTION` and imports `Iso`
and `ggplot2`.  This work uses the package as an independent executable
reference; it does not redistribute its source.  `Iso` 0.0-21 declares `GPL
(>= 2)`.  The app page names Yanhong Zhou, Haitao Pan, Ruitao Lin, and Ying
Yuan and links to both the 2017 single-agent paper and the 2020 combination
paper.

## Algorithm findings

The Pan et al. paper describes five possible combination movement algorithms.
The CRAN package implements the fixed non-diagonal algorithm (key1): from
`(j,k)`, escalation candidates are `(j+1,k)` and `(j,k+1)` and de-escalation
candidates are `(j-1,k)` and `(j,k-1)`.  Diagonal candidates are never used.
Out-of-matrix candidates are assigned posterior mass zero.  For each eligible
candidate, the package evaluates

```text
Pr(target - marginL < p_jk < target + marginR | y_jk + 0.5, n_jk - y_jk + 0.5)
  + 0.0005 * n_jk
```

and moves to the candidate with the largest value.  The `0.0005*n` term favors
a previously treated candidate when posterior masses tie.  Exact remaining
ties are sampled uniformly with `runif`; `next.comb.kb` calls `set.seed(1)` at
the beginning of every invocation, so its random tie is repeatable per call.
The first movement fixture captures the resulting equal-mass escalation tie.

The boundary generator uses a uniform Beta(1,1) prior and full-width keys,
with endpoint keys compensated by `(marginL + marginR) / endpoint_width`.  The
strongest-key tie is resolved to the highest key.  Elimination requires at
least three treated patients and, in the package code, uses the strict test
`1 - pbeta(target, y + 1, n - y + 1) > cutoff.eli`.  The generated boundary
fixture has 18 patients (six cohorts of three) and includes the extra-safe
lowest-dose stopping cutoff.

During `next.comb.kb`, every treated cell satisfying the elimination cutoff
marks an upper-right rectangle `i:nrow, j:ncol` as unavailable.  If `(1,1)` is
overly toxic, the function returns `NA, NA` and emits a stop warning.  The
extra-safe check applies only at `(1,1)`, after at least three patients, with
the cutoff lowered by `offset`.  The package also has a reproducible defect:
the internal call to `get.boundary.comb.kb` passes `marginL`, `marginR`, and
`cutoff.eli` positionally after named arguments.  For small `n`, this can leave
the de-escalation cutoff `NA`, producing `missing value where TRUE/FALSE was
needed`; this is retained in the movement fixture and should not be mistaken
for the intended statistical rule.

For final selection, `select.mtd.comb.kb` first applies safety.  Its elimination
closure is a cross: for each offending `(i,j)`, it marks `i:nrow,j` and
`i,j:ncol`, and breaks the inner column loop.  If the lowest cell is marked,
it returns `MTD = 99` and an all-`NA` estimate matrix.  Otherwise it computes

```text
raw = (y + 0.05) / (n + 0.1)
isotonic = Iso::biviso(raw, n + 0.1)
```

Untreated estimates are reported as `NA`; reported isotonic estimates are
rounded to two decimals.  Eliminated cells are assigned 1.1 for selection,
untreated cells 10, and a `1e-5 * (row + column)` perturbation is added before
minimizing absolute distance to the target.  Thus the package's final tie
selection is generally deterministic despite its documentation saying ties are
randomized.  The equal-matrix fixture records the observed first-cell result;
the all-untreated fixture records the package's `(1,1)` result.

`get.oc.comb.kb` fixes the R RNG seed to 6, samples each complete cohort as
`sum(runif(cohortsize) < p.true[current])`, applies safety after each cohort,
and records mean patient/toxicity counts plus selection percentages.  Its
boundary construction hardcodes `cutoff.eli = 0.95` and does not forward
custom margins; movement posterior masses use the supplied margins.  The small
2-by-3, 20-trial fixture exercises these rules without embedding a large Monte
Carlo output.

The paper's simulation study uses a separate random scenario generator.  It
chooses a pivotal cell uniformly from the dose matrix and sets its true rate to
the target, constructs a monotone path from `(1,1)` to `(J,K)`, samples rates
below and above the target along that path, then fills the upper and lower
blocks with uniforms bounded by neighboring path values.  The upper bound is
`pmax = 1 - exp(-(J*K)/8)` in the worked example.  This generator is a paper
simulation device; the CRAN `get.oc.comb.kb` function accepts a supplied
`p.true` matrix and does not generate scenarios itself.

## Fixture contract

* `keyboard-combination-boundaries.csv`: one row per treated-patient count,
  including escalation, de-escalation, elimination, and extra-safe cutoffs.
* `keyboard-combination-movements.csv`: package calls for escalation,
  retention, de-escalation, safety, precision-stop, eliminated-neighbor, and
  the observed low-count boundary error cases.  Matrices are row-major,
  semicolon-delimited integer strings.
* `keyboard-combination-selection.csv`: final MTD and rounded isotonic matrix
  for documented, safety, extra-safe, tie, and untreated cases.  `NA` is a
  literal missing-value token in the row-major matrix string.
* `keyboard-combination-simulation.csv` and
  `keyboard-combination-simulation-summary.csv`: bounded operating
  characteristics for the deterministic 2-by-3 scenario.
