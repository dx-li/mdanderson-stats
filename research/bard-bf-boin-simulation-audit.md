# BARD BF-BOIN repeated-trial simulation audit

The cached primary text is `research/raw/BARD/paper.txt`; the user-interface
workflow is described in `research/raw/BARD/Guide.txt`. The paper's numerical
study specifies 30,000 trials and reports average total sample size, average
duration, per-factor imbalance, arm allocation imbalance, and correct OBD
selection by both the efficacy/noninferiority and utility methods (`paper.txt`
lines 797–823; main paper Section 3, printed pages 18–19).

The per-factor quantity is the absolute difference in the proportion of
patients with factor level 1 between the two arms. Its aggregation uses
stage-one and stage-two data together (`paper.txt` lines 541–567, 803–813).
The published study evaluates all three generated response factors, although
its stage-two minimization targets only the first two (`paper.txt` lines
763–775). Allocation imbalance is the absolute difference in arm counts
(`paper.txt` lines 815–817). The implementation computes these on the
combined, stage-eligible two-dose cohort. Count imbalance has a denominator
whenever a non-rejected pair is present; factor proportion imbalance requires
nonempty counts in both arms. Missing pair or factor metrics are excluded and
their counts are returned separately.

True OBD labels for the noninferiority and utility methods are separate
caller-supplied one-based dose indices. The boldface true OBD labels in the
cached table extraction are not reliably recoverable from plain text, and the
design's clinical dose pair and eligibility rules may be protocol decisions.
The simulator therefore does not infer either true OBD or pair from table
position. Dose-selection probabilities, no-selection rates, and unconditional
correct-selection probabilities use all trials; selected-only accuracy has a
separate denominator.

The repeated-trial driver calls the single-trial BF-BOIN/BARD orchestration
serially and updates fixed-size sufficient statistics. It retains no array of
all trial ledgers. Enrollment and duration means use Welford summaries; binary
event MCSEs use the binomial plug-in formula. This is an OC estimator, not a
claim that a bounded demonstration reproduces the paper's 30,000-trial values.

## Explicit Python conventions and remaining parity limits

The cached app guide asks for population toxicity and response probabilities
per dose (`Guide.txt` lines 224–240), while the primary paper provides a
covariate-dependent response model. The response model and profile law must be
provided explicitly. Neither the inspected paper nor guide supplies a joint
DLT-response generation law. The Python stage-one/stage-two hooks therefore
use conditional independence when no joint table is provided and validate a
caller-supplied joint table against its marginal Frechet bounds otherwise.
This is a disclosed simulation choice, not recovered native parity.

The primary paper describes integration of stage-one and stage-two data and a
total target including eligible stage-one carryover; the app asks users for
per-arm target counts. A universal hard-quota scheduling rule is not recovered.
The trial runner uses its documented inclusive total-target convention and
returns shortfall/status information.

The source does not establish stage-two arrival or assessment-time laws or
native random-number stream ordering. Python stage two begins after complete
stage-one follow-up, uses the configured renewal arrival law, and uses the
shared BF-BOIN Weibull DLT-window calibration. These implementation settings
are recorded separately from the paper-defined OC estimands. No native file
format or printed-column parity is claimed.
