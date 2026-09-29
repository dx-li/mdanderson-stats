# Randomized paired BOP2-DC source mapping

Primary cached source: `research/raw/BOP2-DC/paper.txt`, the 2021 preprint,
[arXiv 2112.10880](https://arxiv.org/abs/2112.10880).
The [published article](https://onlinelibrary.wiley.com/doi/10.1002/pst.2296)
confirms single-arm/randomized binary, continuous, survival and multiple/co-primary
scope in its abstract. Its supporting information lists
`pst2296-sup-0001-Tables.docx` (95.7 KB). On September 28 the direct public
supplement link returned HTTP 403; the app URL timed out through the web reader.
No supplementary contents or native application behavior were obtained from
those requests, and no S3/S4 numerical parity is claimed.

- Section 2.1.4, text lines 239–302: multinomial–Dirichlet cell model,
  posterior shape update, four-category binary examples, and marginal Beta
  posterior for any 0/1 sum of cell probabilities.
- Section 2.2, lines 369–411: OR composition for multiple endpoints and AND
  composition for co-primary endpoints, with the opposite no-go composition.
- Section 2.4, lines 466–495: independent arm posteriors, signed
  experimental-minus-control differences, and optional interim O'Brien–Fleming
  superiority rules.
- Section 3.2, lines 663–667: randomized multiple-endpoint and efficacy/toxicity
  simulations are reported in Supplement Tables S3 and S4.

The implementation's raw toxicity margin is experimental minus control,
with favorable probability below the margin. Equivalently, one can negate both
the difference and its margin to work in a positive-benefit direction. Keeping
raw margins avoids silently switching the observed category convention.

The paired graduation rule is the composition of §2.2 endpoint logic with
§2.4's scalar superiority rule. The main text does not supply separate paired
superiority pseudocode; document this composition and avoid a native-default
claim. All priors, margins, joint truths, allocation schedules and calibration
grids are explicit inputs. Supplement simulation settings are not defaults.

For two arbitrary 0/1 indicator functions on a larger categorical state space,
coarsening into the four indicator combinations preserves the joint Dirichlet
posterior and all needed marginal tails. This supports the stated categorical
model through exact aggregation; it is not an assumption that endpoints are
independent. More than two endpoint decisions remain a separate generalization.
