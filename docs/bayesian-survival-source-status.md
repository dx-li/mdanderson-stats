# Bayesian Survival Function Posterior Estimates: source status

Catalog **146** remains pending. The live application was inspected on
2026-09-10, reporting PID 1029, version 1.0.6.0, updated 2020-06-08.

Despite its general title, this is a fitted clinical prediction model. Inputs
include sex, smoking, primary site, T and N stage, chemotherapy group, IMRT,
radiation dose, neck dissection, age and time. The app explicitly conditions on
having already survived at least five years. Its outputs are a survival
probability distribution and mean survival with 95% intervals.

The associated paper is Dahlstrom et al., *Conditional survival among patients
with oropharyngeal cancer treated with radiation therapy and alive without
recurrence 5 years after diagnosis*, Cancer (2021),
[DOI 10.1002/cncr.33370](https://doi.org/10.1002/cncr.33370).
The paper's indexed text identifies a fitted Bayesian piecewise exponential
survival regression and links this app. This inspection has not established
access to its fitted posterior draws, baseline hazards, covariate encoding or
complete prediction implementation. The app's Reference dialog did not return
reference text during inspection.

Those fitted model artifacts and exact age/time conventions must be obtained
before claiming a Python reproduction. A generic exponential posterior or a
regression built from published marginal hazard-ratio summaries would not
reproduce this software. No patient predictions or substitute fitted parameters
have been added to the package.

## October12 technical supplement recovery

Parent research has now retrieved the actual supporting DOCX linked by the
[primary article](https://pmc.ncbi.nlm.nih.gov/articles/PMC8058232/). It specifies
seven piecewise-exponential intervals, half-open intervals, independent hazard
priors (no autoregression), Gamma(shape0.0001, scale0.0001) in its explicit wording,
and Normal(0, variance1e6) coefficient priors. The shape/scale wording must be
preserved and investigated rather than silently converted to a rate convention.
The earlier download challenge was an access limitation, not missing technical
content. Artifact/hash verification in this executor is pending transfer.

This supplies useful published-method evidence. It does not supply the fitted
cutpoints or joint posterior needed for the deployed conditional-survival
predictor. A generic method implementation and exact fitted-tool reproduction
must remain distinct claims; no application status change has yet occurred.
