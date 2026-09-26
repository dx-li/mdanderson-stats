# Next uncovered method: UAROET

Entry 92 remains pending. The official desktop page was verified on 2026-09-26:
https://biostatistics.mdanderson.org/SoftwareDownload/SingleSoftware/Index/92

It identifies UAROET 1.8 (2016-06-28), a command-line Windows implementation of
Thall and Nguyen (2012), *Adaptive randomization to improve utility-based
dose-finding with bivariate ordinal outcomes*, Journal of Biopharmaceutical
Statistics 22:785–801. The public institutional paper is accessible:
https://odin.mdacc.tmc.edu/~pfthall/main/JBS_AR_Utility2012.pdf

An alternative full-text primary record is:
https://pmc.ncbi.nlm.nih.gov/articles/PMC3385658/

The method has saturated marginal ordinal models, optional monotonic dose
effects, a copula joint distribution, elicited outcome utilities, posterior
toxicity/efficacy admissibility and adaptive randomization among near-optimal
doses. The utility tolerance decreases with sample size. This is distinct from
the existing U2OET combination model; inspect its numerical primitives for
reuse but do not alias its likelihood or decision rule.

Next: read Sections 3–5 completely, verify the copula and marginal link,
parameter/prior conventions and near-optimal admissibility rules. Start with
the probability model and a bounded posterior/decision workflow using explicit
priors; defer native elicitation calibration only if its specification remains
unavailable. Delegate to one Luna agent and preserve serial numerical checks.
The native archive has not been retrieved or executed in this audit.

Other pending-entry triage during MTADF integration: the BayesianSurvival app
rendered only its title/version, with no mathematical specification. ComPAS
could not be opened by the web reader. Its citation is Tang, Shen and Yuan,
Statistics in Medicine 38:1120–1134, DOI 10.1002/sim.8026 (PubMed 30419609), not
the distinct BPCC phase I/II method. These entries should not receive guessed
models based only on their titles.
