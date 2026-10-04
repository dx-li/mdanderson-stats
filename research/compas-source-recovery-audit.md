# ComPAS source recovery audit — 2026-10-04

## Scope

Bounded search for an independent primary author/institutional manuscript,
dissertation, or author code for Tang, Shen, and Yuan, “ComPAS: A Bayesian drug
combination platform trial design with adaptive shrinkage,” *Statistics in
Medicine* 38 (2019), 1120–1134, DOI [10.1002/sim.8026](https://doi.org/10.1002/sim.8026).
The search was limited to three targeted queries for the title/DOI and author
code, after reading the scoped source-status note. No exhausted publisher,
vendor, or MD Anderson route was retried.

## Outcome

No new source recovered the ComPAS method contract. The [PubMed record](https://pubmed.ncbi.nlm.nih.gov/30419609/)
supplies the abstract and citation:
adaptive borrowing uses Bayesian model selection and hierarchical models, and
the platform may drop futile combinations, graduate effective combinations,
and add combinations. It does not state the outcome likelihood, model-space
prior, hyperpriors, shrinkage calculation, posterior computation, allocation,
or stopping thresholds.

The promising independent institutional result was an Indiana University
dissertation deposited in the university's Indianapolis repository:
[*Innovative Bayesian Designs for Clinical
Trials*](https://scholarworks.indianapolis.iu.edu/server/api/core/bitstreams/d64d0c29-0b5b-4c73-be52-95f296beb8d9/content).
The front matter identifies Indiana University as the degree-granting
institution (PDF p. 1). Its introductory overview cites ComPAS as prior work
(PDF pp. 5–6; repository text lines 367–377), then describes the dissertation's own PMED and survival
platform designs. The reference list cites Tang, Shen, and Yuan (PDF p. 144),
but the dissertation does not provide ComPAS equations or implementation
details. It therefore cannot support a ComPAS port.

The author-hosted/publication links and blocked or exhausted routes are already
documented in `research/compas-source-status.md`; this audit adds no claim that
those sources were retried or newly accessed. The fresh search yielded no
author repository or independent full-text copy with the missing contract.

## Implementation conclusion

Do not implement a generic beta-binomial monitor, infer ComPAS priors from
another platform design, or claim ComPAS coverage from the abstract. A faithful
implementation remains blocked pending a primary source that specifies the
model-selection/shrinkage hierarchy and its trial conduct rules. No code was
changed.
