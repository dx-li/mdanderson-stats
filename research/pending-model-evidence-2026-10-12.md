# Six pending models: complementary public-source pass

The initial parent research pass recovered no complete fitted contract. A deeper
October12 pass has since recovered the BLESS supplement and BayesianSurvival
technical DOCX; see the dated update below. All six currently remain pending
until implementation and source validation justify a status change. No outreach was authorized or performed; challenged/403 downloads were
not bypassed. Findings below distinguish readable primary evidence from indexed
or inaccessible supplements.

The table records the initial pass; the update below supersedes its BLESS and
BayesianSurvival access and parameter gaps.

| Application | Evidence recovered | Exact missing evidence |
|---|---|---|
| BayesianSurvival | [Primary article](https://pmc.ncbi.nlm.nih.gov/articles/PMC8058232/) specifies seven piecewise-exponential intervals selected by minimum DIC among 1–10, time origin five years after diagnosis, gamma hazards and normal coefficient priors. | Cutpoints, full coding/centering, joint fitted posterior, exact prior hyperparameters, tail behavior and prediction/interval algorithm. [Supporting DOCX](https://pmc.ncbi.nlm.nih.gov/articles/instance/8058232/bin/NIHMS1662865-supplement-Supporting_material.docx) hit a download challenge; Wiley returned403. |
| K-COMPASS | [Primary abstract](https://pubmed.ncbi.nlm.nih.gov/41765697/) readable. An indexed [first-author interview](https://www.urotoday.com/video-lectures/asco-gu-2026/video/5407-k-compass-study-investigates-ctdna-and-kim-1-in-oligometastatic-kidney-cancer-chad-tang.html) describes continuous log10 KIM-1 and binary detectable ctDNA; direct interview retrieval failed. | Exact fit, centering, zero/missing handling, Weibull shape/intercept and uncertainty. |
| MDS-DPSS | [Lab publications](https://www.mdanderson.org/research/departments-labs-institutes/labs/ziyi-li-laboratory/publications.html) and [author papers](https://ziyili20.github.io/papers.html) still list submitted work; verified author yuelyu21 public repositories supplied no artifact. | Dynamic fitted model, calibration and complete output contract. |
| BLESS | [Primary article](https://pmc.ncbi.nlm.nih.gov/articles/PMC8449006/) points to essential e-Appendix1/e-Tables1–8 in mmc1.pdf. [2022 validation PDF](https://journal.chestnet.org/article/S0012-3692%2822%2902819-7/pdf) returned403 and yielded no coefficients. | Exact coefficients, coding, baseline and uncertainty from the supplements/fitted object. |
| CNSRISK | [Author bibliography](https://faculty.mdanderson.org/profiles/michael_davies.html) still points only to DOI10.1200/JCO.2023.41.16_suppl.2012. | Final competing-risk model, baseline functions and uncertainty. Primary site/subtype/Breslow/mitotic-rate predictor names are insufficient. |
| ComPAS | [PNOC022 application](https://academic.oup.com/neuro-oncology/article/26/Supplement_2/S125/7485712) supplies trial-specific schedules. | General priors, likelihood, decision defaults and complete application output contract. |

If permitted public sources remain exhausted, author-provided artifacts with
appropriate reuse permission may be necessary. This is a possible future path,
not permission to contact authors or a basis to substitute guessed models.

## Deeper October12 evidence now available

Parent research recovered the actual47-page BLESS mmc1.pdf, including e-Appendix1
and coefficient/baseline e-Tables1–8 for all four published models. Baseline
survivors are available at0/30/60/90/120/150/180 days. This can support a
published-precision grid model, subject to explicit source inconsistencies,
rounding and deployed-parity limits. Artifact/hash and structured extraction are
being transferred for executor verification; no implementation is claimed yet.
Do not continue describing this supplement as unavailable.

The actual BayesianSurvival DOCX was also recovered. It specifies seven intervals,
Gamma(shape.0001,scale.0001) hazards in the source's wording, independent rather
than autoregressive hazard priors, Normal(0,variance1e6) coefficients and half-open
intervals. Shape/scale wording needs careful preservation; fitted cutpoints and
the joint posterior still remain missing. The previous download challenge was
an access failure, not evidence that the document lacked these parameters.

K-COMPASS's full article/supplements are publisher-gated and uninspected; no claim
that they omit coefficients is justified. CNSRISK's final2023 abstract explicitly
has no full-text component and refines the2021 model. Detailed current app inputs
and remaining targets are in their respective source-status files. No outreach
is authorized; research and implementation may continue without it.

## Current recovered-core update

BLESS149 and ComPAS140 now have source-backed published workflows and are partial,
not pending. Current counts102 implemented/32 partial/4 pending. Their remaining
deployed/full-platform contracts are documented in docs/bless.md and docs/compas.md.
The four pending applications are BayesianSurvival, CNSRISK, K-COMPASS and MDS-DPSS;
no missing fitted parameters were replaced by approximations. Earlier six-model
tables above are historical evidence of the initial source boundary.
