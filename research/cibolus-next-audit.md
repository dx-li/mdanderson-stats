# Next uncovered method: CiBolus

Catalog entry 86 is pending, with a recorded `CIBOLUS_V1.1.zip` archive.
Its official detail URL did not open through the web reader during the
Dose Schedule Finder pass. The [primary article record](https://pmc.ncbi.nlm.nih.gov/articles/PMC3137757/)
explicitly names CiBolus as the software implementing this method.

The author-hosted [published paper](https://odin.mdacc.tmc.edu/~pfthall/main/Biometrics_IAtPA_2011.pdf)
is accessible: Thall, Szabo, Nguyen, Amlie-Lefond and Zaidat (2011),
*Optimizing the Concentration and Bolus of a Drug Delivered by Continuous
Infusion*, Biometrics 67:1638–1646, DOI 10.1111/j.1541-0420.2011.01580.x.

This is a distinct model for time to response and binary toxicity. Response
can occur immediately after a bolus or during infusion; periodic monitoring
produces interval censoring. Infusion stops upon response, affecting exposure.
Posterior expected utility selects concentration and bolus fraction.

Next: audit all model, likelihood, utility and admissibility equations before
delegating implementation. The accessible article is nine pages and the PMC
record identifies supplementary tables. Preserve immediate-response mass,
continuous/interval/right-censoring distinctions and the response-dependent
toxicity model; do not substitute a generic independent efficacy/toxicity fit.
Native input/report workflows and executable parity have not been inspected.
