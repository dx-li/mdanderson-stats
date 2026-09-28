# Proportional-density full-data bootstrap

The [primary paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC2721282/), section 3.1,
equation (3.1), defines a goodness-of-fit statistic integrating the weighted
squared difference between the fitted disease-conditional control CDF and its
nonparametric estimate. Its integration endpoint is the smaller arm-specific
maximum follow-up.

The full-data bootstrap description samples diagnosis times from fitted F1/G1,
the distributions **conditional on an observed diagnosis**, and separately
resamples censoring times within each arm from its original data. Both curves
are refitted before calculating each bootstrap statistic. It does not specify
a cure indicator or a probability atom at infinity. Replacing F1/G1 with
disease-conditional distributions and inventing latent failure/censoring pairing
would change the stated procedure.

The prose leaves the counts and pairing/assembly of these samples insufficiently
explicit for a verified native generator. The archived R files contain no
bootstrap driver. These conventions need resolution before claiming a complete
port. The existing failure-only bootstrap is a separate approximation.

This procedure calibrates a goodness-of-fit test. The inspected description
does not establish bootstrap parameter confidence intervals or disease-curve
bands; those require their own stated statistical construction. No new generator
or uncertainty interface was implemented in this audit.
