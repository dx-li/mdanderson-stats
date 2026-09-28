# Proportional-density bootstrap scope

The primary paper's section 3.1, equation (3.1), defines the full-data
goodness-of-fit statistic

\[
\Delta_n=\int_0^\tau w(t)\{\widehat F_c(t)-\widetilde F_c(t)\}^2dt,
\qquad \tau=\min\{x_{(n_0)},x_{(n_1)}\}.
\]

It then specifies arm-specific resampling: draw the observed diagnosis times
with replacement from the fitted observed-case distributions \(d\widehat F_1\)
and \(d\widehat G_1\); separately draw censoring times with replacement from
the raw censoring records in each arm; refit the semiparametric and
nonparametric disease curves; and calculate \(\Delta_n^*\). The paper's
notation gives \(m_0\) control failures and \(m-m_0\) treatment failures. The
implementation keeps each original arm's event and censor counts fixed, labels
the separately sampled records by their original status, and does not pair
sampled failures with sampled censor times. This is an explicit status-stratified
reading of the paper's sampling description; it does not introduce latent event
times, a cure indicator, or an atom at infinity.

The source archive's `modelchecking.R` `pepetest` removes a weight function and
sets \(w(t)=1\), as does this implementation. That archived helper evaluates a
right-endpoint rectangle sum. The full bootstrap instead integrates the
right-continuous step curves exactly, using each post-jump value through the
next event time and the original-data \(\tau\) as the fixed endpoint. An
explicit smaller positive endpoint is also accepted. The archive has no
bootstrap driver, so this is a paper-described workflow rather than an
unchanged-native bootstrap reproduction.

The existing `proportional_density_bootstrap` implements the paper's cheaper
failure-only \(\Delta_{1n}^*\) alternative. It keeps the original censoring
offset and is not the full-data \(\Delta_n^*\) method.

Section 3.2 says a bootstrap can provide critical values for the conditional
likelihood-ratio treatment-effect test under unequal censoring, but does not
specify the null-generation or restricted-fit algorithm. This code does not
claim that calibration. The paper also does not define bootstrap confidence
intervals or disease-curve bands; bootstrap draws are not exposed as parameter
uncertainty intervals.

Primary source: Shen, Qin and Costantino (2007), *JASA* 102:1235–1244,
section 3.1, equation (3.1), and section 3.2. In the cached extraction
`research/raw/PropDen/paper.txt`, the relevant paragraphs are lines 1189–1199
(full-data bootstrap), 1200–1208 (failure-only alternative), and 1215
(unequal-censoring likelihood-ratio calibration statement).
