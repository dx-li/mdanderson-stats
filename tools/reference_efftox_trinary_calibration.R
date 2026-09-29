#!/usr/bin/env Rscript
# Independent base-R reference for continuation-ratio prior probability moments.
# The reference integrates independent normal intercepts and uses fixed positive
# slopes, so no implementation quadrature or optimizer code is shared.

logistic_moments <- function(mu, sd, slope, x) {
  f <- function(z) plogis(mu + sd * z + slope * x) * dnorm(z)
  m <- integrate(f, -Inf, Inf, rel.tol=1e-12, abs.tol=1e-14)$value
  g <- function(z) (plogis(mu + sd * z + slope * x)-m)^2 * dnorm(z)
  variance <- integrate(g, -Inf, Inf, rel.tol=1e-12, abs.tol=1e-15)$value
  c(mean=m, variance=variance)
}

cases <- data.frame(
  case=c("moderate", "high_toxicity", "low_efficacy", "near_fixed"),
  mu_t=c(-1.1, -0.2, -1.8, -1.0), sd_t=c(0.45, 0.3, 0.7, 1e-4), beta_t=c(0.8, 0.4, 1.1, 0.8),
  mu_q=c(0.25, 0.9, -0.6, 0.2), sd_q=c(0.6, 0.5, 0.35, 1e-4), beta_q=c(1.2, 0.7, 0.9, 1.1)
)
doses <- c(1, 2, 4)
x <- log(doses) - mean(log(doses))
rows <- list()
for (i in seq_len(nrow(cases))) {
  spec <- cases[i,]
  for (j in seq_along(x)) {
    t <- logistic_moments(spec$mu_t, spec$sd_t, spec$beta_t, x[j])
    q <- logistic_moments(spec$mu_q, spec$sd_q, spec$beta_q, x[j])
    e_mean <- (1-t["mean"]) * q["mean"]
    # Independent product variance, avoiding subtraction of nearly equal
    # second-moment terms in the near-fixed coefficient case.
    e_var <- (1-t["mean"])^2 * q["variance"] +
      q["mean"]^2 * t["variance"] + t["variance"] * q["variance"]
    rows[[length(rows)+1]] <- data.frame(
      case=spec$case, dose=doses[j], x=x[j], toxicity_mean=t["mean"],
      toxicity_variance=t["variance"], conditional_efficacy_mean=q["mean"],
      conditional_efficacy_variance=q["variance"], efficacy_mean=e_mean,
      efficacy_variance=e_var,
      efficacy_ess=e_mean*(1-e_mean)/e_var-1,
      toxicity_ess=t["mean"]*(1-t["mean"])/t["variance"]-1
    )
  }
}
write.csv(do.call(rbind, rows), "tests/fixtures/efftox-trinary-prior-moments.csv", row.names=FALSE)
