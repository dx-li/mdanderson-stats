# Independent probability-moment integration for EffTox prior calibration.
# Thall et al. (2014), DOI 10.1177/1740774514547397, beta-moment ESS.
# This is base-R mathematical validation, not the Windows calibration kernel.
options(digits = 17)

integral <- function(f, lower = -Inf, upper = Inf) {
  integrate(f, lower, upper, rel.tol = 1e-11, abs.tol = 1e-30,
            subdivisions = 300)$value
}

normal_expectation <- function(mean, sd, transform) {
  if (sd == 0) return(transform(plogis(mean)))
  integral(function(z) transform(plogis(mean + sd * z)) * dnorm(z))
}

prior_moments <- function(x, mean, sd, outcome, monotone) {
  if (outcome == "E") {
    location <- mean[3] + x * mean[4] + x^2 * mean[5]
    scale <- sqrt(sd[3]^2 + x^2 * sd[4]^2 + x^4 * sd[5]^2)
    expectation <- function(f) normal_expectation(location, scale, f)
  } else if (!monotone || sd[2] == 0) {
    location <- mean[1] + x * mean[2]
    scale <- sqrt(sd[1]^2 + x^2 * sd[2]^2)
    expectation <- function(f) normal_expectation(location, scale, f)
  } else {
    # Integrate the intercept conditionally on a positive Gaussian slope.
    lower <- -mean[2] / sd[2]
    log_mass <- pnorm(lower, lower.tail = FALSE, log.p = TRUE)
    expectation <- function(f) integral(function(z) {
      vapply(z, function(zz) {
        slope <- mean[2] + sd[2] * zz
        normal_expectation(mean[1] + x * slope, sd[1], f) *
          exp(dnorm(zz, log = TRUE) - log_mass)
      }, numeric(1))
    }, lower = lower)
  }
  m <- expectation(identity)
  # Center before integrating; avoid subtracting two nearly equal moments.
  v <- expectation(function(p) (p - m)^2)
  stopifnot(is.finite(m), is.finite(v), m > 0, m < 1, v > 0,
            v <= m * (1 - m) * (1 + 1e-10))
  c(mean = m, variance = v, ess = m * (1 - m) / v - 1)
}

cases <- list(
  published_monotone = list(
    doses = c(1, 2, 4, 6.6, 10), monotone = TRUE,
    mean = c(-7.9593, 1.5482, .7367, 3.4181, 0, 0),
    sd = c(3.5487, 3.5018, 2.5423, 2.4406, .2, 1)),
  published_unconstrained = list(
    doses = c(1, 2, 4, 6.6, 10), monotone = FALSE,
    mean = c(-7.9593, 1.5482, .7367, 3.4181, 0, 0),
    sd = c(3.5487, 3.5018, 2.5423, 2.4406, .2, 1)),
  negative_slope_mean = list(
    doses = c(.5, 1, 2, 4), monotone = TRUE,
    mean = c(-1, -.8, .3, 1.2, -.1, 0),
    sd = c(.6, .7, .5, .8, .35, 1)),
  near_fixed = list(
    doses = c(1, 2, 4), monotone = TRUE,
    mean = c(-1, .8, 0, 0, 0, 0),
    sd = c(1e-5, 0, 1e-5, 1e-5, 0, 1))
)

rows <- list()
for (name in names(cases)) {
  case <- cases[[name]]
  x <- log(case$doses) - mean(log(case$doses))
  for (k in seq_along(x)) for (outcome in c("E", "T")) {
    result <- prior_moments(x[k], case$mean, case$sd, outcome, case$monotone)
    rows[[length(rows) + 1]] <- data.frame(
      case = name, dose_index = k, dose = case$doses[k], dose_code = x[k],
      monotone_toxicity = case$monotone, outcome = outcome,
      mu_T = case$mean[1], beta_T = case$mean[2], mu_E = case$mean[3],
      beta_E1 = case$mean[4], beta_E2 = case$mean[5],
      sd_mu_T = case$sd[1], sd_beta_T = case$sd[2], sd_mu_E = case$sd[3],
      sd_beta_E1 = case$sd[4], sd_beta_E2 = case$sd[5],
      probability_mean = unname(result["mean"]),
      probability_variance = unname(result["variance"]),
      beta_moment_ess = unname(result["ess"]))
  }
}
output <- do.call(rbind, rows)
write.csv(output, "tests/fixtures/efftox-prior-moments.csv", row.names = FALSE)
print(aggregate(cbind(probability_mean, beta_moment_ess) ~ case + outcome,
                output, mean))
