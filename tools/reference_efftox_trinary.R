# Independent continuation-ratio EffTox calculations, Thall and Cook (2004).
# No native EffTox executable or implementation is used.
options(digits = 17)
softplus <- function(x) pmax(x, 0) + log1p(exp(-abs(x)))
log_cells <- function(x, theta) {
  eta_t <- theta[1] + theta[2] * x
  eta_q <- theta[3] + theta[4] * x
  cbind(
    neither = -softplus(eta_t) - softplus(eta_q),
    efficacy = -softplus(eta_t) - softplus(-eta_q),
    toxicity = -softplus(-eta_t)
  )
}

theta <- c(-1, .6, .4, .9)
doses <- c(1, 2, 4)
x <- log(doses) - mean(log(doses))
counts <- rbind(c(2, 2, 1), c(1, 3, 1), c(1, 2, 2))
cells <- log_cells(x, theta)
joint <- data.frame(
  dose = doses, dose_code = x,
  count_neither = counts[, 1], count_efficacy = counts[, 2], count_toxicity = counts[, 3],
  mu_t = theta[1], beta_t = theta[2], mu_e = theta[3], beta_e = theta[4],
  log_neither = cells[, 1], log_efficacy = cells[, 2], log_toxicity = cells[, 3],
  prob_neither = exp(cells[, 1]), prob_efficacy = exp(cells[, 2]),
  prob_toxicity = exp(cells[, 3]), log_likelihood = sum(counts * cells)
)
write.csv(joint, "tests/fixtures/efftox-trinary-joint.csv", row.names = FALSE)

# With independent coefficient priors, the full CR likelihood factorizes
# into toxicity vs non-toxicity and efficacy vs neither among non-toxic cases.
# Fix positive slopes; integrate each Gaussian intercept independently.
prior_sd <- c(.7, 0, .8, 0)
integral <- function(f) integrate(f, -Inf, Inf, rel.tol = 1e-11,
                                abs.tol = 1e-25, subdivisions = 250)$value
posterior_block <- function(mu, sd, slope, success, failure) {
  kernel <- function(z) vapply(z, function(v) {
    eta <- mu + sd * v + slope * x
    log_lik <- sum(-success * softplus(-eta) - failure * softplus(eta))
    exp(log_lik) * dnorm(v)
  }, numeric(1))
  normalizer <- integral(kernel)
  expectation <- function(f) integral(function(z) kernel(z) * f(mu + sd * z)) / normalizer
  probabilities <- vapply(x, function(code) {
    expectation(function(intercept) plogis(intercept + slope * code))
  }, numeric(1))
  second <- vapply(x, function(code) {
    expectation(function(intercept) plogis(intercept + slope * code)^2)
  }, numeric(1))
  mu_post <- expectation(identity)
  list(mean = probabilities, second = second, intercept_mean = mu_post,
       intercept_sd = sqrt(expectation(function(a) (a - mu_post)^2)))
}
t_block <- posterior_block(theta[1], prior_sd[1], theta[2], counts[, 3],
                           counts[, 1] + counts[, 2])
q_block <- posterior_block(theta[3], prior_sd[3], theta[4], counts[, 2], counts[, 1])
post_e <- (1 - t_block$mean) * q_block$mean
post_t <- t_block$mean
result <- data.frame(
  dose = doses, dose_code = x,
  prior_mu_t = theta[1], prior_beta_t = theta[2],
  prior_mu_e = theta[3], prior_beta_e = theta[4],
  prior_sd_mu_t = prior_sd[1], prior_sd_beta_t = prior_sd[2],
  prior_sd_mu_e = prior_sd[3], prior_sd_beta_e = prior_sd[4],
  posterior_mu_t_mean = t_block$intercept_mean,
  posterior_mu_t_sd = t_block$intercept_sd,
  posterior_mu_e_mean = q_block$intercept_mean,
  posterior_mu_e_sd = q_block$intercept_sd,
  posterior_efficacy_mean = post_e, posterior_toxicity_mean = post_t,
  posterior_neither_mean = (1 - post_t) * (1 - q_block$mean),
  posterior_conditional_efficacy_mean = q_block$mean,
  posterior_efficacy_toxicity_covariance =
    -q_block$mean * (t_block$second - t_block$mean^2)
)
write.csv(result, "tests/fixtures/efftox-trinary-posterior.csv", row.names = FALSE)
stopifnot(max(abs(rowSums(exp(cells)) - 1)) < 1e-14,
          max(post_e + post_t) < 1,
          all(result$posterior_efficacy_toxicity_covariance < 0))
cat("Wrote trinary likelihood and independently integrated posterior references.\n")
