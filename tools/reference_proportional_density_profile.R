# Independent common-censoring profile-likelihood reference for PropDen §3.2.
# The source conditions on observed failures.  Under common censoring, the
# censoring-survival ratio is one, leaving the offset log(m1/m0) in the
# failure-arm logistic regression.  Confidence limits below are a Python-port
# validation extension obtained by profile-likelihood inversion.
options(warn = 2, digits = 17)

cases <- list(
  tied_unbalanced = list(
    failure_time = c(1, 1, 2, 3, 3, 4, 5, 5, 6, 7, 8, 9, 10),
    arm = c(0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1),
    censor_time = c(2.5, 6.5, 11.5),
    null_beta = 0
  ),
  moderate_effect = list(
    failure_time = 1:14,
    arm = c(0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1),
    censor_time = c(3.5, 8.5, 15),
    null_beta = 0.2
  ),
  reverse_effect = list(
    failure_time = c(1, 2, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11),
    arm = c(1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0),
    censor_time = c(2.5, 7.5, 12.5),
    null_beta = -0.2
  ),
  separated_counts = list(
    failure_time = 1:15,
    arm = c(0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1),
    censor_time = c(3.5, 10.5, 16),
    null_beta = 0
  ),
  rescaled_time = list(
    failure_time = 10 * (1:14),
    arm = c(0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1),
    censor_time = 10 * c(3.5, 8.5, 15),
    null_beta = 0.02
  )
)

input_rows <- list()
summary_rows <- list()
interval_rows <- list()

profile <- function(y, time, offset, beta) {
  model <- glm(y ~ 1 + offset(offset + beta * time), family = binomial(),
    control = glm.control(epsilon = 1e-13, maxit = 100))
  if (!model$converged) stop("profile intercept GLM did not converge")
  list(alpha = unname(coef(model)[1]), loglik = as.numeric(logLik(model)))
}

for (case_id in names(cases)) {
  case <- cases[[case_id]]
  event_time <- case$failure_time
  event_arm <- case$arm
  censor_time <- case$censor_time
  # Identical censor-time samples in the two arms make the validation design's
  # common-censoring assumption explicit.  Censored rows do not enter §3.2's
  # conditional failure likelihood.
  times <- c(event_time, censor_time, censor_time)
  events <- c(rep(1L, length(event_time)), rep(0L, 2L * length(censor_time)))
  arms <- c(event_arm, rep(0L, length(censor_time)), rep(1L, length(censor_time)))
  for (i in seq_along(times)) {
    input_rows[[length(input_rows) + 1L]] <- data.frame(case_id = case_id,
      row = i, time = times[i], event = events[i], arm = arms[i])
  }

  use <- events == 1L
  y <- arms[use]
  x <- times[use]
  m0 <- sum(y == 0L)
  m1 <- sum(y == 1L)
  if (m0 == 0L || m1 == 0L) stop("both arms need observed failures")
  base_offset <- rep(log(m1 / m0), length(x))
  fitted <- glm(y ~ x + offset(base_offset), family = binomial(),
    control = glm.control(epsilon = 1e-13, maxit = 100))
  if (!fitted$converged) stop("unrestricted GLM did not converge")
  alpha_hat <- unname(coef(fitted)[1])
  beta_hat <- unname(coef(fitted)[2])
  ll_hat <- as.numeric(logLik(fitted))
  at_null <- profile(y, x, base_offset, case$null_beta)
  lr_null <- max(0, 2 * (ll_hat - at_null$loglik))
  summary_rows[[length(summary_rows) + 1L]] <- data.frame(case_id = case_id, failures_arm0 = m0,
    failures_arm1 = m1, alpha_hat = alpha_hat, beta_hat = beta_hat,
    loglik_hat = ll_hat, beta_null = case$null_beta,
    alpha_at_null = at_null$alpha, loglik_at_null = at_null$loglik,
    lr_at_null = lr_null, p_at_null = pchisq(lr_null, df = 1, lower.tail = FALSE))

  profile_lr <- function(beta) {
    fit <- profile(y, x, base_offset, beta)
    max(0, 2 * (ll_hat - fit$loglik))
  }
  for (level in c(0.80, 0.95, 0.99)) {
    cutoff <- qchisq(level, df = 1)
    endpoint <- function(direction) {
      step <- max(0.01, abs(beta_hat) * 0.1, 1 / max(diff(range(x)), 1))
      edge <- beta_hat + direction * step
      for (iteration in seq_len(100)) {
        if (profile_lr(edge) >= cutoff) break
        step <- step * 2
        edge <- beta_hat + direction * step
        if (!is.finite(edge)) stop("could not bracket profile limit")
      }
      if (profile_lr(edge) < cutoff) stop("could not bracket profile limit")
      root <- uniroot(function(b) profile_lr(b) - cutoff,
        sort(c(edge, beta_hat)), tol = 1e-12)$root
      root
    }
    interval_rows[[length(interval_rows) + 1L]] <- data.frame(case_id = case_id,
      confidence = level, lower = endpoint(-1), upper = endpoint(1),
      lr_cutoff = cutoff)
  }
}

write.csv(do.call(rbind, input_rows),
  "tests/fixtures/proportional-density-profile-input.csv", row.names = FALSE)
write.csv(do.call(rbind, summary_rows),
  "tests/fixtures/proportional-density-profile.csv", row.names = FALSE)
write.csv(do.call(rbind, interval_rows),
  "tests/fixtures/proportional-density-profile-intervals.csv", row.names = FALSE)
cat("Wrote independent R GLM/profile-likelihood references.\n")
