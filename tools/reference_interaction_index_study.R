# Independent base-R reference for the interaction-index Scenario 1 fixture.
# Error rows come from tests/fixtures/interaction_index_study_noise.csv;
# noise generation is recorded in the fixture provenance and is not repeated.
args <- commandArgs(trailingOnly = TRUE)
noise_path <- if (length(args) >= 1L) args[[1L]] else "tests/fixtures/interaction_index_study_noise.csv"
out_dir <- if (length(args) >= 2L) args[[2L]] else "."
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

noise <- as.matrix(read.csv(noise_path, check.names = FALSE))
stopifnot(identical(dim(noise), c(5L, 19L)))
taus <- c(0.2, 1.0, 2.5)
confidence <- 0.95
median_doses <- c(1, 2, 4)
dose_grids <- lapply(median_doses, function(dm) seq(0.1, 3 * dm, length.out = 6))
combination_doses <- median_doses / 3

fit_curve <- function(dose, response) {
  fit <- lm(qlogis(response) ~ log(dose))
  list(
    beta = unname(coef(fit)),
    covariance = vcov(fit),
    df = length(dose) - 2,
    residual_mse = deviance(fit) / df.residual(fit)
  )
}

calculate_one <- function(tau, errors) {
  responses <- vector("list", 4)
  for (drug in seq_len(3)) {
    dose <- dose_grids[[drug]]
    logit_mean <- log(median_doses[drug]) - log(dose)
    errors_for_drug <- errors[((drug - 1L) * 6L + 1L):(drug * 6L)]
    responses[[drug]] <- plogis(logit_mean + errors_for_drug)
  }
  responses[[4]] <- plogis(log(tau) + errors[19L])
  fits <- lapply(seq_len(3), function(drug) fit_curve(dose_grids[[drug]], responses[[drug]]))

  response <- responses[[4]]
  inverse_logdose <- vapply(
    fits,
    function(fit) (qlogis(response) - fit$beta[1L]) / fit$beta[2L],
    numeric(1)
  )
  log_terms <- log(combination_doses) - inverse_logdose
  log_estimate <- max(log_terms) + log(sum(exp(log_terms - max(log_terms))))
  shares <- exp(log_terms - log_estimate)
  variance <- 0
  for (drug in seq_len(3)) {
    gradient <- c(
      shares[drug] / fits[[drug]]$beta[2L],
      shares[drug] * inverse_logdose[drug] / fits[[drug]]$beta[2L]
    )
    variance <- variance + drop(t(gradient) %*% fits[[drug]]$covariance %*% gradient)
  }
  dfs <- vapply(fits, `[[`, numeric(1), "df")
  pooled_mse <- sum(dfs * vapply(fits, `[[`, numeric(1), "residual_mse")) / sum(dfs)
  response_gradient <- -sum(shares / vapply(fits, function(fit) fit$beta[2L], numeric(1)))
  variance <- variance + response_gradient^2 * pooled_mse
  df <- sum(dfs)
  log_se <- sqrt(variance)
  critical <- qt((1 + confidence) / 2, df)
  log_limits <- log_estimate + c(-1, 1) * critical * log_se
  estimate <- exp(log_estimate)
  raw_se <- estimate * log_se
  raw_limits <- estimate + c(-1, 1) * critical * raw_se
  c(
    estimate = estimate,
    log_lower = log_limits[1L],
    log_upper = log_limits[2L],
    raw_lower = raw_limits[1L],
    raw_upper = raw_limits[2L]
  )
}

ledger <- do.call(rbind, lapply(taus, function(tau) {
  do.call(rbind, lapply(seq_len(nrow(noise)), function(replicate) {
    values <- calculate_one(tau, noise[replicate, ])
    data.frame(tau = tau, replicate = replicate, t(values), check.names = FALSE)
  }))
}))
summaries <- do.call(rbind, lapply(taus, function(tau) {
  rows <- ledger[ledger$tau == tau, ]
  data.frame(
    tau = tau,
    replicates = nrow(rows),
    mean_estimated_index = mean(rows$estimate),
    raw_ci_coverage = mean(rows$raw_lower <= tau & rows$raw_upper >= tau),
    log_ci_coverage = mean(rows$log_lower <= log(tau) & rows$log_upper >= log(tau)),
    mean_raw_ci_length = mean(rows$raw_upper - rows$raw_lower),
    mean_log_ci_length_on_index_scale = mean(exp(rows$log_upper) - exp(rows$log_lower)),
    fraction_log_ci_below_one = mean(rows$log_upper < 0),
    fraction_log_ci_contains_one = mean(rows$log_lower <= 0 & rows$log_upper >= 0),
    fraction_log_ci_above_one = mean(rows$log_lower > 0)
  )
}))
write.csv(ledger, file.path(out_dir, "interaction-index-study-replicates.csv"), row.names = FALSE)
write.csv(summaries, file.path(out_dir, "interaction-index-study-summary.csv"), row.names = FALSE)
print(summaries, row.names = FALSE)
