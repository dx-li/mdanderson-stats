#!/usr/bin/env Rscript

# Independent base-R oracle for BARD's categorical logistic response model.
# Source anchors: cached paper.txt main text pp. 17-18 and supplement S1;
# cached Guide.txt remarks 1 (categorical odds-ratio definition).

profile_offsets <- function(profiles, odds_ratios) {
  profiles <- as.matrix(profiles)
  odds_ratios <- as.matrix(odds_ratios)
  if (ncol(profiles) != nrow(odds_ratios) || any(profiles < 1) ||
      any(profiles != floor(profiles)) || any(!is.finite(odds_ratios)) ||
      any(odds_ratios <= 0) || any(odds_ratios[, 1] != 1)) {
    stop("profiles and baseline-coded positive odds ratios are inconsistent")
  }
  if (any(profiles > ncol(odds_ratios))) {
    stop("a profile level has no supplied odds ratio")
  }
  offsets <- numeric(nrow(profiles))
  for (factor in seq_len(ncol(profiles))) {
    offsets <- offsets + log(odds_ratios[factor, profiles[, factor]])
  }
  offsets
}

population_response <- function(intercept, offsets, profile_probabilities) {
  if (length(offsets) != length(profile_probabilities) ||
      any(!is.finite(offsets)) || any(!is.finite(profile_probabilities)) ||
      any(profile_probabilities < 0) ||
      abs(sum(profile_probabilities) - 1) > 1e-12) {
    stop("offsets and normalized joint profile probabilities are required")
  }
  sum(profile_probabilities * plogis(intercept + offsets))
}

calibrate_intercept <- function(target, offsets, profile_probabilities) {
  if (length(target) != 1 || !is.finite(target) || target <= 0 || target >= 1) {
    stop("target must be strictly between zero and one")
  }
  fn <- function(intercept) {
    population_response(intercept, offsets, profile_probabilities) - target
  }
  lo <- -1
  hi <- 1
  for (i in seq_len(60)) {
    if (fn(lo) <= 0 && fn(hi) >= 0) break
    lo <- 2 * lo
    hi <- 2 * hi
  }
  if (!(fn(lo) <= 0 && fn(hi) >= 0)) stop("failed to bracket the monotone calibration root")
  uniroot(fn, c(lo, hi), tol = 1e-13)$root
}

profile_probability <- function(intercept, profiles, odds_ratios) {
  plogis(intercept + profile_offsets(profiles, odds_ratios))
}

conditional_and_marginal_or <- function(profiles, profile_probabilities,
                                        odds_ratios, response_probability) {
  profiles <- as.matrix(profiles)
  odds_ratios <- as.matrix(odds_ratios)
  response_probability <- as.numeric(response_probability)
  if (length(response_probability) != nrow(profiles)) stop("profile probability shape mismatch")

  conditional <- lapply(seq_len(nrow(odds_ratios)), function(factor) {
    levels <- sort(unique(profiles[, factor]))
    vapply(levels[-1], function(level) {
      matched <- which(profiles[, factor] == level)
      reference <- profiles[matched, , drop = FALSE]
      reference[, factor] <- 1
      reference_index <- match(
        apply(reference, 1, paste, collapse = ":"),
        apply(profiles, 1, paste, collapse = ":")
      )
      if (anyNA(reference_index)) stop("profile table is not a complete Cartesian product")
      observed_odds <- response_probability[matched] / (1 - response_probability[matched])
      reference_odds <- response_probability[reference_index] /
        (1 - response_probability[reference_index])
      ratios <- observed_odds / reference_odds
      if (max(ratios) - min(ratios) > 1e-10 * max(1, max(ratios))) {
        stop("conditional OR varies by other factors; main-effects model assumption failed")
      }
      mean(ratios)
    }, numeric(1))
  })

  marginal <- lapply(seq_len(nrow(odds_ratios)), function(factor) {
    levels <- sort(unique(profiles[, factor]))
    level_probability <- vapply(levels, function(level) {
      rows <- profiles[, factor] == level
      sum(profile_probabilities[rows] * response_probability[rows]) /
        sum(profile_probabilities[rows])
    }, numeric(1))
    reference_odds <- level_probability[1] / (1 - level_probability[1])
    level_odds <- level_probability[-1] / (1 - level_probability[-1])
    level_odds / reference_odds
  })
  list(conditional = conditional, marginal = marginal)
}

run_reference_checks <- function() {
  # Closed-form binary reduction of the monotone marginal calibration.
  prevalence <- 0.4
  target <- 0.35
  or <- 3
  binary_profiles <- expand.grid(x = 1:2)
  binary_weights <- c(1 - prevalence, prevalence)
  binary_or <- matrix(c(1, or), nrow = 1)
  binary_offset <- profile_offsets(binary_profiles, binary_or)
  binary_intercept <- calibrate_intercept(target, binary_offset, binary_weights)
  a <- or * (target - 1)
  b <- target * (1 + or) - ((1 - prevalence) + prevalence * or)
  c0 <- target
  odds_baseline <- 2 * c0 / (sqrt(b^2 - 4 * a * c0) - b)
  stopifnot(abs(binary_intercept - log(odds_baseline)) < 2e-12)

  # A correlated 2x3 joint profile distribution distinguishes conditional
  # logistic ORs from marginal ORs after averaging over the other factor.
  profiles <- expand.grid(f1 = 1:2, f2 = 1:3)
  joint <- c(0.40, 0.05, 0.10, 0.15, 0.05, 0.25)
  joint <- joint / sum(joint)
  odds_ratios <- rbind(c(1, 2.5, 1), c(1, 0.6, 3.0))
  offsets <- profile_offsets(profiles, odds_ratios)
  correlated_target <- 0.42
  intercept <- calibrate_intercept(correlated_target, offsets, joint)
  probabilities <- profile_probability(intercept, profiles, odds_ratios)
  or_results <- conditional_and_marginal_or(profiles, joint, odds_ratios, probabilities)
  stopifnot(abs(population_response(intercept, offsets, joint) - correlated_target) < 2e-13)
  stopifnot(max(abs(unlist(or_results$conditional) - c(2.5, 0.6, 3.0))) < 2e-12)
  stopifnot(any(abs(unlist(or_results$marginal) - unlist(or_results$conditional)) > 1e-3))

  # Supplement S1: eight five-dose intercept vectors and common binary-factor
  # effects. Table 3 gives population-marginal response rates rounded to .001.
  intercepts <- rbind(
    c(-2.197, -1.099, -0.619, -0.201, 0.201),
    c(-2.442, -2.197, -1.099, -0.619, -0.201),
    c(-2.944, -2.442, -2.197, -1.099, -0.619),
    c(-3.892, -2.944, -2.442, -2.197, -1.099),
    c(-1.099, -1.099, -1.046, -1.046, -1.046),
    c(-2.197, -1.099, -1.099, -1.046, -1.046),
    c(-2.442, -2.197, -1.099, -1.099, -1.046),
    c(-2.944, -2.442, -2.197, -1.099, -1.099)
  )
  table3_response <- rbind(
    c(.181, .349, .439, .519, .596),
    c(.152, .181, .349, .439, .519),
    c(.103, .152, .181, .349, .439),
    c(.046, .103, .152, .181, .349),
    c(.349, .349, .359, .359, .359),
    c(.181, .349, .349, .359, .359),
    c(.152, .181, .349, .349, .359),
    c(.103, .152, .181, .349, .349)
  )
  source_beta <- c(1.7, -1.5, 0.4)
  source_profiles <- expand.grid(x1 = 1:2, x2 = 1:2, x3 = 1:2)
  source_weights <- rep(1 / 8, nrow(source_profiles))
  source_or <- rbind(exp(c(0, source_beta[1])),
                     exp(c(0, source_beta[2])),
                     exp(c(0, source_beta[3])))
  source_offsets <- profile_offsets(source_profiles, source_or)
  reconstructed <- matrix(NA_real_, nrow(intercepts), ncol(intercepts))
  recalibrated <- reconstructed
  recalibrated_rates <- reconstructed
  for (scenario in seq_len(nrow(intercepts))) {
    for (dose in seq_len(ncol(intercepts))) {
      reconstructed[scenario, dose] <- population_response(
        intercepts[scenario, dose], source_offsets, source_weights
      )
      recalibrated[scenario, dose] <- calibrate_intercept(
        table3_response[scenario, dose], source_offsets, source_weights
      )
      recalibrated_rates[scenario, dose] <- population_response(
        recalibrated[scenario, dose], source_offsets, source_weights
      )
    }
  }
  # Three-decimal publication rounding contributes at most .0005 to each
  # displayed value; rounded intercepts contribute at most .0005*.25=.000125
  # to a logistic probability. Allow a small numerical root-solve cushion.
  stopifnot(max(abs(reconstructed - table3_response)) < 0.000625 + 1e-12)
  stopifnot(max(abs(recalibrated_rates - table3_response)) < 1e-12)
  published_reference <- data.frame(
    scenario = rep(seq_len(nrow(intercepts)), each = ncol(intercepts)),
    dose = rep(seq_len(ncol(intercepts)), times = nrow(intercepts)),
    target_rate = as.vector(t(table3_response)),
    source_intercept = as.vector(t(intercepts)),
    source_rate = as.vector(t(reconstructed)),
    recalibrated_intercept = as.vector(t(recalibrated)),
    recalibrated_rate = as.vector(t(recalibrated_rates))
  )
  list(
    binary_intercept = binary_intercept,
    correlated_intercept = intercept,
    conditional_or = or_results$conditional,
    marginal_or = or_results$marginal,
    published_reference = published_reference,
    source_rates = reconstructed,
    recalibrated_intercepts = recalibrated,
    source_rate_max_error = max(abs(reconstructed - table3_response)),
    source_intercept_max_error = max(abs(recalibrated - intercepts))
  )
}

if (sys.nframe() == 0) {
  options(digits = 17)
  result <- run_reference_checks()
  csv_path <- Sys.getenv("BARD_REFERENCE_CSV", unset = "")
  if (nzchar(csv_path)) {
    write.csv(result$published_reference, csv_path, row.names = FALSE)
  }
  cat("BARD independent response reference checks passed\n")
  print(result)
}
