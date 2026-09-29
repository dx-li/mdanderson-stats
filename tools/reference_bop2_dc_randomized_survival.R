# Independent base-R reference for randomized exponential-survival BOP2-DC.
# Rscript tools/reference_bop2_dc_randomized_survival.R [path-prefix]
# Nonzero-margin posterior comparisons integrate the standardized CONTROL
# reciprocal-mean Gamma density over x in (0, Inf), not a Gamma-quantile map.
options(digits = 17)
args <- commandArgs(trailingOnly = TRUE)
prefix <- if (length(args)) args[[1]] else "tests/fixtures/bop2-dc-randomized-survival-"
cases <- read.csv(paste0(prefix, "cases.csv"), stringsAsFactors = FALSE,
                  colClasses = c("character", rep("numeric", 11), rep("character", 4), rep("numeric", 2)))
parse_num <- function(x) as.numeric(strsplit(x, ";", fixed = TRUE)[[1]])
parse_int <- function(x) as.integer(strsplit(x, ";", fixed = TRUE)[[1]])

# P(log(2)*(mean_E-mean_C)>margin), where reciprocal means have independent
# Gamma(shape, rate=scale) posteriors. Standardize X=scale_C/lambda_C~Gamma(shape_C,1).
median_difference_probability <- function(shape_e, scale_e, shape_c, scale_c, margin) {
  if (margin == 0) {
    probability <- pbeta(scale_e / (scale_e + scale_c), shape_e, shape_c)
    return(list(probability = probability, abs_error = 0))
  }
  delta <- margin / log(2)
  integrand <- function(x) {
    denominator <- scale_c / x + delta
    density <- dgamma(x, shape = shape_c, rate = 1)
    conditional <- ifelse(denominator <= 0, 1,
                          pgamma(scale_e / denominator, shape = shape_e, rate = 1))
    conditional * density
  }
  if (margin < 0) {
    critical <- scale_c / (-delta)
    if (!is.finite(critical) || critical <= 0) stop("invalid negative-margin split")
    inner <- integrate(integrand, lower = 0, upper = critical,
                       rel.tol = 2e-12, abs.tol = 2e-13,
                       subdivisions = 2000L, stop.on.error = TRUE)
    tail <- pgamma(critical, shape = shape_c, rate = 1, lower.tail = FALSE)
    probability <- inner$value + tail
    error <- inner$abs.error
  } else {
    whole <- integrate(integrand, lower = 0, upper = Inf,
                       rel.tol = 2e-12, abs.tol = 2e-13,
                       subdivisions = 2000L, stop.on.error = TRUE)
    probability <- whole$value
    error <- whole$abs.error
  }
  if (!is.finite(probability) || !is.finite(error) || probability < 0 || probability > 1)
    stop("invalid independent Gamma-density comparison integral")
  list(probability = probability, abs_error = error)
}

obf_cutoff <- function(lambda, look, n) {
  2 * pnorm(qnorm((1 + lambda) / 2) / sqrt(look / n)) - 1
}
decision_at <- function(pl, pc, err_l, err_c, look, row) {
  # Match the shared Python rule's four-corner numerical-uncertainty safeguard.
  low_l <- max(0, pl - err_l); high_l <- min(1, pl + err_l)
  low_c <- max(0, pc - err_c); high_c <- min(1, pc + err_c)
  classify <- function(l, c) {
    if (look < row$max_subjects) {
      no_go <- l < row$lambda_lrv * (look / row$max_subjects)^row$gamma_lrv &&
        c < row$lambda_cmv * (look / row$max_subjects)^row$gamma_cmv
      graduate <- FALSE
      if (row$graduate_at_interim == 1) {
        graduate <- l > obf_cutoff(row$lambda_lrv, look, row$max_subjects) &&
          c > obf_cutoff(row$lambda_cmv, look, row$max_subjects)
      }
      if (no_go && graduate) stop("overlapping source decision regions")
      return(if (no_go) "stop_no_go" else if (graduate) "graduate" else "continue")
    }
    go <- l > row$lambda_lrv && c > row$lambda_cmv
    no_go <- l < row$lambda_lrv && c < row$lambda_cmv
    if (go) return("final_go")
    if (no_go) return("final_no_go")
    "final_consider"
  }
  actions <- c(classify(low_l, low_c), classify(low_l, high_c),
               classify(high_l, low_c), classify(high_l, high_c))
  if (length(unique(actions)) != 1) stop("quadrature uncertainty straddles a decision boundary")
  action <- actions[[1]]
  list(action = action,
       no_go_cutoff_lrv = if (look == row$max_subjects) row$lambda_lrv else row$lambda_lrv * (look / row$max_subjects)^row$gamma_lrv,
       no_go_cutoff_cmv = if (look == row$max_subjects) row$lambda_cmv else row$lambda_cmv * (look / row$max_subjects)^row$gamma_cmv,
       graduate_cutoff_lrv = if (look < row$max_subjects && row$graduate_at_interim == 1) obf_cutoff(row$lambda_lrv, look, row$max_subjects) else NA_real_,
       graduate_cutoff_cmv = if (look < row$max_subjects && row$graduate_at_interim == 1) obf_cutoff(row$lambda_cmv, look, row$max_subjects) else NA_real_)
}

rows <- list(); out_i <- 0L
for (i in seq_len(nrow(cases))) {
  r <- cases[i, , drop = FALSE]
  n <- r$max_subjects
  arm <- strsplit(r$assignments, "", fixed = TRUE)[[1]]
  looks <- parse_int(r$looks)
  arrivals <- parse_num(r$enrollment_times)
  durations <- parse_num(r$event_durations)
  if (length(arm) != n || length(arrivals) != n || length(durations) != n ||
      !all(arm %in% c("0", "1")) || !any(arm == "0") || !any(arm == "1") ||
      any(!is.finite(arrivals)) || any(arrivals < 0) || any(diff(arrivals) < 0) ||
      any(is.na(durations) | durations < 0) ||
      any(!is.finite(looks)) || any(looks < 1 | looks > n) || tail(looks, 1) != n ||
      any(diff(looks) <= 0)) stop(paste("invalid replay tape:", r$case_id))
  if (!is.finite(r$final_followup) || r$final_followup < 0) stop("invalid final_followup")
  terminal <- "continue"
  for (look in looks) {
    final <- look == n
    last_enrollment <- arrivals[[look]]
    clock <- last_enrollment + if (final) r$final_followup else 0
    if (!is.finite(clock) || (final && r$final_followup > 0 && clock <= last_enrollment))
      stop("unrepresentable analysis calendar time")
    elapsed <- last_enrollment - arrivals[seq_len(look)]
    if (final) elapsed <- elapsed + r$final_followup
    if (any(!is.finite(elapsed)) || any(elapsed < 0)) stop("invalid as-of follow-up")
    observed <- pmin(durations[seq_len(look)], elapsed)
    events <- durations[seq_len(look)] <= elapsed
    arm_prefix <- arm[seq_len(look)]
    c_mask <- arm_prefix == "0"; e_mask <- arm_prefix == "1"
    d_c <- sum(events[c_mask]); d_e <- sum(events[e_mask])
    exposure_c <- sum(observed[c_mask]); exposure_e <- sum(observed[e_mask])
    shape_c <- r$control_a + d_c; scale_c <- r$control_b + exposure_c
    shape_e <- r$treatment_a + d_e; scale_e <- r$treatment_b + exposure_e
    if (any(!is.finite(c(shape_c, scale_c, shape_e, scale_e))) ||
        any(c(shape_c, scale_c, shape_e, scale_e) <= 0)) stop("invalid posterior IG parameters")
    lrv <- median_difference_probability(shape_e, scale_e, shape_c, scale_c, r$median_lrv)
    cmv <- median_difference_probability(shape_e, scale_e, shape_c, scale_c, r$median_cmv)
    decision <- decision_at(lrv$probability, cmv$probability, lrv$abs_error,
                            cmv$abs_error, look, r)
    out_i <- out_i + 1L
    rows[[out_i]] <- data.frame(
      case_id = r$case_id, look = look, analysis_time = clock,
      enrolled = look, control_n = sum(c_mask), treatment_n = sum(e_mask),
      control_events = d_c, treatment_events = d_e,
      control_exposure = exposure_c, treatment_exposure = exposure_e,
      control_shape = shape_c, control_scale = scale_c,
      treatment_shape = shape_e, treatment_scale = scale_e,
      posterior_lrv = lrv$probability, error_lrv = lrv$abs_error,
      posterior_cmv = cmv$probability, error_cmv = cmv$abs_error,
      no_go_cutoff_lrv = decision$no_go_cutoff_lrv,
      no_go_cutoff_cmv = decision$no_go_cutoff_cmv,
      graduate_cutoff_lrv = decision$graduate_cutoff_lrv,
      graduate_cutoff_cmv = decision$graduate_cutoff_cmv,
      decision = decision$action
    )
    terminal <- decision$action
    if (terminal != "continue") break
  }
}

# Nonsymmetric shape/scale zero-margin identity: Beta(1,1) ratio cutoff 2/3.
analytic_zero_margin <- data.frame(shape_e = 1, scale_e = 2, shape_c = 1,
                                   scale_c = 1, median_margin = 0,
                                   probability = median_difference_probability(1, 2, 1, 1, 0)$probability,
                                   exact_probability = 2 / 3)
write.csv(do.call(rbind, rows), paste0(prefix, "calendar-replay.csv"), row.names = FALSE)
write.csv(analytic_zero_margin, paste0(prefix, "analytic-checks.csv"), row.names = FALSE)
