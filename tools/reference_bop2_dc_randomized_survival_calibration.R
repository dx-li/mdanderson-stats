# Independent base-R randomized-exponential-survival BOP2-DC calibration oracle.
# Uses Gamma-density integration and supplied arrival/event-duration tapes;
# it does not call Python functions or reproduce its random streams.
# Usage: Rscript reference_bop2_dc_randomized_survival_calibration.R [prefix]
options(digits = 17)
args <- commandArgs(trailingOnly = TRUE)
prefix <- if (length(args)) args[[1]] else ""
settings <- read.csv(paste0(prefix, "settings.csv"), stringsAsFactors = FALSE,
                     colClasses = "character", check.names = FALSE)
grid <- read.csv(paste0(prefix, "grid.csv"), stringsAsFactors = FALSE,
                 colClasses = "character", check.names = FALSE)
paths <- read.csv(paste0(prefix, "paths.csv"), stringsAsFactors = FALSE,
                  colClasses = "character", check.names = FALSE)
parse_ints <- function(x) as.integer(strsplit(x, ";", fixed = TRUE)[[1]])

# P(log(2)*(mean_E - mean_C) > margin) for independent Gamma rate
# posteriors. Standardize X=rate_C * exposure_C ~ Gamma(shape_C, rate=1).
median_difference_probability <- function(shape_e, rate_e, shape_c, rate_c, margin) {
  if (margin == 0) {
    return(list(probability = pbeta(rate_e / (rate_e + rate_c), shape_e, shape_c),
                abs_error = 0))
  }
  delta <- margin / log(2)
  integrand <- function(x) {
    denominator <- rate_c / x + delta
    density <- dgamma(x, shape = shape_c, rate = 1)
    conditional <- ifelse(denominator <= 0, 1,
                          pgamma(rate_e / denominator, shape = shape_e, rate = 1))
    conditional * density
  }
  if (margin < 0) {
    critical <- rate_c / (-delta)
    if (!is.finite(critical) || critical <= 0) stop("invalid negative-margin split")
    inner <- integrate(integrand, lower = 0, upper = critical,
                       rel.tol = 2e-12, abs.tol = 2e-13,
                       subdivisions = 2000L, stop.on.error = TRUE)
    tail <- pgamma(critical, shape = shape_c, rate = 1, lower.tail = FALSE)
    probability <- inner$value + tail; error <- inner$abs.error
  } else {
    whole <- integrate(integrand, lower = 0, upper = Inf,
                       rel.tol = 2e-12, abs.tol = 2e-13,
                       subdivisions = 2000L, stop.on.error = TRUE)
    probability <- whole$value; error <- whole$abs.error
  }
  if (!is.finite(probability) || !is.finite(error) || probability < 0 || probability > 1)
    stop("invalid independent Gamma-density comparison integral")
  list(probability = probability, abs_error = error)
}
obf_cutoff <- function(lambda, look, n) 2 * pnorm(qnorm((1 + lambda) / 2) / sqrt(look / n)) - 1
classify_corner <- function(pl, pc, look, s, p) {
  n <- as.integer(s$max_subjects)
  if (look < n) {
    no_go <- pl < p[["lambda_lrv"]] * (look / n)^p[["gamma_lrv"]] &&
      pc < p[["lambda_cmv"]] * (look / n)^p[["gamma_cmv"]]
    graduate <- FALSE
    if (as.integer(s$graduate_at_interim) == 1L)
      graduate <- pl > obf_cutoff(p[["lambda_lrv"]], look, n) &&
        pc > obf_cutoff(p[["lambda_cmv"]], look, n)
    if (no_go && graduate) stop("overlapping interim decision regions")
    if (no_go) return("stop_no_go")
    if (graduate) return("graduate")
    return("continue")
  }
  go <- pl > p[["lambda_lrv"]] && pc > p[["lambda_cmv"]]
  no_go <- pl < p[["lambda_lrv"]] && pc < p[["lambda_cmv"]]
  if (go) return("final_go")
  if (no_go) return("final_no_go")
  "final_consider"
}
classify_with_error <- function(pl, pc, err_l, err_c, look, s, p) {
  bounds_l <- c(max(0, pl - err_l), min(1, pl + err_l))
  bounds_c <- c(max(0, pc - err_c), min(1, pc + err_c))
  actions <- c(classify_corner(bounds_l[[1]], bounds_c[[1]], look, s, p),
               classify_corner(bounds_l[[1]], bounds_c[[2]], look, s, p),
               classify_corner(bounds_l[[2]], bounds_c[[1]], look, s, p),
               classify_corner(bounds_l[[2]], bounds_c[[2]], look, s, p))
  if (length(unique(actions)) != 1L)
    stop("quadrature uncertainty straddles a candidate decision boundary")
  actions[[1]]
}

# As-of sufficient statistics at each scheduled look. At interim look n, the
# analysis clock is arrival[n]; final follow-up is added only at max N.
precompute_tails <- function(s, phase, scenario, one, arms, looks) {
  n <- as.integer(s$max_subjects)
  trial_ids <- sort(unique(as.integer(one$trial)))
  if (!length(trial_ids) || !identical(trial_ids, seq_len(length(trial_ids))))
    stop(paste("trial IDs must be consecutive from one:", phase, scenario))
  rows <- list(); k <- 0L
  for (trial in trial_ids) {
    dat <- one[as.integer(one$trial) == trial, , drop = FALSE]
    dat <- dat[order(as.integer(dat$patient)), , drop = FALSE]
    patient <- as.integer(dat$patient)
    arrival_time <- as.numeric(dat$arrival_time)
    event_duration <- as.numeric(dat$event_duration)
    if (length(patient) != n || !identical(patient, seq_len(n)) ||
        any(!is.finite(arrival_time)) || any(arrival_time < 0) ||
        any(diff(arrival_time) < 0) || any(is.na(event_duration)) ||
        any(event_duration < 0) || any(event_duration == -Inf))
      stop(paste("invalid/incomplete calendar tape:", phase, scenario, trial))
    for (look in looks) {
      final <- look == n
      last <- arrival_time[[look]]
      followup <- as.numeric(s$final_followup)
      clock <- last + if (final) followup else 0
      if (!is.finite(clock) || (final && followup > 0 && clock <= last))
        stop("unrepresentable analysis calendar time")
      elapsed <- last - arrival_time[seq_len(look)]
      if (final) elapsed <- elapsed + followup
      if (any(!is.finite(elapsed)) || any(elapsed < 0)) stop("invalid as-of follow-up")
      duration <- event_duration[seq_len(look)]
      observed <- pmin(duration, elapsed)
      event <- duration <= elapsed
      prefix_arm <- arms[seq_len(look)]
      cmask <- prefix_arm == 0L; emask <- prefix_arm == 1L
      d_c <- sum(event[cmask]); d_e <- sum(event[emask])
      exposure_c <- sum(observed[cmask]); exposure_e <- sum(observed[emask])
      shape_c <- as.numeric(s$control_prior_shape) + d_c
      rate_c <- as.numeric(s$control_prior_rate) + exposure_c
      shape_e <- as.numeric(s$treatment_prior_shape) + d_e
      rate_e <- as.numeric(s$treatment_prior_rate) + exposure_e
      if (any(!is.finite(c(shape_c, rate_c, shape_e, rate_e))) ||
          any(c(shape_c, rate_c, shape_e, rate_e) <= 0) ||
          any(!is.finite(c(exposure_c, exposure_e)))) stop("invalid posterior statistics")
      lrv <- median_difference_probability(shape_e, rate_e, shape_c, rate_c,
                                            as.numeric(s$median_lrv))
      cmv <- median_difference_probability(shape_e, rate_e, shape_c, rate_c,
                                            as.numeric(s$median_cmv))
      k <- k + 1L
      rows[[k]] <- data.frame(config_id = s$config_id[[1]], phase = phase,
        scenario = scenario, trial = trial, look = look, analysis_time = clock,
        control_n = sum(cmask), treatment_n = sum(emask), control_events = d_c,
        treatment_events = d_e, control_exposure = exposure_c,
        treatment_exposure = exposure_e, posterior_lrv = lrv$probability,
        error_lrv = lrv$abs_error, posterior_cmv = cmv$probability,
        error_cmv = cmv$abs_error, stringsAsFactors = FALSE)
    }
  }
  do.call(rbind, rows)
}
terminalize <- function(tails, p, s, looks) {
  ids <- sort(unique(tails$trial)); out <- vector("list", length(ids))
  for (i in seq_along(ids)) {
    trial <- ids[[i]]; one <- tails[tails$trial == trial, , drop = FALSE]
    one <- one[match(looks, one$look), , drop = FALSE]
    action <- "continue"; terminal_look <- tail(looks, 1)
    for (j in seq_along(looks)) {
      action <- classify_with_error(one$posterior_lrv[[j]], one$posterior_cmv[[j]],
        one$error_lrv[[j]], one$error_cmv[[j]], looks[[j]], s, p)
      if (action != "continue") { terminal_look <- looks[[j]]; break }
    }
    final_row <- one[one$look == terminal_look, , drop = FALSE]
    out[[i]] <- data.frame(trial = trial, terminal_look = terminal_look,
      terminal_decision = action, analysis_time = final_row$analysis_time,
      control_events = final_row$control_events, treatment_events = final_row$treatment_events,
      control_exposure = final_row$control_exposure,
      treatment_exposure = final_row$treatment_exposure,
      stringsAsFactors = FALSE)
  }
  do.call(rbind, out)
}
summarize_terminal <- function(term, looks) {
  decisions <- c("stop_no_go", "graduate", "final_go", "final_consider", "final_no_go")
  nt <- nrow(term); mass <- matrix(0, length(looks), length(decisions),
    dimnames = list(as.character(looks), decisions)); n_mass <- numeric(length(looks))
  for (i in seq_len(nt)) {
    li <- match(term$terminal_look[[i]], looks)
    mass[li, term$terminal_decision[[i]]] <- mass[li, term$terminal_decision[[i]]] + 1
    n_mass[[li]] <- n_mass[[li]] + 1
  }
  counts <- mass
  mass <- counts / nt; n_mass <- n_mass / nt
  en <- mean(term$terminal_look)
  list(count = counts, probability = mass, mcse = sqrt(mass * (1 - mass) / nt),
    sample_size_probability = n_mass, expected_n = en,
    expected_n_mcse = if (nt > 1L) sd(term$terminal_look) / sqrt(nt) else NA_real_,
    trials = nt)
}
metric_summary <- function(futile, effective) {
  fgr <- mean(futile$terminal_decision %in% c("graduate", "final_go"))
  fngr <- mean(effective$terminal_decision %in% c("stop_no_go", "final_no_go"))
  cgr <- mean(effective$terminal_decision %in% c("graduate", "final_go"))
  fcf <- mean(futile$terminal_decision == "final_consider")
  fce <- mean(effective$terminal_decision == "final_consider")
  list(fgr = fgr, fgr_mcse = sqrt(fgr * (1-fgr) / nrow(futile)),
       fngr = fngr, fngr_mcse = sqrt(fngr * (1-fngr) / nrow(effective)),
       cgr = cgr, cgr_mcse = sqrt(cgr * (1-cgr) / nrow(effective)),
       fcr_futile = fcf, fcr_futile_mcse = sqrt(fcf * (1-fcf) / nrow(futile)),
       fcr_effective = fce, fcr_effective_mcse = sqrt(fce * (1-fce) / nrow(effective)),
       fcr = max(fcf, fce), fcr_mcse = NA_real_,
       en_futile = mean(futile$terminal_look),
       en_futile_mcse = if (nrow(futile) > 1L) sd(futile$terminal_look) / sqrt(nrow(futile)) else NA_real_,
       en_effective = mean(effective$terminal_look),
       en_effective_mcse = if (nrow(effective) > 1L) sd(effective$terminal_look) / sqrt(nrow(effective)) else NA_real_)
}
selected_arm_summaries <- function(term) {
  value <- cbind(term$control_events, term$treatment_events,
                 term$control_exposure, term$treatment_exposure)
  colnames(value) <- c("mean_control_events", "mean_treatment_events",
                       "mean_control_exposure", "mean_treatment_exposure")
  means <- colMeans(value)
  mcse <- apply(value, 2, function(x) if (length(x) > 1L) sd(x) / sqrt(length(x)) else NA_real_)
  as.list(c(means, setNames(mcse, paste0(names(means), "_mcse"))))
}

candidate_rows <- list(); decision_rows <- list(); validation_rows <- list()
tail_rows <- list(); selected_rows <- list()
for (si in seq_len(nrow(settings))) {
  s <- settings[si, , drop = FALSE]; id <- s$config_id[[1]]
  n <- as.integer(s$max_subjects); looks <- parse_ints(s$looks)
  arms <- as.integer(strsplit(s$arm_assignments, "", fixed = TRUE)[[1]])
  if (length(arms) != n || any(!arms %in% c(0L, 1L)) || !any(arms == 0L) || !any(arms == 1L) ||
      tail(looks, 1) != n || any(diff(looks) <= 0) || any(looks < 1 | looks > n))
    stop(paste("invalid allocation/look schedule:", id))
  truth_medians <- c(as.numeric(s$futile_control_median), as.numeric(s$futile_treatment_median),
                     as.numeric(s$effective_control_median), as.numeric(s$effective_treatment_median))
  prior_values <- c(as.numeric(s$control_prior_shape), as.numeric(s$control_prior_rate),
                    as.numeric(s$treatment_prior_shape), as.numeric(s$treatment_prior_rate))
  if (any(!is.finite(truth_medians)) || any(truth_medians <= 0) ||
      any(!is.finite(prior_values)) || any(prior_values <= 0) ||
      as.numeric(s$final_followup) < 0 || as.numeric(s$accrual_rate) <= 0 ||
      !s$arrival[[1]] %in% c("fixed", "poisson"))
    stop(paste("invalid truth, prior, or calendar configuration:", id))
  fg <- as.numeric(s$false_go_limit); fn <- as.numeric(s$false_no_go_limit)
  if (fg < 0 || fg > 1 || fn < 0 || fn > 1 ||
      !s$objective[[1]] %in% c("cgr", "ess_futile"))
    stop(paste("invalid error limits/objective:", id))
  candidates <- grid[grid$config_id == id, , drop = FALSE]
  candidates$candidate_index <- as.integer(candidates$candidate_index)
  if (!nrow(candidates) || !identical(candidates$candidate_index, seq_len(nrow(candidates)) - 1L))
    stop(paste("candidate indices must be consecutive zero-based input order:", id))
  for (nm in c("lambda_lrv", "lambda_cmv", "gamma_lrv", "gamma_cmv")) candidates[[nm]] <- as.numeric(candidates[[nm]])
  if (any(candidates$lambda_lrv <= 0 | candidates$lambda_lrv >= 1) ||
      any(candidates$lambda_cmv <= 0 | candidates$lambda_cmv >= 1) ||
      any(candidates$gamma_lrv < 0 | candidates$gamma_lrv > 1) ||
      any(candidates$gamma_cmv < 0 | candidates$gamma_cmv > 1)) stop(paste("bad cutoff grid:", id))
  cache <- list()
  for (phase in c("calibration", "validation")) for (scenario in c("futile", "effective")) {
    dat <- paths[paths$config_id == id & paths$phase == phase & paths$scenario == scenario, , drop = FALSE]
    if (!nrow(dat)) stop(paste("missing tape:", id, phase, scenario))
    dat$trial <- as.integer(dat$trial); dat$patient <- as.integer(dat$patient)
    cache[[paste(phase, scenario, sep = "_")]] <- precompute_tails(s, phase, scenario, dat, arms, looks)
    tail_rows[[length(tail_rows) + 1L]] <- cache[[paste(phase, scenario, sep = "_")]]
  }
  for (phase in c("calibration", "validation")) {
    nf <- length(unique(cache[[paste(phase, "futile", sep = "_")]]$trial))
    ne <- length(unique(cache[[paste(phase, "effective", sep = "_")]]$trial))
    if (nf != ne) stop(paste("truth scenarios must have equal trial counts:", id, phase))
  }
  per_config <- vector("list", nrow(candidates))
  for (ci in seq_len(nrow(candidates))) {
    p <- unlist(candidates[ci, c("lambda_lrv", "lambda_cmv", "gamma_lrv", "gamma_cmv")], use.names = TRUE)
    calibration <- list()
    for (scenario in c("futile", "effective")) {
      term <- terminalize(cache[[paste("calibration", scenario, sep = "_")]], p, s, looks)
      sm <- summarize_terminal(term, looks); calibration[[scenario]] <- list(terminals = term, summary = sm)
      for (li in seq_along(looks)) for (decision in colnames(sm$probability)) {
        decision_rows[[length(decision_rows) + 1L]] <- data.frame(
          config_id = id, phase = "calibration", scenario = scenario,
          candidate_index = candidates$candidate_index[[ci]], look = looks[[li]],
          decision = decision, count = sm$count[li, decision],
          probability = sm$probability[li, decision],
          mcse = sm$mcse[li, decision], sample_size_probability = sm$sample_size_probability[[li]],
          stringsAsFactors = FALSE)
      }
    }
    m <- metric_summary(calibration$futile$terminals, calibration$effective$terminals)
    fcl_raw <- s$false_consider_limit[[1]]; fcl <- if (is.na(fcl_raw) || fcl_raw == "NA") NA_real_ else as.numeric(fcl_raw)
    feasible <- m$fgr <= as.numeric(s$false_go_limit) && m$fngr <= as.numeric(s$false_no_go_limit) &&
      (is.na(fcl) || m$fcr <= fcl)
    row <- data.frame(config_id = id, candidate_index = candidates$candidate_index[[ci]],
      lambda_lrv = p[["lambda_lrv"]], lambda_cmv = p[["lambda_cmv"]],
      gamma_lrv = p[["gamma_lrv"]], gamma_cmv = p[["gamma_cmv"]],
      false_go_rate = m$fgr, false_go_mcse = m$fgr_mcse,
      false_no_go_rate = m$fngr, false_no_go_mcse = m$fngr_mcse,
      correct_go_rate = m$cgr, correct_go_mcse = m$cgr_mcse,
      false_consider_futile_rate = m$fcr_futile, false_consider_futile_mcse = m$fcr_futile_mcse,
      false_consider_effective_rate = m$fcr_effective, false_consider_effective_mcse = m$fcr_effective_mcse,
      false_consider_rate = m$fcr, false_consider_rate_mcse = NA_real_,
      expected_n_futile = m$en_futile, expected_n_futile_mcse = m$en_futile_mcse,
      expected_n_effective = m$en_effective, expected_n_effective_mcse = m$en_effective_mcse,
      feasible = feasible, stringsAsFactors = FALSE)
    candidate_rows[[length(candidate_rows) + 1L]] <- row; per_config[[ci]] <- row
  }
  cm <- do.call(rbind, per_config); eligible <- which(cm$feasible); selected_index <- NA_integer_
  if (length(eligible)) {
    if (s$objective[[1]] == "cgr") {
      ord <- order(-cm$correct_go_rate[eligible], cm$expected_n_futile[eligible], eligible)
    } else if (s$objective[[1]] == "ess_futile") {
      ord <- order(cm$expected_n_futile[eligible], -cm$correct_go_rate[eligible], eligible)
    } else {
      stop(paste("unsupported objective:", s$objective[[1]]))
    }
    selected_index <- cm$candidate_index[eligible[ord[[1]]]]
  }
  fcl_raw <- s$false_consider_limit[[1]]; fcl <- if (is.na(fcl_raw) || fcl_raw == "NA") NA_real_ else as.numeric(fcl_raw)
  if (is.na(selected_index)) {
    selected_rows[[length(selected_rows) + 1L]] <- data.frame(
      config_id = id, objective = s$objective[[1]], selected_index = NA_integer_,
      selected_lambda_lrv = NA_real_, selected_lambda_cmv = NA_real_,
      selected_gamma_lrv = NA_real_, selected_gamma_cmv = NA_real_,
      feasible_candidates = 0L, no_feasible_candidate = TRUE,
      validation_false_go_rate = NA_real_, validation_false_go_mcse = NA_real_,
      validation_false_no_go_rate = NA_real_, validation_false_no_go_mcse = NA_real_,
      validation_correct_go_rate = NA_real_, validation_correct_go_mcse = NA_real_,
      validation_false_consider_rate = NA_real_, validation_false_consider_rate_mcse = NA_real_,
      validation_false_consider_futile_rate = NA_real_, validation_false_consider_futile_mcse = NA_real_,
      validation_false_consider_effective_rate = NA_real_, validation_false_consider_effective_mcse = NA_real_,
      validation_feasible = NA, validation_expected_n_futile = NA_real_,
      validation_expected_n_futile_mcse = NA_real_,
      validation_expected_n_effective = NA_real_,
      validation_expected_n_effective_mcse = NA_real_,
      validation_futile_mean_control_events = NA_real_, validation_futile_mean_control_events_mcse = NA_real_,
      validation_futile_mean_treatment_events = NA_real_, validation_futile_mean_treatment_events_mcse = NA_real_,
      validation_futile_mean_control_exposure = NA_real_, validation_futile_mean_control_exposure_mcse = NA_real_,
      validation_futile_mean_treatment_exposure = NA_real_, validation_futile_mean_treatment_exposure_mcse = NA_real_,
      validation_effective_mean_control_events = NA_real_, validation_effective_mean_control_events_mcse = NA_real_,
      validation_effective_mean_treatment_events = NA_real_, validation_effective_mean_treatment_events_mcse = NA_real_,
      validation_effective_mean_control_exposure = NA_real_, validation_effective_mean_control_exposure_mcse = NA_real_,
      validation_effective_mean_treatment_exposure = NA_real_, validation_effective_mean_treatment_exposure_mcse = NA_real_,
      stringsAsFactors = FALSE)
    next
  }
  p <- unlist(candidates[candidates$candidate_index == selected_index,
    c("lambda_lrv", "lambda_cmv", "gamma_lrv", "gamma_cmv")][1, ], use.names = TRUE)
  validation <- list()
  for (scenario in c("futile", "effective")) {
    term <- terminalize(cache[[paste("validation", scenario, sep = "_")]], p, s, looks)
    sm <- summarize_terminal(term, looks); validation[[scenario]] <- list(terminals = term, summary = sm)
    for (li in seq_along(looks)) for (decision in colnames(sm$probability)) {
      validation_rows[[length(validation_rows) + 1L]] <- data.frame(
        config_id = id, scenario = scenario, candidate_index = selected_index,
        look = looks[[li]], decision = decision, count = sm$count[li, decision],
        probability = sm$probability[li, decision], mcse = sm$mcse[li, decision],
        sample_size_probability = sm$sample_size_probability[[li]],
        stringsAsFactors = FALSE)
    }
  }
  vm <- metric_summary(validation$futile$terminals, validation$effective$terminals)
  vf <- vm$fgr <= as.numeric(s$false_go_limit) && vm$fngr <= as.numeric(s$false_no_go_limit) &&
    (is.na(fcl) || vm$fcr <= fcl)
  fs <- selected_arm_summaries(validation$futile$terminals)
  es <- selected_arm_summaries(validation$effective$terminals)
  selected_rows[[length(selected_rows) + 1L]] <- data.frame(
    config_id = id, objective = s$objective[[1]], selected_index = selected_index,
    selected_lambda_lrv = p[["lambda_lrv"]], selected_lambda_cmv = p[["lambda_cmv"]],
    selected_gamma_lrv = p[["gamma_lrv"]], selected_gamma_cmv = p[["gamma_cmv"]],
    feasible_candidates = length(eligible), no_feasible_candidate = FALSE,
    validation_false_go_rate = vm$fgr, validation_false_go_mcse = vm$fgr_mcse,
    validation_false_no_go_rate = vm$fngr, validation_false_no_go_mcse = vm$fngr_mcse,
    validation_correct_go_rate = vm$cgr, validation_correct_go_mcse = vm$cgr_mcse,
    validation_false_consider_rate = vm$fcr, validation_false_consider_rate_mcse = NA_real_,
    validation_false_consider_futile_rate = vm$fcr_futile,
    validation_false_consider_futile_mcse = vm$fcr_futile_mcse,
    validation_false_consider_effective_rate = vm$fcr_effective,
    validation_false_consider_effective_mcse = vm$fcr_effective_mcse, validation_feasible = vf,
    validation_expected_n_futile = vm$en_futile,
    validation_expected_n_futile_mcse = vm$en_futile_mcse,
    validation_expected_n_effective = vm$en_effective,
    validation_expected_n_effective_mcse = vm$en_effective_mcse,
    validation_futile_mean_control_events = fs$mean_control_events,
    validation_futile_mean_control_events_mcse = fs$mean_control_events_mcse,
    validation_futile_mean_treatment_events = fs$mean_treatment_events,
    validation_futile_mean_treatment_events_mcse = fs$mean_treatment_events_mcse,
    validation_futile_mean_control_exposure = fs$mean_control_exposure,
    validation_futile_mean_control_exposure_mcse = fs$mean_control_exposure_mcse,
    validation_futile_mean_treatment_exposure = fs$mean_treatment_exposure,
    validation_futile_mean_treatment_exposure_mcse = fs$mean_treatment_exposure_mcse,
    validation_effective_mean_control_events = es$mean_control_events,
    validation_effective_mean_control_events_mcse = es$mean_control_events_mcse,
    validation_effective_mean_treatment_events = es$mean_treatment_events,
    validation_effective_mean_treatment_events_mcse = es$mean_treatment_events_mcse,
    validation_effective_mean_control_exposure = es$mean_control_exposure,
    validation_effective_mean_control_exposure_mcse = es$mean_control_exposure_mcse,
    validation_effective_mean_treatment_exposure = es$mean_treatment_exposure,
    validation_effective_mean_treatment_exposure_mcse = es$mean_treatment_exposure_mcse,
    stringsAsFactors = FALSE)
}
write.csv(do.call(rbind, candidate_rows), paste0(prefix, "candidate-metrics.csv"), row.names = FALSE)
write.csv(do.call(rbind, decision_rows), paste0(prefix, "calibration-decisions.csv"), row.names = FALSE)
write.csv(do.call(rbind, selected_rows), paste0(prefix, "selected-validation.csv"), row.names = FALSE)
write.csv(do.call(rbind, validation_rows), paste0(prefix, "validation-decisions.csv"), row.names = FALSE)
write.csv(do.call(rbind, tail_rows), paste0(prefix, "posterior-tail-cache.csv"), row.names = FALSE)
