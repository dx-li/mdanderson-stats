# Independent base-R randomized-Normal BOP2-DC calibration oracle.
# Uses direct control-Student-t density convolution and direct NIG updates.
# Usage: Rscript reference_bop2_dc_randomized_normal_calibration.R [prefix]
# Required inputs: settings.csv, grid.csv, paths.csv at prefix. Path values are
# centered by the truth-specific control mean; priors are shifted accordingly.
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

# Direct NIG update for (mu0,kappa0,a0,b0); marginal mean is
# t(df=2*a_n, location=mu_n, scale=sqrt(b_n/(a_n*k_n))).
update_nig <- function(mu0, kappa0, a0, b0, residuals) {
  n <- length(residuals)
  if (n == 0L) return(list(location = mu0, precision = kappa0, shape = a0,
                           scale_ig = b0, df = 2 * a0,
                           t_scale = sqrt(b0 / (a0 * kappa0))))
  ybar <- mean(residuals); ss <- sum((residuals - ybar)^2)
  kn <- kappa0 + n; an <- a0 + n / 2; delta <- ybar - mu0
  mun <- mu0 + (n / kn) * delta
  bn <- b0 + ss / 2 + (kappa0 * n / kn) * delta^2 / 2
  if (!is.finite(mun) || !is.finite(bn) || bn <= 0) stop("unrepresentable NIG posterior")
  list(location = mun, precision = kn, shape = an, scale_ig = bn,
       df = 2 * an, t_scale = sqrt(bn / (an * kn)))
}

# P(T-C > margin), conditioning on standardized control mean X~t(df_C).
difference_tail <- function(mu_c, scale_c, df_c, mu_t, scale_t, df_t, margin) {
  delta <- mu_t - mu_c
  if (!is.finite(delta)) stop("posterior mean difference is not representable")
  if (margin == delta) return(list(probability = 0.5, abs_error = 0))
  z <- integrate(function(x) {
    dt(x, df = df_c) * pt((margin - delta + scale_c * x) / scale_t,
                           df = df_t, lower.tail = FALSE)
  }, lower = -Inf, upper = Inf, rel.tol = 2e-12, abs.tol = 2e-13,
  subdivisions = 2000L, stop.on.error = TRUE)
  if (!is.finite(z$value) || !is.finite(z$abs.error) || z$value < 0 || z$value > 1)
    stop("invalid posterior difference integral")
  list(probability = z$value, abs_error = z$abs.error)
}
obf_cutoff <- function(lambda, look, n) 2 * pnorm(qnorm((1 + lambda) / 2) / sqrt(look / n)) - 1
decision_at <- function(pl, pc, look, s, p) {
  n <- as.integer(s$max_subjects)
  if (look < n) {
    cl <- p[["lambda_lrv"]] * (look / n)^p[["gamma_lrv"]]
    cc <- p[["lambda_cmv"]] * (look / n)^p[["gamma_cmv"]]
    no_go <- pl < cl && pc < cc; graduate <- FALSE
    if (as.integer(s$graduate_at_interim) == 1L) {
      graduate <- pl > obf_cutoff(p[["lambda_lrv"]], look, n) &&
        pc > obf_cutoff(p[["lambda_cmv"]], look, n)
    }
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
truth_parameters <- function(s, scenario) {
  prefix <- if (scenario == "futile") "futile" else "effective"
  c(control_mean = as.numeric(s[[paste0(prefix, "_control_mean")]]),
    treatment_mean = as.numeric(s[[paste0(prefix, "_treatment_mean")]]),
    control_sd = as.numeric(s[[paste0(prefix, "_control_sd")]]),
    treatment_sd = as.numeric(s[[paste0(prefix, "_treatment_sd")]]))
}

# Precompute posterior tails once per phase/truth/trial/look, then apply every
# candidate to the shared cache. Outcomes are centered by truth control mean.
precompute_tails <- function(s, phase, scenario, dat, arms, looks) {
  trial_ids <- sort(unique(as.integer(dat$trial)))
  if (!length(trial_ids) || !identical(trial_ids, seq_len(length(trial_ids))))
    stop(paste("trial IDs must be consecutive from one:", phase, scenario))
  truth <- truth_parameters(s, scenario)
  cp_mean0 <- as.numeric(s$control_prior_mean) - truth[["control_mean"]]
  tp_mean0 <- as.numeric(s$treatment_prior_mean) - truth[["control_mean"]]
  rows <- list(); k <- 0L
  for (trial in trial_ids) {
    one <- dat[as.integer(dat$trial) == trial, , drop = FALSE]
    one <- one[order(as.integer(one$patient)), , drop = FALSE]
    patient <- as.integer(one$patient); value <- as.numeric(one$value)
    if (length(patient) != as.integer(s$max_subjects) ||
        !identical(patient, seq_len(as.integer(s$max_subjects))) || any(!is.finite(value)))
      stop(paste("incomplete/nonfinite path:", phase, scenario, trial))
    offset <- value[[1]]; relative <- value - offset
    for (look in looks) {
      ids <- seq_len(look); cidx <- ids[arms[ids] == 0L]; tidx <- ids[arms[ids] == 1L]
      cp <- update_nig(cp_mean0 - offset, as.numeric(s$control_prior_precision),
        as.numeric(s$control_prior_shape), as.numeric(s$control_prior_scale), relative[cidx])
      tp <- update_nig(tp_mean0 - offset, as.numeric(s$treatment_prior_precision),
        as.numeric(s$treatment_prior_shape), as.numeric(s$treatment_prior_scale), relative[tidx])
      lrv <- difference_tail(cp$location, cp$t_scale, cp$df,
        tp$location, tp$t_scale, tp$df, as.numeric(s$theta_lrv))
      cmv <- difference_tail(cp$location, cp$t_scale, cp$df,
        tp$location, tp$t_scale, tp$df, as.numeric(s$theta_cmv))
      k <- k + 1L
      rows[[k]] <- data.frame(config_id = s$config_id[[1]], phase = phase,
        scenario = scenario, trial = trial, look = look, control_n = length(cidx),
        treatment_n = length(tidx), control_location_centered = cp$location,
        control_df = cp$df, control_t_scale = cp$t_scale,
        treatment_location_centered = tp$location, treatment_df = tp$df,
        treatment_t_scale = tp$t_scale, difference_location = tp$location - cp$location,
        posterior_lrv = lrv$probability, error_lrv = lrv$abs_error,
        posterior_cmv = cmv$probability, error_cmv = cmv$abs_error,
        stringsAsFactors = FALSE)
    }
  }
  do.call(rbind, rows)
}
terminalize <- function(tails, p, s, looks) {
  trial_ids <- sort(unique(tails$trial)); rows <- vector("list", length(trial_ids))
  for (i in seq_along(trial_ids)) {
    trial <- trial_ids[[i]]; one <- tails[tails$trial == trial, , drop = FALSE]
    one <- one[match(looks, one$look), , drop = FALSE]
    action <- "continue"; terminal_look <- tail(looks, 1)
    for (j in seq_along(looks)) {
      action <- decision_at(one$posterior_lrv[[j]], one$posterior_cmv[[j]], looks[[j]], s, p)
      if (action != "continue") { terminal_look <- looks[[j]]; break }
    }
    rows[[i]] <- data.frame(trial = trial, terminal_look = terminal_look,
                            terminal_decision = action, stringsAsFactors = FALSE)
  }
  do.call(rbind, rows)
}
summarize_terminal <- function(term, looks) {
  decisions <- c("stop_no_go", "graduate", "final_go", "final_consider", "final_no_go")
  ntrial <- nrow(term); mass <- matrix(0, length(looks), length(decisions),
    dimnames = list(as.character(looks), decisions)); n_mass <- numeric(length(looks))
  for (i in seq_len(ntrial)) {
    li <- match(term$terminal_look[[i]], looks)
    if (is.na(li)) stop("terminal look missing")
    n_mass[[li]] <- n_mass[[li]] + 1
    mass[li, term$terminal_decision[[i]]] <- mass[li, term$terminal_decision[[i]]] + 1
  }
  counts <- mass
  mass <- counts / ntrial; n_mass <- n_mass / ntrial
  en <- mean(term$terminal_look)
  list(count = counts, probability = mass, mcse = sqrt(mass * (1 - mass) / ntrial),
       sample_size_probability = n_mass, expected_n = en,
       expected_n_mcse = if (ntrial > 1L) sd(term$terminal_look) / sqrt(ntrial) else NA_real_,
       n_trials = ntrial)
}
metric_summary <- function(futile, effective) {
  fg <- mean(futile$terminal_decision %in% c("graduate", "final_go"))
  fn <- mean(effective$terminal_decision %in% c("stop_no_go", "final_no_go"))
  cg <- mean(effective$terminal_decision %in% c("graduate", "final_go"))
  fcf <- mean(futile$terminal_decision == "final_consider")
  fce <- mean(effective$terminal_decision == "final_consider")
  list(fgr = fg, fgr_mcse = sqrt(fg * (1 - fg) / nrow(futile)),
       fngr = fn, fngr_mcse = sqrt(fn * (1 - fn) / nrow(effective)),
       cgr = cg, cgr_mcse = sqrt(cg * (1 - cg) / nrow(effective)),
       fcr_futile = fcf, fcr_futile_mcse = sqrt(fcf * (1 - fcf) / nrow(futile)),
       fcr_effective = fce, fcr_effective_mcse = sqrt(fce * (1 - fce) / nrow(effective)),
       fcr = max(fcf, fce), fcr_mcse = NA_real_,
       en_futile = mean(futile$terminal_look),
       en_futile_mcse = if (nrow(futile) > 1L) sd(futile$terminal_look) / sqrt(nrow(futile)) else NA_real_,
       en_effective = mean(effective$terminal_look),
       en_effective_mcse = if (nrow(effective) > 1L) sd(effective$terminal_look) / sqrt(nrow(effective)) else NA_real_)
}

candidate_rows <- list(); decision_rows <- list(); validation_rows <- list()
tail_rows <- list(); selected_rows <- list()
for (si in seq_len(nrow(settings))) {
  s <- settings[si, , drop = FALSE]; id <- s$config_id[[1]]
  n <- as.integer(s$max_subjects); looks <- parse_ints(s$looks)
  arms <- as.integer(strsplit(s$arm_assignments, "", fixed = TRUE)[[1]])
  if (length(arms) != n || any(!arms %in% c(0L, 1L)) || tail(looks, 1) != n ||
      any(diff(looks) <= 0) || any(looks < 1 | looks > n)) stop(paste("bad schedule:", id))
  diff_f <- as.numeric(s$futile_treatment_mean) - as.numeric(s$futile_control_mean)
  diff_e <- as.numeric(s$effective_treatment_mean) - as.numeric(s$effective_control_mean)
  if (!(diff_f < diff_e && diff_e >= as.numeric(s$theta_cmv))) stop(paste("bad truth pair:", id))
  cand <- grid[grid$config_id == id, , drop = FALSE]
  cand$candidate_index <- as.integer(cand$candidate_index)
  if (!nrow(cand) || !identical(cand$candidate_index, seq_len(nrow(cand)) - 1L))
    stop(paste("candidate indices must be consecutive zero-based input order:", id))
  for (nm in c("lambda_lrv", "lambda_cmv", "gamma_lrv", "gamma_cmv")) cand[[nm]] <- as.numeric(cand[[nm]])
  if (any(cand$lambda_lrv <= 0 | cand$lambda_lrv >= 1) ||
      any(cand$lambda_cmv <= 0 | cand$lambda_cmv >= 1) ||
      any(cand$gamma_lrv < 0 | cand$gamma_lrv > 1) || any(cand$gamma_cmv < 0 | cand$gamma_cmv > 1))
    stop(paste("invalid candidate:", id))
  cache <- list()
  for (phase in c("calibration", "validation")) for (scenario in c("futile", "effective")) {
    dat <- paths[paths$config_id == id & paths$phase == phase & paths$scenario == scenario, , drop = FALSE]
    if (!nrow(dat)) stop(paste("missing paths:", id, phase, scenario))
    dat$trial <- as.integer(dat$trial); dat$patient <- as.integer(dat$patient)
    tails <- precompute_tails(s, phase, scenario, dat, arms, looks)
    cache[[paste(phase, scenario, sep = "_")]] <- tails
    tail_rows[[length(tail_rows) + 1L]] <- tails
  }
  for (phase in c("calibration", "validation")) {
    nf <- length(unique(cache[[paste(phase, "futile", sep = "_")]]$trial))
    ne <- length(unique(cache[[paste(phase, "effective", sep = "_")]]$trial))
    if (nf != ne) stop(paste("truth scenarios must have equal trial counts:", id, phase))
  }
  calibration_candidate <- vector("list", nrow(cand))
  per_config_metrics <- vector("list", nrow(cand))
  for (ci in seq_len(nrow(cand))) {
    p <- unlist(cand[ci, c("lambda_lrv", "lambda_cmv", "gamma_lrv", "gamma_cmv")], use.names = TRUE)
    fit <- list()
    for (scenario in c("futile", "effective")) {
      tails <- cache[[paste("calibration", scenario, sep = "_")]]
      term <- terminalize(tails, p, s, looks); sm <- summarize_terminal(term, looks)
      fit[[scenario]] <- list(terminals = term, summary = sm)
      for (li in seq_along(looks)) for (decision in colnames(sm$probability)) {
        decision_rows[[length(decision_rows) + 1L]] <- data.frame(
          config_id = id, phase = "calibration", scenario = scenario,
          candidate_index = cand$candidate_index[[ci]], look = looks[[li]], decision = decision,
          count = sm$count[li, decision], probability = sm$probability[li, decision],
          mcse = sm$mcse[li, decision],
          sample_size_probability = sm$sample_size_probability[[li]], stringsAsFactors = FALSE)
      }
    }
    m <- metric_summary(fit$futile$terminals, fit$effective$terminals)
    fcl_raw <- s$false_consider_limit[[1]]; fcl <- if (is.na(fcl_raw) || fcl_raw == "NA" || fcl_raw == "") NA_real_ else as.numeric(fcl_raw)
    feasible <- m$fgr <= as.numeric(s$false_go_limit) && m$fngr <= as.numeric(s$false_no_go_limit) &&
      (is.na(fcl) || m$fcr <= fcl)
    row <- data.frame(config_id = id, candidate_index = cand$candidate_index[[ci]],
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
    candidate_rows[[length(candidate_rows) + 1L]] <- row
    per_config_metrics[[ci]] <- row
    calibration_candidate[[ci]] <- fit
  }
  cm <- do.call(rbind, per_config_metrics); eligible <- which(cm$feasible); sel <- NA_integer_
  if (length(eligible)) {
    if (s$objective[[1]] == "cgr") {
      ord <- order(-cm$correct_go_rate[eligible], cm$expected_n_futile[eligible], eligible)
    } else if (s$objective[[1]] == "ess_futile") {
      ord <- order(cm$expected_n_futile[eligible], -cm$correct_go_rate[eligible], eligible)
    } else {
      stop(paste("unknown objective:", s$objective[[1]]))
    }
    sel <- cm$candidate_index[eligible[ord[[1]]]]
  }
  if (is.na(sel)) {
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
      validation_expected_n_futile = NA_real_, validation_expected_n_futile_mcse = NA_real_,
      validation_expected_n_effective = NA_real_, validation_expected_n_effective_mcse = NA_real_,
      validation_feasible = NA, stringsAsFactors = FALSE)
    next
  }
  p <- unlist(cand[cand$candidate_index == sel, c("lambda_lrv", "lambda_cmv", "gamma_lrv", "gamma_cmv")][1, ], use.names = TRUE)
  val_fit <- list()
  for (scenario in c("futile", "effective")) {
    term <- terminalize(cache[[paste("validation", scenario, sep = "_")]], p, s, looks)
    sm <- summarize_terminal(term, looks); val_fit[[scenario]] <- list(terminals = term, summary = sm)
    for (li in seq_along(looks)) for (decision in colnames(sm$probability)) {
      validation_rows[[length(validation_rows) + 1L]] <- data.frame(
        config_id = id, scenario = scenario, candidate_index = sel, look = looks[[li]],
        decision = decision, count = sm$count[li, decision],
        probability = sm$probability[li, decision], mcse = sm$mcse[li, decision],
        sample_size_probability = sm$sample_size_probability[[li]], stringsAsFactors = FALSE)
    }
  }
  vm <- metric_summary(val_fit$futile$terminals, val_fit$effective$terminals)
  vf <- vm$fgr <= as.numeric(s$false_go_limit) && vm$fngr <= as.numeric(s$false_no_go_limit) &&
    (is.na(fcl) || vm$fcr <= fcl)
  selected_rows[[length(selected_rows) + 1L]] <- data.frame(
    config_id = id, objective = s$objective[[1]], selected_index = sel,
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
    validation_false_consider_effective_mcse = vm$fcr_effective_mcse,
    validation_expected_n_futile = vm$en_futile,
    validation_expected_n_futile_mcse = vm$en_futile_mcse,
    validation_expected_n_effective = vm$en_effective,
    validation_expected_n_effective_mcse = vm$en_effective_mcse,
    validation_feasible = vf, stringsAsFactors = FALSE)
}
write.csv(do.call(rbind, candidate_rows), paste0(prefix, "candidate-metrics.csv"), row.names = FALSE)
write.csv(do.call(rbind, decision_rows), paste0(prefix, "calibration-decisions.csv"), row.names = FALSE)
write.csv(do.call(rbind, selected_rows), paste0(prefix, "selected-validation.csv"), row.names = FALSE)
write.csv(do.call(rbind, validation_rows), paste0(prefix, "validation-decisions.csv"), row.names = FALSE)
write.csv(do.call(rbind, tail_rows), paste0(prefix, "posterior-tail-cache.csv"), row.names = FALSE)
