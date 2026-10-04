#!/usr/bin/env Rscript

# Independent base-R reference for categorical BOP2-DC posterior probabilities
# and decision composition. Equations follow cached paper §§2.1.4, 2.2 and 2.4.
# This script intentionally does not invoke Python implementation code.

args <- commandArgs(trailingOnly = TRUE)
out_dir <- if (length(args)) args[[1]] else getwd()
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

indicators5 <- rbind(
  endpoint_a = c(1, 1, 0, 0, 1),
  endpoint_b = c(0, 1, 1, 0, 0),
  endpoint_c = c(0, 0, 1, 1, 1)
)
cases <- list(
  single_any = list(
    arm = "single", combination = "any", indicators = indicators5,
    direction = c("greater", "less", "greater"),
    lrv = c(.30, .50, .40), cmv = c(.55, .35, .70),
    prior_e = c(.4, .8, .3, .6, .9), count_e = c(3, 1, 1, 1, 2),
    count_interim_e = c(2, 1, 0, 1, 1), count_interim_c = NULL,
    prior_c = NULL, count_c = NULL,
    lambda_lrv = .70, lambda_cmv = .55, gamma_lrv = .5, gamma_cmv = .4,
    max_n = 8, interim_n = 5
  ),
  randomized_all = list(
    arm = "randomized", combination = "all", indicators = indicators5,
    direction = c("greater", "less", "greater"),
    lrv = c(.05, -.10, .10), cmv = c(.15, -.20, .20),
    prior_e = c(.7, .3, .5, .2, .4), count_e = c(2, 0, 1, 0, 1),
    count_interim_e = c(1, 0, 1, 0, 0),
    prior_c = c(.2, .6, .4, .5, .3), count_c = c(0, 2, 0, 1, 1),
    count_interim_c = c(0, 1, 0, 0, 1),
    lambda_lrv = .70, lambda_cmv = .55, gamma_lrv = .5, gamma_cmv = .4,
    max_n = 8, interim_n = 4
  ),
  two_endpoint_any = list(
    arm = "single", combination = "any",
    indicators = rbind(endpoint_a = c(1, 1, 0, 0), endpoint_b = c(1, 0, 1, 0)),
    direction = c("greater", "greater"), lrv = c(.25, .40), cmv = c(.45, .60),
    prior_e = c(.2, .3, .4, .5), count_e = c(1, 2, 0, 1),
    count_interim_e = c(1, 0, 0, 1), count_interim_c = NULL,
    prior_c = NULL, count_c = NULL,
    lambda_lrv = .65, lambda_cmv = .45, gamma_lrv = .5, gamma_cmv = .5,
    max_n = 4, interim_n = 2
  ),
  prior_only_equal_single = list(
    arm = "single", combination = "all", indicators = rbind(endpoint_a = c(1, 0)),
    direction = "greater", lrv = .4, cmv = .5,
    prior_e = c(1, 1), count_e = c(0, 0), prior_c = NULL, count_c = NULL,
    lambda_lrv = .6, lambda_cmv = .5, gamma_lrv = 0, gamma_cmv = 0,
    max_n = 4, interim_n = 0
  ),
  prior_only_equal_randomized = list(
    arm = "randomized", combination = "all", indicators = rbind(endpoint_a = c(1, 0)),
    direction = "greater", lrv = 0, cmv = .2,
    prior_e = c(1, 1), count_e = c(0, 0), prior_c = c(1, 1), count_c = c(0, 0),
    lambda_lrv = .5, lambda_cmv = .32, gamma_lrv = 0, gamma_cmv = 0,
    max_n = 4, interim_n = 0
  )
)

beta_shapes <- function(prior, counts, indicator) {
  post <- prior + counts
  a <- sum(post[indicator == 1])
  b <- sum(post[indicator == 0])
  c(a = a, b = b)
}

beta_tail <- function(shape, cutoff, direction) {
  if (direction == "greater") {
    pbeta(cutoff, shape[["a"]], shape[["b"]], lower.tail = FALSE)
  } else {
    pbeta(cutoff, shape[["a"]], shape[["b"]], lower.tail = TRUE)
  }
}

beta_difference_tail <- function(e, c, margin, direction) {
  # Integrate f_E(x) * P(C < x-margin) for a greater-direction event,
  # and f_E(x) * P(C > x-margin) directly for a less-direction event.
  conditional <- function(x) {
    z <- x - margin
    vapply(z, function(value) {
      if (direction == "greater") {
        if (value <= 0) return(0)
        if (value >= 1) return(1)
        return(pbeta(value, c[["a"]], c[["b"]], lower.tail = TRUE))
      }
      if (value <= 0) return(1)
      if (value >= 1) return(0)
      pbeta(value, c[["a"]], c[["b"]], lower.tail = FALSE)
    }, numeric(1))
  }
  integrate(
    function(x) dbeta(x, e[["a"]], e[["b"]]) * conditional(x),
    lower = 0, upper = 1, rel.tol = 1e-11, abs.tol = 1e-12, subdivisions = 300L
  )$value
}

endpoint_action <- function(prob, lambda_lrv, lambda_cmv, interim = FALSE) {
  if (interim) {
    if (all(prob < c(lambda_lrv, lambda_cmv))) return("no_go")
    return("continue")
  }
  if (all(prob > c(lambda_lrv, lambda_cmv))) return("go")
  if (all(prob < c(lambda_lrv, lambda_cmv))) return("no_go")
  "consider"
}

combine_action <- function(actions, combination, interim = FALSE) {
  if (interim) {
    rejected <- if (combination == "any") all(actions == "no_go") else any(actions == "no_go")
    return(if (rejected) "stop_no_go" else "continue")
  }
  if (combination == "any") {
    if (any(actions == "go")) return("final_go")
    if (all(actions == "no_go")) return("final_no_go")
    return("final_consider")
  }
  if (all(actions == "go")) return("final_go")
  if (any(actions == "no_go")) return("final_no_go")
  "final_consider"
}

calc_tails <- function(z, counts_e, counts_c = NULL) {
  m <- nrow(z$indicators)
  tails <- matrix(NA_real_, nrow = m, ncol = 2)
  shapes_e <- vector("list", m)
  shapes_c <- vector("list", m)
  for (j in seq_len(m)) {
    se <- beta_shapes(z$prior_e, counts_e, z$indicators[j, ])
    shapes_e[[j]] <- se
    if (z$arm == "single") {
      tails[j, ] <- c(
        beta_tail(se, z$lrv[j], z$direction[j]),
        beta_tail(se, z$cmv[j], z$direction[j])
      )
    } else {
      sc <- beta_shapes(z$prior_c, counts_c, z$indicators[j, ])
      shapes_c[[j]] <- sc
      tails[j, ] <- c(
        beta_difference_tail(se, sc, z$lrv[j], z$direction[j]),
        beta_difference_tail(se, sc, z$cmv[j], z$direction[j])
      )
    }
  }
  list(tails = tails, shapes_e = shapes_e, shapes_c = shapes_c)
}

prob_rows <- list()
decision_rows <- list()
replay_rows <- list()
config_rows <- list()
for (case_name in names(cases)) {
  z <- cases[[case_name]]
  m <- nrow(z$indicators)
  endpoint_names <- rownames(z$indicators)
  if (is.null(endpoint_names)) endpoint_names <- paste0("endpoint_", seq_len(m))
  calculated <- calc_tails(z, z$count_e, z$count_c)
  tail_matrix <- calculated$tails
  for (j in seq_len(m)) {
    se <- calculated$shapes_e[[j]]
    for (q in 1:2) {
      criterion <- c("LRV", "CMV")[[q]]
      prob_rows[[length(prob_rows) + 1L]] <- data.frame(
        case = case_name, stage = "final", arm = z$arm, endpoint = endpoint_names[[j]],
        criterion = criterion, direction = z$direction[[j]],
        lrv = z$lrv[[j]], cmv = z$cmv[[j]],
        posterior_a_experimental = se[["a"]], posterior_b_experimental = se[["b"]],
        posterior_a_control = if (z$arm == "randomized") calculated$shapes_c[[j]][["a"]] else NA_real_,
        posterior_b_control = if (z$arm == "randomized") calculated$shapes_c[[j]][["b"]] else NA_real_,
        probability = tail_matrix[j, q], stringsAsFactors = FALSE
      )
    }
  }
  endpoint_final <- vapply(seq_len(m), function(j) {
    endpoint_action(tail_matrix[j, ], z$lambda_lrv, z$lambda_cmv)
  }, character(1))
  n_interim <- z$interim_n
  if (n_interim > 0 && z$max_n > 0) {
    l1 <- z$lambda_lrv * (n_interim / z$max_n)^z$gamma_lrv
    l2 <- z$lambda_cmv * (n_interim / z$max_n)^z$gamma_cmv
  } else {
    l1 <- z$lambda_lrv
    l2 <- z$lambda_cmv
  }
    interim_e <- if (is.null(z$count_interim_e)) z$count_e else z$count_interim_e
  interim_c <- if (is.null(z$count_interim_c)) z$count_c else z$count_interim_c
  calculated_interim <- calc_tails(z, interim_e, interim_c)
  interim_tails <- calculated_interim$tails
  for (j in seq_len(m)) {
    se <- calculated_interim$shapes_e[[j]]
    for (q in 1:2) {
      criterion <- c("LRV", "CMV")[[q]]
      prob_rows[[length(prob_rows) + 1L]] <- data.frame(
        case = case_name, stage = "interim", arm = z$arm, endpoint = endpoint_names[[j]],
        criterion = criterion, direction = z$direction[[j]],
        lrv = z$lrv[[j]], cmv = z$cmv[[j]],
        posterior_a_experimental = se[["a"]], posterior_b_experimental = se[["b"]],
        posterior_a_control = if (z$arm == "randomized") calculated_interim$shapes_c[[j]][["a"]] else NA_real_,
        posterior_b_control = if (z$arm == "randomized") calculated_interim$shapes_c[[j]][["b"]] else NA_real_,
        probability = interim_tails[j, q], stringsAsFactors = FALSE
      )
    }
  }
  config_rows[[length(config_rows) + 1L]] <- data.frame(
    case = case_name, arm = z$arm, combination = z$combination,
    indicators = paste(apply(z$indicators, 1, paste, collapse = ""), collapse = "|"),
    direction = paste(z$direction, collapse = ";"),
    lrv = paste(z$lrv, collapse = ";"), cmv = paste(z$cmv, collapse = ";"),
    prior_experimental = paste(z$prior_e, collapse = ";"),
    counts_experimental_final = paste(z$count_e, collapse = ";"),
    counts_experimental_interim = paste(interim_e, collapse = ";"),
    prior_control = if (z$arm == "randomized") paste(z$prior_c, collapse = ";") else "",
    counts_control_final = if (z$arm == "randomized") paste(z$count_c, collapse = ";") else "",
    counts_control_interim = if (z$arm == "randomized") paste(interim_c, collapse = ";") else "",
    lambda_lrv = z$lambda_lrv, lambda_cmv = z$lambda_cmv,
    gamma_lrv = z$gamma_lrv, gamma_cmv = z$gamma_cmv,
    looks = if (case_name == "randomized_all") "2;4;6;8" else if (z$interim_n > 0) paste(unique(c(z$interim_n, z$max_n)), collapse = ";") else as.character(z$max_n),
    arm_assignments = if (z$arm == "randomized") paste(c(0, 0, 1, 1, 0, 0, 1, 1)[seq_len(z$max_n)], collapse = ";") else "",
    max_n = z$max_n, interim_n = z$interim_n,
    stringsAsFactors = FALSE
  )
  endpoint_interim <- vapply(seq_len(m), function(j) {
    endpoint_action(interim_tails[j, ], l1, l2, interim = TRUE)
  }, character(1))
  decision_rows[[length(decision_rows) + 1L]] <- data.frame(
    case = case_name, composition = z$combination,
    endpoint_decisions_final = paste(endpoint_final, collapse = ";"),
    action_final = combine_action(endpoint_final, z$combination),
    interim_n = n_interim, interim_lambda_lrv = l1, interim_lambda_cmv = l2,
    endpoint_decisions_interim = paste(endpoint_interim, collapse = ";"),
    action_interim = combine_action(endpoint_interim, z$combination, interim = TRUE),
    stringsAsFactors = FALSE
  )
}

write.csv(do.call(rbind, prob_rows), file.path(out_dir, "bop2-dc-categorical-reference.csv"), row.names = FALSE)
write.csv(do.call(rbind, decision_rows), file.path(out_dir, "bop2-dc-categorical-decisions.csv"), row.names = FALSE)
write.csv(do.call(rbind, config_rows), file.path(out_dir, "bop2-dc-categorical-config.csv"), row.names = FALSE)

# Fixed-arm randomized replay tape: 0=control, 1=experimental; category labels
# are one-based and follow the indicator-matrix columns.
replay_case <- cases$randomized_all
arm_tape <- c(0, 0, 1, 1, 0, 0, 1, 1)
category_tape <- c(2, 5, 1, 3, 2, 4, 1, 5)
for (n in c(2L, 4L, 6L, 8L)) {
  counts_e <- tabulate(category_tape[seq_len(n)][arm_tape[seq_len(n)] == 1], nbins = 5L)
  counts_c <- tabulate(category_tape[seq_len(n)][arm_tape[seq_len(n)] == 0], nbins = 5L)
  current <- replay_case
  current$count_e <- counts_e
  current$count_c <- counts_c
  m <- nrow(current$indicators)
  probs <- matrix(NA_real_, m, 2)
  endpoint_names <- rownames(current$indicators)
  for (j in seq_len(m)) {
    se <- beta_shapes(current$prior_e, current$count_e, current$indicators[j, ])
    sc <- beta_shapes(current$prior_c, current$count_c, current$indicators[j, ])
    probs[j, ] <- c(
      beta_difference_tail(se, sc, current$lrv[j], current$direction[j]),
      beta_difference_tail(se, sc, current$cmv[j], current$direction[j])
    )
  }
  if (n < current$max_n) {
    l1 <- current$lambda_lrv * (n / current$max_n)^current$gamma_lrv
    l2 <- current$lambda_cmv * (n / current$max_n)^current$gamma_cmv
    acts <- vapply(seq_len(m), function(j) endpoint_action(probs[j, ], l1, l2, interim = TRUE), character(1))
    combined <- combine_action(acts, current$combination, interim = TRUE)
  } else {
    acts <- vapply(seq_len(m), function(j) endpoint_action(probs[j, ], current$lambda_lrv, current$lambda_cmv), character(1))
    combined <- combine_action(acts, current$combination)
  }
  replay_rows[[length(replay_rows) + 1L]] <- data.frame(
    look_n = n, count_control = paste(counts_c, collapse = ";"),
    count_experimental = paste(counts_e, collapse = ";"),
    posterior_probabilities = paste(formatC(as.vector(t(probs)), digits = 16, format = "fg"), collapse = ";"),
    endpoint_actions = paste(acts, collapse = ";"),
    action = combined,
    stringsAsFactors = FALSE
  )
}
write.csv(do.call(rbind, replay_rows), file.path(out_dir, "bop2-dc-categorical-replay.csv"), row.names = FALSE)
