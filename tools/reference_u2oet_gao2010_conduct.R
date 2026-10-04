#!/usr/bin/env Rscript

# Independent, base-R decision reference for the 2010 GAO conduct rule.
# The input fixture is a synthetic posterior tensor, not a fitted model.

args <- commandArgs(trailingOnly = TRUE)
out_dir <- if (length(args) >= 1L) args[[1L]] else "tests/fixtures"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

utility <- matrix(c(0, 80, -100, 60), nrow = 2L, ncol = 2L)
cases <- list(
  interim = list(limit = 0.25, stop = 0.80, current = c(2L, 2L),
                 tried = matrix(c(2L, 2L, 3L, 1L), ncol = 2L, byrow = TRUE),
                 pattern = "interim"),
  stop_boundary = list(limit = 0.25, stop = 0.75, current = c(2L, 2L),
                       tried = matrix(c(2L, 2L), ncol = 2L),
                       pattern = "boundary"),
  global_stop = list(limit = 0.25, stop = 0.80, current = c(2L, 2L),
                     tried = matrix(c(2L, 2L), ncol = 2L),
                     pattern = "unsafe"),
  final = list(limit = 0.25, stop = 0.80, current = NULL,
               tried = matrix(integer(), ncol = 2L), pattern = "final")
)

draw_rows <- list()
summary_rows <- list()
decision_rows <- list()
config_rows <- list()
row_id <- 0L
summary_id <- 0L
decision_id <- 0L

for (case_name in names(cases)) {
  cfg <- cases[[case_name]]
  config_rows[[length(config_rows) + 1L]] <- data.frame(
    case = case_name, toxicity_limit = cfg$limit,
    stopping_probability = cfg$stop,
    current_dose1 = if (is.null(cfg$current)) NA_integer_ else cfg$current[[1L]],
    current_dose2 = if (is.null(cfg$current)) NA_integer_ else cfg$current[[2L]],
    treated_pairs = if (nrow(cfg$tried) == 0L) "" else
      paste(apply(cfg$tried, 1L, paste, collapse = ":"), collapse = ";"),
    stringsAsFactors = FALSE
  )
  tensor <- array(NA_real_, dim = c(2L, 8L, 3L, 3L, 2L, 2L))
  for (ch in seq_len(2L)) for (dr in seq_len(8L)) {
    ordinal <- (ch - 1L) * 8L + dr
    for (d1 in seq_len(3L)) for (d2 in seq_len(3L)) {
      if (cfg$pattern == "boundary") {
        p_tox <- if (ordinal <= 12L) 0.40 else 0.20
      } else if (cfg$pattern == "unsafe") {
        p_tox <- 0.55 + 0.01 * ((d1 + d2 + ordinal) %% 3L)
      } else {
        # Cell (1,1) is exactly on the strict toxicity threshold. Other
        # cells vary by dose and draw so the global and local summaries differ.
        p_tox <- if (d1 == 1L && d2 == 1L) 0.25 else
          0.18 + 0.055 * (d1 + d2 - 2L) + 0.01 * (ordinal %% 3L)
      }
      p_eff <- min(0.92, 0.10 + 0.11 * d1 + 0.12 * d2 + 0.005 * (ordinal %% 4L))
      p_eff_tox <- p_eff * p_tox
      probs <- matrix(c(1 - p_eff - p_tox + p_eff_tox,
                        p_eff - p_eff_tox,
                        p_tox - p_eff_tox,
                        p_eff_tox), nrow = 2L, ncol = 2L)
      tensor[ch, dr, d1, d2, , ] <- probs
      for (eff in 0:1) for (tox in 0:1) {
        row_id <- row_id + 1L
        draw_rows[[row_id]] <- data.frame(
          case = case_name, chain = ch, draw = dr, dose1 = d1, dose2 = d2,
          efficacy = eff, toxicity = tox, probability = probs[eff + 1L, tox + 1L]
        )
      }
    }
  }

  n_draw <- dim(tensor)[1L] * dim(tensor)[2L]
  utilities <- matrix(NA_real_, 3L, 3L)
  utility_sd <- matrix(NA_real_, 3L, 3L)
  severe_prob <- matrix(NA_real_, 3L, 3L)
  for (d1 in 1:3) for (d2 in 1:3) {
    u <- numeric(n_draw)
    severe <- logical(n_draw)
    severe_probability <- numeric(n_draw)
    index <- 0L
    for (ch in 1:2) for (dr in 1:8) {
      index <- index + 1L
      joint <- tensor[ch, dr, d1, d2, , ]
      u[[index]] <- sum(joint * utility)
      severe_probability[[index]] <- sum(joint[, 2L])
      severe[[index]] <- severe_probability[[index]] > cfg$limit
    }
    utilities[d1, d2] <- mean(u)
    utility_sd[d1, d2] <- sd(u)
    severe_prob[d1, d2] <- mean(severe)
    summary_id <- summary_id + 1L
    chain_mcse <- function(values) {
      values <- matrix(values, nrow = 2L, byrow = TRUE)
      batch_size <- max(2L, floor(sqrt(ncol(values))))
      batches <- ncol(values) %/% batch_size
      means <- matrix(NA_real_, nrow = nrow(values) * batches, ncol = 1L)
      offset <- 0L
      for (chain in seq_len(nrow(values))) for (batch in seq_len(batches)) {
        ix <- ((batch - 1L) * batch_size + 1L):(batch * batch_size)
        offset <- offset + 1L
        means[offset, 1L] <- mean(values[chain, ix])
      }
      sd(means[, 1L]) / sqrt(length(means))
    }
    summary_rows[[summary_id]] <- data.frame(
      case = case_name, dose1 = d1, dose2 = d2,
      mean_utility = mean(u), utility_sd = sd(u),
      utility_mcse = chain_mcse(u),
      mean_severe_toxicity = mean(severe_probability),
      severe_toxicity_mcse = chain_mcse(severe_probability),
      severe_exceedance_probability = mean(severe),
      exceedance_probability_mcse = chain_mcse(as.numeric(severe))
    )
  }

  globally_stopped <- min(severe_prob) > cfg$stop
  eligible <- matrix(FALSE, 3L, 3L)
  if (is.null(cfg$current)) {
    eligible[,] <- TRUE
    action <- if (globally_stopped) "stop_all_dose_pairs_too_toxic" else "final_select"
  } else {
    for (d1 in 1:3) for (d2 in 1:3) {
      tried <- nrow(cfg$tried) > 0L && any(cfg$tried[, 1L] == d1 & cfg$tried[, 2L] == d2)
      deescalation <- d1 <= cfg$current[[1L]] && d2 <= cfg$current[[2L]]
      upper_neighbor <- (d1 == cfg$current[[1L]] + 1L && d2 == cfg$current[[2L]]) ||
        (d1 == cfg$current[[1L]] && d2 == cfg$current[[2L]] + 1L) ||
        (d1 == cfg$current[[1L]] + 1L && d2 == cfg$current[[2L]] + 1L)
      eligible[d1, d2] <- tried || deescalation || upper_neighbor
    }
    action <- if (globally_stopped) "stop_all_dose_pairs_too_toxic" else "assign_next_cohort"
  }
  selected <- NA_integer_
  if (!globally_stopped && any(eligible)) {
    candidate_values <- utilities
    candidate_values[!eligible] <- -Inf
    # which.max uses column-major order; explicit lexicographic order is dose1,
    # then dose2, to make the independent fixture deterministic.
    candidates <- which(eligible, arr.ind = TRUE)
    candidates <- candidates[order(candidates[, 1L], candidates[, 2L]), , drop = FALSE]
    values <- utilities[candidates]
    chosen <- candidates[which.max(values), ]
    selected <- (chosen[[1L]] - 1L) * 3L + chosen[[2L]]
  }
  decision_id <- decision_id + 1L
  decision_rows[[decision_id]] <- data.frame(
    case = case_name,
    stopped = globally_stopped,
    minimum_exceedance_probability = min(severe_prob),
    action = action,
    selected_dose1 = if (is.na(selected)) NA_integer_ else (selected - 1L) %/% 3L + 1L,
    selected_dose2 = if (is.na(selected)) NA_integer_ else (selected - 1L) %% 3L + 1L,
    eligible_mask_dose1_major = paste(as.integer(as.vector(t(eligible))), collapse = ""),
    stringsAsFactors = FALSE
  )
}

write.csv(do.call(rbind, draw_rows), file.path(out_dir, "u2oet-gao2010-conduct-draws.csv"), row.names = FALSE)
write.csv(do.call(rbind, summary_rows), file.path(out_dir, "u2oet-gao2010-conduct-summaries.csv"), row.names = FALSE)
write.csv(do.call(rbind, decision_rows), file.path(out_dir, "u2oet-gao2010-conduct-decisions.csv"), row.names = FALSE)
write.csv(do.call(rbind, config_rows), file.path(out_dir, "u2oet-gao2010-conduct-config.csv"), row.names = FALSE)
