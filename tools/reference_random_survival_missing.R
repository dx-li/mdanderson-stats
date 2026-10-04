#!/usr/bin/env Rscript

# Deterministic source-rule ledgers for RF-SRC 3.2.2 missing-value handling.
# This is not the native forest engine and uses explicit U(0,1) tapes because
# NumPy and RF-SRC random streams are intentionally different.

args <- commandArgs(trailingOnly = TRUE)
out_dir <- if (length(args) >= 1L) args[[1L]] else
  file.path("tests", "fixtures")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

sample_value <- function(values, u) {
  stopifnot(length(values) > 0L, is.finite(u), u > 0, u <= 1)
  values[[ceiling(u * length(values))]]
}

mode_value <- function(values, u) {
  stopifnot(length(values) > 0L, is.finite(u), u > 0, u <= 1)
  levels <- sort(unique(values))
  counts <- tabulate(match(values, levels), nbins = length(levels))
  winners <- levels[counts == max(counts)]
  winners[[ceiling(u * length(winners))]]
}

nearest_master_time <- function(value, master, u, eps = 1e-9) {
  stopifnot(length(master) > 0L, all(diff(master) > 0),
            value >= master[[1L]] - eps, value <= master[[length(master)]] + eps)
  upper <- which(master >= value)[[1L]]
  if (upper == 1L) return(master[[1L]])
  lower_distance <- value - master[[upper - 1L]]
  upper_distance <- master[[upper]] - value
  if (lower_distance < upper_distance) return(master[[upper - 1L]])
  if (abs(lower_distance - upper_distance) < eps && u <= 0.5) {
    return(master[[upper - 1L]])
  }
  master[[upper]]
}

write_rows <- function(name, rows) {
  write.csv(do.call(rbind, rows), file.path(out_dir, name), row.names = FALSE,
            na = "NA", quote = TRUE)
}

# Bootstrap multiplicities belong to the donor pool: row 1 occurs twice.
# Missing recipients include two OOB cases; only the listed in-bag values can
# be sampled. Source getSampleValue uses ceil(U * pool_size), one-based.
inbag_ids <- c(1L, 1L, 2L, 3L)
inbag_x <- c(1, 1, 9, 100)
donor_tape <- c(0.125, 1.0)
recipient_ids <- c(4L, 5L)
imputed <- vapply(donor_tape, function(u) sample_value(inbag_x, u), numeric(1))
donor_rows <- lapply(seq_along(recipient_ids), function(j) data.frame(
  case = "bootstrap_multiplicity_oob_recipients",
  recipient_id = recipient_ids[[j]],
  recipient_is_inbag = FALSE,
  donor_ids = paste(inbag_ids, collapse = ";"),
  donor_values = paste(inbag_x, collapse = ";"),
  donor_uniform = donor_tape[[j]],
  one_based_donor_index = ceiling(donor_tape[[j]] * length(inbag_ids)),
  imputed_value = imputed[[j]], stringsAsFactors = FALSE
))

# First-pass split eligibility is candidate-column specific. It excludes rows
# with any originally missing Y and rows missing this candidate X, even though
# all-row X imputations remain available for routing. Row 5 is a large missing-X
# outlier whose filled value would corrupt candidate cutpoints if included.
split_x_original <- c(1, 2, 3, 4, NA, 6)
split_y_missing <- c(FALSE, FALSE, TRUE, FALSE, FALSE, FALSE)
split_x_imputed <- c(1, 2, 3, 4, 5, 6)
eligible <- !split_y_missing & !is.na(split_x_original)
split_rows <- data.frame(
  row_id = seq_along(split_x_original),
  original_x_missing = is.na(split_x_original),
  original_y_missing = split_y_missing,
  imputed_x_for_routing = split_x_imputed,
  eligible_for_first_pass_candidate_and_score = eligible,
  stringsAsFactors = FALSE
)
split_rows$candidate_x_values <- ifelse(eligible, split_x_original, NA_real_)
split_candidates <- sort(unique(split_x_original[eligible]))
split_candidates <- split_candidates[-length(split_candidates)]

# A child with no originally observed local donors retains the parent's
# completion. At terminal imputation RF-SRC may then summarize the completed
# values of originally missing in-bag rows; tied categorical modes use a random
# tie choice. The U tape is explicit, not an R/native RNG claim.
parent_completed <- c(2, 8)
terminal_tie_u <- c(0.25, 0.75)
terminal_rows <- lapply(seq_along(terminal_tie_u), function(j) data.frame(
  case = "terminal_mode_tie_from_parent_completed_inbag",
  node_observed_donor_count = 0L,
  parent_completed_inbag_values = paste(parent_completed, collapse = ";"),
  tie_uniform = terminal_tie_u[[j]],
  selected_mode = mode_value(parent_completed, terminal_tie_u[[j]]),
  stringsAsFactors = FALSE
))

# Terminal outcome completion uses observed in-bag outcomes at that node;
# time uses mean then getNearestMasterTime, status is categorical mode.
time_donors <- c(1, 2, 3, 4)
time_mean <- mean(time_donors)
master <- c(1, 2, 3, 4)
time_tie_u <- c(0.25, 0.75)
time_rows <- lapply(seq_along(time_tie_u), function(j) data.frame(
  case = "terminal_time_mean_snap_asymmetric_tie",
  unsnapped_mean = time_mean,
  observed_inbag_times = paste(time_donors, collapse = ";"),
  mean_time = time_mean,
  lower_distance = time_mean - 2,
  upper_distance = 3 - time_mean,
  tie_uniform = time_tie_u[[j]],
  snapped_time = nearest_master_time(time_mean, master, time_tie_u[[j]]),
  stringsAsFactors = FALSE
))
near_values <- c(2.4999999996, 2.5000000004)
near_tapes <- c(0.75, 0.25)
near_snap_rows <- lapply(seq_along(near_values), function(j) data.frame(
  case = c("near_left_prefers_lower_regardless_of_tape",
           "near_right_tie_band_uses_tape")[[j]],
  value = near_values[[j]],
  lower_distance = near_values[[j]] - 2,
  upper_distance = 3 - near_values[[j]],
  uniform = near_tapes[[j]],
  snapped_time = nearest_master_time(near_values[[j]], master, near_tapes[[j]]),
  stringsAsFactors = FALSE
))
completed_time <- c(1, 2, 3, 4, nearest_master_time(time_mean, master, 0.25))
completed_status <- c(1, 0, 1, 0, mode_value(c(0, 1, 0, 1), 0.75))
original_event_grid <- sort(unique(time_donors[c(1, 3)]))
grid_rows <- data.frame(
  original_complete_event_time_grid = paste(original_event_grid, collapse = ";"),
  terminal_completed_event_time_grid = paste(sort(unique(completed_time[completed_status == 1])),
                                              collapse = ";"),
  master_snap_grid_including_censored_times = paste(master, collapse = ";"),
  stringsAsFactors = FALSE
)
event_grid <- sort(unique(completed_time[completed_status == 1]))
risk <- vapply(event_grid, function(value) sum(completed_time >= value), integer(1))
events <- vapply(event_grid, function(value) {
  sum(completed_time == value & completed_status == 1)
}, integer(1))
terminal_count_rows <- lapply(seq_along(event_grid), function(j) data.frame(
  case = "completed_terminal_km_counts",
  event_time = event_grid[[j]],
  at_risk = risk[[j]],
  events = events[[j]],
  hazard_increment = events[[j]] / risk[[j]],
  survival_after_step = prod(1 - events[seq_len(j)] / risk[seq_len(j)]),
  original_missing_outcomes_excluded_from_split = 1L,
  completed_outcomes_in_terminal_summary = 1L,
  stringsAsFactors = FALSE
))

logrank_score <- function(time, event, x, cut) {
  left <- x <= cut
  event_times <- sort(unique(time[event == 1]))
  numerator <- variance <- 0
  for (u in event_times) {
    risk_total <- sum(time >= u)
    event_total <- sum(time == u & event == 1)
    risk_left <- sum(time >= u & left)
    event_left <- sum(time == u & event == 1 & left)
    if (risk_total >= 2) {
      numerator <- numerator + event_left - risk_left * event_total / risk_total
      variance <- variance + (risk_left / risk_total) *
        (1 - risk_left / risk_total) * ((risk_total - event_total) /
        (risk_total - 1)) * event_total
    }
  }
  if (variance <= 0) return(0)
  abs(numerator) / sqrt(variance)
}
candidate_scores <- vapply(split_candidates, function(cut) {
  logrank_score(c(1, 2, 3, 4), c(1, 0, 1, 1), c(1, 2, 4, 6), cut)
}, numeric(1))
selected_cut <- NA_real_
best_score <- -Inf
for (j in seq_along(split_candidates)) {
  if (candidate_scores[[j]] - best_score > 1e-9) {
    best_score <- candidate_scores[[j]]
    selected_cut <- split_candidates[[j]]
  }
}
candidate_score_rows <- lapply(seq_along(split_candidates), function(j) data.frame(
  cut = split_candidates[[j]],
  score = candidate_scores[[j]],
  selected = split_candidates[[j]] == selected_cut,
  candidate_n = 4L,
  candidate_rows = "1;2;4;6",
  stringsAsFactors = FALSE
))

# The internal no-donor case leaves the previous completion in place. At a
# terminal node, if the donor pool of originally observed values is empty,
# RF-SRC retries with originally missing in-bag rows' already-completed values.
# If that remains empty the default is fatal; OPT_OUTC_TYPE is an explicit
# alternate mode that writes NaN. This ledger records the branch contract.
fallback_rows <- list(
  data.frame(case = "internal_no_local_observed_donor", result = "retain_parent_value",
             value = 7, outc_type_enabled = FALSE, stringsAsFactors = FALSE),
  data.frame(case = "terminal_no_observed_donor_parent_values_available",
             result = "summarize_completed_originally_missing_inbag_values",
             value = mode_value(c(3, 9, 9), 0.5), outc_type_enabled = FALSE,
             stringsAsFactors = FALSE),
  data.frame(case = "terminal_no_donor_anywhere_default",
             result = "fatal_native_error", value = NA_real_, outc_type_enabled = FALSE,
             stringsAsFactors = FALSE),
  data.frame(case = "terminal_no_donor_anywhere_outc_type",
             result = "write_NaN", value = NA_real_, outc_type_enabled = TRUE,
             stringsAsFactors = FALSE)
)

# OOB aggregate fallback uses eligible terminal completions from the selected
# tree set, then full-data originally-observed values only if that pool is empty.
oob_rows <- list(
  data.frame(case = "oob_tree_terminal_pool_nonempty", pool = "tree_terminal_values",
             field_type = "continuous", values = "2;8", uniform = NA_real_,
             result = mean(c(2, 8)), stringsAsFactors = FALSE),
  data.frame(case = "oob_tree_terminal_pool_nonempty_categorical",
             pool = "tree_terminal_values", field_type = "categorical",
             values = "2;8;8", uniform = 0.75,
             result = mode_value(c(2, 8, 8), 0.75), stringsAsFactors = FALSE),
  data.frame(case = "oob_tree_terminal_pool_empty_predictor", pool = "full_data_observed",
             field_type = "continuous", values = "4;6", uniform = 0.25,
             result = sample_value(c(4, 6), 0.25), stringsAsFactors = FALSE),
  data.frame(case = "oob_tree_terminal_pool_empty_outcome", pool = "none",
             field_type = "outcome", values = "", uniform = NA_real_,
             result = NA_real_, stringsAsFactors = FALSE)
)

write_rows("random-survival-missing-donors.csv", donor_rows)
write.csv(split_rows, file.path(out_dir, "random-survival-missing-split-mask.csv"),
          row.names = FALSE, na = "NA")
write.csv(data.frame(candidate_cut = split_candidates),
          file.path(out_dir, "random-survival-missing-split-candidates.csv"),
          row.names = FALSE)
write_rows("random-survival-missing-split-scores.csv", candidate_score_rows)
write_rows("random-survival-missing-terminal-mode.csv", terminal_rows)
write_rows("random-survival-missing-time-snap.csv", time_rows)
write_rows("random-survival-missing-time-near-ties.csv", near_snap_rows)
write_rows("random-survival-missing-terminal-counts.csv", terminal_count_rows)
write.csv(grid_rows, file.path(out_dir, "random-survival-missing-time-grids.csv"),
          row.names = FALSE)
write_rows("random-survival-missing-terminal-fallback.csv", fallback_rows)
write_rows("random-survival-missing-oob-fallback.csv", oob_rows)

stopifnot(identical(imputed, c(1, 100)),
          identical(which(eligible), c(1L, 2L, 4L, 6L)),
          identical(vapply(time_rows, `[[`, numeric(1), "snapped_time"), c(2, 3)),
          identical(vapply(near_snap_rows, `[[`, numeric(1), "snapped_time"), c(2, 2)),
          identical(completed_time, c(1, 2, 3, 4, 2)),
          identical(completed_status, c(1, 0, 1, 0, 1)))
cat("Wrote deterministic RF-SRC missing-data ledgers to", out_dir, "\n")
