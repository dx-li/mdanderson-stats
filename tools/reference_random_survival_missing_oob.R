#!/usr/bin/env Rscript

# Independent fixed-tape reference for RF-SRC missing-response OOB scoring.
# This records source equations and aggregation rules; it does not execute or
# redistribute the native forest and makes no R/Python RNG parity claim.

args <- commandArgs(trailingOnly = TRUE)
out_dir <- if (length(args) >= 1L) args[[1L]] else file.path("tests", "fixtures")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

mode_value <- function(values, u) {
  stopifnot(length(values) > 0L, is.finite(u), u > 0, u <= 1)
  levels <- sort(unique(values))
  counts <- tabulate(match(values, levels), nbins = length(levels))
  winners <- levels[counts == max(counts)]
  winners[[ceiling(u * length(winners))]]
}

sample_value <- function(values, u) {
  stopifnot(length(values) > 0L, is.finite(u), u > 0, u <= 1)
  values[[ceiling(u * length(values))]]
}

# Each row is an eligible tree's terminal completion for one target row.
# Terminal time values are already snapped by per-tree terminal imputation;
# their OOB ensemble mean is not snapped a second time. OOB-only values are
# aggregated first; only an empty OOB pool uses original
# complete-data fallback donors. `uniform` is supplied directly to the native
# helper convention, not generated from an R seed.
response_cases <- data.frame(
  case = c("mean_of_snapped_tree_times_left_unsnapped", "status_mode_tie_low", "status_mode_tie_high",
           "empty_oob_time_global_fallback", "empty_oob_status_global_fallback"),
  field = c("time", "status", "status", "time", "status"),
  recipient_id = c(5L, 6L, 7L, 5L, 5L),
  was_originally_missing = TRUE,
  oob_terminal_values = c("1;3", "0;1;0;1", "0;1;0;1", "", ""),
  global_observed_fallback_values = c("1;3;7;9", "0;1;0;1", "0;1;0;1",
                                      "1;3;7;9", "0;1;0;1"),
  inbag_terminal_outlier = c(100, NA, NA, NA, NA),
  uniform = c(NA, 0.25, 0.75, 0.5, 1),
  stringsAsFactors = FALSE
)
response_cases$pool <- ifelse(nchar(response_cases$oob_terminal_values) > 0,
                              "oob_tree_terminal", "full_data_observed_fallback")
response_cases$expected <- c(
  mean(c(1, 3)),
  mode_value(c(0, 1, 0, 1), 0.25),
  mode_value(c(0, 1, 0, 1), 0.75),
  sample_value(c(1, 3, 7, 9), 0.5),
  sample_value(c(0, 1, 0, 1), 1)
)
response_cases$fallback_used <- response_cases$pool == "full_data_observed_fallback"

# An observed component must remain untouched even if the other response
# component is missing. The row is intentionally otherwise eligible for OOB.
preservation <- data.frame(
  case = "observed_component_preserved",
  row_id = 10L,
  time_original = 4,
  event_original = 1,
  time_missing = TRUE,
  event_missing = FALSE,
  expected_time = mean(c(1, 3)),
  expected_event = 1,
  stringsAsFactors = FALSE
)

# The source performs all nonempty tree-pool aggregations first (including a
# tied status mode draw), then performs empty-pool global fallbacks. Therefore
# a status-mode tape value is consumed before a fallback time sample.
draw_order <- data.frame(
  draw_order = 1:2,
  operation = c("status_tied_mode", "time_global_fallback"),
  pool = c("0;1", "1;3;7"),
  uniform = c(0.75, 0.1),
  selected_index_one_based = c(2L, 1L),
  expected_value = c(1, 1),
  stringsAsFactors = FALSE
)

# Concordance follows getConcordanceIndex (C source 32500ff), including its
# asymmetric tied-failure penalty and OOB denominator gate.
eps <- 1e-9
concordance_case <- data.frame(
  case = "eps_ties_and_zero_oob_contributors",
  row_id = 1:6,
  time = c(1, 1 + 0.5 * eps, 2, 2, 3, 4),
  event = c(1, 0, 1, 1, 0, 1),
  risk = c(2, 2 + 0.5 * eps, 0.5, 1.5, 0.1, 9),
  contributors = c(2, 1, 2, 1, 1, 0),
  stringsAsFactors = FALSE
)
pair_rows <- list()
pair_index <- 0L
pair_size <- worse <- 0L
for (i in seq_len(nrow(concordance_case) - 1L)) {
  for (j in seq.int(i + 1L, nrow(concordance_case))) {
    a <- concordance_case[i, ]
    b <- concordance_case[j, ]
    comparable <- a$contributors != 0 && b$contributors != 0 && (
      (a$time - b$time > eps && b$event > 0) ||
        (abs(a$time - b$time) <= eps && b$event > 0 && a$event == 0) ||
        (b$time - a$time > eps && a$event > 0) ||
        (abs(b$time - a$time) <= eps && a$event > 0 && b$event == 0) ||
        (abs(a$time - b$time) <= eps && a$event > 0 && b$event > 0)
    )
    if (!comparable) next
    if ((a$time - b$time > eps && b$event > 0) ||
        (abs(a$time - b$time) <= eps && b$event > 0 && a$event == 0)) {
      loss <- if (b$risk - a$risk > eps) 2L else if (abs(b$risk - a$risk) <= eps) 1L else 0L
      reason <- "later_j_failure_or_tied_failure_censor"
    } else if ((b$time - a$time > eps && a$event > 0) ||
               (abs(b$time - a$time) <= eps && a$event > 0 && b$event == 0)) {
      loss <- if (a$risk - b$risk > eps) 2L else if (abs(a$risk - b$risk) <= eps) 1L else 0L
      reason <- "later_i_failure_or_tied_failure_censor"
    } else {
      loss <- if (abs(a$risk - b$risk) < eps) 2L else 1L
      reason <- "tied_failures"
    }
    pair_index <- pair_index + 1L
    pair_rows[[pair_index]] <- data.frame(
      i = i, j = j, reason = reason, pair_denominator = 2L,
      worse_increment = loss, stringsAsFactors = FALSE
    )
    pair_size <- pair_size + 2L
    worse <- worse + loss
  }
}
concordance_pairs <- if (length(pair_rows)) do.call(rbind, pair_rows) else data.frame()
concordance_summary <- data.frame(
  case = "eps_ties_and_zero_oob_contributors",
  comparable_unordered_pairs = pair_size / 2L,
  denominator = pair_size,
  worse_count = worse,
  concordance_error = if (pair_size == 0) NA_real_ else 1 - worse / pair_size,
  stringsAsFactors = FALSE
)

write.csv(response_cases, file.path(out_dir, "random-survival-missing-oob-responses.csv"),
          row.names = FALSE, na = "NA")
write.csv(preservation, file.path(out_dir, "random-survival-missing-oob-preservation.csv"),
          row.names = FALSE, na = "NA")
write.csv(draw_order, file.path(out_dir, "random-survival-missing-oob-rng-order.csv"),
          row.names = FALSE, na = "NA")
write.csv(concordance_case, file.path(out_dir, "random-survival-missing-oob-concordance-input.csv"),
          row.names = FALSE, na = "NA")
write.csv(concordance_pairs, file.path(out_dir, "random-survival-missing-oob-concordance-pairs.csv"),
          row.names = FALSE, na = "NA")
write.csv(concordance_summary, file.path(out_dir, "random-survival-missing-oob-concordance-summary.csv"),
          row.names = FALSE, na = "NA")

stopifnot(response_cases$expected[[1L]] == 2,
          !(response_cases$expected[[1L]] %in% c(1, 3, 7, 9)),
          response_cases$expected[[2L]] == 0,
          response_cases$expected[[3L]] == 1,
          response_cases$expected[[4L]] == 3,
          response_cases$expected[[5L]] == 1,
          concordance_summary$comparable_unordered_pairs > 0)
cat("Wrote fixed-tape RF-SRC OOB-missing reference fixtures to", out_dir, "\n")
