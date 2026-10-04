#!/usr/bin/env Rscript

# Fixed-tape semantic ledgers for the pinned RF-SRC nimpute pass summaries.
# This is an independent transcription of the source equations, not the
# native forest and not a native R/C RNG replay.

args <- commandArgs(trailingOnly = TRUE)
out_dir <- if (length(args) >= 1L) args[[1L]] else file.path("tests", "fixtures")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

sample_value <- function(values, u) {
  stopifnot(length(values) > 0L, is.finite(u), u > 0, u <= 1)
  values[[ceiling(u * length(values))]]
}

mode_value <- function(values, u) {
  stopifnot(length(values) > 0L)
  levels <- sort(unique(values))
  counts <- tabulate(match(values, levels), nbins = length(levels))
  winners <- levels[counts == max(counts)]
  if (length(winners) == 1L) return(winners[[1L]])
  stopifnot(is.finite(u), u > 0, u <= 1)
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

summary_cases <- data.frame(
  case = c("numeric_by_root_excludes_inbag_outlier", "numeric_by_node_includes_all",
           "categorical_mode_by_root", "categorical_tied_mode_fixed_tape",
           "empty_oob_pool_global_fallback", "time_mean_then_fixed_master_resnap",
           "observed_value_unchanged"),
  field_kind = c("numeric", "numeric", "categorical", "categorical", "numeric",
                 "time", "time"),
  selection = c("OOB", "ALL", "OOB", "OOB", "OOB", "OOB", "OOB"),
  all_tree_terminal_values = c("2;8;100", "2;8;100", "2;8;8;4", "2;8;2;8",
                               "", "1;3", "1;3"),
  inbag_flags = c("FALSE;FALSE;TRUE", "FALSE;FALSE;TRUE", "FALSE;FALSE;FALSE;TRUE",
                  "FALSE;FALSE;FALSE;FALSE", "TRUE;TRUE", "FALSE;FALSE", "FALSE;FALSE"),
  original_observed_fallback = c("1;3;7;9", "1;3;7;9", "1;2;4;8", "1;2;4;8",
                                 "1;3;7;9", "1;3;7;9", "1;3;7;9"),
  uniform = c(NA, NA, NA, 0.75, 0.4, 0.75, NA),
  master_grid = c("", "", "", "", "", "1;3;7;9", "1;3;7;9"),
  original_value = c(NA, NA, NA, NA, NA, NA, 7),
  original_missing = c(TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, FALSE),
  stringsAsFactors = FALSE
)

summary_value <- function(kind, values, fallback, uniform, master) {
  if (!length(values)) {
    if (!length(fallback)) stop("empty response fallback pool")
    value <- sample_value(fallback, uniform)
  } else if (kind %in% c("numeric", "time")) {
    value <- mean(values)
  } else {
    value <- mode_value(values, uniform)
  }
  if (kind == "time") {
    value <- nearest_master_time(value, master, uniform)
  }
  value
}

summary_cases$selected_terminal_values <- vapply(seq_len(nrow(summary_cases)), function(i) {
  values <- as.numeric(strsplit(summary_cases$all_tree_terminal_values[[i]], ";", fixed = TRUE)[[1L]])
  if (!length(values) || is.na(values[[1L]])) return("")
  inbag <- as.logical(strsplit(summary_cases$inbag_flags[[i]], ";", fixed = TRUE)[[1L]])
  selected <- if (summary_cases$selection[[i]] == "OOB") !inbag else rep(TRUE, length(values))
  paste(values[selected], collapse = ";")
}, character(1))

summary_cases$expected <- vapply(seq_len(nrow(summary_cases)), function(i) {
  if (!summary_cases$original_missing[[i]]) return(summary_cases$original_value[[i]])
  selected <- summary_cases$selected_terminal_values[[i]]
  values <- if (nzchar(selected)) as.numeric(strsplit(selected, ";", fixed = TRUE)[[1L]]) else numeric()
  fallback <- as.numeric(strsplit(summary_cases$original_observed_fallback[[i]], ";", fixed = TRUE)[[1L]])
  uniform <- summary_cases$uniform[[i]]
  master <- if (nzchar(summary_cases$master_grid[[i]])) {
    as.numeric(strsplit(summary_cases$master_grid[[i]], ";", fixed = TRUE)[[1L]])
  } else numeric()
  summary_value(summary_cases$field_kind[[i]], values, fallback, uniform, master)
}, numeric(1))
summary_cases$fallback_used <- !nzchar(summary_cases$selected_terminal_values) &
  summary_cases$original_missing

# Ordinary fitting summarizes after every pass except the last. Values from
# the final pass's own terminal summaries are intentionally not applied.
pass_cases <- data.frame(
  pass = 1:3,
  role = c("first_pass_then_summarize", "intermediate_pass_then_summarize",
           "final_pass_no_postfit_summary"),
  missing_numeric = c(NA, 5, 6),
  missing_category = c(NA, 8, 4),
  missing_time = c(NA, 3, 7),
  missing_status = c(NA, 1, 0),
  terminal_summary_numeric = c(5, 6, 999),
  terminal_summary_category = c(8, 4, 999),
  terminal_summary_time_before_resnap = c(2, 5, 999),
  terminal_summary_time_after_resnap = c(3, 7, 999),
  terminal_summary_status = c(1, 0, 999),
  stringsAsFactors = FALSE
)

row_map <- data.frame(
  original_row = c(0L, 2L, 5L, 9L),
  analyzed_row = 0:3,
  original_time = c(1, 3, 7, 9),
  original_event = c(1, 0, NA, 1),
  event_interest_member = c(TRUE, FALSE, FALSE, TRUE),
  master_time_member = c(TRUE, TRUE, TRUE, TRUE),
  stringsAsFactors = FALSE
)

# Coupled response-pooling case. Original row IDs are noncontiguous; all helper
# vectors use the corresponding analyzed-row order. Two rows have tied local
# status modes, one row has no OOB terminal donors and falls back globally, and
# the first time pool mean needs the second, fixed-master-grid snap.
coupled_rows <- data.frame(
  original_row = c(0L, 2L, 4L, 7L, 9L, 12L),
  analyzed_row = 0:5,
  leaf_node = c(2L, 0L, 2L, 1L, 2L, 2L),
  original_time = c(1, NA, NA, 9, 3, 4),
  original_event = c(1, NA, NA, NA, 0, 1),
  missing_time = c(FALSE, TRUE, TRUE, FALSE, FALSE, FALSE),
  missing_event = c(FALSE, TRUE, TRUE, TRUE, FALSE, FALSE),
  expected_completed_time = c(1, 3, 3, 9, 3, 4),
  expected_completed_event = c(1, 0, 1, 1, 0, 1),
  fallback_time = c(FALSE, FALSE, TRUE, FALSE, FALSE, FALSE),
  fallback_event = c(FALSE, FALSE, TRUE, FALSE, FALSE, FALSE),
  stringsAsFactors = FALSE
)
coupled_trees <- data.frame(
  tree = 1:4,
  inbag_mask = c("1;0;1;1;1;1", "1;0;1;1;1;1", "1;1;1;0;1;1", "1;1;1;0;1;1"),
  terminal_time = c("1;9;4", "3;9;4", "1;9;4", "3;9;4"),
  terminal_event = c("0;0;1", "1;1;0", "0;0;1", "1;1;0"),
  terminal_predictor = c("10;20;30", "10;20;30", "10;20;30", "10;20;30"),
  stringsAsFactors = FALSE
)
coupled_tape <- data.frame(
  sequence = 1:5,
  operation = c("local_status_tie_row_2", "local_status_tie_row_4",
                "global_time_fallback_row_3", "global_status_fallback_row_3",
                "pooled_time_second_snap_row_2"),
  uniform = c(0.25, 0.75, 0.6, 0.75, 0.75),
  pool = c("0;1", "0;1", "1;9;3;4", "1;0;1", "1;3;4;9"),
  expected_selected = c(0, 1, 3, 1, 3),
  stringsAsFactors = FALSE
)
master_grid <- sort(unique(coupled_rows$original_time[is.finite(coupled_rows$original_time)]))
original_event_grid <- sort(unique(coupled_rows$original_time[
  is.finite(coupled_rows$original_time) & coupled_rows$original_event == 1
]))
completed_event_grid <- sort(unique(coupled_rows$expected_completed_time[
  coupled_rows$expected_completed_event == 1
]))
coupled_grids <- data.frame(
  master_grid = paste(master_grid, collapse = ";"),
  original_complete_event_grid = paste(original_event_grid, collapse = ";"),
  completed_event_grid_if_rebuilt = paste(completed_event_grid, collapse = ";"),
  expected_fixed_event_grid = paste(original_event_grid, collapse = ";"),
  stringsAsFactors = FALSE
)

write.csv(summary_cases, file.path(out_dir, "random-survival-iterated-summary.csv"),
          row.names = FALSE, na = "NA")
write.csv(pass_cases, file.path(out_dir, "random-survival-iterated-passes.csv"),
          row.names = FALSE, na = "NA")
write.csv(row_map, file.path(out_dir, "random-survival-iterated-rowmap.csv"),
          row.names = FALSE, na = "NA")
write.csv(coupled_rows, file.path(out_dir, "random-survival-iterated-coupled-rows.csv"),
          row.names = FALSE, na = "NA")
write.csv(coupled_trees, file.path(out_dir, "random-survival-iterated-coupled-trees.csv"),
          row.names = FALSE, na = "NA")
write.csv(coupled_tape, file.path(out_dir, "random-survival-iterated-coupled-tape.csv"),
          row.names = FALSE, na = "NA")
write.csv(coupled_grids, file.path(out_dir, "random-survival-iterated-coupled-grids.csv"),
          row.names = FALSE, na = "NA")

stopifnot(summary_cases$expected[[1L]] == 5,
          summary_cases$expected[[2L]] == mean(c(2, 8, 100)),
          summary_cases$expected[[3L]] == 8,
          summary_cases$expected[[4L]] == 8,
          summary_cases$expected[[5L]] == 3,
          summary_cases$expected[[6L]] == 3,
          summary_cases$expected[[7L]] == 7,
          identical(master_grid, c(1, 3, 4, 9)),
          identical(original_event_grid, c(1, 4)),
          identical(completed_event_grid, c(1, 3, 4, 9)),
          pass_cases$missing_numeric[[3L]] == pass_cases$terminal_summary_numeric[[2L]],
          pass_cases$missing_numeric[[3L]] != pass_cases$terminal_summary_numeric[[3L]])
cat("Wrote fixed-tape RF-SRC iterated-imputation ledgers to", out_dir, "\n")
