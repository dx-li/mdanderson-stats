#!/usr/bin/env Rscript

# Generate small, package-native reference fixtures for the drug-combination
# Keyboard implementation.  This script intentionally calls Keyboard 0.1.3;
# it is a research/reference tool and is not imported by the Python package.
#
# Run from the repository root with a task-local R library, for example:
#   R_LIBS_USER=/tmp/keyboard-r-lib Rscript tools/reference_keyboard_combination.R

suppressPackageStartupMessages(library(Keyboard))
stopifnot(as.character(packageVersion("Keyboard")) == "0.1.3")

out_dir <- file.path(getwd(), "tests", "fixtures")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

matrix_text <- function(x) {
  # Row-major text keeps fixtures readable and independent of R's column-major
  # vectorization.  NA is retained as a literal token.
  flat <- as.vector(t(x))
  flat <- ifelse(is.na(flat), "NA", format(flat, trim = TRUE, scientific = FALSE))
  paste(flat, collapse = ";")
}

matrix_int_text <- function(x) {
  paste(as.vector(t(x)), collapse = ";")
}

write_boundaries <- function() {
  b <- get.boundary.comb.kb(
    target = 0.3, ncohort = 6, cohortsize = 3,
    marginL = 0.05, marginR = 0.05, cutoff.eli = 0.95,
    offset = 0.05, extrasafe = TRUE
  )
  tab <- b$boundary
  safe <- b$safe
  out <- data.frame(
    target = 0.3,
    n = as.integer(tab[1, ]),
    escalate_if_dlt_le = as.integer(tab[2, ]),
    deescalate_if_dlt_ge = as.integer(tab[3, ]),
    eliminate_if_dlt_ge = as.integer(tab[4, ]),
    extra_safe_stop_if_dlt_ge = as.character(safe[2, ]),
    stringsAsFactors = FALSE
  )
  write.csv(out, file.path(out_dir, "keyboard-combination-boundaries.csv"), row.names = FALSE, na = "NA")
}

run_next <- function(case, n, y, dose, ..., target = 0.3) {
  warnings <- character()
  result <- tryCatch(
    withCallingHandlers(
      next.comb.kb(target = target, npts = n, ntox = y, dose.curr = dose, ...),
      warning = function(w) {
        warnings <<- c(warnings, conditionMessage(w))
        invokeRestart("muffleWarning")
      }
    ),
    error = function(e) structure(list(message = conditionMessage(e)), class = "reference_error")
  )
  is_error <- inherits(result, "reference_error")
  next_dc <- if (is_error) c(NA_integer_, NA_integer_) else as.integer(result$next_dc)
  data.frame(
    case = case,
    target = target,
    nrow = nrow(n), ncol = ncol(n),
    patients = matrix_int_text(n), toxicities = matrix_int_text(y),
    current_a = as.integer(dose[1]), current_b = as.integer(dose[2]),
    next_a = next_dc[1], next_b = next_dc[2],
    warning = gsub("\\s+", " ", trimws(paste(warnings, collapse = " | "))),
    error = if (is_error) result$message else "",
    stringsAsFactors = FALSE
  )
}

write_movements <- function() {
  z <- matrix(0, nrow = 2, ncol = 3)
  n <- z; n[1, 1] <- 1
  rows <- list(run_next("escalate_equal_target_mass_random_tie", n, z, c(1, 1)))

  n <- z; n[1, 1] <- 3; n[2, 1] <- 3; n[1, 2] <- 3
  y <- z; y[1, 2] <- 1
  rows[[length(rows) + 1L]] <- run_next("escalate_highest_adjacent_target_mass", n, y, c(1, 1))

  # Keyboard 0.1.3's next.comb.kb calls get.boundary.comb.kb with
  # marginL/marginR/cutoff.eli positionally after named arguments.  For n=1
  # this leaves the de-escalation cutoff NA; preserve the observed error.
  n <- z; n[2, 2] <- 1
  y <- z; y[2, 2] <- 1
  rows[[length(rows) + 1L]] <- run_next("deescalate_n1_package_boundary_error", n, y, c(2, 2))

  n <- z; n[2, 2] <- 3
  y <- z; y[2, 2] <- 3
  rows[[length(rows) + 1L]] <- run_next("deescalate_high_toxicity", n, y, c(2, 2))

  n <- z; n[2, 2] <- 3
  y <- z; y[2, 2] <- 2
  rows[[length(rows) + 1L]] <- run_next("retain_at_intermediate_toxicity", n, y, c(2, 2))

  n <- z; n[1, 1] <- 3
  y <- z; y[1, 1] <- 2
  rows[[length(rows) + 1L]] <- run_next("extra_safe_lowest_dose_stop", n, y, c(1, 1), extrasafe = TRUE)

  n <- z; n[1, 1] <- 9
  rows[[length(rows) + 1L]] <- run_next("precision_stop_at_current_dose", n, z, c(1, 1), n.earlystop = 9)

  # Both adjacent escalation candidates are eliminated by their own observed
  # data.  The package then has no admissible escalation candidate and stays.
  n <- z; n[1, 1] <- 3; n[2, 1] <- 3; n[1, 2] <- 3
  y <- z; y[2, 1] <- 3; y[1, 2] <- 3
  rows[[length(rows) + 1L]] <- run_next("all_adjacent_escalation_candidates_eliminated", n, y, c(1, 1))

  write.csv(do.call(rbind, rows), file.path(out_dir, "keyboard-combination-movements.csv"), row.names = FALSE, na = "NA")
}

run_selection <- function(case, n, y, ..., target = 0.3) {
  warnings <- character()
  result <- tryCatch(
    withCallingHandlers(
      select.mtd.comb.kb(target = target, npts = n, ntox = y, ...),
      warning = function(w) {
        warnings <<- c(warnings, conditionMessage(w))
        invokeRestart("muffleWarning")
      }
    ),
    error = function(e) structure(list(message = conditionMessage(e)), class = "reference_error")
  )
  is_error <- inherits(result, "reference_error")
  mtd <- if (is_error || length(result$MTD) == 1L) c(NA_integer_, NA_integer_) else as.integer(result$MTD[1, ])
  data.frame(
    case = case, target = target, nrow = nrow(n), ncol = ncol(n),
    patients = matrix_int_text(n), toxicities = matrix_int_text(y),
    mtd_a = mtd[1], mtd_b = mtd[2],
    isotonic_estimate_rounded = if (is_error) "" else matrix_text(result$p_est),
    warning = gsub("\\s+", " ", trimws(paste(warnings, collapse = " | "))),
    error = if (is_error) result$message else "",
    stringsAsFactors = FALSE
  )
}

write_selection <- function() {
  n <- matrix(c(3, 5, 0, 0, 7, 6, 15, 0, 0, 0, 4, 0), ncol = 4, byrow = TRUE)
  y <- matrix(c(0, 1, 0, 0, 1, 1, 4, 0, 0, 0, 2, 0), ncol = 4, byrow = TRUE)
  rows <- list(run_selection("paper_package_documented_example", n, y))

  n <- matrix(c(3, 3, 3, 0, 3, 3, 3, 0, 3, 3, 3, 0), 3, 4, byrow = TRUE)
  y <- matrix(0, 3, 4); y[1, 2] <- 3
  rows[[length(rows) + 1L]] <- run_selection("safety_cross_closure", n, y)

  n <- matrix(0, 2, 3); n[1, 1] <- 3
  y <- matrix(0, 2, 3); y[1, 1] <- 3
  rows[[length(rows) + 1L]] <- run_selection("lowest_dose_safety_stop", n, y)

  y[1, 1] <- 2
  rows[[length(rows) + 1L]] <- run_selection("extra_safe_lowest_dose_stop", n, y, extrasafe = TRUE)

  n <- matrix(3, 2, 3); y <- matrix(1, 2, 3)
  rows[[length(rows) + 1L]] <- run_selection("equal_matrix_estimates_deterministic_tie", n, y)

  n <- matrix(0, 2, 3); y <- n
  rows[[length(rows) + 1L]] <- run_selection("all_doses_untreated", n, y)

  write.csv(do.call(rbind, rows), file.path(out_dir, "keyboard-combination-selection.csv"), row.names = FALSE, na = "NA")
}

write_simulation <- function() {
  # get.oc.comb.kb resets the R RNG to seed 6 internally.  Keep this small
  # enough for a reference fixture while exercising cohort simulation and MTD
  # selection over a 2 x 3 dose matrix.
  p <- matrix(c(0.05, 0.15, 0.30, 0.10, 0.25, 0.50), nrow = 2, byrow = TRUE)
  result <- get.oc.comb.kb(
    target = 0.3, p.true = p, ncohort = 4, cohortsize = 3,
    n.earlystop = 9, startdose = c(1, 1), ntrial = 20
  )
  cells <- expand.grid(row = seq_len(nrow(p)), col = seq_len(ncol(p)))
  out <- data.frame(
    scenario = "small_2x3_seed6",
    target = 0.3, ncohort = 4, cohortsize = 3, n_earlystop = 9, ntrial = 20,
    row = cells$row, col = cells$col,
    true_toxicity = as.vector(p),
    selection_percent = as.vector(result$selpercent),
    mean_patients = as.vector(result$nptsdose),
    mean_toxicities = as.vector(result$ntoxdose),
    stringsAsFactors = FALSE
  )
  write.csv(out, file.path(out_dir, "keyboard-combination-simulation.csv"), row.names = FALSE)
  write.csv(data.frame(
    scenario = "small_2x3_seed6", target = 0.3, ncohort = 4, cohortsize = 3,
    n_earlystop = 9, ntrial = 20, pcs = result$pcs,
    total_patients = result$totaln, total_toxicities = result$totaltox,
    pct_early_stop = result$pctearlystop, stringsAsFactors = FALSE
  ), file.path(out_dir, "keyboard-combination-simulation-summary.csv"), row.names = FALSE)
}

write_boundaries()
write_movements()
write_selection()
write_simulation()
