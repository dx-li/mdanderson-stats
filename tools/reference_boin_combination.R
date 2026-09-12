#!/usr/bin/env Rscript

# Generate bounded native references for the BOIN package's drug-combination
# methods.  This is a reference tool, not runtime package code.
#
#   R_LIBS_USER=/tmp/boincomb-r-lib Rscript tools/reference_boin_combination.R

suppressPackageStartupMessages(library(BOIN))
stopifnot(as.character(packageVersion("BOIN")) == "2.7.2")

out_dir <- file.path(getwd(), "tests", "fixtures")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

matrix_text <- function(x) {
  flat <- as.vector(t(x))
  flat <- ifelse(is.na(flat), "NA", format(flat, trim = TRUE, scientific = FALSE))
  paste(flat, collapse = ";")
}
matrix_int_text <- function(x) paste(as.vector(t(x)), collapse = ";")

write_boundaries <- function() {
  # Include every patient count because next.comb indexes the full table even
  # when cohorts are larger than one.
  b <- get.boundary(target = .3, ncohort = 6, cohortsize = 1,
                    n.earlystop = 100, extrasafe = TRUE)
  tab <- b$boundary_tab
  out <- data.frame(
    target = .3, n = as.integer(tab[1, ]),
    escalate_if_dlt_le = as.integer(tab[2, ]),
    deescalate_if_dlt_ge = as.integer(tab[3, ]),
    eliminate_if_dlt_ge = as.integer(tab[4, ]),
    extra_safe_stop_if_dlt_ge = as.integer(b$stop_boundary[2, ]),
    stringsAsFactors = FALSE
  )
  write.csv(out, file.path(out_dir, "boin-combination-boundaries.csv"),
            row.names = FALSE, na = "NA")
}

run_next <- function(case, n, y, dose, ..., target = .3) {
  warnings <- character()
  result <- tryCatch(withCallingHandlers(
    next.comb(target = target, npts = n, ntox = y, dose.curr = dose, ...),
    warning = function(w) {
      warnings <<- c(warnings, conditionMessage(w)); invokeRestart("muffleWarning")
    }), error = function(e) structure(list(message = conditionMessage(e)),
                                      class = "reference_error"))
  is_error <- inherits(result, "reference_error")
  next_dc <- if (is_error) c(NA_integer_, NA_integer_) else as.integer(result$next_dc)
  data.frame(case = case, target = target, nrow = nrow(n), ncol = ncol(n),
             patients = matrix_int_text(n), toxicities = matrix_int_text(y),
             current_a = dose[1], current_b = dose[2], next_a = next_dc[1],
             next_b = next_dc[2], warning = gsub("\\s+", " ", trimws(paste(warnings, collapse = " | "))),
             error = if (is_error) result$message else "", stringsAsFactors = FALSE)
}

write_movements <- function() {
  set.seed(1)
  z <- matrix(0, 2, 3); rows <- list()
  n <- z; n[1, 1] <- 3
  rows[[1]] <- run_next("escalate_equal_neighbor_tie", n, z, c(1, 1))
  n <- z; n[1, 1] <- 3; n[1, 2] <- 3; y <- z; y[1, 2] <- 1
  rows[[2]] <- run_next("escalate_prefer_higher_stay_mass", n, y, c(1, 1))
  n <- z; n[2, 2] <- 3; y <- z; y[2, 2] <- 3
  rows[[3]] <- run_next("deescalate_to_higher_posterior_neighbor", n, y, c(2, 2))
  n <- z; n[2, 2] <- 3; y <- z; y[2, 2] <- 1
  rows[[4]] <- run_next("retain_inside_interval", n, y, c(2, 2))
  n <- z; n[1, 1] <- 3; y <- z; y[1, 1] <- 3
  rows[[5]] <- run_next("lowest_dose_elimination_stop", n, y, c(1, 1))
  n <- z; n[1, 1] <- 3; y <- z; y[1, 1] <- 2
  rows[[6]] <- run_next("extra_safe_lowest_dose_stop", n, y, c(1, 1), extrasafe = TRUE)
  n <- z; n[1, 1] <- 9
  rows[[7]] <- run_next("precision_stop_current_dose", n, z, c(1, 1), n.earlystop = 9)
  write.csv(do.call(rbind, rows), file.path(out_dir, "boin-combination-movements.csv"),
            row.names = FALSE, na = "NA")
}

write_simulation <- function() {
  # Small by design: this validates the package's cohort simulation wiring and
  # seed handling without becoming a Monte Carlo specification.
  p <- matrix(c(.05, .15, .30, .10, .25, .50), 2, 3, byrow = TRUE)
  result <- suppressWarnings(get.oc.comb(.3, p, ncohort = 4, cohortsize = 3,
                                          n.earlystop = 9, startdose = c(1, 1),
                                          ntrial = 20, seed = 9))
  cells <- expand.grid(row = seq_len(nrow(p)), col = seq_len(ncol(p)))
  out <- data.frame(scenario = "small_2x3_seed9", target = .3, ncohort = 4,
                    cohortsize = 3, n_earlystop = 9, ntrial = 20,
                    row = cells$row, col = cells$col,
                    true_toxicity = as.vector(p),
                    selection_percent = as.vector(result$selpercent),
                    mean_patients = as.vector(result$npatients),
                    mean_toxicities = as.vector(result$ntox))
  write.csv(out, file.path(out_dir, "boin-combination-simulation.csv"), row.names = FALSE)
  write.csv(data.frame(scenario = "small_2x3_seed9", total_patients = result$totaln,
                       total_toxicities = result$totaltox, pcs = result$pcs,
                       patient_percent_at_mtd = result$npercent,
                       percent_early_stop = result$percentstop),
            file.path(out_dir, "boin-combination-simulation-summary.csv"), row.names = FALSE)
}

write_biviso <- function() {
  # The package reports a rounded fit, but the unrounded fit is useful for
  # validating implementations before formatting and tie selection.
  shapes <- list(c(2, 3), c(3, 4), c(3, 5)); seeds <- c(17, 31, 47)
  rows <- list()
  for (k in seq_along(shapes)) {
    set.seed(seeds[k]); nr <- shapes[[k]][1]; nc <- shapes[[k]][2]
    n <- matrix(sample(0:10, nr * nc, replace = TRUE), nr, nc)
    n[1, 1] <- 0; n[nr, nc] <- 0
    y <- matrix(vapply(as.vector(n), function(nn) if (nn == 0) 0 else sample.int(nn + 1, 1) - 1, numeric(1)), nr, nc)
    raw <- (y + .05) / (n + .1); fit <- Iso::biviso(raw, n + .1, warn = TRUE)
    nm <- sprintf("seed_%d_%dx%d", seeds[k], nr, nc)
    for (i in seq_len(nr)) for (j in seq_len(nc)) rows[[length(rows) + 1L]] <- data.frame(
      case = nm, seed = seeds[k], nrow = nr, ncol = nc, row = i, col = j,
      patients = n[i, j], toxicities = y[i, j], raw_estimate = sprintf("%.17g", raw[i, j]),
      weight = sprintf("%.17g", n[i, j] + .1), biviso_fit = sprintf("%.17g", fit[i, j]))
  }
  write.csv(do.call(rbind, rows), file.path(out_dir, "boin-combination-biviso.csv"), row.names = FALSE)
}

run_selection <- function(case, n, y, ..., target = .3) {
  warnings <- character()
  result <- tryCatch(withCallingHandlers(
    select.mtd.comb(target = target, npts = n, ntox = y, ...),
    warning = function(w) { warnings <<- c(warnings, conditionMessage(w)); invokeRestart("muffleWarning") }),
    error = function(e) structure(list(message = conditionMessage(e)), class = "reference_error"))
  is_error <- inherits(result, "reference_error")
  mtd <- if (is_error || length(result$MTD) == 1L) c(NA, NA) else as.integer(result$MTD[1, ])
  contour <- if (is_error || length(result$MTD) == 1L) "" else {
    paste(sprintf("(%d,%d)", result$MTD[, 1], result$MTD[, 2]), collapse = ";")
  }
  data.frame(case = case, target = target, nrow = nrow(n), ncol = ncol(n),
             patients = matrix_int_text(n), toxicities = matrix_int_text(y),
             mtd_a = mtd[1], mtd_b = mtd[2], contour = contour,
             isotonic_estimate_rounded = if (is_error) "" else matrix_text(result$p_est),
             warning = gsub("\\s+", " ", trimws(paste(warnings, collapse = " | "))),
             error = if (is_error) result$message else "", stringsAsFactors = FALSE)
}

write_selection <- function() {
  n <- matrix(c(3, 5, 0, 0, 7, 6, 15, 0, 0, 0, 4, 0), 3, 4, byrow = TRUE)
  y <- matrix(c(0, 1, 0, 0, 1, 1, 4, 0, 0, 0, 2, 0), 3, 4, byrow = TRUE)
  rows <- list(run_selection("package_documented_single_mtd", n, y))
  rows[[2]] <- run_selection("package_documented_contour", n, y, mtd.contour = TRUE)
  n <- matrix(c(3, 3, 3, 0, 3, 3, 3, 0, 3, 3, 3, 0), 3, 4, byrow = TRUE)
  y <- matrix(0, 3, 4); y[1, 2] <- 3
  rows[[3]] <- run_selection("cross_elimination_closure", n, y)
  n <- matrix(0, 2, 3); n[1, 1] <- 3; y <- matrix(0, 2, 3); y[1, 1] <- 3
  rows[[4]] <- run_selection("all_doses_overly_toxic", n, y)
  n <- matrix(3, 2, 3); y <- matrix(1, 2, 3)
  rows[[5]] <- run_selection("equal_matrix_tie", n, y)
  n <- matrix(0, 2, 3); y <- n
  rows[[6]] <- run_selection("all_doses_untreated", n, y)
  n <- matrix(c(3, 4, 0, 5, 6, 0), 2, 3, byrow = TRUE)
  y <- matrix(c(0, 1, 0, 1, 2, 0), 2, 3, byrow = TRUE)
  rows[[7]] <- run_selection("bound_mtd_rejects_over_target_fit", n, y, boundMTD = TRUE)
  write.csv(do.call(rbind, rows), file.path(out_dir, "boin-combination-selection.csv"), row.names = FALSE, na = "NA")
}

write_waterfall <- function() {
  n <- matrix(c(6, 0, 0, 0, 6, 10, 12, 0, 9, 12, 0, 0), 3, 4, byrow = TRUE)
  y <- matrix(c(0, 0, 0, 0, 1, 1, 4, 0, 2, 3, 0, 0), 3, 4, byrow = TRUE)
  x <- next.subtrial(.3, n, y)
  write.csv(data.frame(case = "package_documented_waterfall", next_subtrial = x$next_subtrial,
                       start_a = x$starting_dose[1], start_b = x$starting_dose[2]),
            file.path(out_dir, "boin-combination-waterfall.csv"), row.names = FALSE, na = "NA")
}

write_boundaries(); write_movements(); write_biviso(); write_selection(); write_waterfall(); write_simulation()
