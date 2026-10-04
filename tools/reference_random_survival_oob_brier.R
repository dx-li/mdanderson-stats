#!/usr/bin/env Rscript

# Small independent fixture generator for the randomForestSRC full-training
# OOB survival Brier calculation. Source the pinned utilities.survival.R and
# invoke get.brier.survival on minimal grow-like objects. The source path is
# supplied as argv[1]; outputs go to argv[2] (default: current directory).

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1L || length(args) > 2L) {
  stop("usage: Rscript reference_random_survival_oob_brier.R <utilities.survival.R> [output-dir]")
}
source_file <- args[[1L]]
output_dir <- if (length(args) == 2L) args[[2L]] else "."
source(source_file, local = .GlobalEnv)

make_grow <- function(time, event, survival_oob, event_grid) {
  n <- length(time)
  if (length(event) != n || nrow(survival_oob) != n || ncol(survival_oob) != length(event_grid)) {
    stop("inconsistent reference case dimensions")
  }
  yvar <- cbind(time = time, cens = event)
  obj <- list(
    family = "surv",
    yvar = yvar,
    time.interest = event_grid,
    predicted = rep(0, n),
    predicted.oob = rep(0, n),
    survival = survival_oob,
    survival.oob = survival_oob,
    forest = list(yvar = yvar, xvar = data.frame(x = seq_len(n)))
  )
  class(obj) <- c("rfsrc", "grow")
  obj
}

cases <- list(
  no_censor_time_zero = list(
    time = c(0, 2, 5, 5), event = c(1, 1, 1, 1),
    grid = c(0, 2, 5),
    survival = rbind(c(.80, .45, .20), c(.90, .60, .30),
                     c(.95, .70, .40), c(.70, .35, .10))
  ),
  ties_and_between_censors = list(
    time = c(1, 1, 2, 1.5, 3, 4, .5), event = c(1, 0, 1, 0, 0, 1, 0),
    grid = c(1, 2, 4),
    survival = rbind(c(.90, .70, .40), c(.85, .65, .35),
                     c(.80, .55, .25), c(.95, .75, .50),
                     c(.75, .50, .20), c(.88, .62, .32),
                     c(.92, .72, .42))
  ),
  uneven_grid_missing_oob = list(
    time = c(.5, 2, 3, 7, 7), event = c(1, 1, 0, 1, 1),
    grid = c(.5, 2, 7),
    survival = rbind(c(.92, .77, .28), c(.84, .61, .31),
                     c(NA, NA, NA), c(.97, .68, .22),
                     c(.81, .52, .17))
  ),
  reduced_event_grid = list(
    time = c(1, 2, 3, 4), event = c(1, 1, 0, 1),
    # Mimic ntime/grid reduction: the event at 2 is deliberately omitted.
    grid = c(1, 4),
    survival = rbind(c(.91, .42), c(.86, .37),
                     c(.80, .29), c(.96, .18))
  )
)

independent_curve <- function(x) {
  # Use the forest's supplied event grid, which may be a reduced subset of
  # observed event times after an ntime setting.
  grid <- x$grid
  censor_times <- sort(unique(x$time[x$event == 0]))
  censor_hazard <- vapply(censor_times, function(cc) {
    risk <- sum(x$time >= cc)
    deaths <- sum(x$time == cc & x$event == 0)
    deaths / (risk + 1 * (risk == 0))
  }, numeric(1))
  if (length(censor_times) == 0L) censor_hazard <- numeric()
  g_grid <- vapply(grid, function(tt) {
    exp(-sum(censor_hazard[censor_times <= tt]))
  }, numeric(1))
  mat <- matrix(NA_real_, nrow = length(x$time), ncol = length(grid))
  for (i in seq_along(x$time)) {
    k <- sum(grid <= x$time[i])
    g_at_tau <- if (k == 0L) 1 else g_grid[k]
    for (j in seq_along(grid)) {
      if (x$event[i] != 0 && x$time[i] <= grid[j]) {
        weight <- 1 / g_at_tau
      } else if (x$time[i] > grid[j]) {
        weight <- 1 / g_grid[j]
      } else {
        weight <- 0
      }
      if (is.finite(x$survival[i, j])) {
        mat[i, j] <- (1 * (x$time[i] > grid[j]) - x$survival[i, j])^2 * weight
      }
    }
  }
  score <- vapply(seq_along(grid), function(j) {
    valid <- is.finite(mat[, j])
    if (!any(valid)) NaN else mean(mat[valid, j])
  }, numeric(1))
  crps <- if (length(grid) < 2L) 0 else sum(diff(grid) * (head(score, -1L) + tail(score, -1L)) / 2)
  list(grid = grid, g = g_grid, mat = mat, score = score, crps = crps,
       crps_std = if (max(grid) == 0) NaN else crps / max(grid))
}

curve_rows <- list()
row_rows <- list()
input_rows <- list()
for (case_name in names(cases)) {
  x <- cases[[case_name]]
  obj <- make_grow(x$time, x$event, x$survival, x$grid)
  ans <- get.brier.survival(obj, subset = seq_along(x$time), cens.model = "km")
  ref <- independent_curve(x)
  if (!isTRUE(all.equal(as.numeric(ans$brier.score$time), ref$grid, tolerance = 0))) {
    stop(paste("source and independent event grids differ for", case_name))
  }
  if (!isTRUE(all.equal(as.numeric(ans$brier.score$brier.score), ref$score, tolerance = 1e-12))) {
    stop(paste("source and independent Brier curves differ for", case_name))
  }
  if (!isTRUE(all.equal(unname(ans$brier.matx), unname(ref$mat), tolerance = 1e-12))) {
    stop(paste("source and independent per-row contributions differ for", case_name))
  }
  if (!isTRUE(all.equal(as.numeric(ans$cens.dist), ref$g, tolerance = 1e-12))) {
    stop(paste("source and independent projected censor survival differ for", case_name))
  }
  if (length(ref$grid) > 1L && !isTRUE(all.equal(ans$crps, ref$crps, tolerance = 1e-12))) {
    stop(paste("source and independent CRPS differ for", case_name))
  }
  if (length(ref$grid) > 1L && !isTRUE(all.equal(ans$crps.std, ref$crps_std, tolerance = 1e-12))) {
    stop(paste("source and independent standardized CRPS differ for", case_name))
  }
  curve_rows[[case_name]] <- data.frame(
    case = case_name,
    time = ref$grid,
    censor_survival = ref$g,
    brier_score = ref$score,
    crps = rep(ref$crps, length(ref$grid)),
    crps_std = rep(ref$crps_std, length(ref$grid)),
    stringsAsFactors = FALSE
  )
  mat <- ref$mat
  row_rows[[case_name]] <- data.frame(
    case = case_name,
    observation = rep(seq_len(nrow(mat)), times = ncol(mat)),
    time = rep(ans$time, each = nrow(mat)),
    brier_contribution = as.vector(mat),
    stringsAsFactors = FALSE
  )
  input_rows[[case_name]] <- data.frame(
    case = case_name,
    observation = rep(seq_along(x$time), times = length(x$grid)),
    time = rep(x$time, times = length(x$grid)),
    event = rep(x$event, times = length(x$grid)),
    grid_index = rep(seq_along(x$grid), each = length(x$time)),
    grid_time = rep(x$grid, each = length(x$time)),
    oob_survival = as.vector(x$survival),
    stringsAsFactors = FALSE
  )
}
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
write.csv(do.call(rbind, curve_rows), file.path(output_dir, "random-survival-oob-brier-curve.csv"), row.names = FALSE, na = "NA")
write.csv(do.call(rbind, row_rows), file.path(output_dir, "random-survival-oob-brier-contributions.csv"), row.names = FALSE, na = "NA")
write.csv(do.call(rbind, input_rows), file.path(output_dir, "random-survival-oob-brier-input.csv"), row.names = FALSE, na = "NA")
