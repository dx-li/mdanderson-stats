#!/usr/bin/env Rscript

# Boundary-stub reference for randomForestSRC's cens.model="rfsrc" Brier path.
# This tests get.brier.survival's wiring and arithmetic, not a censor forest fit
# or its random stream. The pinned helper path is argv[1], output dir argv[2].

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1L || length(args) > 2L) {
  stop("usage: Rscript reference_random_survival_censoring_brier.R <utilities.survival.R> [output-dir]")
}
source(args[[1L]], local = .GlobalEnv)
output_dir <- if (length(args) == 2L) args[[2L]] else "."

case_now <- NULL
mock_calls <- list()
mock_predictions <- list()

rfsrc <- function(formula, data, ntree, nsplit, splitrule, nodesize, perf.type) {
  mock_calls[[length(mock_calls) + 1L]] <<- list(
    case = case_now$name,
    formula = paste(deparse(substitute(formula)), collapse = " "),
    data = data,
    ntree = ntree,
    nsplit = nsplit,
    splitrule = splitrule,
    nodesize = nodesize,
    perf.type = perf.type
  )
  structure(list(time.interest = case_now$censor_grid, case = case_now$name),
            class = "mock_censor_forest")
}

predict <- function(object, newdata, ...) {
  if (!inherits(object, "mock_censor_forest")) {
    stop("unexpected predict call in boundary-stub reference")
  }
  mock_predictions[[length(mock_predictions) + 1L]] <<- list(
    case = object$case,
    newdata = newdata
  )
  rows <- match(newdata$curve_id, case_now$curve_ids)
  if (anyNA(rows)) stop("prediction input did not map to a supplied censor curve")
  list(survival = case_now$censor_survival[rows, , drop = FALSE])
}

make_cases <- function() {
  list(
    no_censor = list(
      name = "no_censor", mode = "grow",
      time = c(0, 1, 2, 4), event = c(1, 1, 1, 1), grid = c(0, 2, 4),
      train_curve_id = 1:4, train_x = c(-1, 0, 1, 2),
      query_curve_id = 1:4, query_x = c(-1, 0, 1, 2),
      survival = rbind(c(.9, .6, .3), c(.8, .5, .2),
                       c(.7, .4, .1), c(.6, .3, .05)),
      censor_grid = numeric(), curve_ids = 1:4,
      censor_survival = matrix(numeric(), nrow = 4, ncol = 0)
    ),
    grow_projection = list(
      name = "grow_projection", mode = "grow",
      time = c(0, 1, 2, 3, 4, 5), event = c(1, 0, 1, 0, 1, 0),
      grid = c(0, 2, 4),
      train_curve_id = 1:6, train_x = c(-2, -1, 0, 1, 2, 3),
      query_curve_id = 1:6, query_x = c(-2, -1, 0, 1, 2, 3),
      survival = rbind(c(.9, .7, .4), c(.8, .6, .3),
                       c(.7, .5, .2), c(1.0, 1.0, .2),
                       c(NA, NA, NA), c(.5, .3, .1)),
      censor_grid = c(.5, 2, 3.5), curve_ids = 1:6,
      censor_survival = rbind(c(.9, .6, .3), c(.8, .5, .2),
                              c(.7, .4, .1), c(.6, 0, 0),
                              c(.95, .75, .5), c(.5, .25, 0))
    ),
    predict_reduced_grid = list(
      name = "predict_reduced_grid", mode = "predict",
      time = c(0, 2, 3, 4), event = c(1, 1, 0, 1),
      # The actual event at 2 is deliberately absent from this evaluation grid.
      grid = c(0, 4),
      train_curve_id = 1:4, train_x = c(10, 20, 30, 40),
      query_curve_id = c(4, 3, 2, 1), query_x = c(400, 300, 200, 100),
      survival = rbind(c(.91, .42), c(.86, .37),
                       c(.80, .29), c(.96, .18)),
      censor_grid = c(1, 2, 3), curve_ids = 1:4,
      censor_survival = rbind(c(.9, .7, .4), c(.8, .6, .3),
                              c(.7, .5, .2), c(.6, .4, .1))
    ),
    zero_censor_survival = list(
      name = "zero_censor_survival", mode = "grow",
      time = c(1, 2, 3, 4), event = c(1, 1, 0, 1), grid = c(1, 2, 4),
      train_curve_id = 1:4, train_x = c(1, 2, 3, 4),
      query_curve_id = 1:4, query_x = c(1, 2, 3, 4),
      survival = rbind(c(.5, .4, .3), c(.6, .5, .4),
                       c(.7, 1.0, .5), c(.8, .7, .2)),
      censor_grid = c(1, 2, 3), curve_ids = 1:4,
      censor_survival = rbind(c(.5, .2, .1), c(.6, 0, 0),
                              c(.7, 0, 0), c(.8, .4, .2))
    )
  )
}

make_object <- function(x) {
  yvar <- cbind(time = x$time, cens = x$event)
  train_x <- data.frame(curve_id = x$train_curve_id, marker = x$train_x)
  query_x <- data.frame(curve_id = x$query_curve_id, marker = x$query_x)
  base <- list(
    family = "surv", time.interest = x$grid,
    survival = x$survival, survival.oob = x$survival,
    predicted = rep(0, length(x$time)), predicted.oob = NULL,
    forest = list(yvar = yvar, xvar = train_x),
    xvar = query_x
  )
  if (x$mode == "grow") {
    base$yvar <- yvar
    class(base) <- c("rfsrc", "grow")
  } else {
    base$yvar <- NULL
    class(base) <- c("rfsrc", "predict")
  }
  base
}

independent_reference <- function(x) {
  n <- length(x$time)
  grid <- x$grid
  if (length(x$censor_grid) == 0L) {
    g <- matrix(1, nrow = n, ncol = length(grid))
  } else {
    projection <- vapply(grid, function(tt) sum(x$censor_grid <= tt), integer(1))
    row_index <- match(x$query_curve_id, x$curve_ids)
    if (anyNA(row_index)) stop("query rows did not map to supplied censor curves")
    projected <- cbind(1, x$censor_survival)[, 1L + projection, drop = FALSE]
    g <- projected[row_index, , drop = FALSE]
  }
  c1 <- c2 <- matrix(NA_real_, nrow = n, ncol = length(grid))
  contribution <- matrix(NA_real_, nrow = n, ncol = length(grid))
  for (i in seq_len(n)) {
    tau <- x$time[i]
    index <- sum(grid <= tau)
    g_tau <- c(1, g[i, ])[1L + index]
    c1[i, ] <- 1 * (tau <= grid & x$event[i] != 0) / g_tau
    c2[i, ] <- 1 * (tau > grid) / g[i, ]
    contribution[i, ] <- (1 * (tau > grid) - x$survival[i, ])^2 * (c1[i, ] + c2[i, ])
  }
  score <- vapply(seq_along(grid), function(j) mean(contribution[, j], na.rm = TRUE), numeric(1))
  valid_finite <- colSums(is.finite(contribution))
  valid_nonmissing <- colSums(!is.na(contribution))
  crps <- trapz(grid, score)
  list(g = g, c1 = c1, c2 = c2, contribution = contribution, score = score,
       valid_finite = valid_finite, valid_nonmissing = valid_nonmissing,
       crps = crps, crps_std = crps / max(grid))
}

all_inputs <- list()
all_projections <- list()
all_terms <- list()
all_scores <- list()
all_calls <- list()
all_predictions <- list()
for (x in make_cases()) {
  case_now <- x
  before_calls <- length(mock_calls)
  before_predictions <- length(mock_predictions)
  obj <- make_object(x)
  ans <- tryCatch(
    get.brier.survival(obj, subset = seq_along(x$time), cens.model = "rfsrc"),
    error = function(e) structure(list(message = conditionMessage(e)), class = "reference_error")
  )
  ref <- independent_reference(x)

  if (x$name == "no_censor") {
    if (length(mock_calls) != before_calls || length(mock_predictions) != before_predictions) {
      stop("no-censor case unexpectedly fitted or predicted a censor forest")
    }
    if (!inherits(ans, "reference_error") || !grepl("incorrect number of dimensions", ans$message)) {
      stop("expected unchanged-helper no-censor vector/matrix indexing defect was not observed")
    }
    helper_contribution <- matrix(NA_real_, nrow = length(x$time), ncol = length(x$grid))
    helper_score <- rep(NA_real_, length(x$grid))
    helper_crps <- helper_crps_std <- NaN
    source_error <- ans$message
  } else {
    if (inherits(ans, "reference_error")) stop(paste(x$name, ans$message))
    if (!isTRUE(all.equal(unname(t(ans$cens.dist)), unname(ref$g), tolerance = 0))) {
      stop(paste("censor projection mismatch", x$name))
    }
    if (!isTRUE(all.equal(unname(ans$brier.matx), unname(ref$contribution), tolerance = 1e-12))) {
      stop(paste("Brier contribution mismatch", x$name))
    }
    if (!isTRUE(all.equal(as.numeric(ans$brier.score$brier.score), ref$score, tolerance = 1e-12))) {
      stop(paste("Brier score mismatch", x$name))
    }
    if (!isTRUE(all.equal(ans$crps, ref$crps, tolerance = 1e-12)) ||
        !isTRUE(all.equal(ans$crps.std, ref$crps_std, tolerance = 1e-12))) {
      stop(paste("integrated score mismatch", x$name))
    }
    helper_contribution <- ans$brier.matx
    helper_score <- ans$brier.score$brier.score
    helper_crps <- ans$crps
    helper_crps_std <- ans$crps.std
    source_error <- ""
    if (length(mock_calls) != before_calls + 1L || length(mock_predictions) != before_predictions + 1L) {
      stop(paste("expected one censor fit and prediction", x$name))
    }
    call <- mock_calls[[length(mock_calls)]]
    expected_nodesize <- if (length(x$time) <= 300 && ncol(obj$forest$xvar) <= length(x$time)) 5 else 2
    if (call$ntree != 50 || call$nsplit != 1 || call$splitrule != "random" ||
        call$nodesize != expected_nodesize || call$perf.type != "none" ||
        nrow(call$data) != length(x$time)) {
      stop(paste("censor forest call contract mismatch", x$name))
    }
    expected_data <- data.frame(
      time = x$time,
      cens = 1 * (x$event == 0),
      data.frame(curve_id = x$train_curve_id, marker = x$train_x)
    )
    if (!isTRUE(all.equal(call$data, expected_data, check.attributes = FALSE))) {
      stop(paste("censor forest was not fitted on the full training cohort", x$name))
    }
    pred_input <- mock_predictions[[length(mock_predictions)]]
    if (!identical(pred_input$newdata, obj$xvar)) {
      stop(paste("prediction inputs did not match supplied full cohort", x$name))
    }
  }

  n <- length(x$time)
  all_inputs[[x$name]] <- data.frame(
    case = x$name,
    observation = rep(seq_len(n), times = length(x$grid)),
    time = rep(x$time, times = length(x$grid)),
    event = rep(x$event, times = length(x$grid)),
    train_curve_id = rep(x$train_curve_id, times = length(x$grid)),
    train_marker = rep(x$train_x, times = length(x$grid)),
    query_curve_id = rep(x$query_curve_id, times = length(x$grid)),
    query_marker = rep(x$query_x, times = length(x$grid)),
    event_grid_index = rep(seq_along(x$grid), each = n),
    event_grid_time = rep(x$grid, each = n),
    forest_survival = as.vector(x$survival),
    stringsAsFactors = FALSE
  )
  all_projections[[x$name]] <- data.frame(
    case = x$name,
    observation = rep(seq_len(n), times = length(x$grid)),
    event_grid_index = rep(seq_along(x$grid), each = n),
    event_grid_time = rep(x$grid, each = n),
    censor_survival = as.vector(ref$g),
    stringsAsFactors = FALSE
  )
  all_terms[[x$name]] <- data.frame(
    case = x$name,
    observation = rep(seq_len(n), times = length(x$grid)),
    event_grid_index = rep(seq_along(x$grid), each = n),
    event_grid_time = rep(x$grid, each = n),
    c1_event_weight = as.vector(ref$c1),
    c2_at_risk_weight = as.vector(ref$c2),
    reference_contribution = as.vector(ref$contribution),
    helper_contribution = as.vector(helper_contribution),
    stringsAsFactors = FALSE
  )
  all_scores[[x$name]] <- data.frame(
    case = x$name,
    event_grid_index = seq_along(x$grid),
    event_grid_time = x$grid,
    reference_score = as.numeric(ref$score),
    helper_score = as.numeric(helper_score),
    finite_contributors = as.integer(ref$valid_finite),
    nonmissing_contributors = as.integer(ref$valid_nonmissing),
    reference_crps = rep(ref$crps, length(x$grid)),
    helper_crps = rep(helper_crps, length(x$grid)),
    reference_crps_std = rep(ref$crps_std, length(x$grid)),
    helper_crps_std = rep(helper_crps_std, length(x$grid)),
    source_error = rep(source_error, length(x$grid)),
    stringsAsFactors = FALSE
  )
}

dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
write.csv(do.call(rbind, all_inputs), file.path(output_dir, "random-survival-cens-rfsrc-input.csv"), row.names = FALSE, na = "NA")
write.csv(do.call(rbind, all_projections), file.path(output_dir, "random-survival-cens-rfsrc-projection.csv"), row.names = FALSE, na = "NA")
write.csv(do.call(rbind, all_terms), file.path(output_dir, "random-survival-cens-rfsrc-terms.csv"), row.names = FALSE, na = "NA")
write.csv(do.call(rbind, all_scores), file.path(output_dir, "random-survival-cens-rfsrc-score.csv"), row.names = FALSE, na = "NA")
write.csv(do.call(rbind, lapply(Filter(function(x) length(x$censor_grid) > 0L, make_cases()), function(x) {
  n <- length(x$time)
  query_rows <- match(x$query_curve_id, x$curve_ids)
  data.frame(
    case = x$name,
    prediction_row = rep(seq_len(n), times = length(x$censor_grid)),
    curve_id = rep(x$query_curve_id, times = length(x$censor_grid)),
    marker = rep(x$query_x, times = length(x$censor_grid)),
    censor_grid_index = rep(seq_along(x$censor_grid), each = n),
    censor_grid_time = rep(x$censor_grid, each = n),
    censor_survival = as.vector(x$censor_survival[query_rows, , drop = FALSE]),
    stringsAsFactors = FALSE
  )
})), file.path(output_dir, "random-survival-cens-rfsrc-predicted-curves.csv"), row.names = FALSE, na = "NA")
write.csv(do.call(rbind, lapply(mock_calls, function(z) data.frame(
  case = z$case, n_train = nrow(z$data), n_covariates = ncol(z$data) - 2L,
  ntree = z$ntree, nsplit = z$nsplit, splitrule = z$splitrule,
  nodesize = z$nodesize, perf_type = z$perf.type, formula = z$formula
))), file.path(output_dir, "random-survival-cens-rfsrc-calls.csv"), row.names = FALSE, na = "NA")
write.csv(do.call(rbind, lapply(mock_predictions, function(z) data.frame(
  case = z$case, prediction_row = seq_len(nrow(z$newdata)),
  curve_id = z$newdata$curve_id, marker = z$newdata$marker
))), file.path(output_dir, "random-survival-cens-rfsrc-prediction-input.csv"), row.names = FALSE, na = "NA")
