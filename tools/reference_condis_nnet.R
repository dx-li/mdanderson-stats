# CondiS-X neural references using the already-installed nnet 7.3-19.
# No installation; explicit starts isolate the fit from cross-language RNGs.
# Run from the repository root with one BLAS/OpenMP thread.
options(digits = 17, warn = 2)
stopifnot(as.character(packageVersion("nnet")) == "7.3.19")
provenance <- jsonlite::fromJSON("research/condis-nnet-sources.json",
                                simplifyVector = FALSE)
root <- normalizePath("research/raw/CondiS")
for (item in provenance$files) {
  path <- file.path(root, provenance$directory, item$path)
  stopifnot(identical(system2("git", c("hash-object", shQuote(path)), stdout = TRUE),
                     item$git_blob))
}
snapshot <- file.path(root, "caret-models.RData")
stopifnot(identical(system2("git", c("hash-object", shQuote(snapshot)), stdout = TRUE),
                   "61ff1c862001ba6ab9b461a4de5d546af20223e9"))
models <- new.env()
load(snapshot, envir = models)
model <- models$models$nnet
input <- jsonlite::fromJSON("tests/fixtures/condis-models-inputs.json")
fold_reference <- jsonlite::fromJSON("tests/fixtures/condis-models-native.json")
answers <- list()
initial <- function(p, size, tag) {
  count <- (p + 1L) * size + size + 1L
  sin((seq_len(count) + tag) * 0.73) * 0.65
}
native_gradient <- function(net, x, y, starts) {
  # Retain the setup .C return value: the original kernel keeps pointers to its
  # training/weight buffers until VR_unset_net. There are no intervening fits.
  symbols <- getDLLRegisteredRoutines("nnet")$.C
  .C(symbols$VR_set_net, as.integer(net$n), as.integer(net$nconn),
     as.integer(net$conn), as.double(rep(net$decay, length.out = length(starts))),
     as.integer(net$nsunits), as.integer(net$entropy),
     as.integer(net$softmax), as.integer(net$censored))
  on.exit(.C(symbols$VR_unset_net))
  setup <- .C(symbols$VR_dovm, as.integer(length(y)),
    as.double(cbind(x, y)), rep(1.0, length(y)), as.integer(length(starts)),
    as.double(starts), value = double(1L), as.integer(0L), FALSE,
    rep(1L, length(starts)), as.double(1e-4), as.double(1e-8), integer(1L))
  derivative <- .C(symbols$VR_dfunc, as.double(starts),
                    gradient = double(length(starts)), objective = double(1L))
  stopifnot(is.finite(setup$value), all(is.finite(derivative$gradient)))
  list(objective = derivative$objective, gradient = derivative$gradient)
}
for (name in c("ordinary", "wide")) {
  case <- input$cases[[name]]
  x <- cbind(status = case$status, as.matrix(case$covariates))
  y <- case$imputed_time
  folds <- fold_reference$cases[[name]]$fold_ids
  grid <- model$sort(model$grid(x, y, len = 3L))
  rmse <- matrix(NA_real_, length(unique(folds)), nrow(grid))
  convergence <- matrix(NA_integer_, nrow(rmse), ncol(rmse))
  fold_fits <- vector("list", nrow(rmse))
  fit_one <- function(rows, j, starts, maxit = 100L) model$fit(
    x[rows, , drop = FALSE], y[rows], wts = NULL, param = grid[j, , drop = FALSE],
    lev = NULL, last = FALSE, classProbs = FALSE, linout = TRUE,
    Wts = starts, maxit = maxit, trace = FALSE)
  for (fold in sort(unique(folds))) {
    test <- which(folds == fold)
    train <- setdiff(seq_along(y), test)
    fold_fits[[fold + 1L]] <- lapply(seq_len(nrow(grid)), function(j) {
      starts <- initial(ncol(x), grid$size[j], 100 * (fold + 1L) + j)
      fit <- fit_one(train, j, starts)
      prediction <- as.vector(predict(fit, as.data.frame(x[test, , drop = FALSE])))
      rmse[fold + 1L, j] <<- sqrt(mean((prediction - y[test])^2))
      convergence[fold + 1L, j] <<- fit$convergence
      list(initial_weights = starts, fitted_weights = fit$wts,
           prediction = prediction, objective = fit$value,
           convergence = fit$convergence)
    })
  }
  means <- colMeans(rmse)
  best <- which.min(means)
  full <- lapply(seq_len(nrow(grid)), function(j) {
    starts <- initial(ncol(x), grid$size[j], 10000 + j)
    fit <- fit_one(seq_along(y), j, starts)
    list(initial_weights = starts, fitted_weights = fit$wts,
         fitted = as.vector(predict(fit, as.data.frame(x))),
         objective = fit$value, convergence = fit$convergence)
  })
  # One trajectory distinguishes objective/gradient errors from optimizer steps.
  trajectory_j <- which(grid$size == 3L & grid$decay == 0.1)[1L]
  starts <- initial(ncol(x), 3L, 30000)
  trajectory <- lapply(c(0L, 1L, 2L, 5L, 10L, 100L), function(limit) {
    fit <- fit_one(seq_along(y), trajectory_j, starts, maxit = limit)
    list(maxit = limit, fitted_weights = fit$wts, objective = fit$value,
         fitted = as.vector(predict(fit, as.data.frame(x))),
         convergence = fit$convergence)
  })
  zero_fit <- fit_one(seq_along(y), trajectory_j, starts, maxit = 0L)
  derivative <- native_gradient(zero_fit, x, y, starts)
  # Larger networks can amplify floating-point differences during BFGS. Save
  # an actual CV fit at intermediate limits to distinguish this from a fold,
  # starting-weight or optimizer-control-flow mismatch.
  cv_fold <- min(folds)
  cv_test <- which(folds == cv_fold)
  cv_train <- which(folds != cv_fold)
  cv_j <- which(grid$size == 5L & grid$decay == 0.1)[1L]
  cv_start <- fold_fits[[cv_fold + 1L]][[cv_j]]$initial_weights
  cv_path <- lapply(c(0L, 1L, 2L, 5L, 10L, 20L, 40L, 60L, 100L), function(limit) {
    fit <- fit_one(cv_train, cv_j, cv_start, maxit = limit)
    derivative <- native_gradient(fit, x[cv_train, , drop = FALSE],
                                  y[cv_train], fit$wts)
    list(maxit = limit, fitted_weights = fit$wts, objective = fit$value,
         gradient = derivative$gradient,
         prediction = as.vector(predict(fit, as.data.frame(x[cv_test, , drop = FALSE]))),
         convergence = fit$convergence)
  })
  answers[[name]] <- list(grid = grid, fold_ids = folds,
    fold_rmse = rmse, mean_rmse = means, convergence = convergence,
    best_index = best - 1L, selected_size = grid$size[best],
    selected_decay = grid$decay[best], fold_fits = fold_fits, full_fits = full,
    trajectory = list(size = 3L, decay = 0.1, initial_weights = starts,
                      initial_derivative = derivative, fits = trajectory),
    cv_trajectory = list(fold_id = cv_fold, size = 5L, decay = 0.1,
      initial_weights = cv_start, train_rows = cv_train - 1L,
      test_rows = cv_test - 1L, fits = cv_path))
}
# A four-row analytical example checks weight order, bias decay and gradient
# independently of BFGS and of the statistical training fixtures above.
analytical_x <- rbind(c(0, 0), c(1, 0), c(0, 1), c(1, 1))
analytical_y <- c(1, 2, 3, 4)
analytical_start <- c(0, log(2), -log(2), 1, 2)
analytical_fit <- nnet::nnet(analytical_x, analytical_y, size = 1L,
  Wts = analytical_start, linout = TRUE, decay = 0.1, maxit = 0L, trace = FALSE)
analytical_derivative <- native_gradient(
  analytical_fit, analytical_x, analytical_y, analytical_start)
expected_fitted <- c(2, 7/3, 5/3, 2)
expected_objective <- 62/9 + 0.1 * (5 + 2 * log(2)^2)
expected_gradient <- c(-17/9, -46/27, -86/27, -4, -13/9) +
  0.2 * analytical_start
stopifnot(max(abs(as.vector(analytical_fit$fitted.values) - expected_fitted)) < 1e-14,
          abs(analytical_derivative$objective - expected_objective) < 1e-13,
          max(abs(analytical_derivative$gradient - expected_gradient)) < 1e-13)
analytical <- list(x = analytical_x, y = analytical_y, size = 1L, decay = 0.1,
  initial_weights = analytical_start, fitted = expected_fitted,
  objective = expected_objective, gradient = expected_gradient)
jsonlite::write_json(list(cases = answers, analytical = analytical,
  provenance = list(nnet = as.character(packageVersion("nnet")),
                    r = as.character(getRversion()), explicit_initial_weights = TRUE)),
  "tests/fixtures/condis-nnet-native.json", auto_unbox = TRUE, digits = 17, pretty = TRUE)
cat("Native nnet references saved with explicit starts:", length(answers), "cases.\n")
for (name in names(answers)) cat(name, "size", answers[[name]]$selected_size,
                               "decay", answers[[name]]$selected_decay,
                               "fold iteration-limit fits", sum(answers[[name]]$convergence != 0),
                               "\n")
