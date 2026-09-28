# Source-only native Gaussian boosting references for CondiS-X.
# Run from the repository root; serial compilation and no package installation.
options(digits = 17, warn = 2)
root <- normalizePath("research/raw/CondiS")
provenance <- jsonlite::fromJSON("research/condis-gbm-sources.json", simplifyVector = FALSE)
source_root <- file.path(root, provenance$directory)
build <- file.path(root, "gbm-reference")
dir.create(build, showWarnings = FALSE)
for (item in provenance$files) {
  path <- file.path(source_root, item$path)
  stopifnot(identical(system2("git", c("hash-object", shQuote(path)), stdout = TRUE),
                     item$git_blob))
  if (startsWith(item$path, "src/")) {
    copy <- file.path(build, basename(item$path))
    if (!file.exists(copy)) stopifnot(file.copy(path, copy))
    stopifnot(unname(tools::md5sum(path)) == unname(tools::md5sum(copy)))
  }
}
snapshot <- file.path(root, "caret-models.RData")
stopifnot(identical(system2("git", c("hash-object", shQuote(snapshot)), stdout = TRUE),
                   "61ff1c862001ba6ab9b461a4de5d546af20223e9"))
makevars <- file.path(build, "Makevars")
writeLines(c("CFLAGS=-O0", "CXXFLAGS=-O0", "CXX11FLAGS=-O0",
             "CXX14FLAGS=-O0", "CXX17FLAGS=-O0", "MAKEFLAGS=-j1"), makevars)
Sys.setenv(R_MAKEVARS_USER = makevars, MAKEFLAGS = "-j1", PKG_CPPFLAGS = "")
library_path <- file.path(build, paste0("gbm", .Platform$dynlib.ext))
compile <- function() {
  previous <- getwd()
  on.exit(setwd(previous))
  setwd(build)
  files <- list.files(pattern = "\\.(cpp|c)$")
  system2(file.path(R.home("bin"), "R"),
    c("CMD", "SHLIB", "-o", basename(library_path), files),
    stdout = "build.log", stderr = "build.log")
}
status <- compile()
if (status != 0L) stop(paste(readLines(file.path(build, "build.log")), collapse = "\n"))
dyn.load(library_path)
for (name in c("utils", "gbm-internals", "gbm.fit", "predict.gbm")) {
  source(file.path(source_root, "R", paste0(name, ".R")))
}
models <- new.env()
load(snapshot, envir = models)
model <- models$models$gbm
redirect <- function(expression) {
  if (identical(expression, quote(gbm::gbm.fit))) return(quote(gbm.fit))
  if (is.call(expression)) return(as.call(lapply(as.list(expression), redirect)))
  expression
}
body(model$fit) <- redirect(body(model$fit))
environment(model$fit) <- .GlobalEnv
input <- jsonlite::fromJSON("tests/fixtures/condis-gbm-inputs.json")
answers <- list()
for (case_index in seq_along(input$cases)) {
  name <- names(input$cases)[case_index]
  case <- input$cases[[case_index]]
  x <- cbind(status = case$status, as.matrix(case$covariates))
  colnames(x) <- c("status", paste0("x", seq_len(ncol(x) - 1L)))
  y <- case$imputed_time
  folds <- case$fold_ids
  grid <- model$sort(model$grid(x, y, len = 3L))
  depths <- sort(unique(grid$interaction.depth))
  fit_one <- function(rows, param) model$fit(
    x[rows, , drop = FALSE], y[rows], wts = NULL, param = param,
    lev = NULL, last = FALSE, classProbs = FALSE, verbose = FALSE)
  fold_rmse <- matrix(NA_real_, case$folds, nrow(grid))
  fold_paths <- lapply(0:(case$folds - 1L), function(fold) {
    test <- which(folds == fold)
    train <- setdiff(seq_along(y), test)
    lapply(depths, function(depth) {
      indices <- which(grid$interaction.depth == depth)
      param <- grid[indices[which.max(grid$n.trees[indices])], , drop = FALSE]
      seed <- 12000L + case_index * 100L + fold * 10L + depth
      set.seed(seed)
      fit <- fit_one(train, param)
      prediction <- as.matrix(predict(fit, x[test, , drop = FALSE],
                                      n.trees = grid$n.trees[indices], type = "response"))
      fold_rmse[fold + 1L, indices] <<- sqrt(colMeans((prediction - y[test])^2))
      list(seed = seed, depth = depth, candidate_indices = indices - 1L,
           prediction = prediction)
    })
  })
  means <- colMeans(fold_rmse)
  best <- which.min(means)
  full_paths <- lapply(depths, function(depth) {
    indices <- which(grid$interaction.depth == depth)
    param <- grid[indices[which.max(grid$n.trees[indices])], , drop = FALSE]
    seed <- 13000L + case_index * 100L + depth
    set.seed(seed)
    fit <- fit_one(seq_along(y), param)
    list(seed = seed, depth = depth, candidate_indices = indices - 1L,
         init = fit$initF, train_error = fit$train.error[grid$n.trees[indices]],
         fitted = as.matrix(predict(fit, x, n.trees = grid$n.trees[indices])))
  })
  refit_seed <- 14000L + case_index
  set.seed(refit_seed)
  refit <- fit_one(seq_along(y), grid[best, , drop = FALSE])

  # Gaussian fitting consumes exactly one uniform draw per row and tree.
  # Compare the RNG state after the actual native fit to verify the saved stream.
  trace_seed <- 15000L + case_index
  set.seed(trace_seed)
  draws <- matrix(runif(nrow(x) * 3L), nrow(x), 3L)
  expected_rng_state <- .Random.seed
  set.seed(trace_seed)
  param <- data.frame(interaction.depth = 3L, n.trees = 3L,
                      shrinkage = 0.1, n.minobsinnode = 10L)
  trace <- fit_one(seq_along(y), param)
  stopifnot(identical(.Random.seed, expected_rng_state))
  trees <- lapply(trace$trees, function(tree) {
    names(tree) <- c("split_variable", "split_point", "left", "right", "missing",
                     "improvement", "weight", "prediction")
    tree
  })
  answers[[name]] <- list(grid = grid, fold_ids = folds, fold_paths = fold_paths,
    fold_rmse = fold_rmse, mean_rmse = means, best_index = best - 1L,
    full_paths = full_paths,
    selected_fit = list(seed = refit_seed, parameters = grid[best, , drop = FALSE],
      fitted = as.vector(predict(refit, x, n.trees = grid$n.trees[best])),
      train_error = tail(refit$train.error, 1L)),
    trace = list(seed = trace_seed, uniform_draws = draws, init = trace$initF,
      shrinkage = 0.1, bag_fraction = 0.5, min_node_observations = 10L,
      trees = trees, fitted = as.matrix(predict(trace, x, n.trees = 1:3)),
      train_error = trace$train.error))
}
small_sample_error <- tryCatch({
  gbm.fit(matrix(seq_len(42L), ncol = 1L), seq_len(42L), distribution = "gaussian",
          n.trees = 1L, shrinkage = 0.1, verbose = FALSE)
  stop("Expected native sample-size rejection did not occur")
}, error = function(error) conditionMessage(error))
stopifnot(grepl("nTrain * bag.fraction", small_sample_error, fixed = TRUE))
edge <- gbm.fit(matrix(seq_len(43L), ncol = 1L), seq_len(43L),
                distribution = "gaussian", n.trees = 1L, shrinkage = 0.1, verbose = FALSE)
stopifnot(all(is.finite(edge$fit)))
jsonlite::write_json(list(cases = answers, small_sample_error = small_sample_error,
  smallest_default_training_sample = 43L,
  provenance = list(gbm = provenance$version, revision = provenance$revision,
                    r = as.character(getRversion()), gaussian_trace_rng_state_verified = TRUE)),
  "tests/fixtures/condis-gbm-native.json", auto_unbox = TRUE, digits = 17, pretty = TRUE)
for (name in names(answers)) cat(name, "best index", answers[[name]]$best_index,
  "mean RMSE", min(answers[[name]]$mean_rmse), "\n")
