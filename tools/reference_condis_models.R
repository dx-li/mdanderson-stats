# CondiS-X learner references from pinned caret and glmnet sources.
# Run from the repository root with BLAS/OpenMP threads limited to one.
# Requires existing Rcpp, RcppEigen, Matrix and jsonlite; installs nothing.
options(warn = 2, digits = 17)
root <- normalizePath("research/raw/CondiS")
provenance <- jsonlite::fromJSON("docs/condis-sources.json", simplifyVector = FALSE)
model_snapshot <- file.path(root, "caret-models.RData")
stopifnot(identical(system2("git", c("hash-object", shQuote(model_snapshot)),
                           stdout = TRUE), provenance$sources[[2L]]$git_blob))
for (package in provenance$learner_reference_sources) {
  for (item in package$files) {
    path <- file.path(root, package$directory, item$path)
    actual <- system2("git", c("hash-object", shQuote(path)), stdout = TRUE)
    stopifnot(identical(actual, item$git_blob))
  }
}
caret <- file.path(root, "caret-7.0-1")
glmnet_root <- file.path(root, "glmnet-4.1-10")
output <- file.path(root, "learner-reference")
dir.create(output, showWarnings = FALSE)

# Keep the native dense Gaussian wrapper and all computational headers intact.
# Restrict the include graph and exported wrapper to this one model family.
wrapper <- readLines(file.path(glmnet_root, "src/elnet_exp.cpp"))
wrapper <- wrapper[seq_len(which(wrapper == "// Gaussian for sparse X.") - 1L)]
headers <- c("elnet_point/internal/gaussian_cov.hpp",
             "elnet_point/internal/gaussian_naive.hpp",
             "elnet_point/gaussian_cov.hpp", "elnet_point/gaussian_naive.hpp",
             "elnet_driver/gaussian.hpp")
wrapper[wrapper == "#include <glmnetpp>"] <- paste(
  sprintf("#include <glmnetpp_bits/%s>", headers), collapse = "\n")
wrapper[wrapper == "void setpb_cpp(SEXP, int);"] <-
  "void setpb_cpp(SEXP, int) {} // Progress display is disabled."
internal <- readLines(file.path(glmnet_root, "src/internal.cpp"))
defaults <- internal[grepl("^(double|int) InternalParams::", internal)]
cpp <- file.path(output, "gaussian-reference.cpp")
writeLines(c("// [[Rcpp::depends(RcppEigen)]]",
             "// [[Rcpp::plugins(cpp17)]]", wrapper, defaults), cpp)
makevars <- file.path(output, "Makevars")
writeLines(c("CXXFLAGS=-O0", "CXX11FLAGS=-O0", "CXX14FLAGS=-O0",
             "CXX17FLAGS=-O0", "MAKEFLAGS=-j1"), makevars)
Sys.setenv(R_MAKEVARS_USER = makevars, MAKEFLAGS = "-j1",
           PKG_CPPFLAGS = paste0('-I"', glmnet_root, '/src/glmnetpp/include" ',
                                 '-I"', glmnet_root, '/src"'))
Rcpp::sourceCpp(cpp, cacheDir = file.path(output, "cpp-cache"),
                rebuild = FALSE, showOutput = FALSE, verbose = FALSE)

suppressPackageStartupMessages(library(Matrix))
for (name in c("glmnet", "elnet", "getcoef", "zeromat", "fix.lam", "jerr",
               "jerr.elnet", "predict.glmnet", "predict.elnet", "lambda.interp")) {
  source(file.path(glmnet_root, "R", paste0(name, ".R")))
}
# This source-only run uses factory controls. The unchanged C++ defaults above
# determine numerical behavior; these are the two fields read by glmnet.R.
glmnet.control <- function(...) {
  stopifnot(length(list(...)) == 0L)
  list(itrace = 0L, big = 9.9e35)
}

c_source <- file.path(caret, "src/caret.c")
library_path <- file.path(output, paste0("caret", .Platform$dynlib.ext))
compile_log <- file.path(output, "caret-build.log")
if (!file.exists(library_path)) {
  # R's make rules do not escape spaces in source/output paths. Compile the
  # unchanged source using local filenames in the reference output directory.
  build_knn <- function() {
    original_directory <- getwd()
    on.exit(setwd(original_directory))
    copy <- file.path(output, "caret.c")
    if (!file.exists(copy)) stopifnot(file.copy(c_source, copy))
    stopifnot(unname(tools::md5sum(copy)) == unname(tools::md5sum(c_source)))
    setwd(output)
    system2(file.path(R.home("bin"), "R"),
      c("CMD", "SHLIB", "-o", basename(library_path), "caret.c"),
      stdout = "caret-build.log", stderr = "caret-build.log")
  }
  status <- build_knn()
  if (status != 0L) stop(paste(readLines(compile_log), collapse = "\n"))
}
dyn.load(library_path)
for (name in c("knnreg", "createDataPartition", "trainControl")) {
  source(file.path(caret, "R", paste0(name, ".R")))
}
model_data <- new.env()
load(model_snapshot, envir = model_data)
models <- model_data$models
# Redirect only glmnet namespace dispatch to the source-loaded native function.
redirect <- function(expression) {
  if (identical(expression, quote(glmnet::glmnet))) return(quote(glmnet))
  if (is.call(expression)) return(as.call(lapply(as.list(expression), redirect)))
  expression
}
body(models$glmnet$fit) <- redirect(body(models$glmnet$fit))
environment(models$glmnet$fit) <- .GlobalEnv
environment(models$knn$fit) <- .GlobalEnv

input <- jsonlite::fromJSON(file.path(output, "inputs.json"))
answers <- list()
for (case in input$cases) {
  x <- as.matrix(case$covariates)
  x <- cbind(status = case$status, x)
  y <- case$imputed_time
  set.seed(729)
  held_out <- createFolds(y, k = case$folds, list = TRUE)
  fold_ids <- integer(length(y))
  for (fold in seq_along(held_out)) fold_ids[held_out[[fold]]] <- fold - 1L
  learners <- list()
  for (method in c("ridge", "lasso", "knn")) {
    is_knn <- method == "knn"
    model <- models[[if (is_knn) "knn" else "glmnet"]]
    grid <- if (is_knn) model$grid(x, y, len = 3L) else
      expand.grid(alpha = as.integer(method == "lasso"),
                  lambda = seq(.01, 10, length.out = 10) * case$time_scale)
    # caret sorts a performance data frame, which always has metric columns.
    # Supply one placeholder to preserve a data frame for the one-column k grid.
    grid <- model$sort(cbind(grid, .ordering = 0))[, names(grid), drop = FALSE]
    tuning <- grid[[if (is_knn) "k" else "lambda"]]
    fit_one <- function(rows, param, last = FALSE) {
      model$fit(x[rows, , drop = FALSE], y[rows], wts = NULL, param = param,
                lev = NULL, last = last, classProbs = FALSE)
    }
    predict_grid <- function(rows, test) {
      if (is_knn) {
        return(vapply(seq_len(nrow(grid)), function(j) {
          fit <- fit_one(rows, grid[j, , drop = FALSE])
          as.vector(predict(fit, x[test, , drop = FALSE]))
        }, numeric(length(test))))
      }
      fit <- fit_one(rows, grid[1L, , drop = FALSE])
      as.matrix(predict(fit, x[test, , drop = FALSE], s = tuning))
    }
    rmse <- matrix(NA_real_, length(held_out), nrow(grid))
    for (fold in seq_along(held_out)) {
      test <- held_out[[fold]]
      predictions <- predict_grid(setdiff(seq_along(y), test), test)
      rmse[fold, ] <- sqrt(colMeans((predictions - y[test])^2))
    }
    mean_rmse <- colMeans(rmse)
    best <- which.min(mean_rmse)
    fit <- fit_one(seq_along(y), grid[best, , drop = FALSE], last = TRUE)
    fitted <- if (is_knn) as.vector(predict(fit, x)) else
      as.vector(predict(fit, x, s = tuning[best]))
    refined <- fitted
    refined[case$status == 1L] <- case$observed_time[case$status == 1L]
    learners[[method]] <- list(tuning = tuning, fold_rmse = rmse,
      mean_rmse = mean_rmse, selected = tuning[best], fitted_time = fitted,
      refined_time = refined, all_fitted = predict_grid(seq_along(y), seq_along(y)))
    if (!is_knn) learners[[method]]$path <- list(lambda = fit$lambda,
      intercept = fit$a0, coefficients = as.matrix(fit$beta), deviance = fit$dev.ratio)
  }
  answers[[case$name]] <- list(fold_ids = fold_ids, learners = learners)
}
# Exact ties and near-ties isolate native insertion-order behavior.
tie_cases <- list(c(1, sqrt(1.00001)), c(1, 1, sqrt(.99999)),
                  c(sqrt(.99999), 1, 1), c(0, 0, 1))
ties <- lapply(tie_cases, function(values) {
  list(train = values, response = seq_along(values) * 10,
       fitted = as.vector(knnregTrain(matrix(values), matrix(0),
                                      seq_along(values) * 10, k = 1L)))
})
result <- list(cases = answers, knn_ties = ties,
               provenance = list(caret = "7.0-1", glmnet = "4.1-10",
                                  r = as.character(getRversion())))
jsonlite::write_json(result, "tests/fixtures/condis-models-native.json",
                     auto_unbox = TRUE, digits = 17, pretty = TRUE)
cat("Native CondiS learner references saved for", length(answers), "cases.\n")
