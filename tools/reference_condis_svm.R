# Native CondiS-X SVM references; run from repository root with one BLAS thread.
# Compiles unchanged kernlab sources and sources original S4 fit/predict methods.
# Uses existing R/jsonlite and the C/C++ toolchain; installs no packages.
options(digits = 17, warn = 2)
root <- normalizePath("research/raw/CondiS")
provenance <- jsonlite::fromJSON("docs/condis-svm-sources.json", simplifyVector = FALSE)
source_root <- file.path(root, provenance$directory)
for (item in provenance$files) {
  actual <- system2("git", c("hash-object", shQuote(file.path(source_root, item$path))),
                    stdout = TRUE)
  stopifnot(identical(actual, item$git_blob))
}
model_snapshot <- file.path(root, "caret-models.RData")
stopifnot(identical(system2("git", c("hash-object", shQuote(model_snapshot)), stdout = TRUE),
                   "61ff1c862001ba6ab9b461a4de5d546af20223e9"))
output <- file.path(root, "svm-reference")
dir.create(output, showWarnings = FALSE)
library_path <- file.path(output, paste0("kernlab-reference", .Platform$dynlib.ext))
native_files <- list.files(file.path(source_root, "src"), pattern = "\\.(c|cpp|h)$")
for (name in native_files) {
  source_path <- file.path(source_root, "src", name)
  copy <- file.path(output, name)
  if (!file.exists(copy)) stopifnot(file.copy(source_path, copy))
  stopifnot(identical(unname(tools::md5sum(copy)), unname(tools::md5sum(source_path))))
}
makevars <- file.path(output, "reference.Makevars")
writeLines(c("CFLAGS=-O0", "CXXFLAGS=-O0", "CXX11FLAGS=-O0",
             "CXX14FLAGS=-O0", "CXX17FLAGS=-O0", "MAKEFLAGS=-j1"), makevars)
r <- file.path(R.home("bin"), "R")
libs <- vapply(c("LAPACK_LIBS", "BLAS_LIBS", "FLIBS"), function(key)
  system2(r, c("CMD", "config", key), stdout = TRUE), character(1))
Sys.setenv(R_MAKEVARS_USER = makevars, MAKEFLAGS = "-j1",
           PKG_LIBS = paste(libs, collapse = " "))
if (!file.exists(library_path)) {
  build <- function() {
    old <- setwd(output)
    on.exit(setwd(old))
    files <- native_files[grepl("\\.(c|cpp)$", native_files)]
    system2(r, c("CMD", "SHLIB", "-o", basename(library_path), files),
            stdout = "build.log", stderr = "build.log")
  }
  status <- build()
  if (status != 0L) stop(paste(readLines(file.path(output, "build.log")), collapse = "\n"))
}
dll <- dyn.load(library_path)
smo_optim <- getNativeSymbolInfo("smo_optim", dll)
for (name in c("aobjects", "kernelmatrix", "kernels", "ksvm", "sigest")) {
  source(file.path(source_root, "R", paste0(name, ".R")))
}
metadata <- new.env()
load(model_snapshot, envir = metadata)
model <- metadata$models$svmRadial
redirect <- function(expression) {
  if (identical(expression, quote(kernlab::sigest))) return(quote(sigest))
  if (identical(expression, quote(kernlab::ksvm))) return(quote(ksvm))
  if (is.call(expression)) return(as.call(lapply(as.list(expression), redirect)))
  expression
}
body(model$grid) <- redirect(body(model$grid))
body(model$fit) <- redirect(body(model$fit))
environment(model$grid) <- environment(model$fit) <- .GlobalEnv
warnings <- character()
native_call <- function(expression) withCallingHandlers(expression, warning = function(w) {
  if (!grepl("constant. Cannot scale data", conditionMessage(w), fixed = TRUE)) stop(w)
  warnings <<- unique(c(warnings, conditionMessage(w)))
  invokeRestart("muffleWarning")
})
certificate <- function(fit, x, y, cost, epsilon = 0.1) {
  beta <- numeric(length(y))
  beta[alphaindex(fit)] <- as.vector(coef(fit))
  scaling <- scaling(fit)
  if (!is.null(scaling)) {
    x <- scale(x, center = scaling$x.scale$`scaled:center`,
                 scale = scaling$x.scale$`scaled:scale`)
    y <- (y - scaling$y.scale$`scaled:center`) / scaling$y.scale$`scaled:scale`
  }
  kernel <- as.matrix(kernelMatrix(kernelf(fit), x))
  prediction <- as.vector(kernel %*% beta - b(fit))
  residual <- y - prediction
  norm <- as.numeric(crossprod(beta, kernel %*% beta))
  dual_min <- norm / 2 + epsilon * sum(abs(beta)) - sum(y * beta)
  primal <- norm / 2 + cost * sum(pmax(abs(residual) - epsilon, 0))
  violation <- pmax(abs(residual) - epsilon, 0)
  violation[beta > 0] <- abs(residual[beta > 0] - epsilon)
  violation[beta < 0] <- abs(residual[beta < 0] + epsilon)
  violation[beta >= cost] <- pmax(epsilon - residual[beta >= cost], 0)
  violation[beta <= -cost] <- pmax(epsilon + residual[beta <= -cost], 0)
  list(primal_objective = primal, dual_min_objective = dual_min,
       primal_dual_gap = primal + dual_min, kkt = max(violation),
       equality_residual = abs(sum(beta)), box_violation = max(pmax(abs(beta) - cost, 0)))
}
input <- jsonlite::fromJSON("tests/fixtures/condis-svm-inputs.json")
answers <- list()
for (case in input$cases) {
  x <- cbind(status = case$status, as.matrix(case$covariates))
  y <- case$imputed_time
  n <- length(y)
  set.seed(829)
  grid <- native_call(model$grid(x, y, len = 3L))
  grid <- model$sort(grid)
  set.seed(829)
  pairs <- cbind(sample(seq_len(n), floor(n / 2), replace = TRUE),
                  sample(seq_len(n), floor(n / 2), replace = TRUE)) - 1L
  folds <- case$fold_ids
  fit_one <- function(rows, param, tolerance) native_call(model$fit(
    x[rows, , drop = FALSE], y[rows], wts = NULL, param = param,
    lev = NULL, last = FALSE, classProbs = FALSE, tol = tolerance))
  tolerances <- list()
  for (tolerance in c(1e-3, 1e-8)) {
    rmse <- matrix(NA_real_, length(unique(folds)), nrow(grid))
    for (fold in sort(unique(folds))) {
      test <- which(folds == fold)
      train <- setdiff(seq_len(n), test)
      for (j in seq_len(nrow(grid))) {
        fit <- fit_one(train, grid[j, , drop = FALSE], tolerance)
        prediction <- as.vector(predict(fit, x[test, , drop = FALSE]))
        rmse[fold + 1L, j] <- sqrt(mean((prediction - y[test])^2))
      }
    }
    means <- colMeans(rmse)
    best <- which.min(means)
    fits <- lapply(seq_len(nrow(grid)), function(j) {
      fit <- fit_one(seq_len(n), grid[j, , drop = FALSE], tolerance)
      beta <- numeric(n)
      beta[alphaindex(fit)] <- as.vector(coef(fit))
      list(fitted = as.vector(predict(fit, x)), coefficients = beta,
           rho = b(fit), dual_objective = obj(fit), scaling = scaling(fit),
           support = alphaindex(fit) - 1L,
           certificate = certificate(fit, x, y, grid$C[j]))
    })
    tolerances[[if (tolerance == 1e-3) "default" else "tight"]] <- list(
      tolerance = tolerance, fold_rmse = rmse, mean_rmse = means,
      best_index = best - 1L, selected_C = grid$C[best], fits = fits)
  }
  answers[[case$name]] <- list(sigma = grid$sigma[1L], C = grid$C,
    sigma_pairs = pairs, fold_ids = folds, fits = tolerances)
}
# A fixed-sigma all-bound fit exercises the midpoint intercept when no support
# vector has a coefficient strictly inside the cost bounds.
bound_x <- cbind(status = rep(1, 4), covariate = 0:3)
bound_y <- c(1, 2, 4, 6)
bound_fit <- native_call(ksvm(bound_x, bound_y, kernel = "rbfdot",
                              kpar = list(sigma = 0.5), C = 0.01, tol = 1e-8))
all_bound <- list(x = bound_x, y = bound_y, sigma = 0.5, C = 0.01,
                 fitted = as.vector(predict(bound_fit, bound_x)),
                 coefficients = as.vector(coef(bound_fit)), rho = b(bound_fit),
                 certificate = certificate(bound_fit, bound_x, bound_y, 0.01))
jsonlite::write_json(list(cases = answers, all_bound = all_bound, warnings = warnings,
  provenance = list(kernlab = "0.9-33", revision = provenance$revision,
                    caret_models_git_blob = "61ff1c862001ba6ab9b461a4de5d546af20223e9",
                    r = as.character(getRversion()))),
  "tests/fixtures/condis-svm-native.json", auto_unbox = TRUE, digits = 17, pretty = TRUE)
cat("Native SVM references saved:", length(answers), "cases, default and tight tolerance.\n")
for (name in names(answers)) cat(name, "sigma", answers[[name]]$sigma,
                               "selected C", answers[[name]]$fits$tight$selected_C, "\n")
