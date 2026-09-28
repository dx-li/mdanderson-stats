# Resolve native stopping error in the ordinary or wide CondiS lasso fixture.
# The source-only setup verifies pins and installs no packages.
source("tools/reference_condis_models.R")
arguments <- commandArgs(trailingOnly = TRUE)
case_name <- if (length(arguments)) arguments[[1L]] else "wide"
stopifnot(case_name %in% c("wide", "ordinary"))
case <- input$cases[[case_name]]
x <- cbind(status = case$status, as.matrix(case$covariates))
y <- case$imputed_time
fold_ids <- answers[[case_name]]$fold_ids
requested <- answers[[case_name]]$learners$lasso$tuning

certificate <- function(fit, x, y) {
  sx <- sqrt(colMeans(sweep(x, 2L, colMeans(x))^2))
  sy <- sqrt(mean((y - mean(y))^2))
  active <- sx > 0
  z <- sweep(sweep(x[, active, drop = FALSE], 2L, colMeans(x)[active]),
             2L, sx[active], "/")
  b <- as.matrix(fit$beta)[active, , drop = FALSE] * sx[active] / sy
  residual <- (y - mean(y)) / sy - z %*% b
  gradient <- crossprod(z, residual) / nrow(x)
  penalty <- fit$lambda / sy
  threshold <- matrix(penalty, nrow(b), ncol(b), byrow = TRUE)
  violation <- ifelse(b != 0, abs(gradient - threshold * sign(b)),
                      pmax(abs(gradient) - threshold, 0))
  list(lambda = fit$lambda, kkt = apply(violation, 2L, max),
       objective = colMeans(residual^2) / 2 + penalty * colSums(abs(b)))
}

default <- glmnet(x, y, alpha = 1)
# Fix the path to exactly the default reported lambdas. This prevents any
# automatic-path differences from obscuring the point-solver comparison.
tight <- glmnet(x, y, alpha = 1, lambda = default$lambda, thresh = 1e-15)
rmse <- matrix(NA_real_, length(unique(fold_ids)), length(requested))
fold_certificates <- list()
for (fold in sort(unique(fold_ids))) {
  test <- which(fold_ids == fold)
  train <- setdiff(seq_along(y), test)
  original <- glmnet(x[train, , drop = FALSE], y[train], alpha = 1)
  fit <- glmnet(x[train, , drop = FALSE], y[train], alpha = 1,
                 lambda = original$lambda, thresh = 1e-15)
  prediction <- predict(fit, x[test, , drop = FALSE], s = requested)
  rmse[fold + 1L, ] <- sqrt(colMeans((prediction - y[test])^2))
  fold_certificates[[as.character(fold)]] <- list(
    default = certificate(original, x[train, , drop = FALSE], y[train]),
    tight = certificate(fit, x[train, , drop = FALSE], y[train]))
}
result <- list(case = case_name, threshold = 1e-15, tuning = requested,
  fold_ids = fold_ids, default = certificate(default, x, y),
  tight = certificate(tight, x, y), fold_certificates = fold_certificates,
  all_fitted = as.matrix(predict(tight, x, s = requested)),
  fold_rmse = rmse, mean_rmse = colMeans(rmse))
fixture <- if (case_name == "wide") "condis-lasso-converged.json" else
  "condis-lasso-ordinary-converged.json"
jsonlite::write_json(result, file.path("tests/fixtures", fixture),
                     auto_unbox = TRUE, digits = 17, pretty = TRUE)
cat("Native final-point KKT default:", tail(result$default$kkt, 1L),
    "tight:", tail(result$tight$kkt, 1L), "\n")
cat("Native final-point objective default:", tail(result$default$objective, 1L),
    "tight:", tail(result$tight$objective, 1L), "\n")
