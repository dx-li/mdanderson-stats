#!/usr/bin/env Rscript

# Run only the function definitions from the cached CRAN arm source, then call
# its private bayesglm.fit directly on author-equivalent Bernoulli-expanded
# data. This is a deterministic source oracle; it does not install arm or run
# the author trial driver. Set MTADF_ARM_ARCHIVE to the cached tarball path.

options(digits = 17)

archive <- Sys.getenv("MTADF_ARM_ARCHIVE", unset = "")
if (!nzchar(archive) || !file.exists(archive)) {
  stop("Set MTADF_ARM_ARCHIVE to the cached arm_1.6-06.01.tar.gz path")
}
temporary <- tempfile("arm-reference-")
dir.create(temporary)
untar(archive, files = "arm/R/bayesglm.R", exdir = temporary)
source_path <- file.path(temporary, "arm/R/bayesglm.R")
expressions <- parse(source_path)
reference_env <- new.env(parent = globalenv())
for (expression in expressions) {
  if (is.call(expression) && identical(expression[[1]], as.name("<-")) &&
      is.call(expression[[3]]) && identical(expression[[3]][[1]], as.name("function"))) {
    eval(expression, envir = reference_env)
  }
}
if (!exists("bayesglm.fit", envir = reference_env, inherits = FALSE)) {
  stop("Could not load bayesglm.fit from the selected archive")
}

cases <- list(
  list(name = "one_dose_zero_events", n = c(6, 0, 0, 0, 0), y = c(0, 0, 0, 0, 0)),
  list(name = "unequal_prefix_counts", n = c(3, 9, 4, 0, 0), y = c(1, 6, 1, 0, 0)),
  list(name = "all_zero_prefix", n = c(3, 6, 9, 0, 0), y = c(0, 0, 0, 0, 0)),
  list(name = "all_response_prefix", n = c(3, 6, 9, 0, 0), y = c(3, 6, 9, 0, 0))
)

fit_case <- function(case) {
  n <- case$n
  y <- case$y
  stopifnot(length(n) >= 2L, length(n) <= 20L, all(n >= 0), all(y >= 0), all(y <= n), sum(n) > 0)
  grid <- seq_along(n)
  xs <- (grid - mean(grid)) / (2 * sd(grid))
  outcomes <- unlist(Map(function(yi, ni) c(rep(1, yi), rep(0, ni - yi)), y, n), use.names = FALSE)
  design <- cbind("(Intercept)" = rep(1, sum(n)), x1reg = rep(xs, n), x2reg = rep(xs^2, n))
  fit <- reference_env$bayesglm.fit(
    x = design,
    y = outcomes,
    family = binomial(link = "logit"),
    control = glm.control(maxit = 100, epsilon = 1e-8),
    Warning = FALSE
  )
  probability <- plogis(cbind(1, xs, xs^2) %*% fit$coefficients)
  data.frame(
    case = case$name,
    dose_index = grid,
    subjects = n,
    responses = y,
    n_vector = paste(n, collapse = ";"),
    response_vector = paste(y, collapse = ";"),
    beta0 = unname(fit$coefficients[1]),
    beta1 = unname(fit$coefficients[2]),
    beta2 = unname(fit$coefficients[3]),
    fitted_efficacy = as.numeric(probability),
    rightmost_peak_dose = tail(which(probability == max(probability)), 1),
    prior_scale0 = unname(fit$prior.scale[1]),
    prior_scale1 = unname(fit$prior.scale[2]),
    prior_scale2 = unname(fit$prior.scale[3]),
    final_prior_sd0 = unname(fit$prior.sd[1]),
    final_prior_sd1 = unname(fit$prior.sd[2]),
    final_prior_sd2 = unname(fit$prior.sd[3]),
    deviance = unname(fit$deviance),
    iterations = unname(fit$iter),
    converged = unname(fit$converged),
    stringsAsFactors = FALSE
  )
}

result <- do.call(rbind, lapply(cases, fit_case))
# Format explicitly: write.table's numeric serialization need not honor the
# display digits option. Preserve round-trip precision in the reference file.
result[] <- lapply(result, function(column) {
  if (is.numeric(column)) sprintf("%.17g", column) else column
})
output <- file.path("tests", "fixtures", "mtadf-author-global-reference.csv")
dir.create(dirname(output), recursive = TRUE, showWarnings = FALSE)
write.csv(result, output, row.names = FALSE, quote = TRUE)
message("Wrote ", output)
unlink(temporary, recursive = TRUE)
