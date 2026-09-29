# Source-pinned icenReg coefficient-bootstrap reference with fixed resamples.
# Run from the repository root. First source the native core reference builder;
# it verifies all icenReg blob hashes before using the ignored raw-source cache.
source("tools/reference_interval_survival.R")
native <- "research/raw/icenReg"
source(file.path(native, "R/internal_utilities.R"))
source(file.path(native, "R/ic_sp.R"))

# Adapt formula-facing helpers to the unchanged native core entry points built
# by reference_interval_survival.R. These wrappers keep source getBS_coef and
# bs_sampleData unchanged for the bootstrap path below.
findMaximalIntersections <- function(lower, upper) {
  values <- sort(unique(as.numeric(c(lower, upper))))
  result <- reference_mi(values, values %in% lower, values %in% upper, lower, upper)
  names(result) <- c("l_inds", "r_inds", "mi_l", "mi_r")
  result
}
fit_ICPH <- function(obsMat, covars, callText = "ic_ph", weights, other_info) {
  mi <- findMaximalIntersections(obsMat[, 1], obsMat[, 2])
  fit_type <- if (callText == "ic_ph") 1L else 2L
  fit <- reference_fit(
    mi$l_inds,
    mi$r_inds,
    as.matrix(covars),
    fit_type,
    as.double(weights),
    other_info$useGA,
    as.integer(other_info$maxIter),
    as.integer(other_info$baselineUpdates),
    as.logical(other_info$useFullHess),
    as.logical(other_info$updateCovars),
    as.double(other_info$regStart)
  )
  list(coefficients = fit[[2]], iterations = fit[[4]], log_likelihood = fit[[3]])
}

input <- read.csv("tests/fixtures/interval-survival-inputs.csv")
input <- input[input$case == "mixed", ]
lower <- input$lower
upper <- input$upper
# ic_sp applies its default open-left epsilon adjustment before creating the
# data environment that is resampled by bs_sampleData.
adjusted <- adjustIntervals(c(0, 1), cbind(lower, upper))
lower <- adjusted[, 1]
upper <- adjusted[, 2]
x_raw <- as.matrix(input[, c("x1", "x2")])
n <- length(lower)
x <- sweep(x_raw, 2, colMeans(x_raw), "-")
unit_weights <- rep(1, n)
weighted <- 1 + (seq_len(n) %% 3)

set.seed(20260928)
resamples <- list(
  unit_a = sample.int(n, n, replace = TRUE),
  unit_b = sample.int(n, n, replace = TRUE),
  unit_singular = rep.int(1L, n),
  weighted_a = sample.int(n, ceiling(sum(weighted)), replace = TRUE, prob = weighted),
  weighted_b = sample.int(n, ceiling(sum(weighted)), replace = TRUE, prob = weighted),
  weighted_singular = rep.int(1L, ceiling(sum(weighted)))
)
original_weights <- list(unit = unit_weights, weighted = weighted)
weight_case <- c(
  unit_a = "unit", unit_b = "unit", unit_singular = "unit",
  weighted_a = "weighted", weighted_b = "weighted", weighted_singular = "weighted"
)
other_info <- list(
  useGA = TRUE,
  maxIter = 10000L,
  baselineUpdates = 5L,
  useFullHess = TRUE,
  updateCovars = TRUE,
  recenterCovars = TRUE,
  regStart = rep(0, ncol(x))
)
coefficient_rows <- list()
resample_rows <- list()
metric_rows <- list()
for (name in names(resamples)) {
  ids <- resamples[[name]]
  counts <- tabulate(ids, nbins = n)
  retained <- which(counts > 0)
  sample_data <- new.env(parent = emptyenv())
  sample_data$x <- x[retained, , drop = FALSE]
  sample_data$y <- cbind(lower[retained], upper[retained])
  sample_data$w <- counts[retained]
  beta <- getBS_coef(sample_data, "ic_ph", other_info)
  status <- if (all(is.finite(beta))) "fit" else "singular_design"
  coefficient_rows[[name]] <- data.frame(
    case = name, coefficient = seq_along(beta), value = beta, status = status
  )
  resample_rows[[name]] <- data.frame(
    case = name, draw = seq_along(ids), row_id_zero_based = ids - 1L
  )
  metric_rows[[name]] <- data.frame(
    case = name,
    original_weight_sum = sum(original_weights[[unname(weight_case[name])]]),
    bootstrap_draw_count = length(ids),
    unique_rows = length(retained),
    status = status
  )
}
write.csv(
  do.call(rbind, resample_rows),
  "tests/fixtures/interval-survival-bootstrap-resamples.csv",
  row.names = FALSE
)
write.csv(
  do.call(rbind, coefficient_rows),
  "tests/fixtures/interval-survival-bootstrap-coefficients.csv",
  row.names = FALSE
)
write.csv(
  do.call(rbind, metric_rows),
  "tests/fixtures/interval-survival-bootstrap-metrics.csv",
  row.names = FALSE
)

covariance_rows <- list()
for (name in c("unit", "weighted")) {
  selected <- grep(paste0("^", name, "_"), names(resamples), value = TRUE)
  selected <- selected[vapply(
    selected,
    function(item) coefficient_rows[[item]]$status[1] == "fit",
    logical(1)
  )]
  beta_matrix <- do.call(rbind, lapply(selected, function(item) coefficient_rows[[item]]$value))
  covariance <- stats::cov(beta_matrix)
  covariance_rows[[name]] <- data.frame(
    case = name,
    row = rep(seq_len(nrow(covariance)), times = ncol(covariance)),
    column = rep(seq_len(ncol(covariance)), each = nrow(covariance)),
    value = as.vector(covariance),
    successful_replicates = nrow(beta_matrix),
    denominator = nrow(beta_matrix) - 1L
  )
}
write.csv(
  do.call(rbind, covariance_rows),
  "tests/fixtures/interval-survival-bootstrap-covariance.csv",
  row.names = FALSE
)
