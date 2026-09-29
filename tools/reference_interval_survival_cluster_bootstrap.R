# Fixed-tape icenReg cluster-bootstrap coefficient reference.
# Run from the repository root after restoring the pinned, ignored source cache.
options(warn = 2, digits = 17)
source("tools/reference_interval_survival_core.R")

cluster_source <- "research/raw/icenReg/R/clusterBootstrap.R"
cluster_hash <- "3b0667806deaab7650d4feb0d69580eaf481bcc8"
actual_hash <- system2("git", c("hash-object", shQuote(cluster_source)), stdout = TRUE)
stopifnot(identical(unname(actual_hash), cluster_hash))
source(cluster_source)

input <- read.csv("tests/fixtures/interval-survival-inputs.csv")
input <- input[input$case == "mixed", , drop = FALSE]
input <- input[order(input$row), , drop = FALSE]
stopifnot(nrow(input) == 96L, input$covariates[[1L]] == 2L)
input$cluster_id <- rep(seq_len(nrow(input) / 4L), each = 4L)
x <- as.matrix(input[, c("x1", "x2")])
lower <- input$lower
upper <- input$upper
cluster_id <- input$cluster_id

# The unchanged native helper chooses cluster groups and concatenates all rows.
# Equal four-row groups make its generated row tape recoverable without changing
# the helper; the fixture records the corresponding zero-based cluster indices.
set.seed(18421)
replicates <- 2L
cluster_draws <- matrix(NA_integer_, replicates, max(cluster_id))
row_tapes <- vector("list", replicates)
for (b in seq_len(replicates)) {
  rows <- make_sample_inds(cluster_id)
  stopifnot(length(rows) == length(cluster_id))
  group_starts <- seq.int(1L, length(rows), by = 4L)
  selected_ids <- cluster_id[rows[group_starts]]
  stopifnot(all(vapply(seq_along(group_starts), function(j) {
    start <- group_starts[[j]]
    all(cluster_id[rows[start:(start + 3L)]] == selected_ids[[j]])
  }, logical(1))))
  cluster_draws[b, ] <- selected_ids - 1L
  row_tapes[[b]] <- rows
}

original <- native_fit(lower, upper, x, rep(1, nrow(input)))
native_slopes <- matrix(NA_real_, replicates, ncol(x))
native_metrics <- vector("list", replicates + 1L)
native_metrics[[1L]] <- data.frame(
  replicate = 0L,
  rows = nrow(input),
  iterations = original$fit$iterations,
  log_likelihood = original$fit$log_likelihood
)
for (b in seq_len(replicates)) {
  rows <- row_tapes[[b]]
  result <- native_fit(lower[rows], upper[rows], x[rows, , drop = FALSE], rep(1, length(rows)))
  native_slopes[b, ] <- result$fit$coefficients
  native_metrics[[b + 1L]] <- data.frame(
    replicate = b,
    rows = length(rows),
    iterations = result$fit$iterations,
    log_likelihood = result$fit$log_likelihood
  )
}
rownames(native_slopes) <- NULL
colnames(native_slopes) <- paste0("beta", seq_len(ncol(native_slopes)))
covariance <- stats::cov(native_slopes)

dir.create("tests/fixtures", showWarnings = FALSE)
write.csv(input[, c("row", "cluster_id", "lower", "upper", "x1", "x2")],
          "tests/fixtures/interval-survival-cluster-inputs.csv", row.names = FALSE)
write.csv(data.frame(
  replicate = rep(seq_len(replicates), each = ncol(cluster_draws)),
  draw = rep(seq_len(ncol(cluster_draws)), times = replicates),
  cluster_index = as.vector(t(cluster_draws)),
  cluster_id = as.vector(t(cluster_draws)) + 1L
), "tests/fixtures/interval-survival-cluster-tapes.csv", row.names = FALSE)
write.csv(data.frame(
  replicate = rep(seq_len(replicates), each = ncol(native_slopes)),
  parameter = rep(seq_len(ncol(native_slopes)), times = replicates),
  value = as.vector(t(native_slopes))
), "tests/fixtures/interval-survival-cluster-coefficients.csv", row.names = FALSE)
write.csv(data.frame(
  row = rep(seq_len(nrow(covariance)), times = ncol(covariance)),
  column = rep(seq_len(ncol(covariance)), each = nrow(covariance)),
  value = as.vector(covariance)
), "tests/fixtures/interval-survival-cluster-covariance.csv", row.names = FALSE)
write.csv(do.call(rbind, native_metrics),
          "tests/fixtures/interval-survival-cluster-metrics.csv", row.names = FALSE)
cat("replicates", replicates, "covariance_df", replicates - 1L,
    "rows", nrow(input), "clusters", max(cluster_id), "\n")
