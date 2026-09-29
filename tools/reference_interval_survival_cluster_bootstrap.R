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
group_labels <- c(20L, 10L, 40L, 30L)
group_sizes <- c(18L, 22L, 26L, 30L)
input$cluster_id <- rep(group_labels, times = group_sizes)
x <- as.matrix(input[, c("x1", "x2")])
lower <- input$lower
upper <- input$upper
cluster_id <- input$cluster_id

# Fixed zero-based indices in Python-sorted label order (10, 20, 30, 40).
# A narrow `sample` adapter feeds these choices to the unchanged native
# make_sample_inds helper, which still performs the actual row expansion.
replicates <- 2L
cluster_draws <- rbind(c(0L, 1L, 1L, 3L), c(2L, 0L, 3L, 0L))
stopifnot(nrow(cluster_draws) == replicates)
sorted_labels <- sort(unique(cluster_id))
sorted_sizes <- as.integer(table(factor(cluster_id, levels = sorted_labels)))
row_tapes <- vector("list", replicates)
sample_call <- 0L
sample <- function(x, size, replace = FALSE, prob = NULL, ...) {
  if (identical(as.integer(x), seq_len(length(sorted_labels))) &&
      size == length(sorted_labels) && isTRUE(replace) && is.null(prob)) {
    sample_call <<- sample_call + 1L
    return(cluster_draws[sample_call, ] + 1L)
  }
  base::sample(x, size = size, replace = replace, prob = prob, ...)
}
for (b in seq_len(replicates)) {
  rows <- make_sample_inds(cluster_id)
  expected_rows <- sum(sorted_sizes[cluster_draws[b, ] + 1L])
  stopifnot(length(rows) == expected_rows)
  cursor <- 1L
  for (selected in cluster_draws[b, ] + 1L) {
    count <- sorted_sizes[[selected]]
    sampled_group <- rows[cursor:(cursor + count - 1L)]
    stopifnot(all(cluster_id[sampled_group] == sorted_labels[[selected]]))
    cursor <- cursor + count
  }
  row_tapes[[b]] <- rows
}
rm(sample)
stopifnot(sample_call == replicates)
stopifnot(identical(as.integer(vapply(row_tapes, length, integer(1))), c(84L, 100L)))

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
    "rows", nrow(input), "clusters", length(sorted_labels), "\n")
