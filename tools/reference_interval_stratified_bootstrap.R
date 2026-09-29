# Independent fixed-resample reference for shared-beta interval-PH bootstrap.
# Current-status intervals (0,1] / (1,Inf) reduce exactly to a binomial
# complementary-log-log likelihood with separate stratum intercepts.
# This is a mathematical reference, not a native stratified-bootstrap API.
options(warn = 2, digits = 17)
input <- data.frame(
  stratum = rep(c("A", "B"), each = 4),
  x = rep(c(0, 0, 1, 1), 2),
  event = rep(c(1, 0, 1, 0), 2),
  weight = c(2, 8, 5, 5, 4, 6, 7, 3)
)
counts <- rbind(
  c(2, 8, 5, 5, 4, 6, 7, 3),
  c(3, 7, 4, 6, 5, 5, 6, 4),
  c(1, 9, 6, 4, 3, 7, 8, 2),
  c(4, 6, 6, 4, 5, 5, 8, 2),
  c(3, 7, 7, 3, 2, 8, 6, 4),
  c(2, 8, 5, 5, 0, 10, 0, 10)
)
stopifnot(all(rowSums(counts[, 1:4]) == 20),
          all(rowSums(counts[, 5:8]) == 20))
fit_counts <- function(weight) {
  # Aggregate events / nonevents so glm receives binomial counts and no
  # non-integer-success warning from fitting weighted Bernoulli rows.
  d <- data.frame(stratum = factor(c("A", "A", "B", "B")),
                  x = c(0, 1, 0, 1),
                  events = weight[c(1, 3, 5, 7)],
                  nonevents = weight[c(2, 4, 6, 8)])
  fit <- glm(cbind(events, nonevents) ~ 0 + stratum + x, data = d,
             family = binomial(link = "cloglog"),
             control = glm.control(epsilon = 1e-12, maxit = 100))
  stopifnot(fit$converged)
  probability <- fitted(fit)
  c(beta = unname(coef(fit)["x"]),
    log_likelihood = sum(d$events * log(probability) +
                         d$nonevents * log1p(-probability)))
}
result <- list()
tapes <- list()
for (i in seq_len(nrow(counts))) {
  tape <- rep(seq_len(nrow(input)) - 1L, counts[i, ])
  tapes[[i]] <- data.frame(replicate = i - 1L,
                          position = seq_along(tape) - 1L,
                          source_index = tape)
  # A group with no events has an unsupported baseline in the Python target.
  # Preserve the requested draw as a failure rather than fitting another model.
  valid <- all(tapply(counts[i, ] * input$event, input$stratum, sum) > 0)
  reference <- if (valid) fit_counts(counts[i, ]) else
    c(beta = NA_real_, log_likelihood = NA_real_)
  result[[i]] <- data.frame(replicate = i - 1L,
                            status = if (valid) "ok" else "fit_failed",
                            beta = reference["beta"],
                            log_likelihood = reference["log_likelihood"])
}
result <- do.call(rbind, result)
original <- fit_counts(input$weight)
successful <- result$beta[result$status == "ok"]
summary <- data.frame(
  quantity = c("original_beta", "original_log_likelihood", "covariance", "standard_error"),
  value = c(original, var(successful), sd(successful))
)
prefix <- "tests/fixtures/interval-stratified-bootstrap-"
write.csv(do.call(rbind, tapes), paste0(prefix, "resamples.csv"), row.names = FALSE)
write.csv(result, paste0(prefix, "coefficients.csv"), row.names = FALSE, na = "NaN")
write.csv(summary, paste0(prefix, "summary.csv"), row.names = FALSE)
cat("Generated six fixed weighted tapes, five shared-beta fits and one explicit unsupported draw.\n")
