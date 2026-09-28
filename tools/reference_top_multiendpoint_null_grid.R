# Independent final-only probabilities for TOP efficacy/toxicity partial nulls.
# The joint outcomes are enumerated exactly using base R, with no Python calls.
options(digits = 17)
prior <- c(.05, .15, .25, .55)
truth <- rbind(
  both_boundary = prior,
  efficacy_null = c(.02, .18, .08, .72),
  toxicity_null = c(.15, .35, .15, .35),
  alternative = c(.05, .45, .05, .45)
)
n <- 12L
margins <- c(sum(prior[1:2]), sum(prior[c(1, 3)]))
complement <- c(sum(prior[c(3, 4)]), sum(prior[c(2, 4)]))
counts <- do.call(rbind, lapply(0:n, function(a) {
  do.call(rbind, lapply(0:(n - a), function(b) {
    do.call(rbind, lapply(0:(n - a - b), function(c) c(a, b, c, n - a - b - c)))
  }))
}))
events <- cbind(counts[, 1] + counts[, 2], counts[, 1] + counts[, 3])
upper <- pbeta(margins[1], margins[1] + events[, 1],
               complement[1] + (n - events[, 1]), lower.tail = FALSE)
lower <- pbeta(margins[2], margins[2] + events[, 2],
               complement[2] + (n - events[, 2]))
rows <- list()
for (cutoff in c(.6, .8, .95)) {
  success <- upper >= cutoff & lower >= cutoff
  for (i in seq_len(nrow(truth))) {
    mass <- apply(counts, 1, dmultinom, prob = truth[i, ])
    stopifnot(abs(sum(mass) - 1) < 1e-12)
    rows[[length(rows) + 1L]] <- data.frame(
      scenario = rownames(truth)[i], maximum = n, cutoff_scale = cutoff,
      prior11 = prior[1], prior10 = prior[2], prior01 = prior[3], prior00 = prior[4],
      truth11 = truth[i, 1], truth10 = truth[i, 2], truth01 = truth[i, 3],
      truth00 = truth[i, 4], success = sum(mass[success])
    )
  }
}
write.csv(do.call(rbind, rows), "tests/fixtures/top-multiendpoint-null-grid.csv",
          row.names = FALSE)
