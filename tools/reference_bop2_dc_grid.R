#!/usr/bin/env Rscript
# Exhaustive 16-path oracle for a six-candidate binary BOP2-DC grid.
paths <- as.matrix(expand.grid(rep(list(0:1), 4)))
grid <- expand.grid(lambda_lrv = c(.7, .8, .9), lambda_cmv = c(.4, .5))
rows <- list()
for (candidate in seq_len(nrow(grid))) {
  ll <- grid$lambda_lrv[candidate]
  lc <- grid$lambda_cmv[candidate]
  outcome <- character(nrow(paths))
  enrolled <- rep(4L, nrow(paths))
  for (i in seq_len(nrow(paths))) {
    interim_y <- sum(paths[i, 1:2])
    interim_tail <- pbeta(c(.2, .5), 1 + interim_y, 3 - interim_y, lower.tail = FALSE)
    if (all(interim_tail < c(ll, lc) * sqrt(.5))) {
      outcome[i] <- "no_go"
      enrolled[i] <- 2L
    } else {
      y <- sum(paths[i, ])
      tail <- pbeta(c(.2, .5), 1 + y, 5 - y, lower.tail = FALSE)
      outcome[i] <- if (all(tail > c(ll, lc))) "go" else if (all(tail < c(ll, lc))) "no_go" else "consider"
    }
  }
  for (p in c(.2, .6)) {
    weight <- p^rowSums(paths) * (1-p)^(4-rowSums(paths))
    go <- sum(weight[outcome == "go"])
    no_go <- sum(weight[outcome == "no_go"])
    consider <- sum(weight[outcome == "consider"])
    stopifnot(abs(go + no_go + consider - 1) < 1e-14)
    rows[[length(rows)+1L]] <- data.frame(
      candidate = candidate, max_subjects = 4L, interim = 2L,
      prior_alpha = 1, prior_beta = 1, lrv = .2, cmv = .5,
      lambda_lrv = ll, lambda_cmv = lc, gamma_lrv = .5, gamma_cmv = .5,
      probability = p, go = go, no_go = no_go, consider = consider,
      expected_sample_size = sum(weight * enrolled)
    )
  }
}
write.csv(do.call(rbind, rows), "tests/fixtures/bop2-dc-grid.csv", row.names = FALSE)
