# Independent within-chain batch-means precision and classical split R-hat.
# These are Python's stated diagnostics, not an undisclosed native batching rule.
options(digits = 17, warn = 2)
trace <- list()
for (draws in c(16, 37, 64)) {
  for (chain in 0:1) {
    i <- seq_len(draws)
    fourth <- if (draws == 16) rep(chain + 1, draws) else if (draws == 37) {
      rep(7, draws)
    } else {
      2 + sin(2 * i) + .1 * chain
    }
    trace[[length(trace) + 1]] <- data.frame(
      case = paste0("n", draws), chain = chain, draw = i - 1,
      corner0 = 3 + sin(i / 4) + .05 * i + .2 * chain,
      corner1 = 1 + .4 * cos(1.1 * i + chain),
      corner2 = 1 + as.numeric(i > draws / 2) + .1 * chain,
      corner3 = fourth
    )
  }
}
trace <- do.call(rbind, trace)
precision <- list()
rhat <- list()
for (name in unique(trace$case)) {
  local <- trace[trace$case == name, ]
  draws <- sum(local$chain == 0)
  batch_length <- max(2, floor(sqrt(draws)))
  batches <- floor(draws / batch_length)
  half <- floor(draws / 2)
  for (corner in 0:3) {
    values <- lapply(0:1, function(chain) {
      local[local$chain == chain, paste0("corner", corner)]
    })
    for (chain in 0:1) {
      x <- values[[chain + 1]]
      batch_means <- vapply(seq_len(batches), function(batch) {
        first <- 1 + (batch - 1) * batch_length
        mean(x[first:(first + batch_length - 1)])
      }, 0.0)
      posterior_sd <- sd(x)
      mcse <- sd(batch_means) / sqrt(batches)
      ratio <- if (posterior_sd > 0) mcse / posterior_sd else NA_real_
      precision[[length(precision) + 1]] <- data.frame(
        case = name, chain = chain, corner = corner,
        batch_length = batch_length, batches = batches, used = batches * batch_length,
        posterior_sd = posterior_sd, mcse = mcse, ratio = ratio
      )
    }
    split <- unlist(lapply(values, function(x) list(head(x, half), tail(x, half))),
                    recursive = FALSE)
    within <- mean(vapply(split, var, 0.0))
    between <- half * var(vapply(split, mean, 0.0))
    statistic <- if (within == 0) {
      if (between > 0) Inf else NA_real_
    } else {
      sqrt(((half - 1) * within + between) / (half * within))
    }
    rhat[[length(rhat) + 1]] <- data.frame(case = name, corner = corner, split_rhat = statistic)
  }
}
write.csv(trace, "tests/fixtures/u2oet-precision-traces.csv", row.names = FALSE)
write.csv(do.call(rbind, precision), "tests/fixtures/u2oet-precision-reference.csv", row.names = FALSE)
write.csv(do.call(rbind, rhat), "tests/fixtures/u2oet-precision-rhat.csv", row.names = FALSE)
cat("Wrote 24 within-chain precision and 12 split R-hat references.\n")
