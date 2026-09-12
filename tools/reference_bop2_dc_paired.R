#!/usr/bin/env Rscript
# Base-R Dirichlet marginal checks. Cell order: both, first only, second only, neither.
x <- data.frame(
  case = c("both_futile", "one_effective", "both_uncertain", "one_uncertain_one_effective",
           "effective_safe", "effective_toxic", "effective_uncertain_safety", "interim_continue", "interim_stop"),
  mode = c(rep("multiple", 4), rep("efftox", 5)),
  both = c(0, 0, 6, 6, 0, 10, 2, 0, 0),
  first_only = c(0, 12, 0, 0, 12, 2, 10, 5, 0),
  second_only = c(0, 0, 0, 6, 0, 0, 0, 0, 5),
  neither = c(20, 8, 14, 8, 8, 8, 8, 5, 5),
  max_subjects = 20, cell_prior = .25,
  lrv_first = c(rep(.2, 4), rep(.3, 5)),
  cmv_first = c(rep(.3, 4), rep(.45, 5)),
  lrv_second = .2,
  cmv_second = c(rep(.3, 4), rep(.15, 5)),
  lambda_lrv = .9, lambda_cmv = .5, gamma_lrv = .5, gamma_cmv = .5
)
for (i in seq_len(nrow(x))) {
  z <- x[i, ]
  n <- z$both + z$first_only + z$second_only + z$neither
  y <- c(z$both + z$first_only, z$both + z$second_only)
  lrv <- c(z$lrv_first, z$lrv_second)
  cmv <- c(z$cmv_first, z$cmv_second)
  pl <- pc <- numeric(2)
  for (j in 1:2) {
    lower <- j == 2 && z$mode == "efftox"
    pl[j] <- pbeta(lrv[j], y[j] + .5, n - y[j] + .5, lower.tail = lower)
    pc[j] <- pbeta(cmv[j], y[j] + .5, n - y[j] + .5, lower.tail = lower)
  }
  no_go <- pl < z$lambda_lrv * (n/20)^z$gamma_lrv & pc < z$lambda_cmv * (n/20)^z$gamma_cmv
  go <- pl > z$lambda_lrv & pc > z$lambda_cmv
  stop <- if (z$mode == "multiple") all(no_go) else any(no_go)
  success <- if (z$mode == "multiple") any(go) else all(go)
  x$decision[i] <- if (stop) "no_go" else if (n < 20) "continue" else if (success) "go" else "consider"
  x$patients[i] <- n
  x$posterior_lrv_first[i] <- pl[1]
  x$posterior_lrv_second[i] <- pl[2]
  x$posterior_cmv_first[i] <- pc[1]
  x$posterior_cmv_second[i] <- pc[2]
}
stopifnot(all(c("go", "no_go", "consider", "continue") %in% x$decision))
write.csv(x, "tests/fixtures/bop2-dc-paired.csv", row.names = FALSE)
