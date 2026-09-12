#!/usr/bin/env Rscript
# Base-R checks of the published BOP2-DC binary posterior decision rule.
x <- data.frame(
  case = c("interim_futile", "interim_continue", "final_nogo", "final_consider", "final_go"),
  patients = c(10, 10, 40, 40, 40),
  responses = c(0, 3, 8, 12, 16),
  max_subjects = 40, prior_alpha = 0.1, prior_beta = 0.1,
  lrv = 0.2, cmv = 0.3, lambda_lrv = 0.9, lambda_cmv = 0.5,
  gamma_lrv = 0.5, gamma_cmv = 0.5
)
x$posterior_lrv <- pbeta(x$lrv, x$prior_alpha + x$responses,
                         x$prior_beta + x$patients - x$responses, lower.tail = FALSE)
x$posterior_cmv <- pbeta(x$cmv, x$prior_alpha + x$responses,
                         x$prior_beta + x$patients - x$responses, lower.tail = FALSE)
x$cutoff_lrv <- x$lambda_lrv * (x$patients / x$max_subjects)^x$gamma_lrv
x$cutoff_cmv <- x$lambda_cmv * (x$patients / x$max_subjects)^x$gamma_cmv
no_go <- x$posterior_lrv < x$cutoff_lrv & x$posterior_cmv < x$cutoff_cmv
go <- x$posterior_lrv > x$cutoff_lrv & x$posterior_cmv > x$cutoff_cmv
x$decision <- ifelse(no_go, "no_go", ifelse(x$patients < x$max_subjects, "continue",
                                          ifelse(go, "go", "consider")))
stopifnot(identical(x$decision, c("no_go", "continue", "no_go", "consider", "go")))
write.csv(x, "tests/fixtures/bop2-dc-binary.csv", row.names = FALSE)
