# Independent small BARPO trial paths under explicit Python scheduling choices.
# Formula source: official BARPO.pdf; scheduling is not native app parity.
options(digits = 17)
assignment_u <- c(.2, .9, .7, .3, .8, .1, .55, .95, .15, .7, .4, .85)
outcome_u <- c(.05, .82, .25, .1, .45, .93, .3, .6, .12, .77, .9, .22)
settings <- data.frame(
  case = c(paste0("complete_", c("barcp", "barn2n", "barmtv", "dbcd")),
           "complete_signal", "stop_arm", "stop_trial"),
  method = c("barcp", "barn2n", "barmtv", "dbcd", rep("barcp", 3)),
  policy = c(rep("arm", 6), "trial"),
  truth1 = c(rep(.2, 4), 0, 0, 0), truth2 = c(rep(.65, 4), 1, 1, 1),
  maximum = 12L, burn_in = 4L, block_size = 4L, cohort_size = 2L,
  min_n = 4L, tau = .7, tau1 = 1.2, target1 = .35, target2 = .65,
  early_monitoring = c(rep(FALSE, 5), TRUE, TRUE)
)
best_probability <- function(a, b) {
  p1 <- integrate(function(x) dbeta(x, a[1], b[1]) * pbeta(x, a[2], b[2]),
                  0, 1, rel.tol = 1e-12)$value
  c(p1, 1 - p1)
}
paths <- analyses <- summaries <- list()
for (case_id in seq_len(nrow(settings))) {
  cfg <- settings[case_id, ]
  successes <- assigned <- c(0L, 0L)
  active <- c(TRUE, TRUE)
  futile <- efficacious <- final <- c(FALSE, FALSE)
  remaining <- c(2L, 2L)
  n <- 0L
  reason <- "maximum"
  while (n < cfg$maximum && any(active)) {
    a <- 1 + successes
    b <- 1 + assigned - successes
    burn <- n < cfg$burn_in
    if (!burn) {
      best <- best_probability(a, b)
      variance <- a * b / ((a + b)^2 * (a + b + 1))
      target <- c(cfg$target1, cfg$target2)
      w <- switch(cfg$method,
        barcp = best^cfg$tau,
        barn2n = best^(n / (2 * cfg$maximum)),
        barmtv = sqrt(best * variance / (assigned + 1)),
        dbcd = (target * (target / (assigned / n))^cfg$tau)^cfg$tau1)
      w[!active] <- 0
      probability <- w / sum(w)
    }
    next_look <- min(c(4L, 8L, 12L)[c(4L, 8L, 12L) > n])
    chunk <- min(if (burn) cfg$burn_in - n else cfg$cohort_size,
                 next_look - n, cfg$maximum - n)
    for (j in seq_len(chunk)) {
      if (burn) probability <- remaining / sum(remaining)
      patient <- n + j
      arm <- which(assignment_u[patient] < cumsum(probability))[1]
      event <- as.integer(outcome_u[patient] < c(cfg$truth1, cfg$truth2)[arm])
      assigned[arm] <- assigned[arm] + 1L
      successes[arm] <- successes[arm] + event
      if (burn) remaining[arm] <- remaining[arm] - 1L
      paths[[length(paths) + 1L]] <- data.frame(
        case = cfg$case, patient = patient, arm = arm, response = event,
        assignment_uniform = assignment_u[patient], outcome_uniform = outcome_u[patient],
        allocation1 = probability[1], allocation2 = probability[2])
    }
    n <- n + chunk
    if (n %in% c(4L, 8L, 12L)) {
      a <- 1 + successes
      b <- 1 + assigned - successes
      p_fut <- pbeta(.4, a, b)
      p_eff <- pbeta(.6, a, b, lower.tail = FALSE)
      p_final <- pbeta(.5, a, b, lower.tail = FALSE)
      new_fut <- new_eff <- c(FALSE, FALSE)
      if (n == cfg$maximum) {
        final <- active & p_final >= .9
      } else if (cfg$early_monitoring && n >= cfg$min_n) {
        new_fut <- active & p_fut > .7
        new_eff <- active & p_eff >= .9
        stopifnot(!any(new_fut & new_eff))
        futile <- futile | new_fut
        efficacious <- efficacious | new_eff
        active[new_fut | new_eff] <- FALSE
      }
      for (arm in 1:2) analyses[[length(analyses) + 1L]] <- data.frame(
        case = cfg$case, patients = n, arm = arm, assigned = assigned[arm],
        successes = successes[arm], futility_probability = p_fut[arm],
        efficacy_probability = p_eff[arm], final_probability = p_final[arm],
        early_futile = futile[arm], early_efficacious = efficacious[arm],
        final_efficacious = final[arm])
      if (n < cfg$maximum && cfg$policy == "trial" && any(new_fut | new_eff)) {
        reason <- "trial_stop"
        break
      }
      if (!any(active)) reason <- "all_arms_closed"
    }
  }
  for (arm in 1:2) summaries[[length(summaries) + 1L]] <- data.frame(
    case = cfg$case, patients = n, arm = arm, assigned = assigned[arm],
    successes = successes[arm], early_futile = futile[arm],
    early_efficacious = efficacious[arm], final_efficacious = final[arm], reason = reason)
}
write.csv(settings, "tests/fixtures/barpo-trial-settings.csv", row.names = FALSE)
write.csv(do.call(rbind, paths), "tests/fixtures/barpo-trial-paths.csv", row.names = FALSE)
write.csv(do.call(rbind, analyses), "tests/fixtures/barpo-trial-analyses.csv", row.names = FALSE)
write.csv(do.call(rbind, summaries), "tests/fixtures/barpo-trial-summary.csv", row.names = FALSE)
