# Independent exact final-look TOP probabilities for correlated binary endpoints.
# Enumerate the four joint response cells; no simulation or Python kernel is used.
options(digits = 17)
cases <- list()
for (p11 in c(.1, .2, .4)) {
  cases[[length(cases) + 1L]] <- list(
    case = paste0("coprimary_joint_", p11), mode = "coprimary", maximum = 6L,
    prior = c(.15, .30, .15, .40),
    truth = c(p11, .5 - p11, .4 - p11, .1 + p11))
}
for (p11 in c(0, .1, .2)) {
  cases[[length(cases) + 1L]] <- list(
    case = paste0("efftox_joint_", p11), mode = "efficacy_toxicity", maximum = 6L,
    prior = c(.05, .10, .25, .60),
    truth = c(p11, .5 - p11, .2 - p11, .3 + p11))
}
cases[[7L]] <- list(case = "coprimary_null", mode = "coprimary", maximum = 10L,
                   prior = c(.15, .30, .15, .40), truth = c(.15, .30, .15, .40))
cases[[8L]] <- list(case = "efftox_null", mode = "efficacy_toxicity", maximum = 10L,
                   prior = c(.05, .10, .25, .60), truth = c(.05, .10, .25, .60))

results <- lapply(cases, function(cfg) {
  n <- cfg$maximum
  null <- c(sum(cfg$prior[c(1, 2)]), sum(cfg$prior[c(1, 3)]))
  beta <- c(sum(cfg$prior[c(3, 4)]), sum(cfg$prior[c(2, 4)]))
  probability <- c(success = 0, stop_futility = 0, stop_toxicity = 0,
                   stop_futility_toxicity = 0)
  total <- 0
  for (n11 in 0:n) for (n10 in 0:(n - n11)) for (n01 in 0:(n - n11 - n10)) {
    counts <- c(n11, n10, n01, n - n11 - n10 - n01)
    mass <- dmultinom(counts, prob = cfg$truth)
    events <- c(n11 + n10, n11 + n01)
    acceptable <- pbeta(null, null + events, beta + (n - events), lower.tail = FALSE)
    if (cfg$mode == "efficacy_toxicity") {
      acceptable[2] <- pbeta(null[2], null[2] + events[2], beta[2] + n - events[2])
    }
    good <- acceptable >= .6
    action <- if (cfg$mode == "coprimary") {
      if (any(good)) "success" else "stop_futility"
    } else if (all(good)) {
      "success"
    } else if (!any(good)) {
      "stop_futility_toxicity"
    } else if (!good[1]) {
      "stop_futility"
    } else {
      "stop_toxicity"
    }
    probability[action] <- probability[action] + mass
    total <- total + mass
  }
  stopifnot(abs(total - 1) < 1e-12, abs(sum(probability) - 1) < 1e-12)
  data.frame(case = cfg$case, mode = cfg$mode, maximum = n, cutoff_scale = .6,
             gamma = .5, prior11 = cfg$prior[1], prior10 = cfg$prior[2],
             prior01 = cfg$prior[3], prior00 = cfg$prior[4], truth11 = cfg$truth[1],
             truth10 = cfg$truth[2], truth01 = cfg$truth[3], truth00 = cfg$truth[4],
             success = unname(probability["success"]),
             stop_futility = unname(probability["stop_futility"]),
             stop_toxicity = unname(probability["stop_toxicity"]),
             stop_futility_toxicity = unname(probability["stop_futility_toxicity"]))
})
write.csv(do.call(rbind, results), "tests/fixtures/top-multiendpoint-power.csv", row.names = FALSE)
