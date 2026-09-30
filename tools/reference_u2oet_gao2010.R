# Independent equations (3)--(9) from Houede et al., Biometrics (2010).
# Center each agent's dose grid; gamma is endpoint-specific and may be negative.
options(warn = 2, digits = 17)

marginal <- function(d1, d2, intercepts, slopes, lambda, gamma) {
  eta <- intercepts + sweep(slopes, 2, c(d1, d2), '*')
  s <- exp(eta[, 1]) + exp(eta[, 2]) + gamma * exp(rowSums(eta))
  stopifnot(all(s > 0))
  hazard <- log1p(lambda * s) / lambda
  continuation <- -expm1(-hazard)
  survival <- c(1, cumprod(continuation))
  c(head(survival, -1) * exp(-hazard), tail(survival, 1))
}

rectangle <- function(ep, tp, rho) {
  ec <- c(0, cumsum(ep)); tc <- c(0, cumsum(tp))
  ec[length(ec)] <- tc[length(tc)] <- 1
  result <- matrix(0, length(ep), length(tp))
  if (rho == 0) return(outer(ep, tp))
  if (abs(rho) == 1) {
    for (e in seq_along(ep)) for (t in seq_along(tp)) {
      lower <- if (rho > 0) tc[t] else 1 - tc[t + 1]
      upper <- if (rho > 0) tc[t + 1] else 1 - tc[t]
      result[e, t] <- max(0, min(ec[e + 1], upper) - max(ec[e], lower))
    }
    return(result)
  }
  ez <- qnorm(ec); tz <- qnorm(tc); residual_sd <- sqrt(1 - rho * rho)
  for (e in seq_along(ep)) for (t in seq_along(tp)) {
    integrand <- function(x) dnorm(x) * (
      pnorm((tz[t + 1] - rho * x) / residual_sd) -
        pnorm((tz[t] - rho * x) / residual_sd))
    result[e, t] <- integrate(integrand, ez[e], ez[e + 1],
                             abs.tol = 1e-13, rel.tol = 1e-11,
                             subdivisions = 1000L)$value
  }
  stopifnot(max(abs(rowSums(result) - ep)) < 1e-10,
            max(abs(colSums(result) - tp)) < 1e-10)
  result
}

cases <- list(
  binary = list(d1 = c(1, 3), d2 = c(2, 5),
    ei = matrix(c(-1, .3), 1, 2), es = matrix(c(.2, -.1), 1, 2),
    ti = matrix(c(-.4, -1.2), 1, 2), ts = matrix(c(-.05, .15), 1, 2),
    el = 1, tl = 1, eg = .4, tg = -.35),
  ordinal = list(d1 = c(.5, 2, 4), d2 = c(1, 4),
    ei = matrix(c(-.8, .2, .3, -.6, -.2, -1), 3, 2, byrow = TRUE),
    es = matrix(c(.3, -.1, -.2, .25, .1, .15), 3, 2, byrow = TRUE),
    ti = matrix(c(-1, .4, -.3, -.7), 2, 2, byrow = TRUE),
    ts = matrix(c(.1, .15, -.1, .2), 2, 2, byrow = TRUE),
    el = .4, tl = 1.8, eg = -.2, tg = .7),
  small_link = list(d1 = c(.5, 2), d2 = c(1, 3),
    ei = matrix(c(-2, -1.5, -1.2, -2.2), 2, 2, byrow = TRUE),
    es = matrix(c(.1, -.1, .2, .05), 2, 2, byrow = TRUE),
    ti = matrix(c(-1.5, -2), 1, 2), ts = matrix(c(-.1, .2), 1, 2),
    el = 1e-8, tl = 1e-7, eg = -.15, tg = -.3),
  zero_interaction = list(d1 = c(2, 3, 5), d2 = c(1, 2),
    ei = matrix(c(-.4, .2), 1, 2), es = matrix(c(.15, -.3), 1, 2),
    ti = matrix(c(-1, .3), 1, 2), ts = matrix(c(.1, .2), 1, 2),
    el = 2, tl = .6, eg = 0, tg = 0))

coefficients <- list(); doses <- list(); probabilities <- list(); likelihoods <- list()
partial_rows <- list(); utilities <- list()
for (name in names(cases)) {
  x <- cases[[name]]
  for (agent in 1:2) {
    grid <- x[[paste0('d', agent)]]
    for (i in seq_along(grid)) doses[[length(doses) + 1L]] <- data.frame(
      scenario = name, agent = agent, index = i - 1L, value = grid[i])
  }
  for (endpoint in c('e', 't')) {
    intercept <- x[[paste0(endpoint, 'i')]]
    slope <- x[[paste0(endpoint, 's')]]
    for (threshold in seq_len(nrow(intercept))) {
      coefficients[[length(coefficients) + 1L]] <- data.frame(
        scenario = name, endpoint = endpoint, threshold = threshold - 1L,
        intercept1 = intercept[threshold, 1], intercept2 = intercept[threshold, 2],
        slope1 = slope[threshold, 1], slope2 = slope[threshold, 2],
        lambda = x[[paste0(endpoint, 'l')]], gamma = x[[paste0(endpoint, 'g')]])
    }
  }
  centered1 <- x$d1 - mean(x$d1); centered2 <- x$d2 - mean(x$d2)
  for (rho in c(-1, -.65, 0, .55, 1)) {
    complete_ll <- 0; partial_ll <- 0
    for (i in seq_along(centered1)) for (j in seq_along(centered2)) {
      ep <- marginal(centered1[i], centered2[j], x$ei, x$es, x$el, x$eg)
      tp <- marginal(centered1[i], centered2[j], x$ti, x$ts, x$tl, x$tg)
      jp <- rectangle(ep, tp, rho)
      utility <- outer(seq_along(ep) - 1, seq_along(tp) - 1,
                       function(e, t) 20 + 30 * e - 5 * t)
      for (e in seq_along(ep)) for (t in seq_along(tp)) {
        n <- (i + 2 * j + e + t) %% 3
        if (n > 0) complete_ll <- complete_ll + n * log(jp[e, t])
        probabilities[[length(probabilities) + 1L]] <- data.frame(
          scenario = name, association = rho, dose1 = i - 1L, dose2 = j - 1L,
          efficacy = e - 1L, toxicity = t - 1L, efficacy_probability = ep[e],
          toxicity_probability = tp[t], joint_probability = jp[e, t], count = n)
      }
      for (t in seq_along(tp)) {
        n <- (i + j + t) %% 2
        if (n > 0) partial_ll <- partial_ll + n * log(tp[t])
        partial_rows[[length(partial_rows) + 1L]] <- data.frame(
          scenario = name, association = rho, dose1 = i - 1L, dose2 = j - 1L,
          toxicity = t - 1L, count = n)
      }
      utilities[[length(utilities) + 1L]] <- data.frame(
        scenario = name, association = rho, dose1 = i - 1L, dose2 = j - 1L,
        expected_utility = sum(jp * utility))
    }
    likelihoods[[length(likelihoods) + 1L]] <- data.frame(
      scenario = name, association = rho, complete = complete_ll,
      toxicity_only = partial_ll, combined = complete_ll + partial_ll)
  }
}
tables <- list(coefficients = coefficients, doses = doses, probabilities = probabilities,
               partial = partial_rows, utilities = utilities, likelihood = likelihoods)
for (name in names(tables)) {
  write.csv(do.call(rbind, tables[[name]]),
            paste0('tests/fixtures/u2oet-gao2010-', name, '.csv'), row.names = FALSE)
}
cat('Generated', length(probabilities), 'joint cells,', length(likelihoods),
    'complete/partial likelihood pairs and', length(utilities), 'utility references.\n')
