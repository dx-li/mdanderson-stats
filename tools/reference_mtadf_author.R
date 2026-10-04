# Independent, dependency-free reference for the cached 2013 MTADF R program.
#
# This script derives the equal-weight unimodal fit by enumerating contiguous
# constant blocks and minimizing squared error over feasible unimodal vectors.
# It does not copy or load Iso's implementation. It writes compact fixtures
# for the source-defined simulator edge cases described in the audit.

pava_equal <- function(y) {
  n <- length(y)
  level <- numeric(n)
  weight <- numeric(n)
  start <- integer(n)
  end <- integer(n)
  blocks <- 0L
  for (i in seq_len(n)) {
    blocks <- blocks + 1L
    level[blocks] <- y[i]
    weight[blocks] <- 1
    start[blocks] <- i
    end[blocks] <- i
    while (blocks > 1L && level[blocks - 1L] > level[blocks]) {
      total <- weight[blocks - 1L] + weight[blocks]
      level[blocks - 1L] <- (weight[blocks - 1L] * level[blocks - 1L] +
                              weight[blocks] * level[blocks]) / total
      weight[blocks - 1L] <- total
      end[blocks - 1L] <- end[blocks]
      blocks <- blocks - 1L
    }
  }
  fitted <- numeric(n)
  for (b in seq_len(blocks)) fitted[start[b]:end[b]] <- level[b]
  fitted
}

unimodal_equal <- function(y) {
  n <- length(y)
  if (n < 1L || n > 20L || any(!is.finite(y))) stop("invalid reference input")
  best_sse <- Inf
  best <- NULL
  # Each mask introduces a boundary between adjacent observations. The block
  # means are the least-squares levels for that partition. Feasibility is the
  # defining rise-then-fall order restriction.
  for (mask in 0:(2^(n - 1L) - 1L)) {
    starts <- c(1L, which(as.logical(intToBits(mask)[seq_len(n - 1L)])) + 1L)
    ends <- c(starts[-1L] - 1L, n)
    levels <- mapply(function(a, b) mean(y[a:b]), starts, ends)
    feasible <- any(vapply(seq_along(levels), function(peak) {
      (peak == 1L || all(diff(levels[seq_len(peak)]) >= 0)) &&
        (peak == length(levels) || all(diff(levels[peak:length(levels)]) <= 0))
    }, logical(1)))
    if (!feasible) next
    candidate <- numeric(n)
    for (b in seq_along(starts)) candidate[starts[b]:ends[b]] <- levels[b]
    sse <- sum((candidate - y)^2)
    if (sse < best_sse) {
      best_sse <- sse
      best <- candidate
    }
  }
  list(fitted = best, sse = best_sse, peak_rightmost = tail(which(best == max(best)), 1L))
}

alpha <- uniroot(function(a) pbeta(0.3, a, 0.5 - a) - 0.22,
                 c(0.01, 0.49), tol = 1e-14)$root
beta <- 0.5 - alpha

admissible_cap <- function(n, ytox, phi = 0.3, ct = 0.8) {
  raw <- 1 - pbeta(phi, alpha + ytox, beta + n - ytox)
  adjusted <- pava_equal(raw)
  max(1L, sum(adjusted <= ct))
}

author_next_dose <- function(current, tried, ndose, peak, cap) {
  if (current == ndose) {
    candidate <- if (peak == ndose) current else current - 1L
  } else if (peak == tried || peak > current) {
    candidate <- current + 1L
  } else if (peak < current) {
    candidate <- current - 1L
  } else {
    candidate <- current
  }
  min(cap, candidate)
}

reference_rows <- list()
add_fit_case <- function(label, y, subjects) {
  fit <- unimodal_equal(y)
  reference_rows[[length(reference_rows) + 1L]] <<- data.frame(
    record_type = "fit", label = label, dose = seq_along(y),
    subjects = subjects, observed_rate = y, fitted_rate = fit$fitted,
    squared_error = (fit$fitted - y)^2, peak_rightmost = fit$peak_rightmost,
    initial_cap = NA_integer_, fresh_cap = NA_integer_,
    author_next_dose = NA_integer_, df_isotonic_next_dose = NA_integer_,
    stringsAsFactors = FALSE
  )
}

# Grouped-dose references, including unequal group sizes. Author ufit calls
# omit w, so all dose-rate points receive unit weight regardless of subjects.
add_fit_case("unequal_n", c(0.10, 0.80, 0.35, 0.70, 0.25), c(3, 6, 3, 12, 3))
add_fit_case("flat_peak_tie", c(0.20, 0.80, 0.80, 0.20), c(3, 3, 3, 3))

# Case A: before the first cohort the prior-based cap is all doses. Under the
# mathematically defined singleton identity fit (the policy used by the
# independent reference), an all-toxic but efficacy-favorable first cohort
# would let the simulator choose the next assignment using that stale cap,
# then update admissibility. This is conditional rather than native execution
# evidence: Iso 0.0-15's unconstrained Fortran search has no candidates at n=1
# and subsequently indexes peak positions -1 and 0. The per-trial
# df.isotonic helper recomputes the cap first.
initial_cap <- admissible_cap(rep(0, 3), rep(0, 3))
n_after <- c(3, 0, 0)
tox_after <- c(3, 0, 0)
eff_after <- c(3, 0, 0)
interim_fit <- unimodal_equal(eff_after[1] / n_after[1])
author_next <- author_next_dose(
  current = 1L, tried = 1L, ndose = 3L,
  peak = interim_fit$peak_rightmost, cap = initial_cap
)
fresh_cap <- admissible_cap(n_after, tox_after)
df_next <- author_next_dose(
  current = 1L, tried = 1L, ndose = 3L,
  peak = interim_fit$peak_rightmost, cap = fresh_cap
)
reference_rows[[length(reference_rows) + 1L]] <- data.frame(
  record_type = "ledger", label = "toxic_first_cohort_lag", dose = 1L,
  subjects = n_after[1], observed_rate = eff_after[1] / n_after[1],
  fitted_rate = interim_fit$fitted[1], squared_error = interim_fit$sse,
  peak_rightmost = interim_fit$peak_rightmost,
  initial_cap = initial_cap, fresh_cap = fresh_cap,
  author_next_dose = author_next, df_isotonic_next_dose = df_next,
  stringsAsFactors = FALSE
)

# Case B: the source's final epsilon denominator turns every unobserved dose's
# zero response count into a zero rate. The rightmost maximum is an untried
# level, and selection is capped only by admissibility, not observed dose.
n_zero <- c(3, 0, 0, 0)
tox_zero <- c(0, 0, 0, 0)
eff_zero <- c(0, 0, 0, 0)
fit_zero <- unimodal_equal(eff_zero / (n_zero + 1e-4))
zero_cap <- admissible_cap(n_zero, tox_zero)
source_selection <- min(fit_zero$peak_rightmost, zero_cap)
reference_rows[[length(reference_rows) + 1L]] <- data.frame(
  record_type = "ledger", label = "zero_final_untried_tie", dose = 1L,
  subjects = n_zero[1], observed_rate = 0,
  fitted_rate = fit_zero$fitted[1], squared_error = fit_zero$sse,
  peak_rightmost = fit_zero$peak_rightmost,
  initial_cap = NA_integer_, fresh_cap = zero_cap,
  author_next_dose = source_selection, df_isotonic_next_dose = NA_integer_,
  stringsAsFactors = FALSE
)

fixture <- do.call(rbind, reference_rows)
out <- if (length(commandArgs(trailingOnly = TRUE))) {
  commandArgs(trailingOnly = TRUE)[1]
} else {
  "tests/fixtures/mtadf-author-reference.csv"
}
write.csv(fixture, out, row.names = FALSE, na = "")
cat(sprintf("alpha=%.16g beta=%.16g\n", alpha, beta))
cat(sprintf("wrote %s (%d rows)\n", out, nrow(fixture)))
