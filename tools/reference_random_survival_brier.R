# Independent compact transcription of RF-SRC 3.2.2's scalar bs.gradient
# helper contract. This emits row gradients and candidate scores for review;
# it does not compile or invoke the native forest implementation.

cases <- list(
  shared_failure_weight = list(
    time = c(1, 1.5, 2, 3), event = c(1, 0, 1, 0), x = c(0, 0, 1, 1),
    prob = 0.9, cut = 0.5
  ),
  threshold_equality_zero = list(
    time = c(1, 2), event = c(1, 0), x = c(0, 1), prob = 0.5, cut = 0.5
  ),
  first_event_after_censor = list(
    time = c(1, 2, 3), event = c(0, 1, 0), x = c(0, 1, 1),
    prob = 0.9, cut = 0.5
  ),
  equality_uses_previous_point = list(
    time = c(1, 2, 3, 4), event = c(1, 1, 0, 0), x = c(0, 0, 1, 1),
    prob = 0.5, cut = 0.5
  ),
  censor_tied_with_event = list(
    time = c(1, 1, 2, 3), event = c(1, 0, 1, 0), x = c(0, 0, 1, 1),
    prob = 0.9, cut = 0.5
  )
)

compute_case <- function(name, z) {
  event_times <- sort(unique(z$time[z$event == 1]))
  failures <- vapply(event_times, function(t) sum(z$time == t & z$event == 1), 0.0)
  at_risk <- vapply(event_times, function(t) sum(z$time >= t), 0.0)
  km <- cumprod(1 - failures / at_risk)
  crossing <- which(km <= 1 - z$prob)
  qe <- if (length(crossing)) crossing[1] - 1L else length(event_times)
  gamma <- rep(0, length(z$time))
  if (qe > 0L) {
    eval_time <- event_times[qe]
    previous_time <- event_times[max(1L, qe - 1L)]
    censor_times <- sort(unique(z$time[z$event == 0]))
    censor_survival <- vapply(censor_times, function(t) {
      d <- sum(z$time == t & z$event == 0)
      risk <- sum(z$time >= t)
      1 - d / risk
    }, 0.0)
    censor_survival <- cumprod(censor_survival)
    g_before <- function(t) {
      prior <- which(censor_times < t)
      if (!length(prior)) 1 else censor_survival[max(prior)]
    }
    y <- as.numeric(z$time > eval_time)
    w <- numeric(length(z$time))
    w[y == 1] <- 1 / g_before(eval_time)
    prior_events <- z$event == 1 & z$time <= eval_time
    w[prior_events] <- 1 / g_before(previous_time)
    denom <- sum(w)
    fhat <- if (denom == 0) 0 else sum(w * y) / denom
    gamma <- -2 * w * (y - fhat)
  }
  left <- z$x <= z$cut
  n_left <- sum(left)
  score <- if (qe == 0L) 0 else {
    (n_left / length(left)) * mean(gamma[left])^2 +
      ((length(left) - n_left) / length(left)) * mean(gamma[!left])^2
  }
  data.frame(
    case = name, row = seq_along(z$time), time = z$time, event = z$event, left = left,
    prob = z$prob, qe_index = qe, gamma = gamma, score = score
  )
}

result <- do.call(rbind, Map(compute_case, names(cases), cases))
output <- if (length(commandArgs(TRUE))) commandArgs(TRUE)[1] else
  "tests/fixtures/random-survival-brier-reference.csv"
write.csv(result, output, row.names = FALSE, quote = FALSE)
