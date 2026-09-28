# Independent patient-ledger arithmetic for PRT paper Section 2.
# Intervals are [0,1), [1,2); an event at 2 is outside the event window.
# This reference does not reproduce a posterior sampler or calendar controller.
options(warn = 2)
patients <- data.frame(
  patient = 0:3,
  arrival = c(0, 0.25, 0.5, 0.75),
  enrollment = c(0, 0.25, 0.5, 0.75),
  dose = c(0, 0, 1, 1),
  toxicity_delay = c(1, 0.25, 2, Inf)
)
endpoints <- c(1, 2)
times <- c(0, 0.25, 0.5, 0.75, 1, 1.5, 1.75, 2, 2.5, 2.75)
result <- list()
row <- 0L
for (time in times) {
  for (dose in 0:1) {
    for (interval in 0:1) {
      survived <- events <- pending <- 0L
      for (i in seq_len(nrow(patients))) {
        p <- patients[i, ]
        if (p$dose != dose || p$enrollment > time) next
        age <- time - p$enrollment
        observed_event <- p$toxicity_delay < 2 && p$toxicity_delay <= age
        # A boundary event survives every interval ending at that boundary.
        survived <- survived + as.integer(
          age >= endpoints[interval + 1] &&
          p$toxicity_delay >= endpoints[interval + 1]
        )
        lower <- if (interval == 0) 0 else endpoints[interval]
        events <- events + as.integer(
          observed_event && p$toxicity_delay >= lower &&
          p$toxicity_delay < endpoints[interval + 1]
        )
        completed <- sum(endpoints <= age)
        pending <- pending + as.integer(!observed_event && completed == interval)
      }
      row <- row + 1L
      result[[row]] <- data.frame(time, dose, interval, survived, events, pending)
    }
  }
}
write.csv(patients, "tests/fixtures/prt-calendar-patients.csv", row.names = FALSE)
write.csv(do.call(rbind, result), "tests/fixtures/prt-calendar-counts.csv", row.names = FALSE)
