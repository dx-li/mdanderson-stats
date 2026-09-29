# Independent polynomial and scenario-summary references for BaCIS OC ESS.
# Inputs are observed counts and retained-draw variances from small Python
# fits; this validates ESS calculation/aggregation, not native JAGS sampling.
# The cubic is the variance equation in bacistool 1.0.0 compESS. Inadmissible
# and complex roots are excluded, correcting the documented native clamping
# defect rather than reproducing it.
options(digits = 17, warn = 2)
input <- read.csv("tests/fixtures/bacis-oc-ess-input.csv")
input$equivalent_sample_size <- vapply(seq_len(nrow(input)), function(i) {
    y <- input$responses[i]
    n <- input$patients[i]
    v <- input$variance[i]
    candidates <- polyroot(c(12 - (y + 1) * (1 - y) / v,
                             16 - (y + 1) / v, 7, 1))
    real <- abs(Im(candidates)) <= 1e-10 * pmax(1, abs(Re(candidates)))
    candidates <- sort(Re(candidates[real & Re(candidates) >= y]))
    stopifnot(length(candidates) > 0)
    rate <- ifelse(candidates == 0 & y == 0, 0, y / candidates)
    selected <- candidates[which.min(abs(rate - y / n))]
    achieved <- (y + 1) * (selected - y + 1) / ((selected + 2)^2 * (selected + 3))
    stopifnot(abs(achieved / v - 1) < 1e-10)
    selected
}, numeric(1))
write.csv(input, "tests/fixtures/bacis-oc-ess-reference.csv", row.names = FALSE)

groups <- split(input, interaction(input$case, input$group, drop = TRUE))
summary <- do.call(rbind, lapply(groups, function(rows) {
    values <- rows$equivalent_sample_size
    data.frame(case = rows$case[1], group = rows$group[1],
               replications = length(values), mean = mean(values),
               mcse = sd(values) / sqrt(length(values)))
}))
write.csv(summary, "tests/fixtures/bacis-oc-ess-summary.csv", row.names = FALSE)
cat(nrow(input), "independent polynomial roots and", nrow(summary), "group summaries\n")
