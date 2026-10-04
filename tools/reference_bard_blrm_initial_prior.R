# Independent prior-only check of the BARD paper's printed raw-dose-ratio model.
# This does not fit data or alter the paper's prior to make a trial enroll.
options(warn = 2, digits = 17)

doses <- c(10, 20, 50, 100, 200)
reference <- 50
prior_mean <- c(-1.1, 0)
prior_sd <- c(2, 1)
overdose <- 0.33
eta <- 0.30

# Positive slope and dose ratio imply p_j > expit(log(alpha)).
lower_bound <- pnorm((qlogis(overdose) - prior_mean[1]) / prior_sd[1],
                     lower.tail = FALSE)
results <- lapply(doses, function(dose) {
  integral <- integrate(function(z) {
    intercept_threshold <- qlogis(overdose) -
      exp(prior_mean[2] + prior_sd[2] * z) * dose / reference
    pnorm((intercept_threshold - prior_mean[1]) / prior_sd[1],
          lower.tail = FALSE) * dnorm(z)
  }, -10, 10, abs.tol = 1e-13, rel.tol = 1e-11, subdivisions = 300L)
  data.frame(dose = dose, reference = reference, eta = eta,
             analytic_lower_bound = lower_bound,
             prior_overdose_probability = integral$value,
             estimated_quadrature_error = integral$abs.error,
             omitted_prior_mass_bound = 2 * pnorm(10, lower.tail = FALSE))
})
table <- do.call(rbind, results)
stopifnot(all(table$prior_overdose_probability >= lower_bound), lower_bound > eta)
write.table(table, stdout(), sep = ",", row.names = FALSE, col.names = TRUE)
