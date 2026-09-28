# Independent base-R references for TOP Supplementary Methods equations 2/6.
# Lin, Coleman and Yuan (2020), doi:10.1093/jnci/djz049.
# Marginal priors aggregate a four-cell Dirichlet prior of total mass one.
# These are deterministic posterior/boundary references, not native simulations.
options(digits = 17)
out <- "tests/fixtures"
settings <- data.frame(
  case = c("coprimary_orr", "coprimary_pfs", "efftox_orr", "efftox_dlt"),
  maximum = c(45L, 45L, 40L, 40L),
  null = c(.45, .30, .15, .30),
  scale = c(.94, .94, .50, .50),
  gamma = c(.50, .50, .60, .60),
  direction = c("efficacy", "efficacy", "efficacy", "toxicity")
)
posterior <- boundaries <- list()
for (i in seq_len(nrow(settings))) {
  cfg <- settings[i, ]
  looks <- if (cfg$maximum == 45L) c(15L, 30L, 45L) else c(10L, 20L, 30L, 40L)
  for (n in looks) {
    cutoff <- cfg$scale * (n / cfg$maximum)^cfg$gamma
    acceptable <- function(r, ess) {
      pbeta(cfg$null, cfg$null + r, 1 - cfg$null + ess - r,
            lower.tail = cfg$direction == "toxicity")
    }
    for (r in 0:n) {
      f <- function(ess) acceptable(r, ess) - cutoff
      lo <- f(r)
      hi <- f(n)
      crossing <- if (lo * hi < 0) {
        uniroot(f, c(r, n), tol = 1e-12)$root
      } else if (lo == 0) r else if (hi == 0) n else NA_real_
      boundaries[[length(boundaries) + 1L]] <- data.frame(
        case = cfg$case, patients = n, events = r,
        acceptable_probability = acceptable(r, n), cutoff = cutoff,
        stop_complete = acceptable(r, n) < cutoff,
        effective_size_crossing = crossing
      )
    }
    for (r in unique(c(0L, as.integer(n / 3), n))) {
      for (ess in unique(c(r, r + .37 * (n - r), n))) {
        posterior[[length(posterior) + 1L]] <- data.frame(
          case = cfg$case, patients = n, events = r, effective_size = ess,
          shape1 = cfg$null + r, shape2 = 1 - cfg$null + ess - r,
          acceptable_probability = acceptable(r, ess), cutoff = cutoff,
          stop = acceptable(r, ess) < cutoff
        )
      }
    }
  }
}
write.csv(settings, file.path(out, "top-endpoints-settings.csv"), row.names = FALSE)
write.csv(do.call(rbind, posterior), file.path(out, "top-endpoints-posterior.csv"),
          row.names = FALSE)
write.csv(do.call(rbind, boundaries), file.path(out, "top-endpoints-boundaries.csv"),
          row.names = FALSE, na = "")
