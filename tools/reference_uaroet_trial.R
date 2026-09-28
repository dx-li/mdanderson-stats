# Independent base-R references for complete-outcome UAROET trial progression.
# One dose and fixed zero association factorize the posterior into two scalar
# logistic-normal integrals. No Python routines, vendor code or R packages.
options(digits=17, warn=2)

truth <- c(.5, .1, .1, .3) # row-major (E,T): 00, 01, 10, 11
coins <- c(.05, .55, .65, .95, .45, .85)
cell <- vapply(coins, function(u) which(u < cumsum(truth))[1]-1L, integer(1))
efficacy <- cell %/% 2L
toxicity <- cell %% 2L
stopifnot(identical(cell, c(0L, 1L, 2L, 3L, 0L, 3L)))

response_mean <- function(events, n, mu, sd) {
  density <- function(z) dnorm(z, mu, sd)*plogis(z)^events*plogis(-z)^(n-events)
  integrate_mean <- function(f) integrate(function(z) f(z)*density(z), -Inf, Inf,
    rel.tol=1e-11, abs.tol=1e-13, subdivisions=1000)$value
  integrate_mean(plogis)/integrate_mean(function(z) rep(1, length(z)))
}

cases <- list(
  select=list(looks=c(2L,4L,6L), toxicity_limit=1, final_n=6L),
  early_stop=list(looks=2L, toxicity_limit=0, final_n=6L),
  final_no_selection=list(looks=6L, toxicity_limit=0, final_n=6L))
rows <- list()
paths <- list()
for (name in names(cases)) {
  s <- cases[[name]]
  for (n in s$looks) {
    pe <- response_mean(sum(efficacy[seq_len(n)]), n, .2, 1.1)
    pt <- response_mean(sum(toxicity[seq_len(n)]), n, -.7, .8)
    # Utility [[.2,0],[1,.4]], good cutoff .4 includes both E=1 cells.
    mean_utility <- .2+.8*pe-.2*pt-.4*pe*pt
    unsafe <- s$toxicity_limit == 0 # strictly positive logistic-normal tail
    terminal <- unsafe || n == s$final_n
    rows[[length(rows)+1L]] <- data.frame(
      scenario=name, enrolled=n, count00=sum(cell[seq_len(n)]==0L),
      count01=sum(cell[seq_len(n)]==1L), count10=sum(cell[seq_len(n)]==2L),
      count11=sum(cell[seq_len(n)]==3L), efficacy_mean=pe, toxicity_mean=pt,
      mean_utility=mean_utility, good_outcome_probability=pe,
      toxicity_limit=s$toxicity_limit, p_L=0, p_U=.8, utility_tolerance=0,
      probability_best=1, toxicity_exceedance=as.integer(unsafe),
      eligible=!unsafe, terminal=terminal, stopped_early=unsafe && n<s$final_n,
      selected_dose=if(terminal && !unsafe) 0L else NA_integer_)
  }
  used <- seq_len(max(s$looks))
  paths[[length(paths)+1L]] <- data.frame(
    scenario=name, patient=used, outcome_uniform=coins[used], assigned_dose=0L,
    efficacy=efficacy[used], toxicity=toxicity[used])
}
write.csv(do.call(rbind, rows), 'tests/fixtures/uaroet-trial-looks.csv', row.names=FALSE)
write.csv(do.call(rbind, paths), 'tests/fixtures/uaroet-trial-patients.csv', row.names=FALSE)
