# Independent integration over historical inverse-mean gamma distribution.
# TTEConduct guide (Cook, November 2006), section 5.
# All time inputs and continuous roots are in months; no day conversion assumed.
probability <- function(events, total_time, delta=1) {
  integrate(function(z) {
    dgamma(z, shape=60) * pgamma((10+total_time)/(295/z+delta), shape=3+events)
  }, 0, Inf, rel.tol=1e-11, abs.tol=1e-13, subdivisions=500)$value
}
rows <- lapply(1:6, function(d) {
  boundary <- if (probability(d,0) >= .03) 0 else
    uniroot(function(t) probability(d,t)-.03, c(0,100), tol=1e-10)$root
  data.frame(events=d,minimum_total_time=boundary,
             probability_at_boundary=probability(d,boundary),
             probability_at_one_month=probability(d,1),
             probability_at_twenty_months=probability(d,20))
})
write.csv(do.call(rbind,rows), 'tests/fixtures/tteconduct-reference.csv',row.names=FALSE)
