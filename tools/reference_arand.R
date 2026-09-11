# Independent integration in the original theta domain, not Python's uniform quantile domain.
cases <- list(
  beta3 = list(family='beta', a=c(.5,2,4), b=c(.5,3,1)),
  beta10 = list(family='beta', a=1:10, b=10:1),
  survival3 = list(family='inverse_gamma', a=c(.5,2,5), b=c(1,3,7)),
  survival10 = list(family='inverse_gamma', a=1:10, b=2:11)
)
rows <- list()
for (name in names(cases)) {
  z <- cases[[name]]
  for (maximize in c(TRUE,FALSE)) {
    for (i in seq_along(z$a)) {
      others <- setdiff(seq_along(z$a),i)
      integrand <- function(x) {
        if (z$family == 'beta') {
          density <- dbeta(x,z$a[i],z$b[i])
          tails <- lapply(others,function(j) pbeta(x,z$a[j],z$b[j],lower.tail=maximize))
        } else {
          density <- dgamma(1/x,z$a[i],rate=z$b[i])/x^2
          tails <- lapply(others,function(j) pgamma(1/x,z$a[j],rate=z$b[j],lower.tail=!maximize))
        }
        density * Reduce('*',tails)
      }
      value <- integrate(integrand,0,if(z$family=='beta') 1 else Inf,
                         rel.tol=1e-11,abs.tol=1e-12,subdivisions=1000)
      rows[[length(rows)+1]] <- data.frame(case=name, family=z$family,
        maximize=maximize, arm=i-1, shape=z$a[i], second=z$b[i],
        probability=value$value, error=value$abs.error)
    }
  }
}
write.csv(do.call(rbind,rows),'tests/fixtures/arand-posterior-r.csv',row.names=FALSE)
