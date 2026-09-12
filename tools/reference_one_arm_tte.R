# Independent gamma integration for ID98 mean/median, both goals, signed margins.
rows <- list()
for (parameterization in c('mean','median')) {
  a_s <- 4; b_s <- 8; a_e <- 2+2
  b_e <- 3 + 5 * if (parameterization=='mean') 1 else log(2)
  for (delta in c(-1,0,1)) for (goal in c('maximize','minimize')) {
    integrand <- function(z) {
      threshold <- b_s/z + delta
      value <- rep(if (goal=='maximize') 1 else 0,length(z))
      valid <- threshold>0
      value[valid] <- pgamma(b_e/threshold[valid],shape=a_e,lower.tail=goal=='maximize')
      dgamma(z,shape=a_s)*value
    }
    p <- integrate(integrand,0,Inf,rel.tol=1e-11,abs.tol=1e-13,subdivisions=500)$value
    rows[[length(rows)+1]] <- data.frame(parameterization=parameterization,goal=goal,
      delta=delta,probability=p,posterior_shape=a_e,posterior_scale=b_e)
  }
}
write.csv(do.call(rbind,rows),'tests/fixtures/one-arm-tte-reference.csv',row.names=FALSE)
