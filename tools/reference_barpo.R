# Independent base-R posterior integrals and four BARPO allocation formulas.
a <- c(2,5,9); b <- c(5,3,2)
assigned <- c(5,8,12); maximum <- 50
best <- vapply(seq_along(a), function(i) integrate(function(z)
  dbeta(z,a[i],b[i])*vapply(z,function(t)prod(pbeta(t,a[-i],b[-i])),numeric(1)),
  0,1,rel.tol=1e-12)$value,numeric(1))
control <- c(.5,vapply(2:3,function(i)integrate(function(z)
  dbeta(z,a[i],b[i])*pbeta(z,a[1],b[1]),0,1,rel.tol=1e-12)$value,numeric(1)))
variance <- a*b/((a+b)^2*(a+b+1))
write.csv(data.frame(arm=1:3,alpha=a,beta=b,assigned=assigned,best=best,
  variance=variance,above_control=control,
  futility=pbeta(.25,a,b),efficacy=pbeta(.65,a,b,lower.tail=FALSE),
  final_efficacy=pbeta(.5,a,b,lower.tail=FALSE)),
  'tests/fixtures/barpo-posterior.csv',row.names=FALSE)
weights <- list(BARCP=best^.7,BARN2N=best^(sum(assigned)/(2*maximum)),
  BARMTV=sqrt(best*variance/(assigned+1)),DBCD=(c(.2,.3,.5)*(c(.2,.3,.5)/(assigned/sum(assigned)))^2)^.5)
# Proportional redistribution with fixed lower bounds. One control floor is
# exactly the guide's stated rescaling; simultaneous floors are an explicit policy.
floor_allocation <- function(p, lower) {
  active <- rep(TRUE,length(p)); result <- numeric(length(p)); remaining <- 1
  repeat {
    candidate <- remaining*p[active]/sum(p[active])
    below <- candidate < lower[active]
    if (!any(below)) {result[active] <- candidate;break}
    ids <- which(active)[below]
    result[ids] <- lower[ids];active[ids] <- FALSE
    remaining <- 1-sum(result)
  }
  result
}
rows <- list()
for (method in names(weights)) for (floor_name in c('none','control','all')) {
  lower <- switch(floor_name,none=c(0,0,0),control=c(.4,0,0),all=rep(.15,3))
  allocation <- floor_allocation(weights[[method]]/sum(weights[[method]]),lower)
  rows[[length(rows)+1]] <- data.frame(method=method,floor=floor_name,arm=1:3,
    probability=allocation)
}
write.csv(do.call(rbind,rows),'tests/fixtures/barpo-allocation.csv',row.names=FALSE)
