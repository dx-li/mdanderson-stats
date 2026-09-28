# Independent base-R beta integrals and PLBARPO active-ledger snapshots.
# Each snapshot is an allocation calculation, not an assumed platform scheduler.
options(warn=2,digits=17)
prior <- rbind(c(1,1),c(1,1),c(2,1))
success <- c(1,0,8); failure <- c(0,1,0); assigned <- c(1,1,8)
a <- prior[,1]+success; b <- prior[,2]+failure
variance <- a*b/((a+b)^2*(a+b+1))
cases <- list(
  barcp=list(active=c(TRUE,TRUE,FALSE),method='barcp',tau=.5,tau1=1,floor=c(0,0,0)),
  barn2n=list(active=c(TRUE,TRUE,FALSE),method='barn2n',tau=.5,tau1=1,floor=c(0,0,0)),
  barmtv=list(active=c(TRUE,TRUE,FALSE),method='barmtv',tau=.5,tau1=1,floor=c(0,0,0)),
  dbcd=list(active=c(TRUE,TRUE,FALSE),method='dbcd',tau=2,tau1=.5,
            floor=c(0,0,0),target=c(.25,.75,0)),
  floor=list(active=c(TRUE,TRUE,FALSE),method='barcp',tau=.5,tau1=1,floor=c(0,.4,0)),
  switch_active=list(active=c(FALSE,TRUE,TRUE),method='barcp',tau=.5,tau1=1,floor=c(0,0,0)),
  all_active=list(active=c(TRUE,TRUE,TRUE),method='barcp',tau=.5,tau1=1,floor=c(0,0,0)),
  single_active=list(active=c(FALSE,TRUE,FALSE),method='barn2n',tau=.5,tau1=1,floor=c(0,0,0)))

floor_weights <- function(weights, lower) {
  answer <- numeric(length(weights)); open <- rep(TRUE,length(weights))
  repeat {
    proposed <- (1-sum(answer))*weights[open]/sum(weights[open])
    below <- proposed<lower[open]
    if(!any(below)) {answer[open]<-proposed;return(answer)}
    fixed <- which(open)[below]
    answer[fixed]<-lower[fixed];open[fixed]<-FALSE
  }
}

rows <- list()
for(name in names(cases)) {
  s <- cases[[name]]; ids <- which(s$active)
  best <- numeric(length(a))
  best[ids] <- vapply(ids,function(i) {
    other <- setdiff(ids,i)
    if(!length(other)) return(1)
    integrate(function(x) dbeta(x,a[i],b[i])*
      vapply(x,function(y)prod(pbeta(y,a[other],b[other])),numeric(1)),
      0,1,rel.tol=1e-12,abs.tol=1e-13)$value
  },numeric(1))
  stopifnot(abs(sum(best)-1)<1e-12)
  target <- if(is.null(s$target)) rep(0,length(a)) else s$target
  exponent <- if(s$method=='barn2n') sum(assigned)/(2*20) else s$tau
  weights <- switch(s$method,
    barcp=best[ids]^s$tau,
    barn2n=best[ids]^exponent,
    barmtv=sqrt(best[ids]*variance[ids]/(assigned[ids]+1)),
    dbcd=(target[ids]*(target[ids]/(assigned[ids]/sum(assigned)))^s$tau)^s$tau1)
  allocation <- numeric(length(a))
  allocation[ids] <- floor_weights(weights,s$floor[ids])
  rows[[length(rows)+1L]] <- data.frame(scenario=name,arm=seq_along(a)-1L,
    prior_alpha=prior[,1],prior_beta=prior[,2],successes=success,failures=failure,
    assigned=assigned,active=s$active,method=s$method,tau=s$tau,tau1=s$tau1,max_n=20,
    target=target,floor=s$floor,posterior_alpha=a,posterior_beta=b,
    posterior_variance=variance,best=best,allocation=allocation,
    global_enrolled=sum(assigned),effective_exponent=exponent)
}
out <- do.call(rbind,rows)
stopifnot(max(abs(out$best[out$scenario=='barcp']-c(5/6,1/6,0)))<1e-12)
all_best <- out$best[out$scenario=='all_active']
stopifnot(max(abs(all_best[1:2]/sum(all_best[1:2])-c(15/16,1/16)))<1e-12)
write.csv(out,'tests/fixtures/plbarpo-active-allocation.csv',row.names=FALSE)
cat('Verified',length(cases),'active-ledger allocation snapshots.\n')
