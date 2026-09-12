# Independent base-R integration of the BaCIS classification model.
# JAGS is not required. Native R and help adaptive cutoffs are both recorded.
options(digits=17)
softplus <- function(x) pmax(x,0)+log1p(exp(-abs(x)))
evidence <- function(y,n,gamma,tau) {
  score <- function(eta) y-n*plogis(eta)-tau*(eta-gamma)
  mode <- uniroot(score,c(gamma+(y-n)/tau,gamma+y/tau),tol=1e-12)$root
  kernel <- function(eta) -y*softplus(-eta)-(n-y)*softplus(eta)-tau*(eta-gamma)^2/2
  peak <- kernel(mode)
  sd <- 1/sqrt(tau+n*plogis(mode)*plogis(-mode))
  integral <- integrate(function(z) exp(kernel(mode+sd*z)-peak),-Inf,Inf,
                        rel.tol=1e-11,abs.tol=1e-12,subdivisions=300)$value
  peak+log(sd)+log(tau/(2*pi))/2+log(integral)+lchoose(n,y)
}
cases <- list(
 native=list(y=c(2,3,7,6,10),n=rep(25,5),low=.1,high=.3,tau=NA),
 help=list(y=c(1,2,3,7,8),n=c(15,18,10,15,20),low=.1,high=.3,tau=NA),
 unequal=list(y=c(0,10,3),n=c(2,100,5),low=.1,high=.3,tau=NA),
 diffuse=list(y=c(0,1000,1,500),n=rep(1000,4),low=.1,high=.3,tau=.1),
 extremes=list(y=c(0,1000),n=rep(1000,2),low=.1,high=.3,tau=NA))
rows <- list()
for(name in names(cases)) {
  x <- cases[[name]]; tau <- x$tau
  if(is.na(tau)) tau <- (6/(qlogis(x$high)-qlogis(x$low)))^2
  cut_group <- plogis(-2*(mean(x$y/x$n)-(x$low+x$high)/2)/(x$high-x$low))
  cut_patient <- plogis(-2*(sum(x$y)/sum(x$n)-(x$low+x$high)/2)/(x$high-x$low))
  for(i in seq_along(x$y)) {
    lo <- evidence(x$y[i],x$n[i],qlogis(x$low),tau)
    hi <- evidence(x$y[i],x$n[i],qlogis(x$high),tau)
    rows[[length(rows)+1]] <- data.frame(case=name,group=i,y=x$y[i],n=x$n[i],
      phi_low=x$low,phi_high=x$high,precision=tau,log_low=lo,log_high=hi,
      p_high=plogis(hi-lo),p_low=plogis(lo-hi),
      cutoff_subgroup=cut_group,cutoff_patient=cut_patient)
  }
}
write.csv(do.call(rbind,rows),'tests/fixtures/bacis-classification.csv',row.names=FALSE)
# Native singleton cluster fallback, exactly Beta(1+y,1+n-y).
rows <- lapply(c(0,2,9),function(y) {
  a <- 1+y; b <- 11-y
  data.frame(y=y,n=10,mean=a/(a+b),sd=sqrt(a*b/((a+b)^2*(a+b+1))),
             efficacy=pbeta(.1,a,b,lower.tail=FALSE),high=pbeta(.3,a,b,lower.tail=FALSE))
})
write.csv(do.call(rbind,rows),'tests/fixtures/bacis-singleton.csv',row.names=FALSE)
# Independent Model-2 conditional limit: mu fixed at the cluster logit,
# tau3 fixed at one. Concentrated hyperpriors approach these values.
rows <- lapply(seq_along(c(2,3,7,10)),function(i) {
  y <- c(2,3,7,10)[i]; center <- c(.1,.1,.3,.3)[i]
  density <- function(eta) dnorm(eta,qlogis(center),1)*dbinom(y,25,plogis(eta))
  norm <- integrate(density,-Inf,Inf,abs.tol=1e-18,rel.tol=1e-11)$value
  mean <- integrate(function(eta) plogis(eta)*density(eta),-Inf,Inf,
                    abs.tol=1e-18,rel.tol=1e-11)$value/norm
  tail <- integrate(density,qlogis(.1),Inf,abs.tol=1e-18,rel.tol=1e-11)$value/norm
  data.frame(y=y,n=25,center=center,mean=mean,efficacy=tail)
})
write.csv(do.call(rbind,rows),'tests/fixtures/bacis-borrowing-limit.csv',row.names=FALSE)
