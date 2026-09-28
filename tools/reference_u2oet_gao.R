# Independent GAO equations from Thall, Nguyen and Zinner (2017), Appendix A.
# Raw doses; one shared positive kappa; endpoint-specific positive lambda.
# Gaussian rectangles use conditional-normal quadrature, not an FGM copula.
options(warn=2, digits=17)

marginal <- function(d1,d2,intercepts,slopes,lambda,kappa) {
  eta <- intercepts + sweep(slopes,2,c(d1,d2),'*')
  s <- exp(eta[,1])+exp(eta[,2])+kappa*exp(rowSums(eta))
  failure <- exp(-log1p(lambda*s)/lambda)
  continuation <- -expm1(-log1p(lambda*s)/lambda)
  survival <- c(1,cumprod(continuation))
  c(survival[-length(survival)]*failure,tail(survival,1))
}

rectangle <- function(ep,tp,rho) {
  ec <- c(0,cumsum(ep)); tc <- c(0,cumsum(tp))
  ec[length(ec)] <- tc[length(tc)] <- 1
  result <- matrix(0,length(ep),length(tp))
  if(rho==0) return(outer(ep,tp))
  if(abs(rho)==1) {
    for(e in seq_along(ep)) for(t in seq_along(tp)) {
      lower <- if(rho>0) tc[t] else 1-tc[t+1]
      upper <- if(rho>0) tc[t+1] else 1-tc[t]
      result[e,t] <- max(0,min(ec[e+1],upper)-max(ec[e],lower))
    }
    return(result)
  }
  ez <- qnorm(ec); tz <- qnorm(tc); sd <- sqrt(1-rho*rho)
  for(e in seq_along(ep)) for(t in seq_along(tp)) {
    integrand <- function(x) dnorm(x)*(
      pnorm((tz[t+1]-rho*x)/sd)-pnorm((tz[t]-rho*x)/sd))
    result[e,t] <- integrate(integrand,ez[e],ez[e+1],
      abs.tol=1e-13,rel.tol=1e-11,subdivisions=1000L)$value
  }
  stopifnot(max(abs(rowSums(result)-ep))<1e-10,
            max(abs(colSums(result)-tp))<1e-10)
  result
}

cases <- list(
  binary=list(d1=c(1,3),d2=c(2,5),ei=matrix(c(-1,.3),1,2),
    es=matrix(c(.2,-.1),1,2),ti=matrix(c(-.4,-1.2),1,2),
    ts=matrix(c(-.05,.15),1,2),el=1,tl=1,k=.4),
  ordinal=list(d1=c(.5,2),d2=c(1,4),
    ei=matrix(c(-.8,.2,.3,-.6,-.2,-1.0),3,2,byrow=TRUE),
    es=matrix(c(.3,-.1,-.2,.25,.1,.15),3,2,byrow=TRUE),
    ti=matrix(c(-1,.4,-.3,-.7),2,2,byrow=TRUE),
    ts=matrix(c(.1,.15,-.1,.2),2,2,byrow=TRUE),el=.4,tl=1.8,k=.7),
  small_link=list(d1=c(.5,2),d2=c(1,3),
    ei=matrix(c(-2,-1.5,-1.2,-2.2),2,2,byrow=TRUE),
    es=matrix(c(.1,-.1,.2,.05),2,2,byrow=TRUE),
    ti=matrix(c(-1.5,-2),1,2),ts=matrix(c(-.1,.2),1,2),
    el=1e-7,tl=1e-6,k=.05))

rows <- list(); summaries <- list()
for(name in names(cases)) {
  x <- cases[[name]]
  for(rho in c(-1,-.65,0,.55,1)) {
    ll <- 0
    for(i in seq_along(x$d1)) for(j in seq_along(x$d2)) {
      ep <- marginal(x$d1[i],x$d2[j],x$ei,x$es,x$el,x$k)
      tp <- marginal(x$d1[i],x$d2[j],x$ti,x$ts,x$tl,x$k)
      jp <- rectangle(ep,tp,rho)
      for(e in seq_along(ep)) for(t in seq_along(tp)) {
        # Deterministic grouped counts; include structural zeros in the fixture.
        n <- (i+2*j+e+t)%%3
        if(n>0) ll <- ll+n*log(jp[e,t])
        rows[[length(rows)+1L]] <- data.frame(scenario=name,association=rho,
          dose1=i-1L,dose2=j-1L,efficacy=e-1L,toxicity=t-1L,
          efficacy_probability=ep[e],toxicity_probability=tp[t],
          joint_probability=jp[e,t],count=n)
      }
    }
    summaries[[length(summaries)+1L]] <- data.frame(
      scenario=name,association=rho,loglikelihood=ll)
  }
}
write.csv(do.call(rbind,rows),'tests/fixtures/u2oet-gao-probabilities.csv',row.names=FALSE)
write.csv(do.call(rbind,summaries),'tests/fixtures/u2oet-gao-likelihood.csv',row.names=FALSE)
cat('Generated',length(rows),'GAO joint-cell references and',length(summaries),
    'grouped likelihood references.\n')
