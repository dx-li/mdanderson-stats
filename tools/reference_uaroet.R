# Independent base-R evaluation of UAROET's published probability equations.
# Conditional integration over a uniform efficacy quantile, rather than the
# Python kernel's normal-coordinate rectangles. No vendor code or R packages.
options(digits=17, warn=2)

marginal <- function(theta) {
  lambda <- plogis(theta)
  survival <- c(1,cumprod(lambda))
  c(survival[-length(survival)]*(1-lambda),tail(survival,1))
}
logits <- function(p) {
  tailmass <- rev(cumsum(rev(p)))
  qlogis(tailmass[-1]/head(tailmass,-1))
}
joint <- function(e,t,rho) {
  if(rho==0) return(outer(e,t))
  ec <- c(0,cumsum(e)); tc <- c(0,cumsum(t)); tb <- qnorm(tc)
  result <- matrix(0,length(e),length(t))
  for(a in seq_along(e)) for(b in seq_along(t)) {
    if(abs(rho)==1) {
      low <- if(rho==1) tc[b] else 1-tc[b+1]
      high <- if(rho==1) tc[b+1] else 1-tc[b]
      result[a,b] <- max(0,min(ec[a+1],high)-max(ec[a],low))
    } else {
      f <- function(u) {
        x <- qnorm(u); sd <- sqrt((1-rho)*(1+rho))
        lo <- (tb[b]-rho*x)/sd; hi <- (tb[b+1]-rho*x)/sd
        ifelse(lo>=0,pnorm(lo,lower.tail=FALSE)-pnorm(hi,lower.tail=FALSE),
               pnorm(hi)-pnorm(lo))
      }
      result[a,b] <- integrate(f,ec[a],ec[a+1],abs.tol=1e-13,
                               rel.tol=1e-11,subdivisions=1000)$value
    }
  }
  result
}
# Table 2, with rows/doses and columns/categories; utility is efficacy x toxicity.
published_e <- rbind(c(.2,.4,.35,.05),c(.1,.3,.45,.15),c(.1,.2,.5,.2))
published_t <- rbind(c(.65,.2,.12,.03),c(.55,.25,.15,.05),c(.4,.3,.23,.07))
utility <- rbind(c(50,25,10,0),c(85,50,15,5),c(92,60,20,7),c(100,75,25,10))
cases <- list(
  published=list(e=t(apply(published_e,1,logits)),t=t(apply(published_t,1,logits)),rho=.1),
  binary=list(e=matrix(c(-.7,.3),ncol=1),t=matrix(c(-1.2,-.1),ncol=1),rho=.6),
  negative=list(e=rbind(c(-1,.5),c(.2,-.3)),t=rbind(c(-.7,.2),c(.3,.6)),rho=-.5),
  independent=list(e=matrix(c(.2,.8,-.4),ncol=1),t=matrix(c(-.4,-.1,.9),ncol=1),rho=0),
  positive_limit=list(e=matrix(c(.4,-.8),nrow=1),t=matrix(c(-.2,.6),nrow=1),rho=1),
  negative_limit=list(e=matrix(c(.4,-.8),nrow=1),t=matrix(c(-.2,.6),nrow=1),rho=-1))
rows <- list(); theta_rows <- list()
for(name in names(cases)) {
  s <- cases[[name]]
  for(dose in seq_len(nrow(s$e))) {
    e <- marginal(s$e[dose,]); t <- marginal(s$t[dose,]); p <- joint(e,t,s$rho)
    for(endpoint in c('efficacy','toxicity')) {
      theta <- if(endpoint=='efficacy') s$e[dose,] else s$t[dose,]
      theta_rows[[length(theta_rows)+1]] <- data.frame(
        scenario=name,dose=dose-1,endpoint=endpoint,threshold=seq_along(theta),logit=theta)
    }
    for(a in seq_along(e)) for(b in seq_along(t)) {
      rows[[length(rows)+1]] <- data.frame(
        scenario=name,dose=dose-1,efficacy=a-1,toxicity=b-1,rho=s$rho,
        efficacy_marginal=e[a],toxicity_marginal=t[b],joint=p[a,b],
        mean_utility=if(name=='published') sum(p*utility) else NA_real_)
    }
  }
}
write.csv(do.call(rbind,theta_rows),'tests/fixtures/uaroet-logits.csv',row.names=FALSE)
write.csv(do.call(rbind,rows),'tests/fixtures/uaroet-probabilities.csv',row.names=FALSE)

# Exact factorization at rho=0: one dose, independent logistic-normal posteriors.
# Joint counts (efficacy x toxicity): [[6,1],[2,3]], N=12.
settings <- data.frame(endpoint=c('efficacy','toxicity'),events=c(5,4),subjects=12,
                       prior_mean=c(.2,-.7),prior_sd=c(1.1,.8))
rows <- list()
for(i in seq_len(nrow(settings))) {
  s <- settings[i,]
  density <- function(theta) dnorm(theta,s$prior_mean,s$prior_sd)*
    plogis(theta)^s$events*plogis(-theta)^(s$subjects-s$events)
  integral <- function(f) integrate(function(z) f(z)*density(z),-Inf,Inf,
                                    rel.tol=1e-11,abs.tol=1e-13,subdivisions=1000)$value
  z <- integral(function(x) rep(1,length(x)))
  mean <- integral(identity)/z
  rows[[i]] <- data.frame(s,log_evidence=log(z),theta_mean=mean,
    theta_sd=sqrt(integral(function(x) (x-mean)^2)/z),
    response_mean=integral(plogis)/z)
}
write.csv(do.call(rbind,rows),'tests/fixtures/uaroet-posterior.csv',row.names=FALSE)
