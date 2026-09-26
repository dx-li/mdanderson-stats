# Independent exact small-pending DA-CRM reference. Enumerate missing outcomes,
# integrate gamma hazards analytically, and integrate scalar alpha in base R.
# Synthetic audit data under the likelihood in Liu, Yin, Yuan (2013), Section2.3.
# No original program code or Monte Carlo sampler is used.
options(digits=17)
cases <- list(
  complete=list(d=c(0,1,1,2),y=c(0,1,0,1),t=c(3,.5,3,2.5)),
  early=list(d=c(0,1,1,2,2),y=c(0,1,-1,-1,-1),t=c(3,.5,0,.5,1)),
  late=list(d=c(0,1,1,2,2),y=c(0,1,-1,-1,-1),t=c(3,.5,2,2.5,2.9)),
  pending_prior=list(d=c(0,1,2),y=c(-1,-1,-1),t=c(0,0,0)),
  no_patients=list(d=numeric(0),y=numeric(0),t=numeric(0)),
  interval_edges=list(d=c(0,0,1,1,2,2),y=c(1,1,1,1,0,-1),t=c(0,1,2,3,3,1.2)))
p <- c(.1,.25,.5)
breaks <- c(0,1,2,3)
shape <- c(.5,.75,1)
rate <- c(1,1,1)
alpha_sd <- sqrt(2)
target <- .3
K <- length(shape)
rows <- list()
pack <- function(x) paste(x,collapse="|")
for (name in names(cases)) {
  case <- cases[[name]]
  pending <- which(case$y==-1)
  P <- length(pending)
  exposure <- matrix(0,nrow=length(case$t),ncol=K)
  for (k in seq_len(K)) exposure[,k] <- pmax(0,pmin(case$t,breaks[k+1])-breaks[k])
  delta <- tabulate(findInterval(case$t[case$y==1],breaks,rightmost.closed=TRUE),K)
  updated_shape <- shape+delta
  states <- if (P==0) matrix(numeric(0),nrow=1,ncol=0) else
    as.matrix(expand.grid(rep(list(c(0,1)),P)))
  fits <- vector("list",nrow(states))
  for (s in seq_len(nrow(states))) {
    y <- case$y
    y[pending] <- states[s,]
    # Bernoulli patient likelihood: no grouped binomial coefficients, since
    # missing-outcome completions must not acquire combinatorial reweighting.
    lp <- function(a) vapply(a,function(v)
      sum(dbinom(y,1,exp(log(p[case$d+1])*exp(v)),log=TRUE))+
        dnorm(v,0,alpha_sd,log=TRUE),numeric(1))
    objective <- function(a) {
      value <- -lp(a)
      if (is.infinite(value) && value>0) .Machine$double.xmax else value
    }
    mode <- optimize(objective,c(-40,40),tol=1e-10)$minimum
    shift <- lp(mode)
    weight <- function(a) exp(lp(a)-shift)
    cuts <- sort(unique(c(-Inf,mode+c(-12,-4,-1,-.1,0,.1,1,4,12)*alpha_sd,Inf)))
    integral <- function(f,lower=-Inf,upper=Inf) {
      limits <- c(lower,cuts[cuts>lower & cuts<upper],upper)
      sum(vapply(seq_len(length(limits)-1),function(i)
        integrate(function(a) weight(a)*f(a),limits[i],limits[i+1],
                  rel.tol=1e-10,abs.tol=1e-13,subdivisions=200)$value,numeric(1)))
    }
    one <- function(a) rep(1,length(a))
    z <- integral(one)
    alpha_mean <- integral(function(a) a)/z
    alpha_second <- integral(function(a) a*a)/z
    mean <- vapply(p,function(prob)
      integral(function(a) exp(log(prob)*exp(a)))/z,numeric(1))
    overdose <- vapply(p,function(prob) {
      cutoff <- log(-log(target))-log(-log(prob))
      if (cutoff<=mode) integral(one,upper=cutoff)/z else 1-integral(one,lower=cutoff)/z
    },numeric(1))
    updated_rate <- rate+colSums(exposure*y)
    hazard_mean <- updated_shape/updated_rate
    hazard_second <- updated_shape*(updated_shape+1)/updated_rate^2
    gamma_log_evidence <- sum(shape*log(rate)-lgamma(shape)+lgamma(updated_shape)-
                              updated_shape*log(updated_rate))
    fits[[s]] <- list(log_weight=shift+log(z)+gamma_log_evidence,
                     alpha_mean=alpha_mean,alpha_second=alpha_second,
                     dose_mean=mean,overdose=overdose,
                     hazard_mean=hazard_mean,hazard_second=hazard_second)
  }
  lw <- vapply(fits,function(f) f$log_weight,numeric(1))
  w <- exp(lw-max(lw)); w <- w/sum(w)
  average <- function(key) Reduce(`+`,Map(function(f,weight) weight*f[[key]],fits,w))
  a_mean <- average('alpha_mean')
  h_mean <- average('hazard_mean')
  quantities <- list(alpha_mean=a_mean,alpha_sd=sqrt(average('alpha_second')-a_mean^2),
                     hazard_mean=h_mean,hazard_sd=sqrt(average('hazard_second')-h_mean^2),
                     dose_mean=average('dose_mean'),overdose=average('overdose'),
                     pending_probability=as.vector(crossprod(w,states)))
  for (quantity in names(quantities)) for (j in seq_along(quantities[[quantity]])) {
    rows[[length(rows)+1]] <- data.frame(
      scenario=name,skeleton=pack(p),doses=pack(case$d),outcomes=pack(case$y),
      times=pack(case$t),breaks=pack(breaks),shape=pack(shape),rate=pack(rate),
      prior_sd=alpha_sd,target=target,quantity=quantity,index=j-1,
      value=quantities[[quantity]][j])
  }
}
write.csv(do.call(rbind,rows),'tests/fixtures/dacrm-posterior.csv',row.names=FALSE)
