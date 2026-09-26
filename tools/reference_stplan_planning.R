# Independent inverse-planning references. Base R only; no Python calls.
args <- commandArgs(trailingOnly=TRUE)
if (length(args)!=1L) stop('Supply output CSV path')
rows <- list()
solve_case <- function(id, f, bounds, target=.8, integer=FALSE, largest=FALSE) {
  previous <- NA_real_
  if (integer) {
    candidates <- seq.int(ceiling(bounds[1]),floor(bounds[2]))
    if (largest) candidates <- rev(candidates)
    powers <- vapply(candidates,f,numeric(1))
    at <- which(powers>=target)[1]
    if (is.na(at)) stop(paste('Unattainable integer case',id))
    value <- candidates[at]
    if (at>1) previous <- powers[at-1]
  } else value <- uniroot(function(x)f(x)-target,bounds,tol=1e-11)$root
  rows[[length(rows)+1L]] <<- data.frame(case_id=id,value=value,target=target,
    achieved=f(value),previous_power=previous)
}
t_power <- function(d,sd,n1,n2=NULL,alpha=.05) {
  df <- if (is.null(n2)) n1-1 else n1+n2-2
  se <- if (is.null(n2)) sd/sqrt(n1) else sd*sqrt(1/n1+1/n2)
  pt(qt(alpha,df,lower.tail=FALSE),df,ncp=abs(d)/se,lower.tail=FALSE)
}
exponential_one <- function(ratio,n,alpha=.05) {
  upper <- ratio>1
  critical <- qchisq(alpha,2*n,lower.tail=!upper)
  pchisq(critical/ratio,2*n,lower.tail=!upper)
}
exponential_two <- function(m1,m2,n1,n2,alpha=.05) {
  if (m1>m2) {dfn<-2*n1;dfd<-2*n2} else {dfn<-2*n2;dfd<-2*n1}
  pf(qf(alpha,dfn,dfd,lower.tail=FALSE)/max(m1/m2,m2/m1),
    dfn,dfd,lower.tail=FALSE)
}
cor_moments <- function(r,n) c(atanh(r)+r/(2*(n-1)),1/(n-1)+(4-r*r)/(2*(n-1)^2))
cor_one <- function(r0,ra,n,alpha=.05) {
  h0<-cor_moments(r0,n);ha<-cor_moments(ra,n)
  pnorm(abs(ha[1]-h0[1])/sqrt(h0[2])+qnorm(alpha))
}
cor_two <- function(r1,r2,n1,n2,alpha=.05) {
  a<-cor_moments(r1,n1);b<-cor_moments(r2,n2)
  pnorm(abs(a[1]-b[1])/sqrt(a[2]+b[2])+qnorm(alpha))
}
arc <- function(p)2*asin(sqrt(p))
arc_power <- function(p1,p2,n1,n2)pnorm(abs(arc(p1)-arc(p2))/sqrt(1/n1+1/n2)+qnorm(.05))
exact_binomial <- function(n,p0,pa,alpha=.05,upper=pa>p0) {
  if (n==0) return(0)
  k<-0:n;null<-dbinom(k,n,p0);alt<-dbinom(k,n,pa)
  tails<-if(upper)rev(cumsum(rev(null))) else cumsum(null)
  sum(alt[tails<=alpha])
}
exposure <- function(f,re,ru) {
  disease<-f*re+(1-f)*ru
  c(f*re/disease,f*(1-re)/(1-disease))
}
matched <- function(re,n) {
  p<-exposure(.3,re,.05);a<-p[1]*(1-p[2]);b<-(1-p[1])*p[2]
  k<-0:n
  sum(dbinom(k,n,a+b)*vapply(k,function(j)exact_binomial(j,.5,a/(a+b),upper=TRUE),numeric(1)))
}
poisson_two <- function(rate2) {
  mu<-10+rate2*10;pa<-10/mu
  n<-0:ceiling(mu+12*sqrt(mu)+30)
  sum(dpois(n,mu)*vapply(n,function(k)exact_binomial(k,.5,pa),numeric(1)))
}
event_probability <- function(mean,accrual,followup) {
  integrate(function(t)-expm1(-t/mean),followup,followup+accrual,
    rel.tol=1e-12,abs.tol=1e-14)$value/accrual
}
censored_one <- function(rate) {
  mu<-rate*12*event_probability(15,12,6)
  n<-seq_len(ceiling(mu+12*sqrt(mu)+30))
  sum(dpois(n,mu)*exponential_one(1.5,n))
}
george_desu <- function(rate) {
  d<-rate*12/2*c(event_probability(10,12,6),event_probability(15,12,6))
  s<-sum(c(10,15)*d)/25
  exponential_two(10,15,s,s)
}
information <- function(followup) {
  d<-20*12/2*(event_probability(10,12,followup)+event_probability(15,12,followup))
  pnorm(log(1.5)*sqrt(d/4)+qnorm(.05))
}
historical <- function(he) {
  future_control<-20*(-expm1(-.1*18))+.2*5*12*event_probability(10,12,6)
  future_experimental<-.8*5*12*event_probability(1/he,12,6)
  boundary<-sqrt(1/(40+future_control)+1/future_experimental)
  spread<-sqrt(future_control/(40+future_control)^2+1/future_experimental)
  pnorm((log(.1/he)+qnorm(.05)*boundary)/spread)
}
piecewise_probability <- function(h1,h2) {
  event<-function(t)-expm1(-h1*pmin(t,8)-h2*pmax(t-8,0))
  (integrate(event,5,8,rel.tol=1e-12)$value+integrate(event,8,15,rel.tol=1e-12)$value)/10
}
piecewise <- function(rate) {
  d<-rate*10/2*(piecewise_probability(.2,.05)+piecewise_probability(.3,.075))
  pnorm(log(1.5)*sqrt(d/4)+qnorm(.05))
}
k_sample <- function(p,n) {
  pooled<-sum(n*p)/sum(n)
  ncp<-sum(n*(p-pooled)^2)/(pooled*(1-pooled))
  pchisq(qchisq(.05,length(p)-1,lower.tail=FALSE),length(p)-1,ncp=ncp,lower.tail=FALSE)
}

solve_case('normal_n',function(n)t_power(.5,1,n),c(2,1000))
solve_case('normal_difference',function(d)t_power(d,1,10),c(0,3))
solve_case('normal_sd',function(sd)t_power(.5,sd,10),c(.1,10))
solve_case('normal_alpha',function(a)t_power(.5,1,20,alpha=a),c(.0001,.49))
solve_case('normal_equal_n',function(n)t_power(.5,1,n,n),c(2,1000))
solve_case('welch_n1',function(n) {
  v<-1/n+4/200;df<-v^2/((1/n)^2/(n-1)+(4/200)^2/199)
  pt(qt(.05,df,lower.tail=FALSE),df,ncp=.5/sqrt(v),lower.tail=FALSE)
},c(2,1000))
solve_case('lognormal_cv',function(cv)t_power(log(1.25),sqrt(log1p(cv^2)),100,100),c(.01,2))
solve_case('exponential_ratio',function(r)exponential_one(r,30),c(1,5))
solve_case('exponential_equal_n',function(n)exponential_two(10,15,n,n),c(2,1000))
solve_case('correlation_n',function(n)cor_one(0,.4,n),c(4,1000))
solve_case('correlation_alternative',function(r)cor_one(.1,r,50),c(.1,.95))
solve_case('correlation_two_n',function(n)cor_two(.1,.5,n,100),c(4,1000))
solve_case('arcsine_equal_n',function(n)arc_power(.2,.4,n,n),c(2,1000))
solve_case('median_n',function(n)arc_power(.2,.4,n/2,n/2),c(4,2000))
solve_case('historical_binomial_n',function(n)
  pnorm((abs(arc(.4)-arc(.2))+qnorm(.05)*sqrt(1/n+1/100))*sqrt(n)),c(2,1000))
solve_case('responder_margin',function(d)pnorm((d-.1)/sqrt(.3*.7/100+.4*.6/80)-qnorm(.95)),c(0,.9))
solve_case('fisher_n',function(n)
  pnorm((abs(n*.2-1)/sqrt(n)+qnorm(.05)*sqrt(2*.3*.7))/sqrt(.2*.8+.4*.6)),c(6,1000))
solve_case('exact_binomial_probability',function(pa)exact_binomial(20,.2,pa),c(.2,.9))
solve_case('exact_binomial_n',function(n)exact_binomial(n,.2,.4),c(1,100),integer=TRUE)
solve_case('exact_poisson_rate',function(rate) {
  k<-0:200;reject<-ppois(k-1,10,lower.tail=FALSE)<=.05
  sum(dpois(k[reject],rate*10))
},c(1,4))
solve_case('retention_n',function(n)pbinom(79,n,.95^3,lower.tail=FALSE),c(80,140),target=.9,integer=TRUE)
solve_case('retention_threshold',function(k)pbinom(k-1,100,.95^3,lower.tail=FALSE),
  c(0,100),target=.9,integer=TRUE,largest=TRUE)
solve_case('unmatched_n',function(n) {
  p<-exposure(.3,.1,.05)
  pnorm((arc(p[1])-arc(p[2]))/sqrt(1/n+1/120)+qnorm(.05))
},c(2,1000))
solve_case('matched_exposed_risk',function(re)matched(re,60),c(.05,.5))
solve_case('matched_n',function(n)matched(.1,n),c(2,200),integer=TRUE)
solve_case('poisson_two_rate',poisson_two,c(1,4))
solve_case('censored_one_accrual',censored_one,c(1,30))
solve_case('george_desu_accrual',george_desu,c(1,100))
solve_case('information_followup',information,c(0,100))
solve_case('historical_survival_hazard',historical,c(.01,.1))
solve_case('piecewise_accrual',piecewise,c(1,100))
solve_case('binomial_k_total',function(n)k_sample(c(.1,.2,.3),n*c(1,2,1)/4),c(1,2000))
solve_case('binomial_k_probability',function(p)k_sample(c(.1,.2,p),c(20,40,20)),c(.2,.9))
options(digits=17)
write.table(do.call(rbind,rows),file=args[1],sep=',',row.names=FALSE,quote=FALSE)
