# Independent base-R calculations and quadrature for STPLAN survival methods.
args <- commandArgs(trailingOnly=TRUE)
if (length(args) != 2L) stop('Supply native-case CSV and output CSV paths')
cases <- read.csv(args[1],check.names=FALSE)
# Additional independent cases exercise both hazard segments within the observed
# age interval, including zero-hazard limits unsupported by the native integral.
for (hazards in list(c(.2,.05),c(0,.05),c(.2,0))) {
  extra <- cases[cases$routine=='piecewise_native',,drop=FALSE][1,,drop=FALSE]
  extra[1,paste0('a',1:10)] <- list(1.5,5,10,5,.05,8,hazards[1],hazards[2],1,0)
  extra$power <- NA_real_
  cases <- rbind(cases,extra)
}
pdeath <- function(mean,accrual,followup) {
  integrate(function(age)-expm1(-age/mean),followup,followup+accrual,
    rel.tol=1e-12,abs.tol=1e-14)$value/accrual
}
piecewise_death <- function(h1,h2,breakpoint,accrual,followup) {
  survival <- function(age)exp(-h1*pmin(age,breakpoint)-h2*pmax(age-breakpoint,0))
  bounds <- sort(unique(c(followup,followup+accrual,
    max(followup,min(followup+accrual,breakpoint)))))
  value <- 0
  for (i in seq_len(length(bounds)-1L)) {
    value <- value+integrate(function(age)1-survival(age),bounds[i],bounds[i+1],
      rel.tol=1e-12,abs.tol=1e-14)$value
  }
  value/accrual
}
output <- list()
for (row in seq_len(nrow(cases))) {
  a <- as.numeric(unlist(cases[row,paste0('a',1:10)],use.names=FALSE))
  name <- cases$routine[row]
  omitted <- 0
  if (name == 'one') {
    mu <- a[3]*a[4]*pdeath(a[2],a[4],a[5])
    n <- seq_len(ceiling(mu+12*sqrt(mu)+30))
    ratio <- a[2]/a[1]
    if (ratio > 1) {
      critical <- qchisq(a[6],2*n,lower.tail=FALSE)
      power_n <- pchisq(critical/ratio,2*n,lower.tail=FALSE)
    } else {
      critical <- qchisq(a[6],2*n)
      power_n <- pchisq(critical/ratio,2*n)
    }
    power <- sum(dpois(n,mu)*power_n)
    omitted <- ppois(max(n),mu,lower.tail=FALSE)
  } else if (name %in% c('george_desu','information')) {
    d1 <- .5*a[3]*a[4]*pdeath(a[1],a[4],a[5])
    d2 <- .5*a[3]*a[4]*pdeath(a[2],a[4],a[5])
    if (name == 'george_desu') {
      s <- (a[1]*d1+a[2]*d2)/(a[1]+a[2])
      ratio <- max(a[1]/a[2],a[2]/a[1])
      critical <- qf(a[6],2*s,2*s,lower.tail=FALSE)
      power <- pf(critical/ratio,2*s,2*s,lower.tail=FALSE)
    } else power <- pnorm(abs(log(a[1]/a[2]))*sqrt((d1+d2)/4)-qnorm(1-a[6]))
  } else if (name == 'historical') {
    he<-a[1];rate<-a[2];at<-a[3];ft<-a[4];alpha<-a[5];hc<-a[6]
    past<-a[7];alive<-a[8];w<-a[9];continued<-a[10]==1
    future_c <- if (continued) alive*(-expm1(-hc*(at+ft)))+
      w*rate*at*pdeath(1/hc,at,ft) else 0
    future_e <- (1-w)*rate*at*pdeath(1/he,at,ft)
    null_sd <- sqrt(1/(past+future_c)+1/future_e)
    alt_sd <- sqrt(future_c/(past+future_c)^2+1/future_e)
    power <- pnorm((log(hc/he)-qnorm(1-alpha)*null_sd)/alt_sd)
  } else if (name == 'piecewise_native') {
    ratio<-a[1];rate<-a[2];at<-a[3];ft<-a[4];alpha<-a[5]
    tb<-a[6];h1<-a[7];h2<-a[8]
    if (a[9]==1) {
      low<-c(h1,h2);high<-ratio*low
      name<-'piecewise_lower'
    } else {
      high<-c(h1,h2);low<-high/ratio
      name<-'piecewise_higher'
    }
    deaths <- rate*at*.5*(piecewise_death(low[1],low[2],tb,at,ft)+
      piecewise_death(high[1],high[2],tb,at,ft))
    power <- pnorm(log(ratio)*sqrt(deaths/4)-qnorm(1-alpha))
  } else stop('Unknown routine')
  result <- cases[row,,drop=FALSE]
  result$routine<-name;result$power<-power;result$omitted<-omitted
  output[[length(output)+1L]]<-result
}
options(digits=17)
write.table(do.call(rbind,output),file=args[2],sep=',',row.names=FALSE,quote=FALSE)
