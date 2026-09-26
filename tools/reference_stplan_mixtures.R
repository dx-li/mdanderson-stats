# Independent finite-sum references for the documented statistical tests.
# This script uses base R only; no original STPLAN source is incorporated.
args <- commandArgs(trailingOnly=TRUE)
if (length(args) != 1L) stop('Supply the output CSV path')
exact_power <- function(n, p0, pa, alpha, direction) {
  if (n == 0) return(0)
  k <- seq.int(0, n)
  null <- dbinom(k, n, p0)
  alt <- dbinom(k, n, pa)
  tails <- if (direction == 'upper') rev(cumsum(rev(null))) else cumsum(null)
  sum(alt[tails <= alpha])
}
exposure <- function(frequency, disease_exposed, disease_unexposed) {
  diseased <- frequency*disease_exposed + (1-frequency)*disease_unexposed
  c(case=frequency*disease_exposed/diseased,
    control=frequency*(1-disease_exposed)/(1-diseased))
}
rows <- list()
emit <- function(routine, a, power, omitted=0) {
  rows[[length(rows)+1L]] <<- data.frame(routine=routine,
    a1=a[1],a2=a[2],a3=a[3],a4=a[4],a5=a[5],a6=a[6],power=power,omitted=omitted)
}
for (a in list(c(.3,.1,.05,60,60,.05), c(.3,.05,.1,60,60,.05),
               c(.4,.4,.2,100.5,75,.025), c(.2,.1,.1,80,120,.01))) {
  p <- exposure(a[1],a[2],a[3])
  difference <- 2*asin(sqrt(p[1]))-2*asin(sqrt(p[2]))
  power <- pnorm(difference/sqrt(1/a[4]+1/a[5])-qnorm(1-a[6]))
  emit('unmatched', a, power)
}
for (a in list(c(.3,.1,.05,60,.05,0), c(.4,.4,.2,80,.025,0),
               c(.3,.05,.1,60,.05,0), c(.3,.1,.1,20,.05,0))) {
  p <- exposure(a[1],a[2],a[3])
  case_only <- p[1]*(1-p[2])
  control_only <- (1-p[1])*p[2]
  discordance <- case_only+control_only
  conditional <- case_only/discordance
  counts <- seq.int(0,a[4])
  conditional_power <- vapply(counts, function(n)
    exact_power(n,.5,conditional,a[5],'upper'), numeric(1))
  emit('matched', a, sum(dbinom(counts,a[4],discordance)*conditional_power))
}
for (a in list(c(1,2,10,10,.05,0), c(2,1,12.5,17,.025,0),
               c(1,1.5,100,100,.05,0), c(1,1,10,20,.05,0),
               c(0,2,3,4,.05,0))) {
  mu <- a[1]*a[3]+a[2]*a[4]
  p0 <- a[3]/(a[3]+a[4])
  pa <- a[1]*a[3]/mu
  hi <- ceiling(mu+12*sqrt(mu)+30)
  counts <- seq.int(0,hi)
  direction <- if (pa > p0) 'upper' else 'lower'
  powers <- vapply(counts, function(n)exact_power(n,p0,pa,a[5],direction),numeric(1))
  emit('poisson', a, sum(dpois(counts,mu)*powers),ppois(hi,mu,lower.tail=FALSE))
}
options(digits=17)
result <- do.call(rbind,rows)
write.table(result,file=args[1],sep=',',row.names=FALSE,quote=FALSE)
