# Independent Dixon-Simon historical-control planning references, base R only.
args <- commandArgs(trailingOnly=TRUE)
if (length(args)!=1L) stop('Supply output CSV path')
event <- function(h,A,F) {
  x <- h*A
  inc <- if (x<1e-3) x*(1/2+x*(-1/6+x*(1/24+x*(-1/120+x/720)))) else 1+expm1(-x)/x
  -expm1(-h*F)+exp(-h*F)*inc
}
power <- function(A,w,he,hc,rate,F,dead,alive,continued,alpha=.05) {
  E <- (1-w)*rate*A*event(he,A,F)
  C <- if(continued) alive*(-expm1(-hc*(A+F)))+w*rate*A*event(hc,A,F) else 0
  pnorm((log(hc/he)-qnorm(1-alpha)*sqrt(1/(dead+C)+1/E))/
    sqrt(C/(dead+C)^2+1/E))
}
rows <- list()
add <- function(id,he,hc,rate,F,dead,alive,continued=TRUE,lo=0,hi=.99) {
  f <- function(A,w) power(A,w,he,hc,rate,F,dead,alive,continued)
  duration <- function(w) {
    if (f(1e5,w)<.8) return(Inf)
    uniroot(function(A)f(A,w)-.8,c(1e-4,1e5),tol=1e-10)$root
  }
  grid <- seq(lo,hi,length.out=129)
  times <- vapply(grid,duration,numeric(1))
  candidates <- c(lo,hi)
  for (i in 2:128) {
    if (is.finite(times[i]) && times[i]<=times[i-1] && times[i]<=times[i+1]) {
      opt <- optimize(duration,grid[c(i-1,i+1)],tol=1e-10)
      candidates <- c(candidates,opt$minimum)
    }
  }
  values <- vapply(candidates,duration,numeric(1))
  w <- candidates[which.min(values)]
  A <- duration(w)
  rows[[length(rows)+1L]] <<- data.frame(case_id=id,experimental_hazard=he,
    control_hazard=hc,accrual_rate=rate,followup_duration=F,historical_deaths=dead,
    historical_alive=alive,continued_followup=continued,alpha=.05,target=.8,
    allocation_lower=lo,allocation_upper=hi,accrual_lower=1e-4,accrual_upper=1e5,
    allocation=w,accrual_duration=A,achieved=f(A,w),no_new_controls_duration=duration(0))
}
add('manual_25_4',log(2)/24,25/433,5,12,25,25)
add('manual_25_5',log(2)/36,-log(.8)/7.75,3,0,10,40)
add('small_history',.05,.1,10,6,5,10)
add('large_history',.05,.1,10,6,1000,1000)
add('no_continued_followup',.05,.1,5,6,40,20,FALSE)
add('small_effect',.09,.1,10,0,10,10)
add('restricted_allocation',log(2)/36,-log(.8)/7.75,3,0,10,40,lo=.4,hi=.7)
options(digits=17)
write.table(do.call(rbind,rows),file=args[1],sep=',',row.names=FALSE,quote=FALSE)
