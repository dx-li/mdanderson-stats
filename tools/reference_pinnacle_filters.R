# Independent base-R evaluation of RWT 2.4 daubcqf minimum-phase filters.
# Algorithm: Rice University daubcqf.m by Ramesh Gopinath. See the preserved
# notices/rice-wavelet-LICENSE.txt for copyright, conditions and disclaimer.
options(digits=17, warn=2)
multiply <- function(a,b) {
  out <- numeric(length(a)+length(b)-1)
  for(i in seq_along(a)) for(j in seq_along(b)) out[i+j-1] <- out[i+j-1]+a[i]*b[j]
  out
}
daub <- function(n) {
  k <- n/2; a <- 1; p <- 1; q <- 1; h <- c(1,1)
  if(k>1) for(j in seq_len(k-1)) {
    a <- -a*.25*(j+k-1)/j
    h <- c(0,h)+c(h,0)
    p <- c(0,-p)+c(p,0)
    p <- c(0,-p)+c(p,0)
    q <- c(0,q,0)+a*p
  }
  if(k>1) {
    roots <- polyroot(rev(q))
    selected <- roots[Mod(roots)<1]
    stopifnot(length(selected)==k-1)
    polynomial <- 1+0i
    for(root in selected) polynomial <- multiply(polynomial,c(1,-root))
    stopifnot(max(abs(Im(polynomial)))<1e-9)
    h <- multiply(h,Re(polynomial))
  }
  h <- sqrt(2)*h/sum(h)
  stopifnot(abs(sum(h*h)-1)<1e-9)
  h
}
rows <- lapply(seq(2,20,2),function(n) data.frame(length=n,
  coefficients=paste(daub(n),collapse='|')))
write.csv(do.call(rbind,rows),'tests/fixtures/pinnacle-filters.csv',row.names=FALSE)
