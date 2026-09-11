# Independent original-probability-coordinate quadrature; no Python algorithms.
cases <- expand.grid(margin=c(-.8,-.2,.1,.7), case=1:4)
shapes <- rbind(c(1,1,1,1),c(.3,.7,2,8),c(5,2,.4,.8),c(100,200,120,180))
rows <- lapply(seq_len(nrow(cases)), function(i) {
  s <- shapes[cases$case[i],]; delta <- cases$margin[i]
  integrate_tail <- function(a,b,c,d,delta) {
    lower <- max(0,delta); upper <- min(1,1+delta)
    p <- integrate(function(x) pbeta(x-delta,a,b)*dbeta(x,c,d),
                   lower,upper,abs.tol=2e-11,rel.tol=2e-11,subdivisions=2000)$value
    if (upper < 1) p <- p + pbeta(upper,c,d,lower.tail=FALSE)
    p
  }
  data.frame(a=s[1],b=s[2],c=s[3],d=s[4],margin=delta,
    above=integrate_tail(s[1],s[2],s[3],s[4],delta),
    below=integrate_tail(s[3],s[4],s[1],s[2],-delta))
})
write.csv(do.call(rbind,rows),'tests/fixtures/beta-difference-r.csv',row.names=FALSE)
