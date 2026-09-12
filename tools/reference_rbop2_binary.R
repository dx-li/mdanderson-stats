# Independent enumeration of every four-outcome path, with exact known beta integrals.
# Rows index experimental events; columns control events. Both priors are Beta(1,1).
q1 <- matrix(c(1/2,5/6,1/6,1/2),2,2)
q2 <- matrix(c(1/2,4/5,19/20,1/5,1/2,4/5,1/20,1/5,1/2),3,3)
# Verify the closed-form fractions by a separate base-R integral.
for (n in 1:2) for (e in 0:n) for (c in 0:n) {
  q <- integrate(function(x) dbeta(x,1+e,1+n-e)*pbeta(x,1+c,1+n-c),0,1)$value
  stopifnot(abs(q-(if(n==1)q1 else q2)[e+1,c+1])<1e-12)
}
paths <- as.matrix(expand.grid(rep(list(0:1),4)))
scenarios <- rbind(c(.3,.3),c(.6,.3),c(.5,.5),c(1,0),c(0,1))
rows <- list()
for (endpoint in c('efficacy','toxicity')) for (s in 1:nrow(scenarios)) {
  pe <- scenarios[s,1]; pc <- scenarios[s,2]
  mass <- rep(0,4) # early futility, early superiority, final negative, final positive
  for (i in 1:nrow(paths)) {
    x <- paths[i,]; e <- x[1:2]; c <- x[3:4]
    w <- prod(ifelse(e==1,pe,1-pe))*prod(ifelse(c==1,pc,1-pc))
    q <- if(endpoint=='efficacy') q1[e[1]+1,c[1]+1] else q1[c[1]+1,e[1]+1]
    if(q<.25) category <- 1 else if(q>=.8) category <- 2 else {
      q <- if(endpoint=='efficacy')q2[sum(e)+1,sum(c)+1] else q2[sum(c)+1,sum(e)+1]
      category <- if(q>=.8)4 else 3
    }
    mass[category] <- mass[category]+w
  }
  stopifnot(abs(sum(mass)-1)<1e-12)
  rows[[length(rows)+1]] <- data.frame(endpoint=endpoint,experimental_rate=pe,control_rate=pc,
    early_futility=mass[1],early_superiority=mass[2],final_negative=mass[3],final_positive=mass[4],
    positive_probability=mass[2]+mass[4],expected_total=2*sum(mass[1:2])+4*sum(mass[3:4]))
}
write.csv(do.call(rbind,rows),'tests/fixtures/rbop2-binary-reference.csv',row.names=FALSE)

# Fractional priors and signed margins exercise the noninteger quadrature route.
rows <- list()
for(endpoint in c('efficacy','toxicity')) for(delta in c(-.1,0,.1)) {
  a_e <- 1.5+2; b_e <- 2.5+(5-2)
  a_c <- 2.25+1; b_c <- .75+(5-1)
  integrand <- if(endpoint=='efficacy')
    function(x)dbeta(x,a_e,b_e)*pbeta(x-delta,a_c,b_c) else
    function(x)dbeta(x,a_c,b_c)*pbeta(x-delta,a_e,b_e)
  q <- integrate(integrand,0,1,rel.tol=1e-11,abs.tol=1e-13,subdivisions=500)$value
  rows[[length(rows)+1]] <- data.frame(endpoint=endpoint,margin=delta,probability=q)
}
write.csv(do.call(rbind,rows),'tests/fixtures/rbop2-binary-margins.csv',row.names=FALSE)
