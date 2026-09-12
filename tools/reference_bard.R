# Independent base-R evaluation of BARD stage-two mathematical rules.
# Explicit prior and isotonic weights are Python configuration choices;
# these are not undisclosed native defaults or native application outputs.
options(digits=17)
cases <- list(
  native_example=list(n=rbind(c(2,2,0,4),c(0,2,4,1)),
                      a=rbind(c(.25,.25,.25,.25),c(.25,.25,.25,.25)),w=c(8,7)),
  reversed_safety=list(n=rbind(c(7,1,2,0),c(0,4,1,5)),
                       a=rbind(c(.1,.2,.3,.4),c(.4,.3,.2,.1)),w=c(1,3)),
  asymmetric_prior=list(n=rbind(c(0,0,0,0),c(1,0,0,1)),
                       a=rbind(c(.1,2,.3,4),c(3,.2,1,.4)),w=c(2,1)))
rows <- list()
for (name in names(cases)) {
  x <- cases[[name]]; post <- x$n+x$a
  utility <- as.vector((post/rowSums(post)) %*% c(0,30,50,100))
  tox <- pbeta(.3,post[,1]+post[,3],post[,2]+post[,4],lower.tail=FALSE)
  eff <- pbeta(.2,post[,3]+post[,4],post[,1]+post[,2])
  adjusted <- tox
  if(tox[1]>tox[2]) adjusted[] <- weighted.mean(tox,x$w)
  for(j in 1:2) rows[[length(rows)+1]] <- data.frame(case=name,arm=j,
    n1=x$n[j,1],n2=x$n[j,2],n3=x$n[j,3],n4=x$n[j,4],
    a1=x$a[j,1],a2=x$a[j,2],a3=x$a[j,3],a4=x$a[j,4],weight=x$w[j],
    utility=utility[j],overdose=tox[j],adjusted=adjusted[j],loweff=eff[j])
}
write.csv(do.call(rbind,rows),'tests/fixtures/bard-posterior.csv',row.names=FALSE)
# Official example history; enumerate every new categorical profile.
h <- read.csv('research/raw/BARD/patient_enroll.csv')
profiles <- expand.grid(Factor1=1:3,Factor2=1:2,Factor3=1:2)
rows <- lapply(seq_len(nrow(profiles)),function(i) {
  v <- as.numeric(profiles[i,])
  scores <- vapply(1:2,function(candidate) {
    arms <- c(h$DoseArm,candidate)
    factors <- rbind(h[,3:5],v)
    sum(vapply(1:3,function(k) {
      matching <- factors[,k]==v[k]
      abs(sum(arms[matching]==1)-sum(arms[matching]==2))
    },numeric(1)))
  },numeric(1))
  p1 <- if(scores[1]==scores[2]) .5 else if(scores[1]<scores[2]) .95 else .05
  data.frame(profiles[i,],score1=scores[1],score2=scores[2],probability1=p1)
})
write.csv(do.call(rbind,rows),'tests/fixtures/bard-minimization.csv',row.names=FALSE)
