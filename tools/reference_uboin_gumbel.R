# Official Gumbel model equation, independently evaluated by base R.
options(digits=17)
cases <- data.frame(tox=c(.02,.3,.3,.2,0,1,.4,.4),
                    eff=c(.2,.6,.6,.4,.8,.2,.7,.7),
                    association=c(.2,5,-5,0,2,-2,1000,-1000))
rows <- lapply(seq_len(nrow(cases)),function(i) {
  x <- cases[i,]; rho <- 2*plogis(x$association)-1
  joint <- vapply(0:3,function(k) {
    e <- k %/% 2; t <- k %% 2
    x$eff^e*(1-x$eff)^(1-e)*x$tox^t*(1-x$tox)^(1-t)+
      x$eff*(1-x$eff)*x$tox*(1-x$tox)*(-1)^(e+t)*rho
  },numeric(1))
  cbind(x,as.data.frame(as.list(setNames(joint,c('p00','p01','p10','p11')))))
})
write.csv(do.call(rbind,rows),'tests/fixtures/uboin-gumbel.csv',row.names=FALSE)
