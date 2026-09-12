# Independent base-R AL likelihood and conditional-imputation references.
# These are mathematical results, not captured native Shiny outputs.
options(digits=17)
d <- data.frame(dose=rep(1:2,each=4),tox=c(1,-1,0,-1,0,1,-1,-1),
 eff=c(0,1,-1,-1,1,-1,0,-1),tf=c(.2,.25,1,.75,1,.3,.6,.4),
 ef=c(2,.4,.5,1,.7,1.2,2,.3))
u <- c(100,30,65,0); phiT <- .35; phiE <- .25
ub <- sum(u*c((1-phiT)*phiE,(1-phiT)*(1-phiE),phiT*phiE,phiT*(1-phiE)))
ub <- (ub+100)/2
endpoint <- function(y,t,window) {
 pending <- y==-1
 ess <- sum(!pending)+sum(t[pending]/window)
 events <- sum(y==1)
 p <- events/ess
 expected <- as.numeric(y)
 expected[pending] <- p*(1-t[pending]/window)/(1-p*t[pending]/window)
 list(ess=ess,y=events,p=p,expected=expected)
}
rows <- list(); patients <- list()
for(j in 1:2) {
 z <- d[d$dose==j,]
 t <- endpoint(z$tox,z$tf,1); e <- endpoint(z$eff,z$ef,2)
 cell <- cbind((1-t$expected)*e$expected,(1-t$expected)*(1-e$expected),
               t$expected*e$expected,t$expected*(1-e$expected))
 joint <- colSums(cell); x <- sum(joint*u)/100; n <- nrow(z)
 rows[[j]] <- data.frame(dose=j,n=n,tox=t$y,eff=e$y,ess_t=t$ess,ess_e=e$ess,
  pi_t=t$p,pi_e=e$p,j01=joint[1],j00=joint[2],j11=joint[3],j10=joint[4],
  utility_events=x,utility_mean=100*(1+x)/(2+n),
  utility_probability=pbeta(ub/100,1+x,1+n-x,lower.tail=FALSE),
  overdose=pbeta(phiT,1+t$y,1+t$ess-t$y,lower.tail=FALSE),
  futility=pbeta(phiE,1+e$y,1+e$ess-e$y))
 patients[[j]] <- cbind(z,expected_t=t$expected,expected_e=e$expected)
}
write.csv(do.call(rbind,rows),'tests/fixtures/tite-boin12-al.csv',row.names=FALSE)
write.csv(do.call(rbind,patients),'tests/fixtures/tite-boin12-patients.csv',row.names=FALSE)
