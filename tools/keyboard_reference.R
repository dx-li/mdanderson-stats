# Independent output fixtures; requires Keyboard 0.1.3, never a Python runtime dependency.
library(Keyboard)
stopifnot(as.character(packageVersion("Keyboard")) == "0.1.3")
rows <- list()
for (phi in c(.15,.2,.25,.3,.35,.4)) {
  b <- get.boundary.kb(phi,10,3)$full_boundary_tab
  rows[[length(rows)+1]] <- data.frame(target=phi,patients=b[1,],escalate=b[2,],
                                      deescalate=b[3,],eliminate=b[4,])
}
write.csv(do.call(rbind,rows),"tests/fixtures/keyboard-boundaries.csv",row.names=FALSE)
set.seed(127)
rows <- list()
for (i in 1:64) {
  n <- c(sample(c(3,6,9,12,15,24,30),5,replace=TRUE),0)
  y <- rbinom(6,n,sort(runif(6,0,.8)))
  phi <- c(.15,.25,.3,.4)[1+i%%4]
  extra <- i%%3==0
  r <- select.mtd.kb(phi,n,y,extrasafe=extra)
  rows[[i]] <- data.frame(target=phi,patients=paste(n,collapse=";"),
    toxicities=paste(y,collapse=";"),extra_safe=extra,mtd=r$MTD,
    estimate=paste(r$p_est$phat,collapse=";"))
}
write.csv(do.call(rbind,rows),"tests/fixtures/keyboard-selection.csv",row.names=FALSE)
r <- get.oc.kb(.3,c(.05,.15,.3,.45,.6),10,3,ntrial=10000)
write.csv(data.frame(dose=1:5,selection_probability=r$selpercent/100,
                     mean_patients=r$npatients,mean_toxicities=r$ntox),
          "tests/fixtures/keyboard-oc.csv",row.names=FALSE)
