# Independent base-R enumeration of all 4^4 full outcome paths.
# Cells: both, first only, second only, neither. Prior: Dirichlet(.25,...,.25).
paths <- as.matrix(expand.grid(rep(list(1:4), 4)))
settings <- list(negative=c(0,.5,.5,0), independent=rep(.25,4),
                 positive=c(.5,0,0,.5), all_both=c(1,0,0,0), all_neither=c(0,0,0,1))
rows <- list()
for (mode in c('multiple_efficacy','efficacy_toxicity')) {
  lrv <- if (mode == 'multiple_efficacy') c(.2,.2) else c(.2,.5)
  cmv <- if (mode == 'multiple_efficacy') c(.5,.5) else c(.5,.2)
  decide <- function(cells, n) {
    y <- c(cells[1]+cells[2],cells[1]+cells[3])
    pl <- pc <- numeric(2)
    for (j in 1:2) {
      lower <- mode == 'efficacy_toxicity' && j == 2
      pl[j] <- pbeta(lrv[j], y[j]+.5,n-y[j]+.5,lower.tail=lower)
      pc[j] <- pbeta(cmv[j], y[j]+.5,n-y[j]+.5,lower.tail=lower)
    }
    no <- pl < .8*(n/4)^.5 & pc < .5*(n/4)^.5
    stop <- if (mode == 'multiple_efficacy') all(no) else any(no)
    if (n < 4) return(if (stop) 'stop' else 'continue')
    go <- pl > .8 & pc > .5
    success <- if (mode == 'multiple_efficacy') any(go) else all(go)
    if (success) 'go' else if (stop) 'no_go' else 'consider'
  }
  for (name in names(settings)) {
    p <- settings[[name]]
    mass <- c(stop=0,go=0,consider=0,no_go=0)
    for (i in seq_len(nrow(paths))) {
      path <- paths[i,]
      weight <- prod(p[path])
      outcome <- decide(tabulate(path[1:2],nbins=4),2)
      if (outcome == 'continue') outcome <- decide(tabulate(path,nbins=4),4)
      mass[outcome] <- mass[outcome]+weight
    }
    stopifnot(abs(sum(mass)-1)<1e-12)
    rows[[length(rows)+1]] <- data.frame(mode=mode,scenario=name,
      both=p[1],first_only=p[2],second_only=p[3],neither=p[4],
      stop_no_go=mass['stop'],final_go=mass['go'],final_consider=mass['consider'],
      final_no_go=mass['no_go'],expected_sample_size=2*mass['stop']+4*(1-mass['stop']))
  }
}
write.csv(do.call(rbind,rows),'tests/fixtures/bop2-dc-paired-oc.csv',row.names=FALSE)
