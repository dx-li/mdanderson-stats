# Independently evaluate a supplied common-path calibration experiment.
# Paths are exported by the companion Python check from the reported seeds;
# this reference derives every as-of observation and posterior with base R.
options(warn=2,digits=17)
args <- commandArgs(trailingOnly=TRUE)
prefix <- if (length(args)) args[1] else 'research/raw/bop2-dc-survival-calibration-'
paths <- read.csv(paste0(prefix,'paths.csv'))
grid <- read.csv(paste0(prefix,'grid.csv'))
settings <- read.csv(paste0(prefix,'settings.csv'))
stopifnot(nrow(settings)==1L)
s <- settings[1,]
looks <- as.integer(strsplit(s$looks,';',fixed=TRUE)[[1]])
labels <- c('stop_no_go','final_go','final_consider','final_no_go')
replay <- function(data,candidate) {
  data <- data[order(data$patient),]
  stopifnot(nrow(data)==s$max_subjects)
  for (n in looks) {
    follow <- if (n==s$max_subjects) s$final_followup else 0
    risk <- data$enrollment[n]-data$enrollment[seq_len(n)]+follow
    events <- sum(data$event[seq_len(n)]<=risk)
    total <- sum(pmin(data$event[seq_len(n)],risk))
    pl <- pgamma((s$prior_scale+total)*log(2)/s$lrv,s$prior_shape+events)
    pc <- pgamma((s$prior_scale+total)*log(2)/s$cmv,s$prior_shape+events)
    ll <- candidate$lambda_lrv*(n/s$max_subjects)^candidate$gamma_lrv
    lc <- candidate$lambda_cmv*(n/s$max_subjects)^candidate$gamma_cmv
    outcome <- if (n<s$max_subjects) {
      if (pl<ll && pc<lc) 'stop_no_go' else 'continue'
    } else if (pl>ll && pc>lc) 'final_go'
    else if (pl<ll && pc<lc) 'final_no_go' else 'final_consider'
    if (outcome!='continue') return(list(outcome=outcome,enrolled=n))
  }
  stop('unresolved terminal decision')
}
evaluate <- function(phase,truth,candidate) {
  subset <- paths[paths$phase==phase & paths$truth==truth,]
  trials <- split(subset,subset$trial)
  result <- lapply(trials,replay,candidate=candidate)
  outcome <- vapply(result,function(x) x$outcome,character(1))
  enrollment <- vapply(result,function(x) x$enrolled,integer(1))
  p <- vapply(labels,function(label) mean(outcome==label),numeric(1))
  c(p,expected_sample_size=mean(enrollment),
    setNames(sqrt(p*(1-p)/length(trials)),paste0(labels,'_mcse')),
    sample_size_mcse=sd(enrollment)/sqrt(length(trials)))
}
records <- metrics <- list()
for (i in seq_len(nrow(grid))) {
  candidate <- grid[i,]
  futile <- evaluate('calibration','futile',candidate)
  effective <- evaluate('calibration','effective',candidate)
  for (truth in c('futile','effective')) {
    value <- if (truth=='futile') futile else effective
    records[[length(records)+1L]] <- data.frame(candidate=i-1L,phase='calibration',
      truth=truth,metric=names(value),value=unname(value))
  }
  fg <- unname(futile['final_go'])
  fn <- unname(sum(effective[c('stop_no_go','final_no_go')]))
  cg <- unname(effective['final_go'])
  fc <- unname(max(futile['final_consider'],effective['final_consider']))
  feasible <- fg<=s$false_go_limit && fn<=s$false_no_go_limit &&
    (is.na(s$false_consider_limit) || fc<=s$false_consider_limit)
  metrics[[i]] <- data.frame(candidate=i-1L,false_go_rate=fg,false_no_go_rate=fn,
    correct_go_rate=cg,false_consider_rate=fc,
    futile_expected_sample_size=unname(futile['expected_sample_size']),feasible=feasible)
}
table <- do.call(rbind,metrics)
selection <- list()
for (objective in c('cgr','ess_futile')) {
  feasible <- table[table$feasible,]
  stopifnot(nrow(feasible)>0L)
  order <- if (objective=='cgr') {
    order(-feasible$correct_go_rate,feasible$futile_expected_sample_size,feasible$candidate)
  } else order(feasible$futile_expected_sample_size,-feasible$correct_go_rate,feasible$candidate)
  selected <- feasible$candidate[order[1]]
  selection[[objective]] <- data.frame(objective=objective,selected_index=selected)
  for (truth in c('futile','effective')) {
    value <- evaluate('validation',truth,grid[selected+1L,])
    records[[length(records)+1L]] <- data.frame(candidate=selected,phase=paste0('validation_',objective),
      truth=truth,metric=names(value),value=unname(value))
  }
}
write_exact <- function(data,path) {
  for (name in names(data)) if (is.numeric(data[[name]])) data[[name]] <- sprintf('%.17g',data[[name]])
  write.csv(data,path,row.names=FALSE)
}
write_exact(do.call(rbind,records),paste0(prefix,'reference.csv'))
write_exact(table,paste0(prefix,'metrics.csv'))
write_exact(do.call(rbind,selection),paste0(prefix,'selection.csv'))
cat('Evaluated every candidate on the supplied paths and selected both objectives.\n')
