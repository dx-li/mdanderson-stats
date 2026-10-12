# Original BOIN desktop1.1.0 simulation wrappers; original source stays external.
# Only RNG services and the independent six-cell isotonic oracle are supplied.
# Usage: R --vanilla --slave -f this_file --args ORIGINAL_R OUTPUT_CSV REPOSITORY
options(warn=2,digits=17)
args<-commandArgs(TRUE)
stopifnot(length(args)==3,
          unname(tools::md5sum(args[1]))=='64708c960eaccbb1c0eff252a1ed1e88')
native<-new.env(parent=globalenv());sys.source(args[1],envir=native)
# Reuse this repository's independently authored exhaustive active-set oracle,
# without executing that generator's original-R tests or loading another package.
expressions<-parse(file.path(args[3],'tools/reference_boin_waterfall_workflow.R'))
for (expr in expressions) {
  if (is.call(expr) && identical(expr[[1]],as.name('<-')) &&
      identical(expr[[2]],as.name('enumerated_biviso'))) eval(expr,envir=native)
}
stopifnot(is.function(native$enumerated_biviso))
native$reference_fit<-function(values,weights=matrix(1,nrow(values),ncol(values)),warn=TRUE) {
  native$enumerated_biviso(values,weights)
}
substitute_fit<-function(expr) {
  if (missing(expr)) return(quote(expr=))
  if (!is.call(expr)) return(expr)
  if (identical(expr[[1]],quote(Iso::biviso))) {
    expr[[1]]<-as.name('reference_fit');return(expr)
  }
  as.call(lapply(as.list(expr),substitute_fit))
}
for (name in ls(native)) {
  if (is.function(native[[name]])) body(native[[name]])<-substitute_fit(body(native[[name]]))
}
native$runif<-function(n,...) rep(.5,n)
native$rbinom<-function(n,size,prob) {
  stopifnot(n==1L,size==1L,all(prob==.5));0L
}
# Never invoke the shipped GUI notification command from a reference run.
native$file.exists<-function(...) FALSE
encode<-function(x) paste(as.vector(t(x)),collapse=';')
rows<-list()
for (mode in c('single','contour')) for (cap in c(2L,3L)) {
 for (titration in c(FALSE,TRUE)) for (runin in c(FALSE,TRUE)) {
  for (safe in c(FALSE,TRUE)) for (upper in c(.35,.5)) for (truth in c('none','all','gradient')) {
   p<-switch(truth,none=matrix(0,2,3),all=matrix(1,2,3),
             gradient=matrix(c(.1,.2,.3,.4,.5,.6),2,3,byrow=TRUE))
   call<-list(target=.25,p.true=p,ncohort=if(mode=='single') 6L else c(4L,2L),
              cohortsize=3L,n.earlystop=12L,ntrial=2L,seed=91L,
              titration=titration,RunIn=runin,titration.cap=cap,extrasafe=safe,p.tox=upper)
   error<-'';result<-NULL
   messages<-capture.output(tryCatch({result<-do.call(
     native[[if(mode=='single') 'get.oc.comb_singleMTD' else 'get.oc.comb_MTDcontour']],call)
   },error=function(e) error<<-conditionMessage(e)))
   rows[[length(rows)+1L]]<-data.frame(mode=mode,cap=cap,titration=titration,runin=runin,
     truth=truth,extra_safe=safe,upper=upper,error=error,selection=if(is.null(result$selpercent)) '' else encode(result$selpercent),
     mean_patients=if(mode=='single') encode(result$nptsdose) else encode(result$npatients),
     mean_toxicities=if(mode=='single') encode(result$ntoxdose) else encode(result$ntox),
     total_patients=if(mode=='single') as.character(result$meanpts) else as.character(result$totaln),
     source_message_lines=length(messages))
  }
 }
}
write.csv(do.call(rbind,rows),args[2],row.names=FALSE)
