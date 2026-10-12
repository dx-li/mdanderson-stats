# Original active final selector; host six-cell isotonic oracle, reference only.
# Usage: R --vanilla --slave -f this_file --args ORIGINAL_R OUTPUT_CSV REPOSITORY
options(warn=2,digits=17)
args<-commandArgs(TRUE);stopifnot(length(args)==3)
stopifnot(unname(tools::md5sum(args[1]))=='40d238f6d19d906a76f062f55a6a4516')
native<-new.env(parent=globalenv());sys.source(args[1],envir=native)
for(expr in parse(file.path(args[3],'tools/reference_boin_waterfall_workflow.R'))) {
 if(is.call(expr) && identical(expr[[1]],as.name('<-')) &&
    identical(expr[[2]],as.name('enumerated_biviso'))) eval(expr,envir=native)
}
native$reference_fit<-function(values,weights=matrix(1,nrow(values),ncol(values)),warn=TRUE)
  native$enumerated_biviso(values,weights)
rewrite<-function(expr) {
 if(missing(expr)) return(quote(expr=))
 if(!is.call(expr)) return(expr)
 if(identical(expr[[1]],quote(Iso::biviso))) expr[[1]]<-as.name('reference_fit')
 as.call(lapply(as.list(expr),rewrite))
}
body(native$select.mtd.comb)<-rewrite(body(native$select.mtd.comb))
# Observe original local report arrays without changing any numeric expression.
original<-body(native$select.mtd.comb)
body(native$select.mtd.comb)<-as.call(c(list(as.name('{'),
 quote(on.exit({native$lastreport<-list(mean=estToxProb,lower=lowerCI,upper=upperCI)}))),
 as.list(original)[-1]))
encode<-function(x) paste(as.vector(t(x)),collapse=';')
set.seed(231)
cases<-list(list(n=matrix(c(3,0,0,0,0,60),2,3,byrow=TRUE),
                 y=matrix(c(0,0,0,0,0,23),2,3,byrow=TRUE)))
for(i in 1:35) {
 n<-matrix(sample(c(0L,3L,6L,12L,30L),6,replace=TRUE),2,3)
 if(sum(n)==0) n[1,1]<-3
 y<-matrix(rbinom(6,as.vector(n),rep(c(.05,.2,.4),2)),2,3)
 cases[[length(cases)+1L]]<-list(n=n,y=y)
}
records<-list()
for(case in cases) for(contour in c(FALSE,TRUE)) for(safe in c(FALSE,TRUE)) {
 native$lastreport<-NULL;result<-NULL;error<-''
 invisible(capture.output(tryCatch({result<-native$select.mtd.comb(.25,case$n,case$y,
     extrasafe=safe,MTD.contour=contour,print=FALSE)},error=function(e) error<<-conditionMessage(e))))
 records[[length(records)+1L]]<-data.frame(patients=encode(case$n),toxicities=encode(case$y),
     contour=contour,extra_safe=safe,selected=encode(result),
     mean=encode(native$lastreport$mean),lower=encode(native$lastreport$lower),
     upper=encode(native$lastreport$upper),error=error)
}
write.csv(do.call(rbind,records),args[2],row.names=FALSE,na='')
