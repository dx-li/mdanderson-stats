# Exhaustive small-set null distributions from unchanged pinned fgsea R scores.
# These provide exact population targets and exhaustive-count transformations,
# not a claim of bitwise R/C++ pilot or adaptive random-stream equivalence.
options(warn=2,digits=17)
path <- 'research/raw/EasyCellType/Bioc-3.18/fgsea/R/fgsea.R'
stopifnot(unname(tools::md5sum(path))=='1c88c66bf72752bd1e182b0ca3ff85e0')
native <- new.env(parent=globalenv())
expr <- Filter(function(x) is.call(x) && identical(x[[1]],as.name('<-')) &&
  identical(x[[2]],as.name('calcGseaStat')),as.list(parse(path)))
stopifnot(length(expr)==1L)
eval(expr[[1]],envir=native)
ranks <- setNames(c(6.3,5.9,4.2,2.8,1.1,.7,-.25,-.8,-1.6,-2.4,-4.3,-7.7),
                  paste0('g',1:12))
sets <- list(top_pair=1:2,top_triplet=1:3,top_quad=1:4,
             bottom_pair=11:12,bottom_triplet=10:12,mixed_triplet=c(2,8,12))
rows <- list()
for(mode in c('std','pos','neg')) for(name in names(sets)) {
  selected <- sets[[name]]; k <- length(selected)
  es <- native$calcGseaStat(ranks,selected,scoreType=mode)
  null <- apply(combn(length(ranks),k),2,function(s)
    native$calcGseaStat(ranks,s,scoreType=mode))
  le <- sum(null<=es); ge <- sum(null>=es)
  lz <- sum(null<=0); gz <- sum(null>=0)
  negative_mean <- if(lz) sum(pmin(null,0))/lz else NA_real_
  positive_mean <- if(gz) sum(pmax(null,0))/gz else NA_real_
  denominator <- switch(mode,std=if(es>0)positive_mean else abs(negative_mean),
                        pos=positive_mean,neg=abs(negative_mean))
  nes <- if(is.finite(denominator) && denominator!=0) es/denominator else NA_real_
  mode_fraction <- switch(mode,std=if(es>=0)gz else lz,pos=gz,neg=lz)
  extreme <- switch(mode,std=if(es>0)ge else le,pos=ge,neg=le)
  exact <- if(is.na(nes)) NA_real_ else min(if(lz)le/lz else Inf,if(gz)ge/gz else Inf)
  pseudo <- if(is.na(nes)) NA_real_ else min((1+le)/(1+lz),(1+ge)/(1+gz))
  rows[[length(rows)+1L]] <- data.frame(scenario=name,score_type=mode,
    selected_ranks=paste(selected-1L,collapse=';'),set_size=k,null_size=length(null),
    ES=es,NES_exact=nes,nLeEs=le,nGeEs=ge,nLeZero=lz,nGeZero=gz,
    leZeroMean=negative_mean,geZeroMean=positive_mean,modeFraction=mode_fraction,
    nMoreExtreme=extreme,exact_conditional_p=exact,exhaustive_pseudocount_p=pseudo)
}
write.csv(data.frame(gene=names(ranks),score=unname(ranks)),
          'tests/fixtures/easycelltype-gsea-null-ranks.csv',row.names=FALSE)
write.csv(do.call(rbind,rows),'tests/fixtures/easycelltype-gsea-null.csv',row.names=FALSE)
cat('Enumerated',length(rows),'small-set signed-null reference cases.\n')

# fgseaSimpleImpl uses this R kernel for exactly one retained gene set.
# Reuse the explicit ordered subset inputs from the C++ reference to expose
# the branch's different mixed-zero behavior without changing random streams.
inputs <- read.csv('tests/fixtures/easycelltype-gsea-cpp-inputs.csv')
single_rows <- list()
for(name in unique(inputs$scenario)) {
  input <- inputs[inputs$scenario==name,]
  input <- input[order(input$rank),]
  keep <- input$selected_order>0
  selected <- (input$rank[keep]+1L)[order(input$selected_order[keep])]
  prepared <- abs(input$score)^input$exponent
  for(mode in c('std','pos','neg')) for(k in seq_along(selected)) {
    es <- native$calcGseaStat(prepared,selected[seq_len(k)],gseaParam=1,scoreType=mode)
    single_rows[[length(single_rows)+1L]] <- data.frame(scenario=name,
      score_type=mode,prefix_size=k,ES=es)
  }
}
write.csv(do.call(rbind,single_rows),
          'tests/fixtures/easycelltype-gsea-single-prefix.csv',row.names=FALSE)
cat('Generated',length(single_rows),'unchanged R single-set pilot scores.\n')
