# Execute the pinned fgsea statistic unchanged, without installing its packages.
# The selected Bioconductor profile and source hashes are documented in
# research/easycelltype-gsea-audit.md. Native sources remain ignored local files.
options(warn=2, digits=17)
path <- 'research/raw/EasyCellType/Bioc-3.18/fgsea/R/fgsea.R'
stopifnot(unname(tools::md5sum(path)) == '1c88c66bf72752bd1e182b0ca3ff85e0')
native <- new.env(parent=globalenv())
expr <- Filter(function(x) is.call(x) && identical(x[[1]], as.name('<-')) &&
  identical(x[[2]], as.name('calcGseaStat')), as.list(parse(path)))
stopifnot(length(expr)==1L)
eval(expr[[1]], envir=native)
core_path <- 'research/raw/EasyCellType/Bioc-3.18/DOSE/R/gsea.R'
stopifnot(unname(tools::md5sum(core_path)) == '1b84819ea24cf9903d493750cd7de768')
for (name in c('gseaScores','leading_edge')) {
  expr <- Filter(function(x) is.call(x) && identical(x[[1]], as.name('<-')) &&
    identical(x[[2]], as.name(name)), as.list(parse(core_path)))
  stopifnot(length(expr)==1L)
  eval(expr[[1]], envir=native)
}

queries <- references <- cases <- results <- list()
run_case <- function(name, genes, scores, sets, score_type='std', exponent=1,
                     min_size=1L, max_size=500L) {
  queries[[length(queries)+1L]] <<- data.frame(case=name, gene=genes, score=scores)
  ref <- do.call(rbind, lapply(names(sets), function(label)
    data.frame(case=name, cell_type=label, gene=sets[[label]])))
  references[[length(references)+1L]] <<- ref
  cases[[length(cases)+1L]] <<- data.frame(case=name, score_type=score_type,
    exponent=exponent, min_size=min_size, max_size=max_size)
  ranked <- setNames(scores, genes)
  ranked <- sort(ranked, decreasing=TRUE)
  # Base match has the same first-occurrence lookup contract as fmatch here.
  # Duplicate associations are removed after matching, as in preparePathwaysAndStats.
  for (label in names(sets)) {
    hits <- unique(na.omit(match(sets[[label]], names(ranked))))
    if (length(hits)<max(1L,min_size) || length(hits)>min(max_size,length(ranked)-1L)) next
    value <- native$calcGseaStat(ranked, hits, gseaParam=exponent,
      returnLeadingEdge=TRUE, scoreType=score_type)
    core <- tryCatch({
      observed <- native$gseaScores(ranked, sets[[label]], exponent=exponent)
      edge <- native$leading_edge(list(observed))
      list(es=observed$ES, genes=paste(edge$core_enrichment[[1]],collapse=';'),
           rank=edge$rank[[1]], status='defined')
    }, error=function(e) {
      if (sum(abs(ranked[names(ranked) %in% sets[[label]]])^exponent)!=0) stop(e)
      list(es=NA_real_,genes=NA_character_,rank=NA_integer_,
           status='undefined_zero_hit_weight')
    })
    results[[length(results)+1L]] <<- data.frame(case=name, cell_type=label,
      set_size=length(hits), es=value$res,
      fgsea_leading_edge=paste(names(ranked)[value$leadingEdge],collapse=';'),
      fgsea_leading_ranks=paste(value$leadingEdge-1L,collapse=';'),
      ranked_genes=paste(names(ranked),collapse=';'), dose_core_es=core$es,
      dose_core_enrichment=core$genes, dose_core_rank=core$rank,
      dose_core_status=core$status)
  }
}

genes <- letters[1:8]
scores <- c(5,3,1,0,0,-1,-3,-5)
sets <- list(Top=c('a','b'), Bottom=c('g','h'), Cross=c('a','h'),
             Zero=c('d','e'), All=genes, Absent='outside')
run_case('signed', genes, scores, sets)
run_case('positive', genes, scores, sets, score_type='pos')
run_case('negative', genes, scores, sets, score_type='neg')
run_case('unweighted', genes, scores, sets, exponent=0)
run_case('squared', genes, scores, sets, exponent=2)
run_case('size_filter', genes, scores, c(sets,list(Single='a',Three=c('a','b','c'))),
         min_size=2L,max_size=2L)
run_case('duplicates_ties', c('a','b','a','c','d','e'), c(3,3,2,0,-1,-1),
         list(Repeated=c('a','a','d','outside'), Tied=c('a','b'), Tail=c('d','e')))
run_case('all_zero', letters[1:6], rep(0,6),
         list(Top=c('a','b'), Bottom=c('e','f'), Balanced=c('b','e')))

write.csv(do.call(rbind,queries),'tests/fixtures/easycelltype-gsea-queries.csv',row.names=FALSE)
write.csv(do.call(rbind,references),'tests/fixtures/easycelltype-gsea-reference.csv',row.names=FALSE)
write.csv(do.call(rbind,cases),'tests/fixtures/easycelltype-gsea-cases.csv',row.names=FALSE)
write.csv(do.call(rbind,results),'tests/fixtures/easycelltype-gsea-results.csv',row.names=FALSE)
cat('Verified',length(cases),'cases and',length(results),'source ES results.\n')
