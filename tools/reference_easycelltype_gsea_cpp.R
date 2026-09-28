# Explicit ordered-subset references from the pinned fgsea C++ pilot kernel.
# Execute from the repository root with Rcpp installed and MAKEFLAGS=-j1.
# No fgsea/BH installation is required: unchanged scalar/cumulative definitions
# precede the random-sampling entrypoint and do not use util.h/Boost.
options(warn=2,digits=17)
path <- 'research/raw/EasyCellType/Bioc-3.18/fgsea/src/fastGSEA.cpp'
stopifnot(unname(tools::md5sum(path))=='ddff4e322ad762b51219839051308a87')
source <- readLines(path,warn=FALSE)
stop <- which(grepl('^NumericVector calcRandomGseaStatCumulative\\(',source))
stopifnot(length(stop)==1L)
kernel <- source[seq_len(stop-1L)]
stopifnot(sum(kernel=='#include "util.h"')==1L)
kernel <- kernel[kernel!='#include "util.h"']
split_path <- 'research/raw/EasyCellType/Bioc-3.18/fgsea/src/esCalculation.cpp'
stopifnot(unname(tools::md5sum(split_path))=='5269032620bab0b25130cff24d2a6a9d')
split_kernel <- readLines(split_path,warn=FALSE)
stopifnot(sum(split_kernel=='#include "esCalculation.h"')==1L)
split_kernel <- split_kernel[split_kernel!='#include "esCalculation.h"']
wrapper <- c('// [[Rcpp::export]]',
  'NumericVector fixture_cumulative(NumericVector stats, IntegerVector selected,',
  '  std::string score_type) {',
  '  return calcGseaStatCumulative(stats, selected, 1.0, score_type);',
  '}',
  '// [[Rcpp::export]]',
  'NumericVector fixture_split_scores(std::vector<double> weights,',
  '  std::vector<int> selected) {',
  '  std::sort(selected.begin(), selected.end());',
  '  return NumericVector::create(calcES(weights, selected),',
  '                               calcPositiveES(weights, selected));',
  '}')
Rcpp::sourceCpp(code=paste(c(kernel,split_kernel,wrapper),collapse='\n'),showOutput=FALSE)

cases <- list(
  ordinary=list(scores=c(6.3,5.9,4.2,2.8,1.1,.7,-.25,-.8,-1.6,-2.4,-4.3,-7.7),
                selected=c(2,8,12,3),exponent=1),
  squared=list(scores=c(4,3,2,0,0,-1,-2,-5),selected=c(4,2,7),exponent=2),
  unweighted=list(scores=c(4,3,2,0,0,-1,-2,-5),selected=c(4,2,7),exponent=0),
  mixed_zero=list(scores=c(4,3,2,0,0,-1,-2,-5),selected=c(4,2,7),exponent=1),
  tiny_later_hit=list(scores=c(4,2,1e-12,0,0,-1,-2,-5),
                      selected=c(4,2,3),exponent=1),
  all_zero_hits=list(scores=c(4,3,0,0,0,-1,-2,-5),selected=c(3,5,4),exponent=1),
  exact_tie=list(scores=c(1,1,1,1,1,1),selected=c(2,5,3),exponent=1))
inputs <- list();rows <- list();split_rows <- list()
for(name in names(cases)) {
  case <- cases[[name]]
  inputs[[length(inputs)+1L]] <- data.frame(scenario=name,
    rank=seq_along(case$scores)-1L,score=case$scores,exponent=case$exponent,
    selected_order=match(seq_along(case$scores),case$selected,nomatch=0L))
  # The R preparation has already powered absolute scores; C++ receives gseaParam=1.
  prepared <- abs(case$scores)^case$exponent
  for(mode in c('std','pos','neg')) {
    es <- fixture_cumulative(prepared,as.integer(case$selected),mode)
    rows[[length(rows)+1L]] <- data.frame(scenario=name,score_type=mode,
      prefix_size=seq_along(es),ES=as.numeric(es))
  }
  for(k in seq_along(case$selected)) {
    scores <- fixture_split_scores(prepared,as.integer(case$selected[seq_len(k)]-1L))
    split_rows[[length(split_rows)+1L]] <- data.frame(scenario=name,prefix_size=k,
      signed_ES=scores[1],positive_ES=scores[2])
  }
}
write.csv(do.call(rbind,inputs),'tests/fixtures/easycelltype-gsea-cpp-inputs.csv',row.names=FALSE)
write.csv(do.call(rbind,rows),'tests/fixtures/easycelltype-gsea-cpp-prefix.csv',row.names=FALSE)
write.csv(do.call(rbind,split_rows),'tests/fixtures/easycelltype-gsea-cpp-split-scores.csv',row.names=FALSE)
cat('Generated',sum(vapply(rows,nrow,integer(1))),
    'unchanged C++ cumulative-kernel scores and',length(split_rows),
    'splitter score pairs; no native RNG comparison.\n')
