# Original CondiS-X regression-forest references, without package installation.
# Run from the repository root with one BLAS/OpenMP thread and MAKEFLAGS=-j1.
options(digits = 17, warn = 2)
root <- normalizePath("research/raw/CondiS")
provenance <- jsonlite::fromJSON("research/condis-random-forest-sources.json",
                                simplifyVector = FALSE)
source_root <- file.path(root, provenance$directory)
for (item in provenance$files) {
  path <- file.path(source_root, item$path)
  stopifnot(identical(system2("git", c("hash-object", shQuote(path)), stdout = TRUE),
                     item$git_blob))
}
snapshot <- file.path(root, "caret-models.RData")
stopifnot(identical(system2("git", c("hash-object", shQuote(snapshot)), stdout = TRUE),
                   "61ff1c862001ba6ab9b461a4de5d546af20223e9"))
build <- file.path(root, "rf-reference")
dir.create(build, showWarnings = FALSE)
sources <- c("regTree.c", "regrf.c", "rfutils.c", "rf.h")
for (name in sources) {
  original <- file.path(source_root, "src", name)
  copy <- file.path(build, name)
  if (!file.exists(copy)) stopifnot(file.copy(original, copy))
  stopifnot(unname(tools::md5sum(original)) == unname(tools::md5sum(copy)))
}

# The original kernels remain unchanged. Redirect their uniform-RNG symbol
# at compilation to a recorder that returns the identical R random draw.
# The R caller retains the recorder's buffer until recording is ended.
recorder <- c(
  "#undef unif_rand",
  "#include <R.h>", "#include <Rmath.h>",
  "static double *recorded = NULL;",
  "static int capacity = 0, used = 0;",
  "double reference_uniform(void) {",
  "  double value = unif_rand();",
  "  if (recorded) {",
  "    if (used >= capacity) error(\"reference draw buffer exhausted\");",
  "    recorded[used++] = value;",
  "  }",
  "  return value;",
  "}",
  "void reference_begin(double *buffer, int *length) {",
  "  recorded = buffer; capacity = *length; used = 0;",
  "}",
  "void reference_end(int *count) {",
  "  *count = used; recorded = NULL; capacity = 0; used = 0;",
  "}")
writeLines(recorder, file.path(build, "reference_rng.c"))
makevars <- file.path(build, "Makevars")
writeLines(c("CFLAGS=-O0", "MAKEFLAGS=-j1"), makevars)
Sys.setenv(R_MAKEVARS_USER = makevars, MAKEFLAGS = "-j1",
           PKG_CPPFLAGS = "-Dunif_rand=reference_uniform")
library_path <- file.path(build, paste0("randomForest", .Platform$dynlib.ext))
compile <- function() {
  previous <- getwd()
  on.exit(setwd(previous))
  setwd(build)
  system2(file.path(R.home("bin"), "R"),
    c("CMD", "SHLIB", "-o", basename(library_path),
      "regTree.c", "regrf.c", "rfutils.c", "reference_rng.c"),
    stdout = "build.log", stderr = "build.log")
}
status <- compile()
if (status != 0L) stop(paste(readLines(file.path(build, "build.log")), collapse = "\n"))
dyn.load(library_path)
for (name in c("randomForest", "randomForest.default", "randomForest.formula",
               "predict.randomForest")) {
  source(file.path(source_root, "R", paste0(name, ".R")))
}
models <- new.env()
load(snapshot, envir = models)
model <- models$models$rf
redirect <- function(expression) {
  if (identical(expression, quote(randomForest::randomForest))) return(quote(randomForest))
  if (is.call(expression)) return(as.call(lapply(as.list(expression), redirect)))
  expression
}
body(model$fit) <- redirect(body(model$fit))
environment(model$fit) <- .GlobalEnv
input <- jsonlite::fromJSON("tests/fixtures/condis-models-inputs.json")
fold_reference <- jsonlite::fromJSON("tests/fixtures/condis-models-native.json")
answers <- list()
for (case_index in seq_along(c("ordinary", "wide"))) {
  name <- c("ordinary", "wide")[case_index]
  case <- input$cases[[name]]
  x <- cbind(status = case$status, as.matrix(case$covariates))
  colnames(x) <- c("status", paste0("x", seq_len(ncol(x) - 1L)))
  y <- case$imputed_time
  folds <- fold_reference$cases[[name]]$fold_ids
  requested_mtry <- sqrt(ncol(x) - 1L)
  param <- data.frame(mtry = requested_mtry)
  fit_one <- function(rows, ntree = 500L) model$fit(
    x[rows, , drop = FALSE], y[rows], wts = NULL, param = param,
    lev = NULL, last = FALSE, classProbs = FALSE, ntree = ntree, keep.inbag = TRUE)
  fold_fits <- lapply(sort(unique(folds)), function(fold) {
    test <- which(folds == fold)
    train <- setdiff(seq_along(y), test)
    seed <- 9000L + case_index * 100L + fold
    set.seed(seed)
    fit <- fit_one(train)
    pred <- as.vector(predict(fit, x[test, , drop = FALSE]))
    list(seed = seed, prediction = pred, rmse = sqrt(mean((pred - y[test])^2)))
  })
  full_seed <- 10000L + case_index
  set.seed(full_seed)
  full <- fit_one(seq_along(y))
  traced_fit <- function() {
    buffer <- .C("reference_begin", values = double(200000L), as.integer(200000L),
                 PACKAGE = "randomForest")
    on.exit(.C("reference_end", count = integer(1L), PACKAGE = "randomForest"))
    seed <- 11000L + case_index
    set.seed(seed)
    fit <- fit_one(seq_along(y), ntree = 5L)
    used <- .C("reference_end", count = integer(1L), PACKAGE = "randomForest")$count
    on.exit(NULL)
    prediction <- predict(fit, x, predict.all = TRUE, nodes = TRUE)
    fields <- c("ndbigtree", "nodestatus", "leftDaughter", "rightDaughter",
                "nodepred", "bestvar", "xbestsplit")
    list(seed = seed, uniform_draws = buffer$values[seq_len(used)],
         inbag_counts = fit$inbag, forest = fit$forest[fields],
         fitted = as.vector(prediction$aggregate), individual = prediction$individual,
         terminal_nodes = attr(prediction, "nodes"))
  }
  trace <- traced_fit()
  # Recording must leave the original random stream and fitted forest intact.
  set.seed(trace$seed)
  plain <- fit_one(seq_along(y), ntree = 5L)
  stopifnot(identical(trace$inbag_counts, plain$inbag),
            identical(trace$fitted, as.vector(predict(plain, x))))
  answers[[name]] <- list(requested_mtry = requested_mtry, effective_mtry = full$mtry,
    ntree = full$ntree, nodesize = 5L, fold_ids = folds, fold_fits = fold_fits,
    mean_rmse = mean(vapply(fold_fits, `[[`, numeric(1L), "rmse")),
    full_fit = list(seed = full_seed, fitted = as.vector(predict(full, x)),
                    out_of_bag = as.vector(full$predicted)), trace = trace)
}
jsonlite::write_json(list(cases = answers,
  provenance = list(randomForest = provenance$version, revision = provenance$revision,
                    r = as.character(getRversion()), recorded_native_uniform_draws = TRUE)),
  "tests/fixtures/condis-rf-native.json", auto_unbox = TRUE, digits = 17, pretty = TRUE)
for (name in names(answers)) cat(name, "mtry", answers[[name]]$effective_mtry,
                               "mean RMSE", answers[[name]]$mean_rmse,
                               "recorded draws", length(answers[[name]]$trace$uniform_draws), "\n")
