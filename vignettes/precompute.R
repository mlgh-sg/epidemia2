#!/usr/bin/env Rscript
#
# Precompute ("bake") the model-fitting tutorial vignettes.
#
# The tutorials fit real Stan models. To keep
# the documentation site fast to build and free of a live CmdStan dependency,
# each tutorial is authored as `<name>.Rmd.orig` (the source, with live fits)
# and this script knits it to `<name>.Rmd` with all model output and figures
# baked in. The committed `<name>.Rmd` and its figures are what pkgdown renders.
#
# Run this WHENEVER the modelling code (R/, inst/stan/) or a `.Rmd.orig` changes,
# then commit the regenerated `.Rmd` files and figures:
#
#     Rscript vignettes/precompute.R            # bake all tutorials
#     Rscript vignettes/precompute.R flu        # bake a subset
#
# Requires a working CmdStan installation (cmdstanr::install_cmdstan()).

args <- commandArgs(trailingOnly = TRUE)
force <- "--force" %in% args
args <- setdiff(args, "--force")
all_vigs <- c("flu", "multiple-obs", "multilevel-multi-obs",
              "flaxman", "b117")
vigs <- if (length(args)) intersect(args, all_vigs) else all_vigs
if (!length(vigs)) stop("No known tutorial requested. Choose from: ",
                        paste(all_vigs, collapse = ", "))

# locate the package root (this script lives in <root>/vignettes)
root <- tryCatch(
  rprojroot::find_root(rprojroot::is_r_package),
  error = function(e) normalizePath(".")
)
if (!file.exists(file.path(root, "DESCRIPTION"))) {
  stop("Run this from the package root: Rscript vignettes/precompute.R")
}

# load the in-development package so the fits use the current source
suppressWarnings(suppressMessages(pkgload::load_all(root, quiet = TRUE)))

vig_dir <- file.path(root, "vignettes")
owd <- setwd(vig_dir)
on.exit(setwd(owd), add = TRUE)

library(knitr)
# keep bookdown figure labels/anchors so \@ref(fig:...) cross-references survive
opts_knit$set(bookdown.internal.label = TRUE)

# A tutorial is stale when its baked .Rmd is older than its source, or older
# than anything in R/ or inst/stan/ that could change the numbers. Everything
# else is skipped, so re-running this after an unrelated edit costs nothing.
# `--force` bakes regardless.
model_mtime <- suppressWarnings(max(file.mtime(c(
  list.files(file.path(root, "R"), full.names = TRUE, pattern = "\\.[Rr]$"),
  list.files(file.path(root, "inst", "stan"), full.names = TRUE,
             recursive = TRUE),
  # the shipped data objects carry the delay kernels, so a change there moves
  # every fitted number even though no code changed
  list.files(file.path(root, "data"), full.names = TRUE)
)), -Inf))

is_stale <- function(src, out) {
  if (force || !file.exists(out)) return(TRUE)
  file.mtime(out) < max(file.mtime(src), model_mtime)
}

# knitr invalidates a cached chunk when ITS OWN code changes, not when the data
# underneath it does. A change to data/ or R/ is therefore replayed from cache
# and the vignette reports stale numbers under new prose -- which is exactly
# what happened when the inf2death kernel was corrected. --force drops the whole
# cache, since chunk labels need not begin with the vignette's name (europe-covid
# labels its chunks `multilevel-*`) and any per-vignette pattern would miss them.
if (force && dir.exists("cache")) {
  n <- length(list.files("cache"))
  unlink("cache", recursive = TRUE)
  message("== Dropped the knitr cache (", n, " files) ==")
}

skipped <- character()
for (v in vigs) {
  src <- paste0(v, ".Rmd.orig")
  out <- paste0(v, ".Rmd")
  if (!file.exists(src)) { warning("missing ", src); next }
  if (!is_stale(src, out)) {
    message("== Skipping ", v, " (up to date; --force to rebake) ==")
    skipped <- c(skipped, v)
    next
  }
  message("== Baking ", v, " ==")
  # namespace each vignette's figures so they do not collide
  opts_chunk$set(fig.path = file.path("figure", paste0(v, "-")))
  knit(input = src, output = out, quiet = TRUE)
  message("   wrote ", out)
}

baked <- setdiff(vigs, skipped)
if (length(baked)) {
  # Record the input fingerprint so `make docs-check` can tell, in about a
  # second and without fitting anything, whether the published tutorials still
  # match the code. Only meaningful once EVERY tutorial has been baked, so skip
  # it for a partial run rather than record a stamp that overstates freshness.
  if (setequal(vigs, all_vigs)) {
    stamp <- file.path(root, "tools", "docs-stamp.sh")
    if (file.exists(stamp)) system2(stamp, c("write", "r"))
  } else {
    message("NOTE: partial bake (", paste(baked, collapse = ", "),
            ") -- docs stamp NOT updated. Run without arguments to refresh it.")
  }
  message("Done. Commit the regenerated .Rmd files and vignettes/figure/*.png")
} else {
  message("Nothing to do -- all requested tutorials are up to date.")
}
