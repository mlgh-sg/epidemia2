#!/usr/bin/env Rscript
#
# Precompute ("bake") the model-fitting tutorial vignettes.
#
# The tutorials (flu, europe-covid, multiple-obs) fit real Stan models. To keep
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
all_vigs <- c("flu", "europe-covid", "multiple-obs")
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

for (v in vigs) {
  src <- paste0(v, ".Rmd.orig")
  out <- paste0(v, ".Rmd")
  if (!file.exists(src)) { warning("missing ", src); next }
  message("== Baking ", v, " ==")
  # namespace each vignette's figures so they do not collide
  opts_chunk$set(fig.path = file.path("figure", paste0(v, "-")))
  knit(input = src, output = out, quiet = TRUE)
  message("   wrote ", out)
}

message("Done. Commit the regenerated .Rmd files and vignettes/figure/*.png")
