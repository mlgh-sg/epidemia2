# Dependencies that renv's code scanning cannot see.
#
# This file is never executed. renv discovers dependencies by scanning source
# files for library()/require()/:: calls, and a few of ours are invisible to it:
# they are used only inside `include = FALSE` chunks of the tutorial sources,
# which are stripped when precompute.R bakes `<name>.Rmd.orig` into `<name>.Rmd`.
# Listing them here is renv's documented escape hatch and keeps `renv::snapshot()`
# honest.
#
# Add to this list if `make tutorials` fails on a clean restore with a missing
# package that is nowhere in the baked vignettes.

library(EpiEstim)    # flu tutorial: Flu1918 incidence data and serial interval
library(extrafont)   # loadfonts() in every tutorial's setup chunk

# Dev tooling. Referenced only from the Makefile, which renv does not scan, but
# `make document` / `make docs` / `make check` should work from a bare restore.
library(roxygen2)
library(devtools)
library(pkgdown)
library(covr)
