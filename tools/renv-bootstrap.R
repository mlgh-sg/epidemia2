#!/usr/bin/env Rscript
#
# One-off: create (or refresh) the renv project library and lockfile.
#
# Kept in the repo because reproducing the environment from scratch (new machine,
# new contributor, CI) should not depend on remembering the exact incantation.
# Day to day, use `make setup`, which restores from the committed renv.lock.
#
# Idempotent: safe to re-run after adding a dependency to DESCRIPTION.

options(
  repos = c(
    stan = "https://stan-dev.r-universe.dev",
    CRAN = "https://cloud.r-project.org"
  ),
  renv.consent = TRUE
)

if (!requireNamespace("renv", quietly = TRUE)) {
  install.packages("renv")
}

if (!file.exists("renv/activate.R")) {
  message("== renv::init(bare = TRUE) ==")
  renv::init(bare = TRUE, restart = FALSE, load = TRUE)
}

# An "explicit" snapshot would record only DESCRIPTION's Depends/Imports, leaving
# out everything the test suite and tutorials need (cmdstanr, testthat, knitr,
# EpiEstim, ...) -- i.e. most of what the lockfile is for. Widening
# package.dependency.fields to include Suggests is the wrong lever: renv applies
# it recursively, so it then demands the Suggests of every dependency too and the
# snapshot aborts on pre-flight validation. Scan the project's code instead.
renv::settings$snapshot.type("implicit")
renv::settings$package.dependency.fields(c("Depends", "Imports", "LinkingTo"))

# Used by vignettes/precompute.R but not by the package itself.
SCRIPT_TOOLS <- c("rprojroot", "pkgload")

message("== renv::hydrate() ==")
# hydrate copies/links from the user library instead of downloading and
# compiling, so this is fast when the packages are already installed.
renv::hydrate()
renv::hydrate(packages = SCRIPT_TOOLS)

# cmdstanr is Suggests-only and lives on the Stan r-universe, so it is the one
# dependency hydrate is most likely to miss.
if (!requireNamespace("cmdstanr", quietly = TRUE)) {
  message("== installing cmdstanr ==")
  renv::install("cmdstanr")
}

message("== renv::snapshot() ==")
renv::snapshot(prompt = FALSE)

lock <- renv::lockfile_read()
# The lockfile covers running the package, its tests and its tutorials. Pure dev
# tooling (roxygen2, devtools, pkgdown, covr) is intentionally left out: it is
# referenced only from the Makefile, so code scanning cannot see it, and CI
# installs it explicitly. Install it yourself for `make document` / `make docs`.
expected <- c("cmdstanr", "testthat", "knitr", "EpiEstim", SCRIPT_TOOLS)
missing <- setdiff(expected, names(lock$Packages))
if (length(missing)) {
  warning("not captured in renv.lock: ", paste(missing, collapse = ", "),
          call. = FALSE)
}

message("Done. renv.lock records ", length(lock$Packages), " packages.")
