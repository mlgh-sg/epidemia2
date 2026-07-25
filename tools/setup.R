#!/usr/bin/env Rscript
#
# Restore the R environment and make sure CmdStan is present at the pinned version.
#
#     Rscript tools/setup.R      (or: make setup)
#
# Two things are pinned separately because renv only manages R packages. CmdStan
# is a C++ toolchain installed outside any R library, so it is version-checked
# here instead. The version below must match the `cmdstan-${{ runner.os }}-<ver>`
# cache keys in .github/workflows/.

CMDSTAN_VERSION <- "2.36.0"

if (!requireNamespace("renv", quietly = TRUE)) {
  install.packages("renv", repos = "https://cloud.r-project.org")
}

message("== renv::restore() ==")
renv::restore(prompt = FALSE)

if (!requireNamespace("cmdstanr", quietly = TRUE)) {
  stop("cmdstanr is not installed even after renv::restore(). ",
       "Check that renv.lock records it and that the Stan r-universe is in ",
       "getOption('repos').", call. = FALSE)
}

installed <- tryCatch(cmdstanr::cmdstan_version(), error = function(e) NULL)

if (is.null(installed)) {
  message("== no CmdStan found; installing ", CMDSTAN_VERSION, " ==")
  cmdstanr::check_cmdstan_toolchain(fix = TRUE)
  cmdstanr::install_cmdstan(version = CMDSTAN_VERSION,
                            cores = parallel::detectCores())
} else if (!identical(installed, CMDSTAN_VERSION)) {
  message("== CmdStan ", installed, " found, but this project pins ",
          CMDSTAN_VERSION, "; installing it alongside ==")
  cmdstanr::check_cmdstan_toolchain(fix = TRUE)
  cmdstanr::install_cmdstan(version = CMDSTAN_VERSION,
                            cores = parallel::detectCores())
} else {
  message("== CmdStan ", installed, " already installed ==")
}

message("CmdStan path: ", cmdstanr::cmdstan_path())
message("Setup complete. Next: make test")
