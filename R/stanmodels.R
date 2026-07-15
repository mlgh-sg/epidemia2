# Compilation and caching of the epidemia Stan programs (CmdStanR backend)
#
# epidemia previously precompiled its Stan models at install time via rstan /
# rstantools. It now uses CmdStanR: the two Stan programs (epidemia_base and
# epidemia_pp_base) are compiled lazily on first use and cached, so no C++
# toolchain work happens at install time.
#
# The compiled CmdStanModel objects are cached both in memory (per session) and
# on disk (via CmdStanR's own hash-based caching), so compilation happens at
# most once per machine per model version.

# Names of the Stan programs shipped in inst/stan.
.stan_model_names <- c("epidemia_base", "epidemia_pp_base")

# Per-session cache of compiled CmdStanModel objects.
.model_cache <- new.env(parent = emptyenv())

# Directory containing the installed Stan sources (and include files).
stan_dir <- function() {
  d <- system.file("stan", package = "epidemia", mustWork = FALSE)
  if (!nzchar(d) || !dir.exists(d)) {
    # during development with devtools::load_all()
    d <- file.path("inst", "stan")
  }
  d
}

# Writable directory in which to place compiled executables.
#
# We do not write executables into the (possibly read-only) package library.
# Instead we use a per-user cache directory.
stan_exe_dir <- function() {
  d <- tools::R_user_dir("epidemia", which = "cache")
  d <- file.path(d, "stan")
  if (!dir.exists(d)) {
    dir.create(d, recursive = TRUE, showWarnings = FALSE)
  }
  d
}

# Check that CmdStanR (and a CmdStan installation) are available, with an
# actionable error message otherwise.
check_cmdstan <- function() {
  if (!requireNamespace("cmdstanr", quietly = TRUE)) {
    stop(
      "The 'cmdstanr' package is required to fit models with epidemia.\n",
      "Install it with:\n",
      "  install.packages('cmdstanr', repos = c('https://stan-dev.r-universe.dev', getOption('repos')))\n",
      "and then install CmdStan with cmdstanr::install_cmdstan().",
      call. = FALSE
    )
  }
  path <- tryCatch(cmdstanr::cmdstan_path(), error = function(e) NULL)
  if (is.null(path)) {
    stop(
      "No CmdStan installation was found. Install it with:\n",
      "  cmdstanr::install_cmdstan()",
      call. = FALSE
    )
  }
  invisible(TRUE)
}

#' Compile (or retrieve) an epidemia Stan model
#'
#' Returns a compiled \code{CmdStanModel} object for one of the two Stan
#' programs used internally by \pkg{epidemia}. The model is compiled on first
#' use and cached for the remainder of the session; CmdStanR additionally caches
#' the compiled executable on disk. Users do not normally need to call this
#' directly.
#'
#' @param name Name of the Stan program, either \code{"epidemia_base"} (used
#'   for model fitting) or \code{"epidemia_pp_base"} (used to generate posterior
#'   draws of latent series).
#' @param quiet If \code{TRUE} (the default), suppress compiler output.
#' @return A \code{CmdStanModel} object.
#' @keywords internal
#' @export
epidemia_stan_model <- function(name = c("epidemia_base", "epidemia_pp_base"),
                                quiet = TRUE) {
  name <- match.arg(name)
  check_cmdstan()

  if (!is.null(.model_cache[[name]])) {
    return(.model_cache[[name]])
  }

  incl <- stan_dir()
  stan_file <- file.path(incl, paste0(name, ".stan"))
  if (!file.exists(stan_file)) {
    stop("Could not find Stan program '", stan_file, "'.", call. = FALSE)
  }

  mod <- cmdstanr::cmdstan_model(
    stan_file = stan_file,
    include_paths = incl,
    dir = stan_exe_dir(),
    quiet = quiet,
    compile = TRUE
  )

  .model_cache[[name]] <- mod
  mod
}

#' Precompile the epidemia Stan models
#'
#' Compiles both Stan programs used by \pkg{epidemia}. This is optional: models
#' are compiled automatically the first time they are needed. Calling this once
#' after installation (or after a CmdStan upgrade) moves the one-off compilation
#' cost to a convenient time.
#'
#' @param quiet If \code{TRUE} (the default), suppress compiler output.
#' @return Invisibly, a named list of the compiled \code{CmdStanModel} objects.
#' @export
compile_epidemia <- function(quiet = TRUE) {
  models <- lapply(.stan_model_names, epidemia_stan_model, quiet = quiet)
  names(models) <- .stan_model_names
  invisible(models)
}
