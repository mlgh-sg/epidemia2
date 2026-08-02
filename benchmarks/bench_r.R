#!/usr/bin/env Rscript
#
# Time the R (CmdStanR) multilevel fit and report diagnostics as JSON on stdout.
#
#     Rscript benchmarks/bench_r.R --draws 500 --tune 500 --chains 4 --seed 12345
#
# This is the multilevel NPI model the europe-covid vignette used to carry
# (removed in 1.1.0; the model is kept here as a benchmark), fitted with MCMC rather
# than the variational approximation the vignette uses, so that it is comparable
# with the Python port's nutpie fit. See benchmarks/run.py.

suppressWarnings(suppressMessages({
  library(methods)
  rt_root <- tryCatch(rprojroot::find_root(rprojroot::is_r_package),
                      error = function(e) normalizePath("."))
  pkgload::load_all(rt_root, quiet = TRUE)
}))

args <- commandArgs(trailingOnly = TRUE)
getopt <- function(flag, default) {
  i <- match(flag, args)
  if (is.na(i) || i == length(args)) return(default)
  as.numeric(args[[i + 1L]])
}
draws  <- getopt("--draws", 500)
tune   <- getopt("--tune", 500)
chains <- getopt("--chains", 4)
seed   <- getopt("--seed", 12345)

# CmdStanR writes progress and sampler warnings to stdout, so the JSON goes to a
# file rather than being interleaved with it.
json_i <- match("--json", args)
json_path <- if (is.na(json_i) || json_i == length(args)) "" else args[[json_i + 1L]]

data("EuropeCovid2", package = "epidemia")
data("EuropeCovid", package = "epidemia")

d <- EuropeCovid2$data
d <- d[d$date > d$date[which(cumsum(d$deaths) > 10)[1] - 30], ]
d <- as.data.frame(d[d$date < as.Date("2020-05-05"), ])

npis <- c("public_events", "schools_universities", "self_isolating_if_ill",
          "social_distancing_encouraged", "lockdown")

form <- stats::as.formula(paste0(
  "R(country, date) ~ 0 + (1 + ", paste(npis, collapse = " + "), " || country) + ",
  paste(npis, collapse = " + ")
))

rt <- epirt(
  formula = form,
  prior = shifted_gamma(shape = 1 / 6, scale = 1, shift = log(1.05) / 6),
  prior_covariance = decov(shape = c(2, rep(0.5, length(npis))), scale = 0.25),
  link = scaled_logit(6.5)
)
inf <- epiinf(gen = EuropeCovid$si, seed_days = 6)
deaths <- suppressWarnings(epiobs(
  formula = deaths ~ 1, i2o = EuropeCovid2$inf2death,
  prior_intercept = normal(0, 0.2), link = scaled_logit(0.02)
))

# Compilation is cached in tools::R_user_dir("epidemia", "cache") and is a
# one-off per machine, so time it separately rather than charging it to the fit.
t_compile <- system.time(invisible(epidemia_stan_model("epidemia_base")))[["elapsed"]]

t0 <- proc.time()[["elapsed"]]
fm <- suppressWarnings(epim(
  rt = rt, inf = inf, obs = deaths, data = d,
  algorithm = "sampling",
  iter = draws + tune, warmup = tune, chains = chains,
  seed = seed, refresh = 0,
  control = list(max_treedepth = 12, adapt_delta = 0.95)
))
t_sample <- proc.time()[["elapsed"]] - t0

# Diagnostics on the fixed NPI effects -- the parameters the analysis is about,
# and the ones directly comparable with the Python port's `beta`.
mat <- as.matrix(fm, par_models = "R", par_types = "fixed")
drw <- posterior::as_draws_array(fm$stanfit$draws)
summ <- posterior::summarise_draws(drw)
beta_rows <- summ[summ$variable %in% colnames(mat), ]

np <- tryCatch(fm$cmdstanfit$diagnostic_summary(diagnostics = c("divergences", "treedepth"), quiet = TRUE),
               error = function(e) list(num_divergent = NA, num_max_treedepth = NA))

esc <- function(x) gsub('"', '\\\\"', x)
num <- function(x) if (is.null(x) || !length(x) || all(is.na(x))) "null" else
  format(as.numeric(x), digits = 10, scientific = FALSE)

if (nzchar(json_path)) {
  con <- file(json_path, open = "wt")
  sink(con)
  on.exit({ sink(); close(con) }, add = TRUE)
}

cat("{\n")
cat('  "engine": "R / cmdstanr",\n')
cat('  "backend": "stan-nuts",\n')
cat('  "adaptation": "diag",\n')
cat('  "draws": ', draws, ', "tune": ', tune, ', "chains": ', chains, ',\n', sep = "")
cat('  "compile_seconds": ', num(t_compile), ',\n', sep = "")
cat('  "sample_seconds": ', num(t_sample), ',\n', sep = "")
cat('  "divergences": ', num(sum(np$num_divergent)), ',\n', sep = "")
cat('  "max_treedepth_hits": ', num(sum(np$num_max_treedepth)), ',\n', sep = "")
cat('  "max_rhat": ', num(max(beta_rows$rhat, na.rm = TRUE)), ',\n', sep = "")
cat('  "min_ess_bulk": ', num(min(beta_rows$ess_bulk, na.rm = TRUE)), ',\n', sep = "")
cat('  "median_ess_bulk": ', num(stats::median(beta_rows$ess_bulk, na.rm = TRUE)), ',\n', sep = "")
cat('  "effects": {\n')
for (i in seq_len(nrow(beta_rows))) {
  nm <- sub("^R\\|", "", beta_rows$variable[i])
  cat('    "', esc(nm), '": {"mean": ', num(beta_rows$mean[i]),
      ', "sd": ', num(beta_rows$sd[i]),
      ', "ess_bulk": ', num(beta_rows$ess_bulk[i]), '}',
      if (i < nrow(beta_rows)) "," else "", "\n", sep = "")
}
cat("  }\n}\n")
