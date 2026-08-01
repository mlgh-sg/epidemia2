## Regenerate the infection-to-death delay kernel shipped in EuropeCovid and
## EuropeCovid2, and export it for the Python port.
##
## WHY THIS EXISTS
##
## Both shipped kernels are consumed at the same offset: `si[1]` weights
## infections one day back and `inf2death[1]` weights infections one day before
## the death (epidemia's "lag-1-first" convention -- Stan sums over
## `infections[start .. t-1]` for both). Under that convention, entry k should
## carry the probability that the delay falls in (k-1, k]:
##
##     kernel[k] = F(k) - F(k-1)
##
## which puts the kernel's mean lag half a day above the underlying continuous
## mean, and correctly assigns no mass to lag 0 -- a death cannot occur on the
## day of infection.
##
## `si` already follows this rule: its mean lag is 6.997 against a continuous
## mean of 6.5. `inf2death` did NOT. It was discretised as Flaxman's `f` shifted
## one index, giving a mean lag of 22.897 against a continuous mean of 23.900 --
## a day EARLY, and 1.5 days out of step with `si`, with mass at lag 0. Deaths
## were therefore attributed to infections about a day and a half too recent.
##
## THE DISTRIBUTION is the one in Flaxman et al. (2020),
## `nature/utils/process-covariates.r`: infection-to-onset ~ Gamma(mean 5.1,
## cv 0.86) plus onset-to-death ~ Gamma(mean 18.8, cv 0.45), so the continuous
## mean is 23.9 days and the kernel's mean lag should be 24.4.
##
## Note the reproduction vignette (vignettes/flaxman.Rmd.orig) deliberately does
## NOT use this kernel. Flaxman discretises at the midpoint,
## f[k] = F(k+1/2) - F(k-1/2), so reproducing the paper needs the paper's own
## discretisation; the vignette builds it inline and says so.
##
##     Rscript data-raw/inf2death.R

stopifnot(file.exists("DESCRIPTION"))

N <- 101L                      # keep the shipped length

## Deterministic convolution of the two Gammas -- no Monte Carlo, so the result
## is reproducible bit for bit.
h    <- 1e-4
grid <- seq(0, 400, by = h)
dens_gamma <- function(x, mean, cv) {
  dgamma(x, shape = 1 / cv^2, rate = 1 / (mean * cv^2))
}
d1 <- dens_gamma(grid, 5.1, 0.86)      # infection to onset
d2 <- dens_gamma(grid, 18.8, 0.45)     # onset to death
dens <- convolve(d1, rev(d2), type = "open")[seq_along(grid)] * h
cdf  <- cumsum(dens) * h

stopifnot(abs(tail(cdf, 1) - 1) < 1e-5)
stopifnot(abs(sum(dens * grid) * h - 23.9) < 1e-2)

F <- function(q) {
  cdf[pmin(pmax(round(q / h) + 1L, 1L), length(cdf))]
}

inf2death <- F(seq_len(N)) - F(seq_len(N) - 1L)
inf2death <- inf2death / sum(inf2death)      # the shipped kernels sum to 1

mean_lag <- sum(inf2death * seq_along(inf2death))
message(sprintf("inf2death: length %d, sum %.15f, mean lag %.3f (expect 24.4)",
                length(inf2death), sum(inf2death), mean_lag))
stopifnot(abs(mean_lag - 24.4) < 0.05)
stopifnot(inf2death[1] > 0, all(diff(which(inf2death > 1e-12)) == 1))

## ---- write both R data objects -----------------------------------------
for (nm in c("EuropeCovid", "EuropeCovid2")) {
  e <- new.env()
  load(file.path("data", paste0(nm, ".RData")), envir = e)
  obj <- get(nm, envir = e)
  stopifnot(length(obj$inf2death) == N)
  obj$inf2death <- inf2death
  assign(nm, obj, envir = e)
  save(list = nm, file = file.path("data", paste0(nm, ".RData")),
       envir = e, compress = "bzip2")
  message("wrote data/", nm, ".RData")
}

## ---- and the Python port's copies ---------------------------------------
for (f in c("europe_covid_inf2death.csv", "europe_covid2_inf2death.csv")) {
  path <- file.path("python", "src", "epidemia", "data_files", f)
  write.csv(data.frame(inf2death = inf2death), path, row.names = FALSE)
  message("wrote ", path)
}
