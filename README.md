epidemia  <img src='man/figures/logo.png' width="120" align="right"/>
================

The epidemia package allows researchers to flexibly specify and fit
Bayesian epidemiological models in the style of [Flaxman et
al. (2020)](https://www.nature.com/articles/s41586-020-2405-7). The package
itself is described in the software paper [Scott et al. (2021), *Epidemia: An
R Package for Semi-Mechanistic Bayesian Modelling of Infectious Diseases using
Point Processes*](https://arxiv.org/abs/2110.12461), and the framework is
applied in [Mishra et al. (2022), *A COVID-19 Model for Local Authorities of
the United Kingdom*](https://doi.org/10.1111/rssa.12988) (*Journal of the
Royal Statistical Society Series A*). The
package leverages R’s formula interface to parameterize the time-varying
reproduction rate as a function of covariates. Multiple populations can
be modeled simultaneously with hierarchical models. The design of the
package has been inspired by, and has borrowed from,
[rstanarm](https://mc-stan.org/rstanarm/) (Goodrich et al. 2020).
epidemia fits models with [Stan](https://mc-stan.org/) via the
[cmdstanr](https://mc-stan.org/cmdstanr/) backend, and represents
posterior draws with the [posterior](https://mc-stan.org/posterior/)
package. The Stan models are compiled on first use and cached, so no
compilation happens at install time.

## Installation

epidemia fits models with [CmdStanR](https://mc-stan.org/cmdstanr/). Install
CmdStanR and CmdStan first, then install epidemia from GitHub:

```r
# 1. CmdStanR + CmdStan (needs a C++ toolchain)
install.packages("cmdstanr",
  repos = c("https://stan-dev.r-universe.dev", getOption("repos")))
cmdstanr::check_cmdstan_toolchain(fix = TRUE)
cmdstanr::install_cmdstan()

# 2. epidemia
# install.packages("remotes")
remotes::install_github("mlgh-sg/epidemia2")
```

The Stan models are compiled the first time they are used and then cached. You
can optionally precompile them with `epidemia::compile_epidemia()`.

## Disclaimer

This is an early beta release of the package. As a beta release, there will 
be regular updates with additional
features and more extensive testing. Any feedback is greatly appreciated
- in particular if you find bugs, find the documentation unclear, or
have feature requests, please report them
[here](https://github.com/mlgh-sg/epidemia2/issues).

## Package Website

To get started, please see the [package website](https://mlgh-sg.github.io/epidemia2/index.html),
where you can find installation instructions, function documentation,
and vignettes.