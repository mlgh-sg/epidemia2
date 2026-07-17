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

To get started, please see the [package website](https://mlgh-sg.com/epidemia2/index.html),
where you can find installation instructions, function documentation,
and vignettes.

## Authors and credit

This package — **epidemia2**, and the Python port under
[`python/`](python/) — is maintained by **Swapnil Mishra**
([ORCID](https://orcid.org/0000-0002-8759-5902)), with
**[Claude Code](https://claude.com/claude-code)** (Anthropic).

It continues the original **epidemia**
([ImperialCollegeLondon/epidemia](https://github.com/ImperialCollegeLondon/epidemia)),
whose authors are **James A. Scott, Axel Gandy, Swapnil Mishra, H. Juliette T.
Unwin, Seth Flaxman and Samir Bhatt**. The model, its priors, its formula
interface and the statistical framework are their work and remain credited as
such in `DESCRIPTION`; this repository carries it forward rather than replacing
it.

**Please cite the papers, not the repository:**

- **Software paper** — Scott, J. A., Gandy, A., Mishra, S., Bhatt, S., Flaxman,
  S., Unwin, H. J. T. & Ish-Horowicz, J. (2021). *Epidemia: An R Package for
  Semi-Mechanistic Bayesian Modelling of Infectious Diseases using Point
  Processes.* [arXiv:2110.12461](https://arxiv.org/abs/2110.12461)
- **Framework** — Bhatt, S. et al. (2020). *Semi-mechanistic Bayesian modelling
  of COVID-19 with renewal processes.*
  [arXiv:2012.00394](https://arxiv.org/abs/2012.00394)
- **Application** — Mishra, S. et al. (2022). *A COVID-19 Model for Local
  Authorities of the United Kingdom.* *JRSS-A*.
  [doi:10.1111/rssa.12988](https://doi.org/10.1111/rssa.12988)

`citation("epidemia")` in R, or [`inst/CITATION`](inst/CITATION), gives the
authoritative list. Licensed GPL-3.