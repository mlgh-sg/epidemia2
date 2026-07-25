# epidemia (Python)

A Python counterpart of the R package
[**epidemia**](https://mlgh-sg.com/epidemia2/): Bayesian **semi-mechanistic**
modelling of infectious diseases.

Latent daily infections are propagated by a discrete **renewal process** (a
self-exciting point process); the time-varying reproduction number `R_t` is
modelled with a random walk, and observations are linked to infections through a
delay/ascertainment convolution.

The model is specified with **[PyMC](https://www.pymc.io/)** and fit with
**[nutpie](https://github.com/pymc-devs/nutpie)** — a fast Rust NUTS with
Fisher-information mass-matrix adaptation — returning an ArviZ `InferenceData`.

## Install

Not on PyPI yet — install from the repository. The package lives in the `python/`
subdirectory, hence `#subdirectory=python`; without it pip looks for a
`pyproject.toml` at the repo root and fails.

```bash
pip install "git+https://github.com/mlgh-sg/epidemia2.git#subdirectory=python"
```

Pin to a tag for anything you need to reproduce later, since `main` moves:

```bash
pip install "git+https://github.com/mlgh-sg/epidemia2.git@v0.1.0#subdirectory=python"
```

Python 3.10–3.13. The example datasets ship inside the wheel.

!!! warning "While the repository is private"
    The `https://` forms fail for everyone until the repo is public. Use SSH in
    the meantime — it uses the key you already push with:

    ```bash
    pip install "git+ssh://git@github.com/mlgh-sg/epidemia2.git#subdirectory=python"
    ```

### For development

Using [`uv`](https://docs.astral.sh/uv/):

```bash
git clone git@github.com:mlgh-sg/epidemia2.git
cd epidemia2/python
uv python pin 3.12   # broad wheel availability
uv sync              # PyMC + nutpie, compiled with the numba CPU backend
uv run pytest
```

The default `numba` backend is the recommended optimized path on **all**
platforms, including **Apple Silicon**.

!!! tip "GPU (Linux + NVIDIA)"
    `uv sync --extra gpu` pulls `jax[cuda12]`; then fit with
    `epi.fit(y, config, backend="jax")` to run the log-density on the GPU.

## Quick start

```python
import numpy as np, epidemia as epi

d = epi.flu1918()
y = np.concatenate([[np.nan], d.incidence])            # 1st day explained by seeding
config = epi.EpiConfig(
    gen=d.generation, i2o=np.repeat(0.25, 4), seed_days=6,
    link="log", family="neg_binom",
    intercept_loc=np.log(2.0), intercept_scale=0.2, rw_prior_scale=0.1,
)
idata = epi.fit(y, config, draws=1000, tune=1000, chains=4)

epi.plots.plot_rt(idata)          # reproduction number over time
epi.plots.plot_obs(idata, y)      # posterior predictive vs observed
epi.plots.plot_infections(idata)  # latent infections
```

Or run the packaged example:

```bash
uv run epidemia-flu --save flu
```

## Next steps

- The [**user guide**](guide.md) walks through the model, the configuration
  options, and the performance design.
- The [**API reference**](reference.md) documents every public function and class.

## Authors and credit

This Python package is written and maintained by **Swapnil Mishra**
([ORCID](https://orcid.org/0000-0002-8759-5902)), with
**[Claude Code](https://claude.com/claude-code)** (Anthropic).

!!! note "Cite the original, not this port"
    This is a port, not a new method. The model, its priors and links, the
    example data, and the design it follows all come from the R package
    **epidemia** — by James A. Scott, Axel Gandy, Swapnil Mishra,
    H. Juliette T. Unwin, Seth Flaxman and Samir Bhatt — and the framework is
    Bhatt et al. (2020), applied in Flaxman et al. (2020).

    * Scott, J. A., Gandy, A., Mishra, S., Bhatt, S., Flaxman, S., Unwin, H. J. T.
      & Ish-Horowicz, J. (2021). *Epidemia: An R Package for Semi-Mechanistic
      Bayesian Modelling of Infectious Diseases using Point Processes.*
      [arXiv:2110.12461](https://arxiv.org/abs/2110.12461)
    * Bhatt, S. et al. (2020). *Semi-mechanistic Bayesian modelling of COVID-19
      with renewal processes.* [arXiv:2012.00394](https://arxiv.org/abs/2012.00394)
    * Flaxman, S. et al. (2020). *Estimating the effects of non-pharmaceutical
      interventions on COVID-19 in Europe.*
      [Nature 584, 257–261](https://www.nature.com/articles/s41586-020-2405-7)

Licensed GPL-3.0-or-later, as the R package is.
