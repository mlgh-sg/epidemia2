# Example notebooks

Worked examples ported from the R package's vignettes. Each is a **jupytext**
notebook stored as a plain `.py` file in the *percent* format — cells are
delimited by `# %%`, so the file is a normal Python script **and** a notebook.

| Notebook | R vignette | What it shows |
|----------|------------|---------------|
| [`europe-covid.py`](europe-covid.py) | *Multilevel Modeling* (`europe-covid`) | Hierarchical, partially-pooled model of NPI effects on COVID-19 across 11 European countries; effect sizes, forecasting, counterfactuals. |
| [`partial-pooling.py`](partial-pooling.py) | *Partial Pooling* (`partial-pooling`) | How the R `(expr \| factor)` / `(expr \|\| factor)` pooling maps to hierarchical priors here; a no-pooling vs. partial-pooling demonstration. |

**Estimation uses MCMC.** The R `europe-covid` vignette fits with Variational
Bayes for speed (and notes VB understates uncertainty); these notebooks fit the
same models with full **NUTS / MCMC** via [nutpie](https://github.com/pymc-devs/nutpie),
so the credible intervals are the genuine posterior ones. The hierarchical model
lives in `epidemia.multilevel` (`build_multilevel_model` / `fit_multilevel`), and
the `EuropeCovid2` data is available as `epidemia.europe_covid2()`.

## Running

From the `python/` directory, install the dev tools and register a Jupyter
kernel that points at this project's virtualenv (do the `ipykernel install`
**once**):

```bash
uv sync --dev                                  # jupytext + nbconvert + ipykernel
uv run python -m ipykernel install --user --name epidemia --display-name "Python (epidemia)"
```

The `ipykernel install` step is important: it writes a kernel whose interpreter
is the **absolute** path to `.venv/bin/python`, so the notebook cells can
`import epidemia`. (The default `python3` kernel launches a bare `python` off
your `PATH`, which usually is *not* this venv — that is what causes
`ModuleNotFoundError: No module named 'epidemia'`.)

Then open interactively (pick the **Python (epidemia)** kernel):

```bash
uv run jupytext --to notebook notebooks/europe-covid.py    # -> europe-covid.ipynb
uv run jupyter lab notebooks/europe-covid.ipynb            # or open the .py directly
```

Or execute headless (runs every cell and embeds the outputs in the `.ipynb`):

```bash
uv run jupytext --execute --set-kernel epidemia --to notebook notebooks/partial-pooling.py
```

Headless execution needs `nbconvert` and `ipykernel` (installed by
`uv sync --dev`); without them you get *"make sure that 'nbconvert' and
'ipykernel' are installed"*.

> **Runtime.** Full MCMC on the 11-country model is computationally demanding
> (the renewal equation is evaluated per country at every leapfrog step). Expect
> several minutes for a good `europe-covid` fit; reduce `draws`/`tune`, or the
> number of countries/NPIs, for a quicker pass. `partial-pooling.py` uses a small
> 3-country slice and is fast.
