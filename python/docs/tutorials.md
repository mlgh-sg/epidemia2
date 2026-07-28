# Tutorials

Five worked examples, all ported from the R package's vignettes so you can read
them side by side. They are **jupytext** notebooks stored as plain `.py` files in
percent format, which means each one is simultaneously a runnable script and a
notebook — open it in Jupyter or VS Code, or run it with `python`.

| Tutorial | R counterpart | What it covers |
|---|---|---|
| [Spanish flu in Baltimore](tutorials/flu.md) | [Spanish Flu](https://mlgh-sg.com/epidemia2/articles/flu.html) | The smallest interesting model: one population, one series, a daily random walk on `R_t`. Compares the deterministic renewal recursion against the latent (`latent=True`) one, and forecasts from it. |
| [Assessing the effects of interventions](tutorials/europe-covid.md) | [Multilevel Modeling](https://mlgh-sg.com/epidemia2/articles/europe-covid.html) | A partially pooled model of five NPIs across 11 European countries, fitted to daily deaths. Effect sizes, per-region reproduction numbers, counterfactuals. |
| [Tracking SARS-CoV-2 in England](tutorials/multiple-obs.md) | [Multiple Observations](https://mlgh-sg.com/epidemia2/articles/multiple-obs.html) | Daily case counts and weekly ONS positivity fitted jointly to one region. Series observed on different days, day-of-week reporting effects, and an `i2o` that deliberately does not sum to one. |
| [Partial pooling](tutorials/partial-pooling.md) | [Partial Pooling](https://mlgh-sg.com/epidemia2/articles/partial-pooling.html) | How R's `(expr \| factor)` and `(expr \|\| factor)` map onto hierarchical priors here, and what no pooling, partial pooling and full pooling each do to the estimates. |
| [Several observation series](tutorials/multilevel-multi-obs.md) | [Multilevel + Multiple Observations](https://mlgh-sg.com/epidemia2/articles/multilevel-multi-obs.html) | Deaths and cases fitted jointly, with correlated region effects, a weekly random walk per region, and a susceptibility adjustment. Scored with `epidemia.scoring`. |

The pages above are **precomputed**: `scripts/precompute.py` executes each
notebook once and commits the rendered markdown and figures, exactly as the R
side's `vignettes/precompute.R` bakes the vignettes. Nothing is fitted when the
documentation is built. Re-bake after changing the modelling code:

```bash
uv run --group dev python scripts/precompute.py            # all five
uv run --group dev python scripts/precompute.py europe-covid
```

The notebook sources live in
[`notebooks/`](https://github.com/mlgh-sg/epidemia2/tree/main/python/notebooks)
as jupytext percent files — each is a runnable script *and* a notebook.

## One difference from the R vignettes worth knowing

The R `europe-covid` vignette fits with **variational Bayes** for speed, and says
so plainly — it notes that the resulting intervals are "relatively narrow ... an
artifact of using Variational Bayes". These notebooks fit the same model with
**full NUTS**, so the credible intervals are the genuine posterior ones and are
wider than the vignette's. That is the intervals being right, not the model
disagreeing.

If you want to know what that costs in wall-clock, see [Performance](performance.md).

## Running them

From the `python/` directory:

```bash
uv sync --dev
uv run python -m ipykernel install --user --name epidemia \
    --display-name "Python (epidemia)"
```

Then open either notebook and select the **Python (epidemia)** kernel, or run it
as a script:

```bash
uv run python notebooks/europe-covid.py
```

!!! note "These fit real models"
    The multilevel notebook runs NUTS on eleven regions and takes minutes, not
    seconds. `fit_multilevel` prints the compile step separately from the
    sampling, because compilation alone can take as long as the sampling and has
    no progress bar of its own.

## Before you start

Read [What this port does and does not do](parity.md) first if you are coming
from the R package. Everything the R tutorials use routinely — multiple
observation series, per-region random walks on `R_t`, the population and
susceptibility adjustment, forecasting and scoring — is available here; the
parity table maps each R feature onto its Python spelling and records the few
places the two genuinely differ.
