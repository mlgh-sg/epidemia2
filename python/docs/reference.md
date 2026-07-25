# API reference

The public API is re-exported from the top-level `epidemia` package.

## Configuration & model

::: epidemia.model.EpiConfig

::: epidemia.model.build_model

## Inference

::: epidemia.infer.fit

## Multilevel (multi-region, partially pooled)

The hierarchical model behind the Europe/COVID and partial-pooling examples:
several regions fit jointly, with covariate effects partially pooled across them.

::: epidemia.multilevel.MultilevelConfig

::: epidemia.multilevel.MultilevelData

::: epidemia.multilevel.prepare_panel

::: epidemia.multilevel.build_multilevel_model

::: epidemia.multilevel.fit_multilevel

::: epidemia.multilevel.effect_table

## Renewal dynamics (NumPy reference)

::: epidemia.renewal

## Data

::: epidemia.data

## Plotting

::: epidemia.plots
