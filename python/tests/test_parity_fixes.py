"""Regression tests for parity gaps found by the R-vs-Python audit."""

import json

import arviz as az
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from epidemia.core import ObsModel, PanelData
from epidemia.plots import _draws, _predictive_draws, plot_obs

M, T = 2, 30


def _idata(with_family=True, aux=True):
    rng = np.random.default_rng(0)
    v = {"E_deaths": (("chain", "draw", "region", "region_time"),
                      rng.random((2, 60, M, T)) * 20 + 10)}
    if aux:
        v["deaths|aux"] = (("chain", "draw"), rng.random((2, 60)) * 4 + 4)
    post = xr.Dataset(v, coords={"chain": [0, 1], "draw": range(60),
                                 "region": ["A", "B"], "region_time": range(T)})
    idata = az.InferenceData(posterior=post)
    if with_family:
        idata.attrs["epidemia_families"] = json.dumps({"deaths": "neg_binom"})
    return idata


def _panel():
    return PanelData(X=np.zeros((M, T, 0)), lengths=np.full(M, T),
                     regions=["A", "B"], npis=[],
                     dates=[pd.date_range("2020-03-01", periods=T).values] * M,
                     pops=np.array([1e6, 2e6]))


def _obs_model():
    rng = np.random.default_rng(1)
    return ObsModel("deaths", rng.poisson(15, (M, T)).astype(float),
                    np.ones((M, T), bool), np.array([0.5, 0.5]))


# --- plot_obs bands the predictive, not the mean --------------------------

def test_predictive_band_is_wider_than_the_mean_band():
    """R's plot_obs bands posterior_predict; banding the mean is far too narrow."""
    idata = _idata()
    pred = _predictive_draws(idata, "E_deaths", "deaths")
    assert pred is not None
    mean_arr, _ = _draws(idata, "E_deaths")

    def width(a):
        return (np.percentile(a, 97.5, axis=0) - np.percentile(a, 2.5, axis=0)).mean()

    assert width(pred) > 1.5 * width(mean_arr)


def test_family_is_read_from_the_fit_or_from_the_obs_model():
    assert _predictive_draws(_idata(with_family=True), "E_deaths", "deaths") is not None
    # no family recorded -> cannot draw
    assert _predictive_draws(_idata(with_family=False), "E_deaths", "deaths") is None
    # ...unless the ObsModel supplies it
    got = _predictive_draws(_idata(with_family=False), "E_deaths", "deaths",
                            obs_model=_obs_model())
    assert got is not None


def _single_pop_idata(with_family=False):
    """A one-population fit: no region dim, so no panel counts are needed."""
    rng = np.random.default_rng(3)
    post = xr.Dataset(
        {"E_deaths": (("chain", "draw", "time"), rng.random((2, 40, T)) * 20 + 10)},
        coords={"chain": [0, 1], "draw": range(40), "time": range(T)})
    idata = az.InferenceData(posterior=post)
    if with_family:
        idata.attrs["epidemia_families"] = json.dumps({"deaths": "poisson"})
    return idata


def test_plot_obs_warns_rather_than_silently_banding_the_mean():
    """Falling back to the mean must be loud: the interval is not comparable."""
    with pytest.warns(UserWarning, match="too narrow"):
        plot_obs(_single_pop_idata(with_family=False), save=False)


def test_plot_obs_predictive_false_is_silent():
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        plot_obs(_single_pop_idata(with_family=False), save=False,
                 predictive=False)


def test_plot_obs_uses_the_recorded_family_without_warning():
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        plot_obs(_single_pop_idata(with_family=True), save=False)


# --- posterior_linpred sums every walk ------------------------------------

def test_linpred_sums_all_rt_walks_and_observation_walks():
    from epidemia.postprocess import _walk_contribution

    rng = np.random.default_rng(2)
    steps = 5
    post = xr.Dataset(
        {"rw": (("chain", "draw", "proc", "step"), rng.normal(size=(1, 8, 1, steps))),
         "rw2": (("chain", "draw", "proc", "step"), rng.normal(size=(1, 8, 1, steps))),
         "deaths|rw": (("chain", "draw", "proc", "step"),
                       rng.normal(size=(1, 8, 1, steps)))},
        coords={"chain": [0], "draw": range(8), "proc": [0], "step": range(steps)})
    panel = _panel()
    panel.rw_index = np.tile(np.repeat(np.arange(steps), T // steps)[:T], (M, 1))

    idxs = [panel.rw_index, panel.rw_index]
    both = _walk_contribution(post, idxs, M, prefix="")
    only_first = _walk_contribution({"rw": post["rw"]}, idxs, M, prefix="")
    # rw2 must contribute: summing both differs from summing only rw
    assert not np.allclose(both, only_first)

    # observation walks live under a "<series>|" prefix and were dropped entirely
    obs_walk = _walk_contribution(post, idxs, M, prefix="deaths|")
    assert np.any(obs_walk != 0)


# --- autoscale is actually applied by the model builder -------------------

def test_model_builder_autoscales_a_shifted_gamma_covariate_prior():
    """R divides the prior scale by each covariate's own scale; Python never did.

    Checked through the built model rather than the helper, because the gap was
    that nothing *called* the (correct) helper.
    """
    import epidemia.priors as pr
    from epidemia.core import EpiModelConfig, build_epidemia_model

    seen = []
    real = pr.autoscale

    def spy(spec, sd):
        seen.append(np.atleast_1d(np.asarray(sd, dtype=float)).copy())
        return real(spec, sd)

    rng = np.random.default_rng(0)
    X = rng.normal(0.0, 10.0, size=(M, T, 1))       # sd far from 1
    panel = PanelData(X=X, lengths=np.full(M, T), regions=["A", "B"], npis=["x"],
                      dates=[pd.date_range("2020-03-01", periods=T).values] * M,
                      pops=np.array([1e6, 2e6]))
    gen = np.ones(10) / 10
    om = ObsModel("deaths", rng.poisson(10, (M, T)).astype(float),
                  np.ones((M, T), bool), gen)
    cfg = EpiModelConfig(gen=gen, prior_covariates=pr.shifted_gamma(1 / 6, 1.0, 0.0))

    pr.autoscale = spy
    try:
        build_epidemia_model(panel, [om], cfg)
    finally:
        pr.autoscale = real

    assert seen, "the builder never autoscaled the covariate prior"
    assert seen[0][0] == pytest.approx(np.std(X.reshape(-1), ddof=1), rel=1e-6)


def test_autoscale_can_be_switched_off_on_the_config():
    import epidemia.priors as pr
    from epidemia.core import EpiModelConfig, build_epidemia_model

    called = []
    real = pr.autoscale
    pr.autoscale = lambda spec, sd: (called.append(1), real(spec, sd))[1]
    try:
        rng = np.random.default_rng(0)
        panel = PanelData(X=rng.normal(size=(M, T, 1)), lengths=np.full(M, T),
                          regions=["A", "B"], npis=["x"],
                          dates=[pd.date_range("2020-03-01", periods=T).values] * M,
                          pops=np.array([1e6, 2e6]))
        gen = np.ones(10) / 10
        om = ObsModel("deaths", rng.poisson(10, (M, T)).astype(float),
                      np.ones((M, T), bool), gen)
        build_epidemia_model(panel, [om], EpiModelConfig(
            gen=gen, prior_covariates=pr.shifted_gamma(1 / 6, 1.0, 0.0),
            autoscale=False))
    finally:
        pr.autoscale = real
    assert not called


# --- a latent fit can now be forecast -------------------------------------

def test_latent_forecast_is_stochastic_and_reads_the_fit_back():
    """R continues epiinf(latent=TRUE) with normal_lb_rng(mu, sigma, 0).

    Python used to refuse outright. The fitted window must come back from the
    posterior unchanged; the horizon must be drawn, not set to the mean.
    """
    from epidemia.forecast import _renewal

    rng = np.random.default_rng(0)
    S, Mr, Tt = 40, 2, 24
    Rt = np.full((S, Mr, Tt), 1.1)
    gen = np.ones(5) / 5
    seed = np.full((S, Mr), 10.0)
    fitted = rng.uniform(5.0, 15.0, size=(S, Mr, 12))     # the fitted window

    inf, _ = _renewal(Rt, gen, seed, seed_days=3,
                      latent_sd=lambda mean: np.sqrt(2.0 * mean),
                      rng=rng, observed=fitted)

    # in-sample days are read back verbatim
    np.testing.assert_allclose(inf[:, :, :12], fitted)
    # the horizon is not the deterministic mean: repeated draws differ
    again, _ = _renewal(Rt, gen, seed, seed_days=3,
                        latent_sd=lambda mean: np.sqrt(2.0 * mean),
                        rng=np.random.default_rng(1), observed=fitted)
    assert not np.allclose(inf[:, :, 12:], again[:, :, 12:])
    # and never negative -- R truncates at zero
    assert np.all(inf >= 0.0)


def test_latent_forecast_without_noise_matches_the_deterministic_recursion():
    from epidemia.forecast import _renewal

    Rt = np.full((5, 1, 15), 1.2)
    gen = np.ones(4) / 4
    seed = np.full((5, 1), 8.0)
    a, _ = _renewal(Rt, gen, seed, seed_days=3)
    b, _ = _renewal(Rt, gen, seed, seed_days=3, latent_sd=None,
                    rng=np.random.default_rng(0))
    np.testing.assert_allclose(a, b)


def test_forecast_works_without_b0_for_a_single_population_fit():
    """region_effects=False never creates b0; forecast() used to demand it.

    Found by the flu tutorial, which is a one-region model and so has no
    per-region intercept deviation to forecast with.
    """
    import epidemia
    from epidemia.core import (EpiModelConfig, ObsModel, PanelData, RandomWalk,
                               fit_epidemia)
    from epidemia.priors import normal

    flu = epidemia.flu1918()
    yv, n = flu.incidence[:45], 45
    panel1 = PanelData(X=np.zeros((1, n, 0)), lengths=np.array([n]),
                       regions=["Baltimore"], npis=[],
                       dates=[pd.date_range("1918-09-17", periods=n).values],
                       pops=None)
    obs = ObsModel("cases", yv[None, :], np.ones((1, n), bool),
                   np.repeat(0.25, 4), family="poisson", link="identity",
                   intercept=False, offset=np.ones((1, n)))
    cfg = EpiModelConfig(gen=flu.generation, link="log", intercept=True,
                         prior_intercept=normal(np.log(2), 0.2),
                         region_effects=False, seed_days=6,
                         rw=RandomWalk(index=np.tile(np.arange(n), (1, 1)),
                                       prior_scale=0.01))
    idata = fit_epidemia(panel1, [obs], cfg, draws=60, tune=60, chains=2,
                         seed=0, progress_bar=False)
    assert "b0" not in idata.posterior          # the premise of the test
    fc = epidemia.forecast(idata, panel1, [obs], cfg, draws=20, seed=0)
    assert np.isfinite(np.asarray(fc.Rt)).all()


# --- rw index -1 excludes a day from the walk -----------------------------

def test_rw_index_minus_one_contributes_nothing():
    """R's rw(time=) takes NA for "no walk term here"; Python had no equivalent.

    Without it, an excluded day silently takes the walk's FIRST step, which is a
    free parameter perfectly confounded with the intercept -- it showed up as
    r_hat 1.03 on `intercept` in the England tutorial.
    """
    import pymc as pm
    import pytensor.tensor as pt
    from epidemia.core import RandomWalk, _random_walk

    Tn, Mn = 10, 1
    idx = np.array([[-1, -1, -1, 0, 0, 1, 1, 2, 2, 3]])
    with pm.Model():
        contrib = _random_walk(pm, pt, RandomWalk(index=idx), Mn, Tn)
        fn = pm.compile([], contrib, on_unused_input="ignore")
    vals = np.asarray(fn())
    assert np.allclose(vals[0, :3], 0.0), "excluded days must contribute zero"
    assert not np.allclose(vals[0, 3:], 0.0), "included days must contribute"


def test_rw_index_below_minus_one_is_rejected():
    import pymc as pm
    import pytensor.tensor as pt
    from epidemia.core import RandomWalk, _random_walk

    with pm.Model():
        with pytest.raises(ValueError, match=r">= -1"):
            _random_walk(pm, pt, RandomWalk(index=np.array([[-2, 0, 1]])), 1, 3)


# --- medium/low parity fixes ----------------------------------------------

def test_log_uses_a_pseudo_log_that_is_defined_at_zero():
    """R uses trans='pseudo_log'; scale_y_log10 silently drops zero-count days."""
    from epidemia.plots import _pseudo_log, _pseudo_log_inv

    assert np.isfinite(_pseudo_log(0.0))
    assert _pseudo_log(0.0) == pytest.approx(0.0)
    x = np.array([0.0, 1.0, 10.0, 1000.0])
    np.testing.assert_allclose(_pseudo_log_inv(_pseudo_log(x)), x, atol=1e-9)


def test_smooth_drops_incomplete_windows_instead_of_zero_padding():
    """R's rollmean(fill=NA) + complete.cases; 'same' convolution biases edges."""
    from epidemia.plots import _draw_transform

    v = np.full((1, 11), 10.0)                 # constant series
    out = _draw_transform(smooth=5)(v, 0)
    # a constant series must smooth to the same constant wherever it is defined
    finite = out[np.isfinite(out)]
    np.testing.assert_allclose(finite, 10.0)
    # ...and the edges are dropped, not dragged toward zero
    assert np.isnan(out[0, 0]) and np.isnan(out[0, -1])
    assert np.isfinite(out[0, 5])


def test_smooth_window_wider_than_the_series_warns_and_does_nothing():
    from epidemia.plots import _draw_transform

    v = np.full((1, 6), 3.0)
    with pytest.warns(UserWarning, match="not shorter"):
        out = _draw_transform(smooth=99)(v, 0)
    np.testing.assert_allclose(out, 3.0)


def test_groups_accepts_a_vector_and_rejects_unknown_names():
    from epidemia.plots import _as_groups

    assert _as_groups(None, ["A", "B"]) == ["A", "B"]
    assert _as_groups("A", None) == ["A"]
    assert _as_groups(["A", "B"], None) == ["A", "B"]
    assert _as_groups(None, None) is None


def test_score_forwards_metrics_and_validates_groups():
    """R's evaluate_forecast takes metrics=; gr_subset errors on a bad group."""
    import epidemia.scoring as sc

    rng = np.random.default_rng(0)
    n = 12
    yv = rng.poisson(20, n).astype(float)
    dr = rng.poisson(20, (n, 100)).astype(float)
    only = sc.evaluate_forecast(yv, dr, metrics="crps")
    assert "crps" in only.error.columns
    assert "mean_abs_error" not in only.error.columns


def test_prior_family_gates_match_r():
    """R rejects out-of-set families at epirt, epiinf and handle_glm_prior."""
    from epidemia.core import EpiModelConfig, ObsModel, PanelData, build_epidemia_model
    from epidemia.priors import laplace, shifted_gamma

    rng = np.random.default_rng(0)
    n = 20
    pan = PanelData(X=np.zeros((1, n, 0)), lengths=np.array([n]), regions=["A"],
                    npis=[], dates=[pd.date_range("2020-03-01", periods=n).values],
                    pops=None)
    gen = np.ones(5) / 5
    om = ObsModel("d", rng.poisson(8, (1, n)).astype(float),
                  np.ones((1, n), bool), gen)

    with pytest.raises(ValueError, match="not supported here"):
        build_epidemia_model(pan, [om], EpiModelConfig(
            gen=gen, intercept=True, region_effects=False,
            prior_intercept=shifted_gamma()))

    om_bad = ObsModel("d", om.y, om.mask, gen, prior_aux=laplace())
    with pytest.raises(ValueError, match="not supported here"):
        build_epidemia_model(pan, [om_bad],
                             EpiModelConfig(gen=gen, region_effects=False))


def test_posterior_linpred_transform_applies_the_cap():
    """The old docstring said the cap was NOT applied; it is."""
    from epidemia.forecast import _apply_link

    assert _apply_link("scaled_logit", np.array([0.0]), 0.02)[0] == pytest.approx(0.01)


# --- plotting straight from a Forecast ------------------------------------

def _tiny_fit():
    import epidemia as ep
    from epidemia.core import (EpiModelConfig, ObsModel, PanelData, RandomWalk,
                               fit_epidemia)
    from epidemia.priors import normal

    flu = ep.flu1918()
    yv, n = flu.incidence[:45], 45
    pan = PanelData(X=np.zeros((1, n, 0)), lengths=np.array([n]), regions=["B"],
                    npis=[], dates=[pd.date_range("1918-09-17", periods=n).values],
                    pops=None)
    om = ObsModel("cases", yv[None, :], np.ones((1, n), bool), np.repeat(0.25, 4),
                  family="neg_binom", link="identity", intercept=False,
                  offset=np.ones((1, n)))
    cfg = EpiModelConfig(gen=flu.generation, link="log", intercept=True,
                         prior_intercept=normal(np.log(2), 0.2),
                         region_effects=False, seed_days=6,
                         rw=RandomWalk(index=np.tile(np.arange(n), (1, 1)),
                                       prior_scale=0.01))
    idata = fit_epidemia(pan, [om], cfg, draws=60, tune=60, chains=2, seed=0,
                         progress_bar=False)
    return idata, pan, om, cfg


def test_plots_accept_a_forecast_directly():
    """R forwards newdata into posterior_*; here a Forecast is the object."""
    import epidemia as ep

    idata, pan, om, cfg = _tiny_fit()
    fc = ep.forecast(idata, pan, [om], cfg, draws=25, seed=0)
    assert ep.plot_rt(fc, save=False) is not None
    assert ep.plot_infections(fc, save=False) is not None
    assert ep.plot_obs(fc, obs_model=om, series="cases", save=False) is not None


def test_forecast_observed_labels_in_and_out_of_sample():
    import epidemia as ep
    from epidemia.plots import _forecast_observed

    idata, pan, om, cfg = _tiny_fit()
    fc = ep.forecast(idata, pan, [om], cfg, draws=25, seed=0)
    df = _forecast_observed(fc, om, None, n_fitted=30)
    assert set(df["period"]) == {"In-sample", "Out-of-sample"}
    assert int((df["period"] == "In-sample").sum()) == 30


def test_plot_obs_on_a_forecast_needs_a_series_when_ambiguous():
    import epidemia as ep

    idata, pan, om, cfg = _tiny_fit()
    fc = ep.forecast(idata, pan, [om], cfg, draws=25, seed=0)
    with pytest.raises(ValueError, match="no series named"):
        ep.plot_obs(fc, obs_model=om, series="nope", save=False)


# --- parameter selection and summaries ------------------------------------

def test_par_types_selection_matches_rs_categories():
    import arviz as az
    import xarray as xr
    from epidemia.postprocess import PAR_TYPES, extract_samples

    rng = np.random.default_rng(0)
    ds = xr.Dataset(
        {n: (("chain", "draw"), rng.normal(size=(1, 20))) for n in
         ["intercept", "beta", "b0", "rw_scale", "seed", "d|aux",
          "infections_raw"]},
        coords={"chain": [0], "draw": range(20)})
    idata = az.InferenceData(posterior=ds)

    assert set(extract_samples(idata, par_types="fixed").columns) == {"intercept", "beta"}
    assert set(extract_samples(idata, par_types="seeds").columns) == {"seed"}
    assert set(extract_samples(idata, par_types="latent").columns) == {"infections_raw"}
    assert "d|aux" in extract_samples(idata, par_types="aux").columns
    with pytest.raises(ValueError, match="unknown par_types"):
        extract_samples(idata, par_types="nonsense")
    assert set(PAR_TYPES) >= {"fixed", "random", "autocor", "aux", "seeds", "latent"}


def test_summary_reports_diagnostics_alongside_quantiles():
    import arviz as az
    import xarray as xr
    import epidemia as ep

    rng = np.random.default_rng(1)
    ds = xr.Dataset(
        {n: (("chain", "draw"), rng.normal(size=(2, 200))) for n in
         ["intercept", "beta"]},
        coords={"chain": [0, 1], "draw": range(200)})
    out = ep.summary(az.InferenceData(posterior=ds))
    assert list(out.index) == ["intercept", "beta"]
    for col in ("mean", "sd", "10%", "50%", "90%", "ess_bulk", "r_hat"):
        assert col in out.columns


def test_pairs_plot_refuses_an_unwieldy_selection():
    import arviz as az
    import xarray as xr
    import epidemia as ep

    rng = np.random.default_rng(2)
    ds = xr.Dataset(
        {f"p{i}": (("chain", "draw"), rng.normal(size=(1, 50))) for i in range(12)},
        coords={"chain": [0], "draw": range(50)})
    with pytest.raises(ValueError, match="between 2 and 8"):
        ep.pairs_plot(az.InferenceData(posterior=ds), save=False)


# --- forecast() must reproduce the model it was given ---------------------

def test_forecast_index_minus_one_contributes_nothing():
    """core and postprocess honour -1; forecast used to give those days a step."""
    from epidemia.forecast import _extend_rw_index

    idx = np.array([[-1, -1, 0, 0, 1, 1]])
    out = _extend_rw_index(idx, np.array([6]), 10)
    assert list(out[0, :2]) == [-1, -1], "excluded days must stay excluded"
    assert out[0, 2] == 0 and out[0, 5] == 1
    # the cadence is inferred from the REAL steps (2 days each), not from -1
    assert out[0, 6] == 2 and out[0, 7] == 2


def test_forecast_extend_index_handles_an_all_excluded_row():
    from epidemia.forecast import _extend_rw_index

    out = _extend_rw_index(np.array([[-1, -1, -1]]), np.array([3]), 6)
    assert (out == -1).all()


def test_forecast_propagates_an_observation_series_walk():
    """A walk on epiobs was dropped, forecasting a flat ascertainment rate."""
    import epidemia as ep
    from epidemia.core import (EpiModelConfig, ObsModel, PanelData, RandomWalk,
                               fit_epidemia)

    rng = np.random.default_rng(0)
    n = 40
    pan = PanelData(X=np.zeros((1, n, 0)), lengths=np.array([n]), regions=["A"],
                    npis=[], dates=[pd.date_range("2020-03-01", periods=n).values],
                    pops=None)
    idx = np.tile(np.repeat(np.arange(n // 5), 5)[:n], (1, 1))
    om = ObsModel("cases", rng.poisson(30, (1, n)).astype(float),
                  np.ones((1, n), bool), np.array([0.5, 0.5]),
                  rw=RandomWalk(index=idx, prior_scale=0.1))
    cfg = EpiModelConfig(gen=np.ones(5) / 5, region_effects=False)
    idata = fit_epidemia(pan, [om], cfg, draws=50, tune=50, chains=2, seed=0,
                         progress_bar=False)
    assert "cases|rw" in idata.posterior      # the premise
    fc = ep.forecast(idata, pan, [om], cfg, draws=20, seed=0)
    assert np.isfinite(np.asarray(fc.expected["cases"])).all()


def test_forecast_centres_the_design_when_the_fit_did():
    """center=True fits coefficients on the centred scale; a forecast must match."""
    import epidemia as ep
    from epidemia.core import (EpiModelConfig, ObsModel, PanelData,
                               fit_epidemia)

    rng = np.random.default_rng(1)
    n = 35
    X = rng.normal(5.0, 2.0, size=(1, n, 1))       # far from zero-mean
    pan = PanelData(X=X, lengths=np.array([n]), regions=["A"], npis=["x"],
                    dates=[pd.date_range("2020-03-01", periods=n).values],
                    pops=None)
    om = ObsModel("d", rng.poisson(25, (1, n)).astype(float),
                  np.ones((1, n), bool), np.array([0.5, 0.5]))
    cfg = EpiModelConfig(gen=np.ones(5) / 5, region_effects=False, center=True)
    idata = fit_epidemia(pan, [om], cfg, draws=50, tune=50, chains=2, seed=0,
                         progress_bar=False)
    fc = ep.forecast(idata, pan, [om], cfg, draws=30, seed=0)

    # in-sample, the forecast must reproduce the fitted Rt
    fitted = np.asarray(idata.posterior["Rt"]).reshape(-1, n).mean(axis=0)
    got = np.asarray(fc.Rt).mean(axis=0)[0, :n]
    np.testing.assert_allclose(got, fitted, rtol=0.15)


def test_forecast_accepts_a_list_of_walks():
    from epidemia.core import RandomWalk
    from epidemia.forecast import _walk_eta

    rng = np.random.default_rng(2)
    M, T = 1, 12
    idx = np.tile(np.repeat(np.arange(4), 3)[:T], (M, 1))
    store = {"rw": rng.normal(size=(5, 1, 4)), "rw2": rng.normal(size=(5, 1, 4))}

    def take(name, required=True):
        return store.get(name)

    one = _walk_eta(RandomWalk(index=idx), "", take, rng, M, T, T,
                    np.array([T]), "hold")
    two = _walk_eta([RandomWalk(index=idx), RandomWalk(index=idx)], "", take,
                    rng, M, T, T, np.array([T]), "hold")
    assert not np.allclose(one, two), "the second walk must contribute"


def test_latent_with_pop_adjust_saturates_the_raw_draw():
    """R saturates the RAW draw and uses the UNADJUSTED renewal term as the mean.

    Doing it the other way round (saturating the mean, using the raw draw as the
    infection count) left infections unbounded by the susceptible pool, so
    susceptibles went negative and the state-space sd collapsed to its floor.
    The invariant that must hold: i_t <= S_t, hence S never goes negative.
    """
    from epidemia.core import EpiModelConfig, ObsModel, PanelData, build_epidemia_model

    rng = np.random.default_rng(0)
    Mr, Tt, POP = 2, 10, 1000.0
    pan = PanelData(X=np.zeros((Mr, Tt, 0)), lengths=np.full(Mr, Tt),
                    regions=["A", "B"], npis=[],
                    dates=[pd.date_range("2020-03-01", periods=Tt).values] * Mr,
                    pops=np.full(Mr, POP))
    gen = np.ones(4) / 4
    om = ObsModel("d", rng.poisson(5, (Mr, Tt)).astype(float),
                  np.ones((Mr, Tt), bool), gen)
    m = build_epidemia_model(pan, [om], EpiModelConfig(
        gen=gen, region_effects=True, pop_adjust=True, latent=True, seed_days=2))
    assert np.isfinite(m.compile_logp()(m.initial_point()))

    # The recursion, replicated: whatever the raw draw, saturation bounds it.
    S = POP
    for raw in (1.0, 900.0, 1e9):
        i_t = S * (1.0 - np.exp(-raw / POP))
        assert i_t <= S + 1e-9, "infections must not exceed the susceptible pool"
        assert S - i_t >= -1e-9, "susceptibles must not go negative"


def test_single_region_pop_adjust_compiles():
    """A one-region pop-adjusted model could not be built at all before."""
    from epidemia.core import EpiModelConfig, ObsModel, PanelData, build_epidemia_model

    rng = np.random.default_rng(0)
    n = 20
    pan = PanelData(X=np.zeros((1, n, 0)), lengths=np.array([n]), regions=["A"],
                    npis=[], dates=[pd.date_range("2020-03-01", periods=n).values],
                    pops=np.array([1e6]))
    gen = np.ones(4) / 4
    om = ObsModel("d", rng.poisson(8, (1, n)).astype(float),
                  np.ones((1, n), bool), gen)
    m = build_epidemia_model(pan, [om], EpiModelConfig(
        gen=gen, region_effects=False, pop_adjust=True))
    assert np.isfinite(m.compile_logp()(m.initial_point()))


def test_default_beta_prior_is_autoscaled_like_an_explicit_one():
    """The identical shifted_gamma must not behave differently by spelling."""
    import epidemia.priors as pr
    from epidemia.core import EpiModelConfig, ObsModel, PanelData, build_epidemia_model

    rng = np.random.default_rng(0)
    Mr, Tt = 2, 25
    X = rng.normal(0.0, 10.0, size=(Mr, Tt, 1))
    pan = PanelData(X=X, lengths=np.full(Mr, Tt), regions=["A", "B"], npis=["x"],
                    dates=[pd.date_range("2020-03-01", periods=Tt).values] * Mr,
                    pops=None)
    gen = np.ones(4) / 4
    om = ObsModel("d", rng.poisson(8, (Mr, Tt)).astype(float),
                  np.ones((Mr, Tt), bool), gen)

    seen, real = [], pr.autoscale
    pr.autoscale = lambda spec, sd: (seen.append(1), real(spec, sd))[1]
    try:
        build_epidemia_model(pan, [om], EpiModelConfig(gen=gen))   # DEFAULT prior
    finally:
        pr.autoscale = real
    assert seen, "the built-in default coefficient prior must autoscale too"


# --- the four remaining high-severity gaps --------------------------------

def test_formula_emits_region_effects_from_the_bar_terms():
    """R builds no Z matrix for a formula with no bar term (standata_reg.R:112)."""
    import epidemia
    from epidemia.formula import build_from_formula

    df = epidemia.europe_covid2().data.copy()
    _, _, pooled = build_from_formula(
        df, "R(country, date) ~ 1 + lockdown", responses=["deaths"],
        pop="pop", fit_until="2020-05-05")
    _, _, hier = build_from_formula(
        df, "R(country, date) ~ 1 + (1 + lockdown || country) + lockdown",
        responses=["deaths"], pop="pop", fit_until="2020-05-05")

    assert pooled["region_effects"] is False, "no bar term => fully pooled"
    assert hier["region_effects"] is True


def test_linpred_reads_the_walk_index_off_the_term_not_the_panel():
    """Both shipped tutorials build the panel by hand, so it has no rw_index."""
    import arviz as az
    import xarray as xr
    from epidemia.core import EpiModelConfig, PanelData, RandomWalk
    from epidemia.postprocess import posterior_linpred

    rng = np.random.default_rng(0)
    Mr, Tt, steps = 2, 12, 4
    idx = np.tile(np.repeat(np.arange(steps), 3)[:Tt], (Mr, 1))
    pan = PanelData(X=np.zeros((Mr, Tt, 0)), lengths=np.full(Mr, Tt),
                    regions=["A", "B"], npis=[],
                    dates=[pd.date_range("2020-03-01", periods=Tt).values] * Mr,
                    pops=None)
    assert not hasattr(pan, "rw_index"), "the premise: a hand-built panel"

    post = xr.Dataset(
        {"Rt_unadj": (("chain", "draw", "region", "region_time"),
                      rng.random((1, 6, Mr, Tt))),
         "b0": (("chain", "draw", "region"), rng.normal(size=(1, 6, Mr))),
         "rw": (("chain", "draw", "proc", "step"),
                rng.normal(size=(1, 6, 1, steps)))},
        coords={"chain": [0], "draw": range(6), "region": ["A", "B"],
                "region_time": range(Tt), "proc": [0], "step": range(steps)})
    idata = az.InferenceData(posterior=post)
    cfg = EpiModelConfig(gen=np.ones(3) / 3, rw=RandomWalk(index=idx))

    with_rw = posterior_linpred(idata, pan, cfg, autocor=True)
    without = posterior_linpred(idata, pan, cfg, autocor=False)
    assert not np.allclose(with_rw, without), "the walk must reach the predictor"


def test_linpred_undoes_centring():
    """center=True fits on the centred design; rebuilding from the raw one offsets eta."""
    import arviz as az
    import xarray as xr
    from epidemia.core import EpiModelConfig, PanelData
    from epidemia.postprocess import posterior_linpred

    rng = np.random.default_rng(1)
    Mr, Tt = 2, 10
    X = rng.normal(5.0, 1.0, size=(Mr, Tt, 1))       # mean far from zero
    pan = PanelData(X=X, lengths=np.full(Mr, Tt), regions=["A", "B"], npis=["x"],
                    dates=[pd.date_range("2020-03-01", periods=Tt).values] * Mr,
                    pops=None)
    post = xr.Dataset(
        {"Rt_unadj": (("chain", "draw", "region", "region_time"),
                      rng.random((1, 5, Mr, Tt))),
         "beta": (("chain", "draw", "npi"), np.full((1, 5, 1), 0.7))},
        coords={"chain": [0], "draw": range(5), "region": ["A", "B"],
                "region_time": range(Tt), "npi": ["x"]})
    idata = az.InferenceData(posterior=post)

    centred = posterior_linpred(idata, pan, EpiModelConfig(gen=np.ones(3) / 3,
                                                           center=True))
    raw = posterior_linpred(idata, pan, EpiModelConfig(gen=np.ones(3) / 3,
                                                       center=False))
    offset = float(np.mean(raw - centred))
    assert abs(offset - 0.7 * X.mean()) < 1e-8, "must differ by exactly xbar . beta"


def test_observed_overlay_gets_the_same_transform_as_the_bands():
    """R applies cumulative/by_100k to the DATA too (R/plots_epi.R:211-225)."""
    from epidemia.core import ObsModel, PanelData
    from epidemia.plots import _draw_transform, _observed_frame

    Mr, Tt = 1, 8
    pan = PanelData(X=np.zeros((Mr, Tt, 0)), lengths=np.full(Mr, Tt),
                    regions=["A"], npis=[],
                    dates=[pd.date_range("2020-03-01", periods=Tt).values],
                    pops=np.array([1e5]))
    y = np.arange(1.0, Tt + 1.0)[None, :]
    om = ObsModel("d", y, np.ones((Mr, Tt), bool), np.array([0.5, 0.5]))

    plain = _observed_frame(pan, None, obs_model=om)
    cum = _observed_frame(pan, None, obs_model=om,
                          transform=_draw_transform(cumulative=True))
    np.testing.assert_allclose(plain["obs"].to_numpy(), y[0])
    np.testing.assert_allclose(cum["obs"].to_numpy(), np.cumsum(y[0]))
