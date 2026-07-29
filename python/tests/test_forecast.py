"""One-call forecasting from a fitted model (:mod:`epidemia.forecast`).

Nothing is fitted here. A forecast is a deterministic function of the posterior
draws, so the fixtures build an ``InferenceData`` with the variable names, dims
and coords that :func:`epidemia.core.build_epidemia_model` produces, and the
reference values come from evaluating that model's own deterministics at the
same parameter values (via ``pm.do``, which pins the free RVs and leaves a
purely deterministic graph).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from epidemia.core import (
    EpiModelConfig,
    ObsModel,
    PanelData,
    RandomWalk,
    build_epidemia_model,
)
from epidemia.forecast import Forecast, forecast

M, T, K = 2, 36, 2
SEED_DAYS = 5
HORIZON = 21
POPS = np.array([5.0e6, 1.0e7])
REGIONS = ["Italy", "Sweden"]
NPIS = ["lockdown", "schools"]
START = pd.Timestamp("2020-03-01")
CHAINS, DRAWS = 2, 2


def _gen():
    g = np.exp(-np.arange(1, 16) / 4.5)
    return g / g.sum()


def _i2o_deaths():
    d = np.exp(-((np.arange(1, 26) - 16) ** 2) / 30.0)
    return d / d.sum()


def _i2o_cases():
    return np.concatenate([np.zeros(3), np.full(6, 1.0 / 6.0)])


# --------------------------------------------------------------------------
# fixtures: panel, model configuration, and a hand-built posterior
# --------------------------------------------------------------------------


def _X():
    """Two step-function NPIs: switched on part-way and never switched off."""
    X = np.zeros((M, T, K))
    X[0, 12:, 0] = 1.0
    X[0, 18:, 1] = 1.0
    X[1, 15:, 0] = 1.0
    X[1, 22:, 1] = 1.0
    return X


@pytest.fixture(scope="module")
def panel():
    return PanelData(
        X=_X(), lengths=np.full(M, T), regions=list(REGIONS), npis=list(NPIS),
        dates=[np.asarray(pd.date_range(START, periods=T))] * M, pops=POPS,
    )


@pytest.fixture(scope="module")
def obs_models():
    rng = np.random.default_rng(11)
    mask = np.ones((M, T), bool)
    mask[:, :SEED_DAYS] = False
    # A second, covariate-carrying series with a different family: the forecast
    # has to carry each series' own design forward, not just the R_t one.
    cases_X = np.zeros((M, T, 1))
    cases_X[:, 20:, 0] = 1.0                       # e.g. "testing expanded"
    return [
        ObsModel(name="deaths", y=rng.poisson(30, (M, T)).astype(float),
                 mask=mask, i2o=_i2o_deaths(), family="neg_binom", link_K=0.02),
        ObsModel(name="cases", y=rng.poisson(300, (M, T)).astype(float),
                 mask=mask, i2o=_i2o_cases(), family="poisson", link_K=0.5,
                 X=cases_X),
    ]


@pytest.fixture(scope="module")
def config():
    index = np.broadcast_to(np.arange(T) // 7, (M, T)).copy()
    return EpiModelConfig(
        gen=_gen(), rw=RandomWalk(index=index, by_region=False, prior_scale=0.2),
        seed_days=SEED_DAYS, seed_pooling=True, pop_adjust=True,
    )


def _free_values(rng, n):
    """`n` draws of every free RV of the model, on their natural supports."""
    n_steps = int(T - 1) // 7 + 1
    out = []
    for _ in range(n):
        out.append({
            "sd": np.array([0.30, 0.10, 0.15]),
            # b0 = sd[0] * z0 must land near logit(1/6.5) for a plausible R ~ 1.
            "z0": rng.normal(-5.0, 0.4, size=M),
            "z": rng.normal(0.0, 1.0, size=(M, K)),
            "g_beta": rng.gamma(2.0, 0.15, size=K),
            "rw_scale": np.array([0.12]),
            "rw_noise": rng.normal(0.0, 1.0, size=(1, n_steps)),
            # Seeded large enough that the observation series land in the
            # hundreds: at counts of ~1 the integer predictive quantiles are too
            # chunky to compare against the continuous expected ones.
            "seed_tau": 4000.0 + rng.gamma(2.0, 500.0),
            "seed_raw": rng.gamma(3.0, 0.4, size=M),
            "deaths|intercept": rng.normal(0.0, 0.2),
            "deaths|aux_raw": abs(rng.normal(0.0, 1.0)),
            "cases|intercept": rng.normal(0.0, 0.2),
            "cases|coef": rng.normal(0.0, 0.3, size=1),
        })
    return out


def _derived(free, config):
    """The deterministics the forecast reads, computed from the free values."""
    sd = free["sd"]
    return {
        "b0": sd[0] * free["z0"],
        "b": sd[1:] * free["z"],
        "beta": config.beta_shift - free["g_beta"],
        "rw": np.cumsum(free["rw_scale"][:, None] * free["rw_noise"], axis=1),
        # a real fit records the walk's scale; the forecast needs it to draw
        # increments past the fitted window
        "rw_scale": free["rw_scale"],
        "seed": free["seed_tau"] * free["seed_raw"],
        "deaths|intercept": np.asarray(free["deaths|intercept"]),
        "deaths|aux": 10.0 + 5.0 * free["deaths|aux_raw"],
        "cases|intercept": np.asarray(free["cases|intercept"]),
        "cases|coef": free["cases|coef"],
    }


def _make_idata(frees, config, chains):
    """Pack per-sample deterministics into a posterior with the real dim names."""
    import arviz as az
    import xarray as xr

    n = len(frees)
    assert n % chains == 0
    per_draw = [_derived(f, config) for f in frees]

    def block(name):
        arr = np.stack([np.asarray(d[name], dtype=float) for d in per_draw])
        return arr.reshape((chains, n // chains) + arr.shape[1:])

    n_steps = per_draw[0]["rw"].shape[1]
    data = {
        "b0": (("chain", "draw", "region"), block("b0")),
        "b": (("chain", "draw", "region", "npi"), block("b")),
        "beta": (("chain", "draw", "npi"), block("beta")),
        "rw": (("chain", "draw", "rw_dim_0", "rw_dim_1"), block("rw")),
        "rw_scale": (("chain", "draw", "rw_dim_0"), block("rw_scale")),
        "seed": (("chain", "draw", "region"), block("seed")),
        "deaths|intercept": (("chain", "draw"), block("deaths|intercept")),
        "deaths|aux": (("chain", "draw"), block("deaths|aux")),
        "cases|intercept": (("chain", "draw"), block("cases|intercept")),
        "cases|coef": (("chain", "draw", "cases|coef_dim_0"), block("cases|coef")),
    }
    coords = {
        "chain": np.arange(chains), "draw": np.arange(n // chains),
        "region": list(REGIONS), "npi": list(NPIS),
        "rw_dim_0": np.arange(1), "rw_dim_1": np.arange(n_steps),
        "cases|coef_dim_0": np.arange(1),
    }
    return az.InferenceData(posterior=xr.Dataset(data, coords=coords))


@pytest.fixture(scope="module")
def frees(config):
    return _free_values(np.random.default_rng(4), CHAINS * DRAWS)


@pytest.fixture(scope="module")
def idata(frees, config):
    return _make_idata(frees, config, CHAINS)


@pytest.fixture(scope="module")
def big_idata(config):
    """1000 draws, for the statements that are statistical rather than exact."""
    return _make_idata(_free_values(np.random.default_rng(9), 1000), config, 4)


@pytest.fixture(scope="module")
def tight_idata(config):
    """1000 draws with a deliberately *narrow* posterior.

    Whether predictive intervals are visibly wider than intervals on the mean
    depends on which uncertainty dominates. With the wide fixture above, the
    parameter spread swamps the observation noise and the comparison is decided
    by Monte-Carlo error. Shrinking the parameter spread makes the observation
    noise -- the thing under test -- the dominant term.
    """
    rng = np.random.default_rng(21)
    base = _free_values(np.random.default_rng(2), 1)[0]
    frees = []
    for _ in range(1000):
        f = dict(base)
        f["z0"] = base["z0"] + rng.normal(0.0, 0.01, size=M)
        frees.append(f)
    return _make_idata(frees, config, 4)


def _newdata(panel, npi_tail="hold", horizon=HORIZON):
    """A long frame covering the fitted window plus `horizon` extra days.

    ``npi_tail="hold"`` writes the last observed NPI values into the future rows
    explicitly (what R's vignettes do by hand); ``"nan"`` leaves them missing, so
    the carry-forward rule has to supply them.
    """
    rows = []
    X = panel.X
    for m, region in enumerate(panel.regions):
        dates = pd.date_range(START, periods=T + horizon)
        vals = np.zeros((T + horizon, K))
        vals[:T] = X[m]
        if npi_tail == "hold":
            vals[T:] = X[m, -1]
        else:
            vals[T:] = np.nan
        frame = pd.DataFrame({
            "country": region, "date": dates,
            "deaths": np.r_[np.full(T, 10.0), np.full(horizon, np.nan)],
            "cases": np.r_[np.full(T, 100.0), np.full(horizon, np.nan)],
            "pop": POPS[m],
        })
        for k, name in enumerate(panel.npis):
            frame[name] = vals[:, k]
        rows.append(frame)
    return pd.concat(rows, ignore_index=True)


# --------------------------------------------------------------------------
# the model itself is the reference
# --------------------------------------------------------------------------


def _model_values(panel, obs_models, config, free):
    """Evaluate the fitted model's own deterministics at one parameter draw."""
    import pymc as pm

    model = build_epidemia_model(panel, obs_models, config)
    fixed = pm.do(model, {k: v for k, v in free.items()})
    names = ["Rt", "Rt_unadj", "infections", "susceptible", "E_deaths", "E_cases"]
    vals = pm.draw([fixed[n] for n in names])
    return dict(zip(names, [np.asarray(v) for v in vals]))


def test_in_sample_reproduces_the_models_own_deterministics(
    panel, obs_models, config, idata, frees
):
    """Feeding the fitted design back in must give back the fit, exactly.

    This is the load-bearing test: it pins the linear predictor, the seeding
    convention, the susceptibility recursion and the 1e-15 floor on E to what
    `build_epidemia_model` actually computes.
    """
    fc = forecast(idata, panel, obs_models, config, seed=0)
    assert isinstance(fc, Forecast)
    assert fc.Rt.shape == (CHAINS * DRAWS, M, T)

    for s, free in enumerate(frees):          # chain-major flattening
        ref = _model_values(panel, obs_models, config, free)
        np.testing.assert_allclose(fc.Rt_unadj[s], ref["Rt_unadj"], rtol=1e-10)
        np.testing.assert_allclose(fc.Rt[s], ref["Rt"], rtol=1e-10)
        np.testing.assert_allclose(fc.infections[s], ref["infections"], rtol=1e-9)
        np.testing.assert_allclose(fc.susceptible[s], ref["susceptible"], rtol=1e-10)
        np.testing.assert_allclose(fc.expected["deaths"][s], ref["E_deaths"],
                                   rtol=1e-8)
        np.testing.assert_allclose(fc.expected["cases"][s], ref["E_cases"],
                                   rtol=1e-8)


def test_newdata_over_the_fitted_window_only_matches_in_sample(
    panel, obs_models, config, idata
):
    """Rebuilding the design from a frame must recover the fitted arrays."""
    nd = _newdata(panel, horizon=0)
    same = forecast(idata, panel, obs_models, config, newdata=nd, seed=0)
    base = forecast(idata, panel, obs_models, config, seed=0)
    np.testing.assert_allclose(same.Rt, base.Rt, rtol=1e-12)
    np.testing.assert_allclose(same.expected["deaths"], base.expected["deaths"],
                               rtol=1e-12)


# --------------------------------------------------------------------------
# extending the window
# --------------------------------------------------------------------------


@pytest.mark.parametrize("npi_tail", ["hold", "nan"])
def test_longer_newdata_extends_without_revising(
    panel, obs_models, config, idata, npi_tail
):
    """A forecast appends days; it must not change the ones already there."""
    base = forecast(idata, panel, obs_models, config, seed=0)
    fc = forecast(idata, panel, obs_models, config,
                  newdata=_newdata(panel, npi_tail), seed=0)

    assert fc.Rt.shape == (CHAINS * DRAWS, M, T + HORIZON)
    np.testing.assert_array_equal(fc.lengths, np.full(M, T + HORIZON))
    assert len(fc.dates[0]) == T + HORIZON
    assert fc.dates[0][-1] == np.datetime64(START + pd.Timedelta(days=T + HORIZON - 1))

    for name in ("Rt", "Rt_unadj", "infections", "susceptible"):
        np.testing.assert_allclose(getattr(fc, name)[..., :T],
                                   getattr(base, name), rtol=1e-10)
    for name in ("deaths", "cases"):
        np.testing.assert_allclose(fc.expected[name][..., :T],
                                   base.expected[name], rtol=1e-10)


def test_missing_covariates_carry_the_last_observed_row_forward(
    panel, obs_models, config, idata
):
    """`nan` future NPIs must give exactly what writing them out by hand gives."""
    held = forecast(idata, panel, obs_models, config,
                    newdata=_newdata(panel, "hold"), seed=0)
    filled = forecast(idata, panel, obs_models, config,
                      newdata=_newdata(panel, "nan"), seed=0)
    np.testing.assert_allclose(filled.Rt_unadj, held.Rt_unadj, rtol=1e-12)


def test_rw_forecast_hold_freezes_the_walk_at_its_final_fitted_step(
    panel, obs_models, config, idata
):
    """rw_forecast="hold": with covariates held, R_t stays where the fit left it."""
    fc = forecast(idata, panel, obs_models, config,
                  newdata=_newdata(panel, "hold"), seed=0, rw_forecast="hold")
    tail = fc.Rt_unadj[..., T:]
    last = np.broadcast_to(fc.Rt_unadj[..., T - 1, None], tail.shape)
    np.testing.assert_allclose(tail, last, rtol=1e-12)
    # Rt itself still moves: susceptibles keep being depleted.
    assert np.all(fc.Rt[..., -1] <= fc.Rt[..., T - 1] + 1e-12)


def test_rw_forecast_draw_keeps_the_walk_walking(
    panel, obs_models, config, idata
):
    """The default matches R: new increments are drawn past the fitted window.

    R's new_rw_stanmat draws rnorm(n) * sigma per future period and cumulates
    (R/pp_eta.R:118-140), so forecast R_t fans out rather than going flat.
    """
    nd = _newdata(panel, "hold")
    drawn = forecast(idata, panel, obs_models, config, newdata=nd, seed=0)
    held = forecast(idata, panel, obs_models, config, newdata=nd, seed=0,
                    rw_forecast="hold")

    # the median no longer flatlines at the last fitted value
    tail = drawn.Rt_unadj[..., T:]
    last = np.broadcast_to(drawn.Rt_unadj[..., T - 1, None], tail.shape)
    assert not np.allclose(tail, last)

    # and the forecast fans out: wider spread than the frozen walk, growing
    def spread(fc, t):
        return np.percentile(fc.Rt_unadj[..., t], 97.5) - \
               np.percentile(fc.Rt_unadj[..., t], 2.5)

    assert spread(drawn, -1) > spread(held, -1)
    assert spread(drawn, -1) > spread(drawn, T)


def test_rw_forecast_rejects_an_unknown_mode(panel, obs_models, config, idata):
    with pytest.raises(ValueError, match="rw_forecast must be"):
        forecast(idata, panel, obs_models, config,
                 newdata=_newdata(panel, "hold"), seed=0, rw_forecast="freeze")


def test_forecast_only_needs_the_regions_it_was_fitted_on(
    panel, obs_models, config, idata
):
    nd = _newdata(panel)
    with pytest.raises(ValueError, match="no rows for region"):
        forecast(idata, panel, obs_models, config,
                 newdata=nd[nd["country"] != REGIONS[1]], seed=0)


def test_newdata_must_cover_the_fitted_window(panel, obs_models, config, idata):
    nd = _newdata(panel)
    late = nd[nd["date"] > START]
    with pytest.raises(ValueError, match="fitted start date"):
        forecast(idata, panel, obs_models, config, newdata=late, seed=0)


def test_newdata_must_carry_the_covariate_columns(panel, obs_models, config, idata):
    nd = _newdata(panel).drop(columns=[NPIS[1]])
    with pytest.raises(ValueError, match="missing covariate"):
        forecast(idata, panel, obs_models, config, newdata=nd, seed=0)


# --------------------------------------------------------------------------
# susceptibility
# --------------------------------------------------------------------------


def test_susceptibles_decline_and_stay_positive(panel, obs_models, config, idata):
    fc = forecast(idata, panel, obs_models, config,
                  newdata=_newdata(panel, "hold"), seed=0)
    susc = fc.susceptible
    assert susc is not None
    assert np.all(susc > 0.0)
    assert np.all(np.diff(susc, axis=-1) <= 1e-9)         # never rebounds
    assert np.all(susc[..., -1] < susc[..., 0])           # depletion actually bites
    assert np.all(fc.infections >= 0.0)
    # The pool starts at the FULL population and the recursion removes each
    # day's infections, seeding days included -- as R's Stan does. So after the
    # first day it is the population less that day's (saturated) infections,
    # not the population less seed_days * seed.
    expected0 = POPS[None, :] - fc.infections[:, :, 0]
    np.testing.assert_allclose(susc[:, :, 0], expected0, rtol=1e-12)


def test_no_pop_adjust_gives_no_susceptible_series_and_Rt_equals_Rt_unadj(
    panel, obs_models, config, idata
):
    plain = EpiModelConfig(**{**config.__dict__, "pop_adjust": False})
    fc = forecast(idata, panel, obs_models, plain, seed=0)
    assert fc.susceptible is None
    np.testing.assert_allclose(fc.Rt, fc.Rt_unadj)
    # Without depletion the same R_t always produces at least as many infections.
    adj = forecast(idata, panel, obs_models, config, seed=0)
    assert np.all(fc.infections >= adj.infections - 1e-6)


# --------------------------------------------------------------------------
# predictive draws
# --------------------------------------------------------------------------


def test_predictive_draws_track_the_expected_mean(
    panel, obs_models, config, big_idata
):
    fc = forecast(big_idata, panel, obs_models, config, seed=3)
    for name in ("deaths", "cases"):
        E, y = fc.expected[name], fc.predicted[name]
        assert y.shape == E.shape
        assert np.all(y >= 0)
        busy = E.mean(axis=0) > 20.0
        assert busy.sum() > 10
        np.testing.assert_allclose(y.mean(axis=0)[busy], E.mean(axis=0)[busy],
                                   rtol=0.15)


def test_predictive_intervals_are_wider_than_intervals_on_the_mean(
    panel, obs_models, config, tight_idata
):
    """The point of `predicted`: bands on `expected` alone exclude noise."""
    fc = forecast(tight_idata, panel, obs_models, config, seed=3)
    for name in ("deaths", "cases"):
        E, y = fc.expected[name], fc.predicted[name]
        busy = E.mean(axis=0) > 20.0      # where integer draws are informative
        assert busy.sum() > 10
        q_e = np.percentile(E, [5, 95], axis=0)
        q_y = np.percentile(y, [5, 95], axis=0)
        assert np.all(q_y[0][busy] < q_e[0][busy])
        assert np.all(q_y[1][busy] > q_e[1][busy])


def test_observation_noise_survives_a_degenerate_posterior(
    panel, obs_models, config
):
    """With one parameter value repeated, E has no spread but the draws must."""
    free = _free_values(np.random.default_rng(2), 1)[0]
    idata = _make_idata([free] * 400, config, chains=4)
    fc = forecast(idata, panel, obs_models, config, seed=1)
    for name in ("deaths", "cases"):
        assert np.allclose(fc.expected[name].std(axis=0), 0.0)
        assert fc.predicted[name][:, :, -1].std() > 0.0


def test_poisson_series_needs_no_aux(panel, obs_models, config, idata):
    """A Poisson series has no `|aux` in the posterior; it must still predict."""
    assert "cases|aux" not in idata.posterior
    fc = forecast(idata, panel, obs_models, config, series="cases", seed=0)
    assert set(fc.expected) == {"cases"}
    assert fc.families == {"cases": "poisson"}


# --------------------------------------------------------------------------
# draws, seeding, series selection, tidying
# --------------------------------------------------------------------------


def test_draws_subsamples_the_posterior(panel, obs_models, config, big_idata):
    fc = forecast(big_idata, panel, obs_models, config, draws=25, seed=0)
    assert fc.n_draws == 25
    assert fc.Rt.shape == (25, M, T)
    assert fc.draw_index.shape == (25,)
    assert len(set(fc.draw_index.tolist())) == 25       # without replacement
    assert fc.draw_index.max() < 1000

    full = forecast(big_idata, panel, obs_models, config, seed=0)
    # A subsample is a subset of the full posterior, not a recomputation.
    np.testing.assert_allclose(fc.Rt, full.Rt[fc.draw_index], rtol=1e-12)
    # Asking for more draws than exist just returns them all.
    assert forecast(big_idata, panel, obs_models, config,
                    draws=10_000, seed=0).n_draws == 1000


def test_same_seed_reproduces_and_a_different_one_does_not(
    panel, obs_models, config, big_idata
):
    a = forecast(big_idata, panel, obs_models, config, draws=40, seed=7)
    b = forecast(big_idata, panel, obs_models, config, draws=40, seed=7)
    c = forecast(big_idata, panel, obs_models, config, draws=40, seed=8)

    np.testing.assert_array_equal(a.draw_index, b.draw_index)
    np.testing.assert_array_equal(a.Rt, b.Rt)
    np.testing.assert_array_equal(a.predicted["deaths"], b.predicted["deaths"])
    assert not np.array_equal(a.draw_index, c.draw_index)
    assert not np.array_equal(a.predicted["deaths"], c.predicted["deaths"])


def test_series_selects_a_subset(panel, obs_models, config, idata):
    fc = forecast(idata, panel, obs_models, config, series=["deaths"], seed=0)
    assert set(fc.expected) == set(fc.predicted) == {"deaths"}
    assert fc.series == ["deaths"]
    with pytest.raises(ValueError, match="unknown series"):
        forecast(idata, panel, obs_models, config, series="hospit", seed=0)


def test_to_frame_is_tidy_and_trims_padding(panel, obs_models, config, idata):
    fc = forecast(idata, panel, obs_models, config,
                  newdata=_newdata(panel, "hold"), seed=0)
    df = fc.to_frame(probs=(0.1, 0.5, 0.9))
    assert {"region", "date", "variable", "mean", "q10", "q50", "q90"} <= set(df)
    variables = {"Rt", "Rt_unadj", "infections", "susceptible",
                 "E_deaths", "E_cases", "deaths", "cases"}
    assert set(df["variable"]) == variables
    assert len(df) == len(variables) * M * (T + HORIZON)
    assert np.all(df["q10"] <= df["q90"] + 1e-9)
    assert set(df["region"]) == set(REGIONS)


def test_missing_posterior_variable_is_a_clear_error(
    panel, obs_models, config, idata
):
    stripped = idata.copy()
    stripped.posterior = stripped.posterior.drop_vars("seed")
    with pytest.raises(KeyError, match="seed"):
        forecast(stripped, panel, obs_models, config, seed=0)
