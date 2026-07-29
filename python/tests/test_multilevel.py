"""Tests for the partially-pooled multi-region renewal model.

The model build + logp are checked directly (fast, no sampling); a full nutpie
fit of this joint model is exercised in the example notebook.
"""

from dataclasses import replace

import numpy as np
import pytest

import epidemia as epi


def _panel():
    ec = epi.europe_covid2()
    # EuropeCovid2's two NaN deaths (Denmark 13/05, Italy 25/06) both fall after
    # this cutoff, so nothing is missing inside the modelled windows here.
    data = epi.prepare_panel(
        ec.data, epi.EUROPE_COVID_NPIS,
        seed_offset=30, death_threshold=10, fit_until="2020-05-05",
    )
    config = epi.MultilevelConfig(gen=ec.si, i2o=ec.inf2death, seed_days=6)
    return data, config


def test_europe_data_loads():
    ec = epi.europe_covid2()
    assert ec.data["country"].nunique() == 11
    assert set(epi.EUROPE_COVID_NPIS).issubset(ec.data.columns)
    np.testing.assert_allclose(ec.si.sum(), 1.0, atol=1e-6)
    np.testing.assert_allclose(ec.inf2death.sum(), 1.0, atol=1e-6)


def test_prepare_panel_shapes_and_masking():
    data, _ = _panel()
    M, T, K = data.X.shape
    assert M == 11 and K == len(epi.EUROPE_COVID_NPIS)
    # mask marks LIKELIHOOD cells: genuine and observed. Nothing is missing
    # inside this window, so it coincides with the lengths here.
    assert (data.mask.sum(axis=1) == data.lengths).all()
    assert data.lengths.max() == T
    # padded region series are zero / masked out beyond their genuine length
    for m in range(M):
        assert not data.mask[m, data.lengths[m]:].any()
        assert not data.deaths[m, data.lengths[m]:].any()


def test_missing_observations_are_masked_out_not_treated_as_zero():
    """A NaN count must not enter the likelihood as an observed zero death."""
    ec = epi.europe_covid2()
    sub = ec.data[ec.data["country"] == "Italy"].copy()
    sub.loc[sub.index[60:65], "deaths"] = np.nan
    with pytest.warns(UserWarning, match="missing"):
        data = epi.prepare_panel(sub, ["lockdown"], seed_offset=30,
                                 death_threshold=10, fit_until="2020-05-05")
    n = int(data.lengths[0])
    assert data.mask[0, :n].sum() == n - 5, "the 5 NaN days must be excluded"
    assert data.mask[0, n:].sum() == 0


def test_build_multilevel_model_logp_finite():
    data, config = _panel()
    model = epi.build_multilevel_model(data, config)
    for var in ("beta", "b0", "b", "Rt", "infections", "E_deaths", "ifr"):
        assert var in model.named_vars, f"missing {var}"
    logp = model.compile_logp()(model.initial_point())
    assert np.isfinite(logp)


def test_multilevel_priors_match_r_defaults():
    ec = epi.europe_covid2()
    config = epi.MultilevelConfig(gen=ec.si, i2o=ec.inf2death)
    # scaled_logit(6.5) on R, scaled_logit(0.02) on IFR, shifted-gamma shift
    assert config.R_link_K == 6.5
    assert config.ifr_link_K == 0.02
    np.testing.assert_allclose(config.beta_shift, np.log(1.05) / 6.0)


def test_prepare_panel_clamps_negative_start_row():
    """A short lead-in must clamp to day 0, not wrap to the END of the series.

    ``crossings[0] - seed_offset`` used to go negative and index from the back
    via ``.iloc[-k]``, silently keeping only the last few days of a region.
    """
    ec = epi.europe_covid2()
    sub = ec.data[ec.data["country"] == "Italy"].copy()
    # Italy crosses 10 cumulative deaths at row 54, so seed_offset=200 underflows.
    with pytest.warns(UserWarning, match="seed_offset"):
        data = epi.prepare_panel(sub, ["lockdown"], seed_offset=200,
                                 death_threshold=10, fit_until="2020-05-05")
    full = sub[sub["date"] < "2020-05-05"]
    assert data.lengths[0] == len(full), "must keep the whole series, not the tail"


def test_pooling_regimes_fix_slopes_but_keep_intercept_sd_estimated():
    """Pooling the covariate effects must not delete each region's own R_0."""
    data, config = _panel()
    for regime, expected in [("partial", None), ("none", 5.0), ("full", 1e-6)]:
        cfg = config.pooling(regime)
        assert cfg.sd_slope_fixed == expected
        model = epi.build_multilevel_model(data, cfg)
        free = {v.name for v in model.free_RVs}
        # the region-intercept SD is a free parameter in EVERY regime ...
        assert "sd" in free or "sd_intercept" in free, regime
        # ... and b0 is never pinned to zero
        assert "z0" in free, regime
        assert np.isfinite(model.compile_logp()(model.initial_point()))


def test_pooling_regimes_only_touch_the_slope_sds():
    """Under a fixed regime, sd[0] must still vary across prior draws."""
    import pymc as pm

    data, config = _panel()
    K = len(data.npis)
    for regime in ("none", "full"):
        model = epi.build_multilevel_model(data, config.pooling(regime))
        with model:
            pr = pm.sample_prior_predictive(draws=200, random_seed=0, var_names=["sd"])
        sd = np.asarray(pr.prior["sd"])[0]           # (draws, K+1)
        assert sd[:, 0].std() > 0.05, f"{regime}: intercept SD got pinned"
        assert np.allclose(sd[:, 1:].std(axis=0), 0.0), f"{regime}: slopes not fixed"
        assert sd.shape[1] == K + 1


def test_pooling_rejects_unknown_regime():
    _, config = _panel()
    with pytest.raises(ValueError, match="partial"):
        config.pooling("sort-of")


def test_seeds_are_hierarchical_by_default_like_R():
    """R's epiinf default is prior_seeds = hexp(prior_aux = exponential(0.03))."""
    data, config = _panel()
    assert config.seed_pooling and config.seed_aux_rate == 0.03
    model = epi.build_multilevel_model(data, config)
    free = {v.name for v in model.free_RVs}
    assert {"seed_tau", "seed_raw"} <= free, "seeds must share a pooled mean tau"
    assert "seed" in model.named_vars
    # tau ~ Exponential(0.03) => prior mean 1/0.03 ~ 33.3
    import pymc as pm
    with model:
        pr = pm.sample_prior_predictive(draws=4000, random_seed=0,
                                        var_names=["seed_tau", "seed"])
    tau = np.asarray(pr.prior["seed_tau"])[0]
    assert 25 < tau.mean() < 42, f"tau prior mean {tau.mean():.1f}, expected ~33.3"
    # turning pooling off restores an independent per-region prior
    off = epi.build_multilevel_model(data, replace(config, seed_pooling=False))
    assert "seed_tau" not in {v.name for v in off.free_RVs}


def test_dispersion_prior_matches_r():
    """R: prior_aux = normal(10, 5) on a lower=0 parameter => 10 + 5*HalfNormal(1)."""
    import pymc as pm

    data, config = _panel()
    assert (config.dispersion_loc, config.dispersion_scale) == (10.0, 5.0)
    model = epi.build_multilevel_model(data, config)
    with model:
        pr = pm.sample_prior_predictive(draws=4000, random_seed=0,
                                        var_names=["reciprocal_dispersion"])
    phi = np.asarray(pr.prior["reciprocal_dispersion"])[0]
    assert phi.min() >= 10.0, "R's prior has support [10, inf), not (0, inf)"
    # mean = 10 + 5*sqrt(2/pi) ~ 13.99
    assert 13.0 < phi.mean() < 15.0, f"prior mean {phi.mean():.2f}, expected ~13.99"


def test_i2o_convolution_starts_at_lag_one_like_R():
    """E_deaths[t] must not depend on infections[t] (R/Stan never uses lag 0)."""
    import pymc as pm

    ec = epi.europe_covid2()
    sub = ec.data[ec.data["country"] == "Italy"].copy()
    data = epi.prepare_panel(sub, ["lockdown"], fit_until="2020-05-05")
    # a kernel with all mass at lag 1 => E_deaths[t] = ifr * infections[t-1]
    config = epi.MultilevelConfig(gen=ec.si, i2o=np.array([1.0]), seed_days=6)
    model = epi.build_multilevel_model(data, config)
    with model:
        pr = pm.sample_prior_predictive(draws=1, random_seed=0,
                                        var_names=["E_deaths", "infections", "ifr"])
    E = np.asarray(pr.prior["E_deaths"])[0, 0, 0]
    inf = np.asarray(pr.prior["infections"])[0, 0, 0]
    ifr = float(np.asarray(pr.prior["ifr"])[0, 0])
    np.testing.assert_allclose(E[1:], ifr * inf[:-1] + 1e-15, rtol=1e-6)
    np.testing.assert_allclose(E[0], 1e-15, atol=1e-18)  # nothing precedes day 0
