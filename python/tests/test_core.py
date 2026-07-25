"""Tests for the unified renewal model in :mod:`epidemia.core`.

These check STRUCTURE and the deterministic recursions rather than inference:
building a graph and drawing from the prior is fast, whereas fitting these
models takes minutes. The one thing that needs a sampler -- that a joint
two-series fit runs at all -- is kept deliberately tiny.
"""

from __future__ import annotations

import numpy as np
import pymc as pm
import pytest

from epidemia.core import (
    EpiModelConfig,
    ObsModel,
    PanelData,
    RandomWalk,
    build_epidemia_model,
)

M, T, K = 3, 40, 2
SEED_DAYS = 6


def _gen():
    g = np.exp(-np.arange(1, 21) / 5.0)
    return g / g.sum()


def _i2o_deaths():
    d = np.exp(-((np.arange(1, 31) - 18) ** 2) / 40.0)
    return d / d.sum()


def _i2o_cases():
    return np.concatenate([np.zeros(4), np.full(7, 1 / 7)])


def _panel(pops=True):
    rng = np.random.default_rng(0)
    X = rng.integers(0, 2, size=(M, T, K)).astype(float)
    return PanelData(
        X=X,
        lengths=np.full(M, T),
        regions=["A", "B", "C"],
        npis=["npi1", "npi2"],
        dates=[np.arange(T)] * M,
        pops=np.array([8e6, 6e7, 8.3e7]) if pops else None,
    )


def _mask():
    mask = np.ones((M, T), bool)
    mask[:, :SEED_DAYS] = False       # seeding period is not observed
    return mask


def _deaths():
    rng = np.random.default_rng(1)
    return ObsModel("deaths", rng.poisson(20, (M, T)), _mask(), _i2o_deaths(),
                    family="neg_binom", link_K=0.02)


def _cases():
    rng = np.random.default_rng(2)
    # `~ 0 + region`: a full-rank region design with no intercept, which is how
    # R writes a per-region ascertainment rate (epiobs has no prior_covariance,
    # so this cannot be partially pooled).
    X = np.eye(M)[:, None, :].repeat(T, 1)
    return ObsModel("cases", rng.poisson(600, (M, T)), _mask(), _i2o_cases(),
                    family="quasi_poisson", link_K=0.4, X=X, intercept=False)


def _rw(by_region=False):
    return RandomWalk(index=np.tile(np.arange(T) // 7, (M, 1)),
                      by_region=by_region)


CONFIGS = {
    "independent": {},
    "correlated": {"correlated": True},
    "rw_shared": {"rw": _rw()},
    "rw_by_region": {"rw": _rw(by_region=True)},
    "pop_adjust": {"pop_adjust": True},
    "pop_adjust_prior_susc": {"pop_adjust": True, "prior_susc_mean": 0.9},
    "everything": {
        "correlated": True, "pop_adjust": True, "prior_susc_mean": 0.9,
        "rw": _rw(by_region=True),
    },
}


@pytest.mark.parametrize("name", list(CONFIGS))
def test_every_configuration_builds_with_finite_logp(name):
    cfg = EpiModelConfig(gen=_gen(), seed_days=SEED_DAYS, **CONFIGS[name])
    model = build_epidemia_model(_panel(), [_deaths(), _cases()], cfg)
    logp = float(model.compile_logp()(model.initial_point()))
    assert np.isfinite(logp), f"{name} gave a non-finite logp"


def test_each_series_gets_its_own_rate_and_auxiliary_parameter():
    cfg = EpiModelConfig(gen=_gen(), seed_days=SEED_DAYS)
    model = build_epidemia_model(_panel(), [_deaths(), _cases()], cfg)
    names = {v.name for v in model.named_vars.values()}

    # If the two series were being collapsed into one, these would not coexist.
    for series in ("deaths", "cases"):
        assert f"{series}|rate" in names
        assert f"E_{series}" in names
        assert f"{series}|aux" in names

    # and the ascertainment regressions differ: deaths has an intercept only,
    # cases has one coefficient per region and no intercept
    assert "deaths|intercept" in names
    assert "cases|intercept" not in names
    assert "cases|coef" in names


def test_a_single_series_may_be_passed_bare():
    cfg = EpiModelConfig(gen=_gen(), seed_days=SEED_DAYS)
    model = build_epidemia_model(_panel(), _deaths(), cfg)
    assert "E_deaths" in {v.name for v in model.named_vars.values()}


def test_correlated_effects_estimate_a_covariance_independent_ones_do_not():
    panel, obs = _panel(), _deaths()

    indep = build_epidemia_model(
        panel, obs, EpiModelConfig(gen=_gen(), seed_days=SEED_DAYS))
    corr = build_epidemia_model(
        panel, obs, EpiModelConfig(gen=_gen(), seed_days=SEED_DAYS, correlated=True))

    indep_names = {v.name for v in indep.named_vars.values()}
    corr_names = {v.name for v in corr.named_vars.values()}

    assert "sd" in indep_names and "Sigma_chol" not in indep_names
    assert "Sigma_chol" in corr_names


def test_correlated_effects_need_a_covariate():
    panel = _panel()
    panel.X = np.zeros((M, T, 0))
    panel.npis = []
    cfg = EpiModelConfig(gen=_gen(), seed_days=SEED_DAYS, correlated=True)
    with pytest.raises(ValueError, match="at least one covariate"):
        build_epidemia_model(panel, _deaths(), cfg)


def test_group_level_walk_has_one_process_per_region():
    for by_region, expected in [(False, 1), (True, M)]:
        cfg = EpiModelConfig(gen=_gen(), seed_days=SEED_DAYS,
                             rw=_rw(by_region=by_region))
        model = build_epidemia_model(_panel(), _deaths(), cfg)
        with model:
            (walk,) = pm.draw([model["rw"]], draws=1, random_seed=0)
        assert walk.shape[0] == expected, (
            f"by_region={by_region} should give {expected} walk process(es)"
        )
        # index // 7 over 40 days spans 6 steps
        assert walk.shape[1] == T // 7 + 1


def test_population_adjustment_depletes_susceptibles_and_lowers_rt():
    cfg = EpiModelConfig(gen=_gen(), seed_days=SEED_DAYS, pop_adjust=True)
    model = build_epidemia_model(_panel(), _deaths(), cfg)
    with model:
        S, R, R_un, inf = pm.draw(
            [model["susceptible"], model["Rt"], model["Rt_unadj"],
             model["infections"]],
            draws=1, random_seed=0,
        )

    # Susceptibles only ever decrease, and never go negative.
    after_seed = S[:, SEED_DAYS:]
    assert np.all(np.diff(after_seed, axis=1) <= 1e-6)
    assert np.all(S > 0), "susceptible pool went negative"

    # The realised R_t is the unadjusted one scaled by the susceptible
    # fraction, so it can never exceed it.
    assert np.all(R <= R_un + 1e-9)
    assert np.all(np.isfinite(inf)) and np.all(inf >= 0)


def test_population_adjustment_vanishes_for_a_huge_population():
    """With nobody meaningfully depleted, the adjustment is a no-op.

    Checked within one model rather than by comparing two: `pm.draw` seeds each
    graph's variables independently, so two separately-built models do not share
    prior draws even under the same seed.
    """
    panel = _panel()
    panel.pops = np.full(M, 1e15)
    model = build_epidemia_model(
        panel, _deaths(),
        EpiModelConfig(gen=_gen(), seed_days=SEED_DAYS, pop_adjust=True),
    )
    with model:
        S, R, R_un = pm.draw(
            [model["susceptible"], model["Rt"], model["Rt_unadj"]],
            draws=1, random_seed=7,
        )

    assert np.all(S / panel.pops[:, None] > 1 - 1e-6)
    assert np.max(np.abs(R - R_un) / R_un) < 1e-5


def test_population_adjustment_bites_for_a_small_population():
    """The converse: a small population visibly depletes and holds R_t down."""
    panel = _panel()
    panel.pops = np.full(M, 5e4)
    model = build_epidemia_model(
        panel, _deaths(),
        EpiModelConfig(gen=_gen(), seed_days=SEED_DAYS, pop_adjust=True),
    )
    with model:
        S, R, R_un = pm.draw(
            [model["susceptible"], model["Rt"], model["Rt_unadj"]],
            draws=1, random_seed=7,
        )

    frac = S / panel.pops[:, None]
    assert frac.min() < 0.99, "a 50k population should deplete measurably"
    assert np.all(R <= R_un + 1e-9)
    # and the gap is not merely numerical
    assert np.max((R_un - R) / R_un) > 1e-3


def test_rejects_malformed_input():
    panel, cfg = _panel(), EpiModelConfig(gen=_gen(), seed_days=SEED_DAYS)

    with pytest.raises(ValueError, match="at least one observation series"):
        build_epidemia_model(panel, [], cfg)

    dup = [_deaths(), _deaths()]
    with pytest.raises(ValueError, match="unique"):
        build_epidemia_model(panel, dup, cfg)

    bad = _deaths()
    bad.y = bad.y[:, :-1]
    with pytest.raises(ValueError, match="must both be"):
        build_epidemia_model(panel, bad, cfg)

    unknown = _deaths()
    unknown.family = "gamma"
    with pytest.raises(ValueError, match="unknown family"):
        build_epidemia_model(panel, unknown, cfg)

    with pytest.raises(ValueError, match="requires PanelData.pops"):
        build_epidemia_model(
            _panel(pops=False), _deaths(),
            EpiModelConfig(gen=_gen(), seed_days=SEED_DAYS, pop_adjust=True),
        )


@pytest.mark.parametrize("family", ["poisson", "neg_binom", "quasi_poisson",
                                    "normal", "log_normal"])
def test_all_observation_families_build(family):
    obs = _deaths()
    obs.family = family
    if family in ("normal", "log_normal"):
        obs.y = obs.y.astype(float) + 1.0     # log_normal needs positive support
    cfg = EpiModelConfig(gen=_gen(), seed_days=SEED_DAYS)
    model = build_epidemia_model(_panel(), obs, cfg)
    assert np.isfinite(float(model.compile_logp()(model.initial_point())))


@pytest.mark.slow
def test_a_joint_two_series_model_actually_samples():
    """The smallest possible end-to-end check that the joint likelihood works."""
    cfg = EpiModelConfig(gen=_gen(), seed_days=SEED_DAYS)
    model = build_epidemia_model(_panel(), [_deaths(), _cases()], cfg)
    with model:
        idata = pm.sample(draws=25, tune=25, chains=1, progressbar=False,
                          random_seed=0, compute_convergence_checks=False)
    for series in ("deaths", "cases"):
        assert f"E_{series}" in idata.posterior
        assert np.all(np.isfinite(idata.posterior[f"E_{series}"].values))
