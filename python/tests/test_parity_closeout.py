"""The last batch of R-parity features, and the guards that go with them."""

from __future__ import annotations

import numpy as np
import pytest

import epidemia
from epidemia import priors as pr
from epidemia.core import (
    EpiModelConfig,
    ObsModel,
    PanelData,
    RandomWalk,
    build_epidemia_model,
    prepare_panel,
)

M, T, K = 3, 40, 2
SEED_DAYS = 6


def _gen():
    g = np.exp(-np.arange(1, 21) / 5.0)
    return g / g.sum()


def _panel(pops=(1e5, 2e5, 3e5)):
    rng = np.random.default_rng(0)
    return PanelData(X=rng.integers(0, 2, (M, T, K)).astype(float),
                     lengths=np.full(M, T), regions=list("ABC"),
                     npis=["a", "b"], dates=[np.arange(T)] * M,
                     pops=np.array(pops))


def _obs(**kw):
    rng = np.random.default_rng(1)
    mask = np.ones((M, T), bool)
    mask[:, :SEED_DAYS] = False
    i2o = np.exp(-((np.arange(1, 31) - 18) ** 2) / 40.0)
    return ObsModel("deaths", rng.poisson(20, (M, T)), mask, i2o / i2o.sum(),
                    family="neg_binom", link_K=0.02, **kw)


def _eval(model, names):
    outs = model.replace_rvs_by_values([model[n] for n in names])
    fn = model.compile_fn(outs, inputs=model.value_vars, point_fn=True,
                          on_unused_input="ignore")
    return fn(model.initial_point())


# --- observation-level random walk ----------------------------------------

def test_a_series_can_have_its_own_random_walk():
    """R allows rw() inside an epiobs formula; the walk is the series' own."""
    idx = np.tile(np.arange(T) // 7, (M, 1))
    model = build_epidemia_model(
        _panel(), _obs(rw=RandomWalk(index=idx, by_region=True)),
        EpiModelConfig(gen=_gen(), seed_days=SEED_DAYS,
                       rw=RandomWalk(index=idx)))
    names = {v.name for v in model.free_RVs}
    # namespaced, so it cannot collide with the walk on R_t
    assert {"rw_scale", "rw_noise"} <= names
    assert {"deaths|rw_scale", "deaths|rw_noise"} <= names
    assert np.isfinite(float(model.compile_logp()(model.initial_point())))


# --- removal from the susceptible pool ------------------------------------

def test_removal_reduces_infections():
    """epiinf(rm=): vaccination takes people out of the susceptible pool."""
    panel, obs = _panel(), _obs()
    base = build_epidemia_model(
        panel, obs, EpiModelConfig(gen=_gen(), seed_days=SEED_DAYS,
                                   pop_adjust=True))
    with_rm = build_epidemia_model(
        panel, obs, EpiModelConfig(gen=_gen(), seed_days=SEED_DAYS,
                                   pop_adjust=True, rm=np.full((M, T), 0.01)))
    (i0,) = _eval(base, ["infections"])
    (i1,) = _eval(with_rm, ["infections"])
    assert i1.sum() < i0.sum()


def test_removal_requires_the_population_adjustment():
    with pytest.raises(ValueError, match="pop_adjust"):
        build_epidemia_model(_panel(), _obs(),
                             EpiModelConfig(gen=_gen(), rm=np.zeros((M, T))))


def test_removal_shape_is_checked():
    with pytest.raises(ValueError, match="shape"):
        build_epidemia_model(
            _panel(), _obs(),
            EpiModelConfig(gen=_gen(), pop_adjust=True, rm=np.zeros((M, 3))))


def test_removal_noise_is_optional_and_builds():
    model = build_epidemia_model(
        _panel(), _obs(),
        EpiModelConfig(gen=_gen(), seed_days=SEED_DAYS, pop_adjust=True,
                       rm=np.full((M, T), 0.01),
                       prior_rm_noise=pr.normal(1.0, 0.1)))
    assert "veps" in {v.name for v in model.free_RVs}


# --- prior predictive ------------------------------------------------------

def test_prior_PD_drops_the_likelihood_but_keeps_the_latent_series():
    """R's epim(prior_PD = TRUE)."""
    fitted = build_epidemia_model(
        _panel(), _obs(), EpiModelConfig(gen=_gen(), seed_days=SEED_DAYS))
    prior = build_epidemia_model(
        _panel(), _obs(),
        EpiModelConfig(gen=_gen(), seed_days=SEED_DAYS, prior_PD=True))

    assert [v.name for v in fitted.observed_RVs] == ["deaths"]
    assert prior.observed_RVs == []
    # the deterministics are still there, so a prior predictive check can look
    # at exactly the series the fit would have produced
    names = {v.name for v in prior.deterministics}
    assert {"Rt", "infections", "E_deaths", "deaths|rate"} <= names


# --- group_subset ----------------------------------------------------------

def test_group_subset_restricts_the_regions():
    ec = epidemia.europe_covid2()
    panel, _ = prepare_panel(ec.data, npis=["lockdown"], responses=["deaths"],
                             group_subset=["Austria", "Italy"],
                             fit_until="2020-05-05")
    assert panel.regions == ["Austria", "Italy"]


def test_group_subset_rejects_unknown_groups():
    ec = epidemia.europe_covid2()
    with pytest.raises(ValueError, match="unknown group"):
        prepare_panel(ec.data, responses=["deaths"], group_subset=["Atlantis"])


# --- the susceptibility recursion matches R's, seeding window included -----

def test_seeded_infections_are_saturated_like_R():
    """R applies the saturating step on EVERY modelled day.

    ``inst/stan/tparameters/gen_infections.stan`` runs
    ``infections[i] = susc[i] * (1 - exp(-infections[i] / pops))`` inside
    ``for (i in n0:n2)``, so the seeded days go through it too and come out
    slightly below the raw seed -- and they decline as the pool depletes.
    """
    panel = _panel(pops=(1e4, 2e4, 3e4))    # small enough for saturation to bite
    model = build_epidemia_model(
        panel, _obs(),
        EpiModelConfig(gen=_gen(), seed_days=SEED_DAYS, pop_adjust=True))
    S, I, seed = _eval(model, ["susceptible", "infections", "seed"])

    assert np.all(I[:, 0] < seed), "seeded infections were not saturated"
    # and they fall over the seeding window, because the pool is shrinking
    assert np.all(np.diff(I[:, :SEED_DAYS], axis=1) < 0)


def test_the_pool_starts_at_the_full_population():
    """R sets susc[n0] = pops * S0 and lets the recursion remove the seeds."""
    panel = _panel(pops=(1e4, 2e4, 3e4))
    model = build_epidemia_model(
        panel, _obs(),
        EpiModelConfig(gen=_gen(), seed_days=SEED_DAYS, pop_adjust=True))
    S, I = _eval(model, ["susceptible", "infections"])

    # after day 0 the pool is the population less that day's infections --
    # not the population less seed_days * seed, which is what it used to be
    np.testing.assert_allclose(S[:, 0], panel.pops - I[:, 0], rtol=1e-10)
    assert np.all(np.diff(S, axis=1) <= 1e-9)
    assert np.all(S > 0)


def test_removal_applies_from_the_first_modelled_day():
    """R applies vacc[i] from n0, not from the end of the seeding window."""
    panel = _panel(pops=(1e4, 2e4, 3e4))
    rm = np.zeros((M, T))
    rm[:, :SEED_DAYS] = 0.05            # removal ONLY during seeding
    base = build_epidemia_model(
        panel, _obs(), EpiModelConfig(gen=_gen(), seed_days=SEED_DAYS,
                                      pop_adjust=True))
    with_rm = build_epidemia_model(
        panel, _obs(), EpiModelConfig(gen=_gen(), seed_days=SEED_DAYS,
                                      pop_adjust=True, rm=rm))
    (s0,) = _eval(base, ["susceptible"])
    (s1,) = _eval(with_rm, ["susceptible"])

    # if removal were applied only after seeding, these would be identical
    assert np.all(s1[:, SEED_DAYS - 1] < s0[:, SEED_DAYS - 1])
