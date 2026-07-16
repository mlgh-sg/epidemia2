"""Tests for the partially-pooled multi-region renewal model.

The model build + logp are checked directly (fast, no sampling); a full nutpie
fit of this joint model is exercised in the example notebook.
"""

import numpy as np

import epidemia as epi


def _panel():
    ec = epi.europe_covid2()
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
    # genuine days per region equal the mask count, and never exceed T
    assert (data.mask.sum(axis=1) == data.lengths).all()
    assert data.lengths.max() == T
    # padded region series are zero beyond their genuine length
    for m in range(M):
        assert not data.deaths[m, data.lengths[m]:].any()


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
