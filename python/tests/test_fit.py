"""End-to-end smoke test: a tiny fit produces a well-shaped InferenceData."""

import numpy as np
import pytest

import epidemia as epi


def _tiny_problem():
    d = epi.flu1918()
    y = np.concatenate([[np.nan], d.incidence[:30]]).astype(float)
    config = epi.EpiConfig(
        gen=d.generation, i2o=np.repeat(0.25, 4), seed_days=6,
        link="log", family="poisson", rw_prior_scale=0.1,
        intercept_loc=float(np.log(2.0)), intercept_scale=0.2,
    )
    return y, config


@pytest.mark.parametrize("sampler", ["numpyro", "blackjax"])
def test_fit_runs_and_exposes_latent_series(sampler):
    y, config = _tiny_problem()
    idata = epi.fit(y, config, sampler=sampler, draws=80, tune=80, chains=2, seed=1)
    N = y.shape[0]
    for var in ("Rt", "infections", "E_obs"):
        assert var in idata.posterior, f"missing {var}"
        assert idata.posterior[var].shape[-1] == N
    # sensible reproduction numbers for an epidemic that clearly grows
    rt = np.asarray(idata.posterior["Rt"]).reshape(-1, N)
    assert np.isfinite(rt).all()
    assert np.median(rt) > 0.5
