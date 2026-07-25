"""Variational inference (ADVI) on a tiny conjugate-ish model.

Deliberately NOT an epidemia model: a two-parameter normal-mean/scale fit on 50
simulated points runs in well under a second, so these tests exercise the
``fit_variational`` plumbing (algorithm mapping, InferenceData shape, ELBO
group, convergence warning) without paying for a renewal-model fit.
"""

import warnings

import numpy as np
import pytest

from epidemia.variational import (
    _resolve_algorithm,
    elbo_relative_improvement,
    fit_variational,
)

TRUE_MU, TRUE_SIGMA, N_OBS = 3.0, 2.0, 50


def _tiny_model():
    """``y ~ Normal(mu, sigma)`` with weak priors; returns (model, y)."""
    import pymc as pm

    y = np.random.default_rng(0).normal(TRUE_MU, TRUE_SIGMA, N_OBS)
    with pm.Model() as model:
        mu = pm.Normal("mu", 0.0, 10.0)
        sigma = pm.HalfNormal("sigma", 5.0)
        pm.Deterministic("mu_doubled", mu * 2)  # deterministics must survive
        pm.Normal("y", mu, sigma, observed=y)
    return model, y


def _fit(**kw):
    """Fit, muting the unconditional 'VB understates uncertainty' warning."""
    model, y = _tiny_model()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        idata = fit_variational(model, iter=30000, draws=500, seed=1,
                                progress_bar=False, **kw)
    return idata, y


@pytest.mark.parametrize("algorithm", ["meanfield", "fullrank"])
def test_both_algorithms_recover_the_truth(algorithm):
    import arviz as az

    idata, y = _fit(algorithm=algorithm)
    assert isinstance(idata, az.InferenceData)
    assert idata.posterior["mu"].shape == (1, 500)

    # ADVI's *mean* is reliable even though its spread is not; compare against
    # the sample mean/sd, which is what the likelihood actually points at.
    mu = float(idata.posterior["mu"].mean())
    sigma = float(idata.posterior["sigma"].mean())
    assert mu == pytest.approx(y.mean(), abs=0.4), f"{algorithm}: mu off"
    assert sigma == pytest.approx(y.std(), rel=0.3), f"{algorithm}: sigma off"
    assert abs(mu - TRUE_MU) < 1.0


def test_deterministics_are_in_the_posterior():
    """Downstream code reads Rt/infections, which are Deterministics."""
    idata, _ = _fit(algorithm="meanfield")
    np.testing.assert_allclose(
        np.asarray(idata.posterior["mu_doubled"]),
        2 * np.asarray(idata.posterior["mu"]),
    )


def test_elbo_history_is_returned_and_improves():
    idata, _ = _fit(algorithm="meanfield")
    assert "elbo" in idata.groups()
    elbo = np.asarray(idata.elbo["elbo"])
    assert elbo.shape == (30000,)
    assert np.isfinite(elbo).all()
    assert elbo[-1000:].mean() > elbo[:1000].mean()  # ELBO is maximised
    assert idata.attrs["inference_method"] == "variational"
    assert idata.attrs["algorithm"] == "meanfield"
    assert idata.attrs["elbo_converged"] is True


@pytest.mark.parametrize("bad", ["nuts", "sampling", "MEANFIELD", "", None, 3])
def test_bad_algorithm_raises_with_the_valid_names(bad):
    with pytest.raises(ValueError, match="meanfield.*fullrank"):
        _resolve_algorithm(bad)


def test_algorithm_names_map_onto_pymc_methods():
    assert _resolve_algorithm("meanfield") == "advi"
    assert _resolve_algorithm("fullrank") == "fullrank_advi"
    # the PyMC spellings are accepted too
    assert _resolve_algorithm("advi") == "advi"
    assert _resolve_algorithm("fullrank_advi") == "fullrank_advi"


def test_it_always_warns_that_vb_understates_uncertainty():
    model, _ = _tiny_model()
    with pytest.warns(UserWarning, match="understates posterior uncertainty"):
        fit_variational(model, algorithm="meanfield", iter=2000, draws=20,
                        seed=1, progress_bar=False)


def test_non_convergence_warns_when_iter_is_absurdly_low():
    model, _ = _tiny_model()
    with pytest.warns(UserWarning, match="maximum number of iterations"):
        fit_variational(model, algorithm="meanfield", iter=200, draws=20,
                        seed=1, progress_bar=False)


def test_plateaued_elbo_does_not_warn():
    """The plateau check must not cry wolf on a converged run."""
    elbo = np.concatenate([np.linspace(-1000, -100, 500), np.full(500, -100.0)])
    assert elbo_relative_improvement(elbo) == pytest.approx(0.0, abs=1e-12)

    from epidemia.variational import _warn_if_not_plateaued

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning here fails the test
        _warn_if_not_plateaued(elbo, iter=1000, rtol=0.01)


def test_relative_improvement_is_positive_while_climbing():
    climbing = np.linspace(-1000.0, -500.0, 1000)
    assert elbo_relative_improvement(climbing) > 0.01
    # ...and is scale-safe when the ELBO passes through zero
    crossing = np.linspace(-1.0, 1.0, 1000)
    assert np.isfinite(elbo_relative_improvement(crossing))
    # too short to split into two blocks -> undefined, not a spurious "converged"
    assert np.isnan(elbo_relative_improvement(np.array([1.0])))


def test_attrs_record_non_convergence():
    model, _ = _tiny_model()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        idata = fit_variational(model, algorithm="meanfield", iter=200, draws=20,
                                seed=1, progress_bar=False)
    assert idata.attrs["elbo_converged"] is False
    assert idata.attrs["elbo_rel_improvement"] > 0.01
    assert idata.attrs["iter"] == 200
