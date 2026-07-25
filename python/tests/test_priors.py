"""Prior specifications: R parity of names/defaults, and what build() produces.

Building random variables is cheap, so these tests declare models freely -- but
they never sample a posterior. Support (e.g. truncation) is checked with logp
rather than draws, which is both exact and free.
"""

import numpy as np
import pymc as pm
import pytest

from epidemia import priors as P
from epidemia.priors import (
    OK_AUX_DISTS,
    OK_COV_DISTS,
    OK_DISTS,
    OK_INT_DISTS,
    build,
    build_covariance,
    cauchy,
    decov,
    exponential,
    hexp,
    laplace,
    lkj,
    normal,
    resolve,
    shifted_gamma,
    student_t,
)


# --------------------------------------------------------------------------
# construction: names, defaults, stored hyperparameters
# --------------------------------------------------------------------------


def test_dist_names_mirror_r():
    # R stores the Student-t family as "t" and shifted_gamma as "gamma".
    assert normal().dist == "normal"
    assert student_t().dist == "t"
    assert cauchy().dist == "cauchy"
    assert exponential().dist == "exponential"
    assert laplace().dist == "laplace"
    assert shifted_gamma().dist == "gamma"
    assert hexp().dist == "hexp"
    assert decov().dist == "decov"
    assert lkj().dist == "lkj"


def test_defaults_match_the_r_constructors():
    assert normal().params() == {"location": 0.0, "scale": 1.0}
    assert student_t().params() == {"df": 1.0, "location": 0.0, "scale": 1.0}
    assert cauchy().params() == {"location": 0.0, "scale": 1.0}
    assert laplace().params() == {"location": 0.0, "scale": 1.0}
    assert exponential().params() == {"rate": 1.0}
    assert shifted_gamma().params() == {"shape": 1.0, "scale": 1.0, "shift": 0.0}
    assert decov().params() == {"regularization": 1.0, "concentration": 1.0,
                                "shape": 1.0, "scale": 1.0}
    assert lkj().params() == {"regularization": 1.0, "scale": 10.0, "df": 1.0}


def test_hyperparameters_are_stored_verbatim():
    p = student_t(df=3, location=-1.5, scale=2.5)
    assert (p.df, p.location, p.scale) == (3, -1.5, 2.5)
    g = shifted_gamma(shape=1 / 6, scale=1.0, shift=0.1)
    assert (g.shape, g.scale, g.shift) == (1 / 6, 1.0, 0.1)


def test_exponential_stores_the_rate_and_reports_r_s_scale():
    """R's exponential() records scale = 1 / rate; keep both accessible."""
    p = exponential(rate=0.03)
    assert p.rate == 0.03
    assert p.scale == pytest.approx(1 / 0.03)


def test_hexp_default_prior_aux_is_exponential_point_oh_three():
    p = hexp()
    assert p.prior_aux.dist == "exponential"
    assert p.prior_aux.rate == 0.03
    # and a user-supplied aux is kept as given
    assert hexp(prior_aux=normal(10, 5)).prior_aux == normal(10, 5)


def test_specs_are_immutable_and_comparable():
    assert normal(0, 2) == normal(0, 2)
    assert normal(0, 2) != normal(0, 3)
    with pytest.raises(Exception):  # frozen dataclass
        normal().scale = 3.0


@pytest.mark.parametrize("bad", [lambda: normal(scale=0), lambda: normal(scale=-1),
                                 lambda: student_t(df=0), lambda: cauchy(scale=-2),
                                 lambda: exponential(rate=0),
                                 lambda: laplace(scale=0),
                                 lambda: shifted_gamma(shape=-1),
                                 lambda: decov(concentration=0),
                                 lambda: lkj(df=-1)])
def test_non_positive_hyperparameters_are_rejected(bad):
    """Mirrors R's validate_parameter_value()."""
    with pytest.raises(ValueError, match="positive"):
        bad()


def test_hexp_rejects_an_aux_family_the_model_cannot_use():
    # R: check_in_set(prior_aux$dist, ok_aux_dists)
    with pytest.raises(ValueError, match="not supported"):
        hexp(prior_aux=shifted_gamma())


# --------------------------------------------------------------------------
# build(): the right family, shape and support
# --------------------------------------------------------------------------


@pytest.mark.parametrize("spec,op_name", [
    (normal(0, 2), "normal"),
    (student_t(3, 1, 2), "t"),
    (cauchy(0, 5), "cauchy"),
    (exponential(0.5), "exponential"),
    (laplace(0, 1), "laplace"),
])
def test_build_produces_the_right_pymc_family(spec, op_name):
    with pm.Model():
        rv = build(spec, "x")
    assert rv.owner.op.name == op_name
    assert rv.name == "x"


def test_build_passes_the_hyperparameters_through():
    with pm.Model() as m:
        build(normal(3.0, 0.25), "x")
    # logp of a Normal(3, 0.25) at its mean
    expected = -np.log(0.25 * np.sqrt(2 * np.pi))
    assert float(m.compile_logp()({"x": 3.0})) == pytest.approx(expected)


@pytest.mark.parametrize("shape,expected", [(None, ()), (4, (4,)), ((2, 3), (2, 3))])
def test_build_honours_shape(shape, expected):
    with pm.Model():
        rv = build(normal(), "x", shape=shape)
    assert tuple(rv.shape.eval()) == expected


def test_build_honours_dims():
    with pm.Model(coords={"region": ["IT", "FR", "DE"]}):
        rv = build(normal(), "x", dims="region")
    assert tuple(rv.shape.eval()) == (3,)


@pytest.mark.parametrize("spec", [normal(10, 5), student_t(3, 10, 5), cauchy(0, 5),
                                  laplace(0, 1)])
def test_positive_truncation(spec):
    """R's `real<lower=0> aux; aux ~ normal(10, 5);` is a truncated normal."""
    with pm.Model():
        free = build(spec, "free")
        trunc = build(spec, "trunc", positive=True)
    assert np.isfinite(pm.logp(free, -1.0).eval())
    assert pm.logp(trunc, -1.0).eval() == -np.inf
    assert np.isfinite(pm.logp(trunc, 1.0).eval())


def test_truncation_is_a_no_op_for_an_already_positive_family():
    with pm.Model():
        rv = build(exponential(0.5), "x", positive=True)
    assert rv.owner.op.name == "exponential"  # not wrapped in a Truncated


def test_truncation_works_with_a_shape():
    with pm.Model():
        rv = build(normal(10, 5), "x", shape=3, positive=True)
    assert tuple(rv.shape.eval()) == (3,)
    assert pm.logp(rv, np.array([-1.0, 1.0, 1.0])).eval()[0] == -np.inf


# --------------------------------------------------------------------------
# the two families that are built by transforming another variable
# --------------------------------------------------------------------------


def test_shifted_gamma_is_shift_minus_a_gamma():
    with pm.Model() as m:
        beta = shifted_gamma(shape=1 / 6, scale=1.0, shift=0.1).build("beta", shape=3)
    assert "beta_gamma" in m.named_vars          # the underlying gamma
    assert [v.name for v in m.free_RVs] == ["beta_gamma"]
    g, b = pm.draw([m["beta_gamma"], beta], draws=1, random_seed=0)
    np.testing.assert_allclose(b, 0.1 - g)
    assert np.all(b <= 0.1)                      # support is (-inf, shift]


def test_shifted_gamma_rejects_positive_truncation():
    with pm.Model():
        with pytest.raises(ValueError, match="shifted_gamma"):
            build(shifted_gamma(), "beta", positive=True)


def test_hexp_builds_tau_then_seeds_given_tau():
    with pm.Model() as m:
        seeds = hexp(prior_aux=exponential(0.03)).build("seeds", shape=4)
    assert [v.name for v in m.free_RVs] == ["seeds_tau", "seeds_raw"]
    assert tuple(seeds.shape.eval()) == (4,)
    # non-centred: seeds == tau * unit-exponential, so seeds | tau ~ Exp(mean=tau)
    tau, raw, s = pm.draw([m["seeds_tau"], m["seeds_raw"], seeds], draws=1,
                          random_seed=0)
    np.testing.assert_allclose(s, tau * raw)
    assert np.all(s > 0)


def test_hexp_aux_is_truncated_when_it_needs_to_be():
    """A normal aux is a *mean*, so it must be positive -- as in R's Stan."""
    with pm.Model() as m:
        hexp(prior_aux=normal(10, 5)).build("seeds")
    assert pm.logp(m["seeds_tau"], -1.0).eval() == -np.inf


# --------------------------------------------------------------------------
# covariance priors
# --------------------------------------------------------------------------


@pytest.mark.parametrize("spec", [decov(), lkj()])
def test_covariance_priors_refuse_to_build_a_scalar_rv(spec):
    with pm.Model():
        with pytest.raises(ValueError, match="build_covariance"):
            build(spec, "cov")


def test_covariance_params_are_exposed_for_the_caller():
    assert decov(2, 3, 4, 5).covariance_params() == {
        "regularization": 2, "concentration": 3, "shape": 4, "scale": 5}
    assert lkj(2, 3, 4).covariance_params() == {
        "regularization": 2, "scale": 3, "df": 4}


@pytest.mark.parametrize("spec", [decov(), lkj()])
def test_build_covariance_returns_a_lower_triangular_cholesky_factor(spec):
    with pm.Model() as m:
        L = build_covariance(spec, "cov", 3)
    assert tuple(L.shape.eval()) == (3, 3)
    assert "cov_corr" in m.named_vars
    draw = pm.draw(L, draws=1, random_seed=0)
    assert np.allclose(np.triu(draw, k=1), 0.0)          # lower triangular
    assert np.all(np.diag(draw) > 0)                     # positive scales
    # L L' is a valid covariance: symmetric with unit-correlation structure
    cov = draw @ draw.T
    np.testing.assert_allclose(cov, cov.T, atol=1e-12)


def test_build_covariance_of_a_single_effect_needs_no_correlations():
    """With one group-specific term, decov reduces to Gamma(shape, scale)."""
    with pm.Model() as m:
        L = build_covariance(decov(shape=2.0, scale=0.25), "cov", 1)
    assert tuple(L.shape.eval()) == (1, 1)
    assert [v.name for v in m.free_RVs] == ["cov_tau"]   # no Dirichlet, no LKJ
    assert m["cov_tau"].owner.op.name == "gamma"
    tau, draw = pm.draw([m["cov_tau"], L], draws=1, random_seed=0)
    np.testing.assert_allclose(draw, [[tau]])


def test_build_covariance_rejects_a_non_covariance_family():
    with pm.Model():
        with pytest.raises(ValueError, match="not supported"):
            build_covariance(normal(), "cov", 2)


# --------------------------------------------------------------------------
# resolve()
# --------------------------------------------------------------------------


def test_resolve_passes_a_spec_through_and_falls_back_to_the_default():
    assert resolve(normal(0, 3), normal(0, 1)) == normal(0, 3)
    assert resolve(None, normal(0, 1)) == normal(0, 1)


def test_resolve_accepts_a_family_name():
    assert resolve("student_t") == student_t()
    assert resolve("t") == student_t()


def test_resolve_without_a_spec_or_default_is_an_error():
    with pytest.raises(ValueError, match="no prior given"):
        resolve(None, what="prior_intercept")


def test_resolve_rejects_an_unknown_family():
    with pytest.raises(ValueError, match="unknown prior family"):
        resolve("horseshoe")


def test_resolve_rejects_a_non_prior():
    with pytest.raises(TypeError, match="prior specification"):
        resolve(0.5)


def test_resolve_enforces_the_family_set_fixed_by_the_model():
    # R: check_in_set(prior_intercept$dist, ok_int_dists)
    assert resolve(cauchy(), allowed=OK_INT_DISTS).dist == "cauchy"
    with pytest.raises(ValueError, match="not supported"):
        resolve(shifted_gamma(), allowed=OK_INT_DISTS, what="prior_intercept")
    with pytest.raises(ValueError, match="not supported"):
        resolve(laplace(), allowed=OK_AUX_DISTS, what="prior_aux")


def test_family_sets_match_r():
    """These mirror ok_*_dists in R/utilities.R (minus the shrinkage families)."""
    assert OK_INT_DISTS == {"normal", "t", "cauchy"}
    assert OK_AUX_DISTS == {"normal", "t", "cauchy", "exponential"}
    assert OK_COV_DISTS == {"decov"}                      # lkj is not usable, as in R
    assert {"normal", "t", "cauchy", "laplace", "gamma", "hexp"} <= OK_DISTS


def test_every_constructor_reports_a_family_the_module_can_look_up():
    """resolve(spec.dist) must round-trip, or an error message would lie."""
    for spec in [normal(), student_t(), cauchy(), exponential(), laplace(),
                 shifted_gamma(), hexp(), decov(), lkj()]:
        assert resolve(spec.dist).dist == spec.dist


def test_build_accepts_a_family_name_directly():
    with pm.Model():
        rv = build("normal", "x")
    assert rv.owner.op.name == "normal"


def test_public_names_are_all_exported():
    """__all__ is the contract the package namespace re-exports."""
    for name in P.__all__:
        assert hasattr(P, name), name
