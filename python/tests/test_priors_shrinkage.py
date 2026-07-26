"""Shrinkage prior families (hs, hs_plus, lasso, product_normal) and autoscaling.

Same discipline as tests/test_priors.py: building a graph is cheap, so these
tests declare models freely, but they never sample a posterior. Where a
distributional claim has to be checked, it is checked with ``pm.logp`` or a
handful of forward draws (which cost nothing), not with MCMC.
"""

import numpy as np
import pymc as pm
import pytest

from epidemia import priors as P
from epidemia.priors import (
    AUTOSCALE_DISTS,
    MIN_PRIOR_SCALE,
    OK_AUX_DISTS,
    OK_DISTS,
    OK_INT_DISTS,
    autoscale,
    build,
    cauchy,
    decov,
    exponential,
    hexp,
    hs,
    hs_plus,
    laplace,
    lasso,
    lkj,
    normal,
    predictor_scale,
    product_normal,
    resolve,
    shifted_gamma,
    student_t,
)

ALL_SHRINKAGE = [hs(), hs_plus(), lasso(), product_normal()]


# --------------------------------------------------------------------------
# construction: names, defaults, stored hyperparameters
# --------------------------------------------------------------------------


def test_dist_names_mirror_r():
    assert hs().dist == "hs"
    assert hs_plus().dist == "hs_plus"
    assert lasso().dist == "lasso"
    assert product_normal().dist == "product_normal"


def test_defaults_match_the_r_constructors():
    # R: hs(df = 1, global_df = 1, global_scale = 0.01, slab_df = 4,
    #       slab_scale = 2.5)
    assert hs().params() == {"df": 1.0, "global_df": 1.0, "global_scale": 0.01,
                             "slab_df": 4.0, "slab_scale": 2.5}
    # R: hs_plus(df1 = 1, df2 = 1, global_df = 1, global_scale = 0.01,
    #            slab_df = 4, slab_scale = 2.5)
    assert hs_plus().params() == {"df1": 1.0, "df2": 1.0, "global_df": 1.0,
                                  "global_scale": 0.01, "slab_df": 4.0,
                                  "slab_scale": 2.5}
    assert lasso().params() == {"df": 1.0, "location": 0.0, "scale": 2.5}
    assert product_normal().params() == {"num_terms": 2, "location": 0.0,
                                         "scale": 1.0}


def test_hyperparameters_are_stored_verbatim():
    p = hs(df=3, global_df=2, global_scale=0.05, slab_df=5, slab_scale=1.5)
    assert (p.df, p.global_df, p.global_scale) == (3, 2, 0.05)
    assert (p.slab_df, p.slab_scale) == (5, 1.5)
    q = hs_plus(df1=2, df2=3)
    assert (q.df1, q.df2) == (2, 3)
    assert lasso(df=2, location=-1.0, scale=0.5).params() == {
        "df": 2, "location": -1.0, "scale": 0.5}


def test_the_r_slot_names_are_mirrored_back():
    """R packs these into generic slots; keep those readable for R users."""
    # R: hs() records location = 0, scale = 1 (fixed by the family).
    assert (hs().location, hs().scale) == (0.0, 1.0)
    # R: hs_plus() records df = df1 and (mis)uses the scale slot for df2.
    p = hs_plus(df1=2, df2=7)
    assert (p.df, p.scale, p.location) == (2, 7, 0.0)
    # R: product_normal()'s first argument is called df.
    assert product_normal(num_terms=3).df == 3


def test_specs_are_immutable_and_comparable():
    assert hs(df=2) == hs(df=2)
    assert hs(df=2) != hs(df=3)
    assert lasso() != product_normal()
    with pytest.raises(Exception):  # frozen dataclass
        hs().slab_scale = 1.0


@pytest.mark.parametrize("bad", [
    lambda: hs(df=0), lambda: hs(df=-1), lambda: hs(global_df=0),
    lambda: hs(global_scale=0), lambda: hs(global_scale=-0.1),
    lambda: hs(slab_df=-4), lambda: hs(slab_scale=0),
    lambda: hs_plus(df1=0), lambda: hs_plus(df2=-1),
    lambda: hs_plus(global_scale=0), lambda: hs_plus(slab_df=0),
    lambda: hs_plus(slab_scale=-2.5),
    lambda: lasso(df=0), lambda: lasso(scale=0), lambda: lasso(scale=-1),
    lambda: product_normal(scale=0), lambda: product_normal(num_terms=0),
    lambda: product_normal(num_terms=-2),
])
def test_non_positive_hyperparameters_are_rejected(bad):
    """Mirrors R's validate_parameter_value()."""
    with pytest.raises(ValueError):
        bad()


def test_num_terms_must_be_a_whole_number():
    # R: stopifnot(all(df >= 1), all(df == as.integer(df)))
    with pytest.raises(ValueError, match="integer"):
        product_normal(num_terms=2.5)
    assert product_normal(num_terms=1).num_terms == 1


def test_the_shrinkage_families_are_registered_with_the_module():
    # resolve(spec.dist) must round-trip, or an error message would lie
    for spec in ALL_SHRINKAGE:
        assert resolve(spec.dist).dist == spec.dist
    # R's ok_dists admits all four for coefficients ...
    assert {"hs", "hs_plus", "lasso", "product_normal"} <= OK_DISTS
    # ... and none of them anywhere else.
    for dist in ("hs", "hs_plus", "lasso", "product_normal"):
        assert dist not in OK_INT_DISTS
        assert dist not in OK_AUX_DISTS


def test_public_names_are_all_exported():
    for name in ["hs", "hs_plus", "lasso", "product_normal", "autoscale",
                 "predictor_scale", "AUTOSCALE_DISTS", "HorseshoePrior",
                 "HorseshoePlusPrior", "LassoPrior", "ProductNormalPrior"]:
        assert name in P.__all__ and hasattr(P, name), name


# --------------------------------------------------------------------------
# build(): sub-parameters, shape, dims
# --------------------------------------------------------------------------


@pytest.mark.parametrize("spec,subs", [
    (hs(), ["global", "local", "slab", "z"]),
    (hs_plus(), ["global", "local1", "local2", "slab", "z"]),
    (lasso(), ["global", "z"]),
    (product_normal(), ["z1", "z2"]),
])
def test_build_registers_the_expected_sub_parameters(spec, subs):
    with pm.Model() as m:
        beta = spec.build("beta", shape=5)
    assert [v.name for v in m.free_RVs] == [f"beta_{s}" for s in subs]
    # beta itself is a Deterministic, not a free RV: all four families are
    # written non-centred.
    assert "beta" in m.named_vars
    assert beta.name == "beta"
    assert "beta" not in [v.name for v in m.free_RVs]


@pytest.mark.parametrize("spec", ALL_SHRINKAGE)
@pytest.mark.parametrize("shape,expected", [(None, ()), (4, (4,)), ((2, 3), (2, 3))])
def test_build_honours_shape(spec, shape, expected):
    with pm.Model():
        beta = spec.build("beta", shape=shape)
    assert tuple(beta.shape.eval()) == expected


@pytest.mark.parametrize("spec", ALL_SHRINKAGE)
def test_build_honours_dims(spec):
    with pm.Model(coords={"npi": ["school", "work", "events"]}) as m:
        beta = spec.build("beta", dims="npi")
    assert tuple(beta.shape.eval()) == (3,)
    assert m.named_vars_to_dims["beta"] == ("npi",)


@pytest.mark.parametrize("spec", ALL_SHRINKAGE)
def test_build_accepts_the_family_name_too(spec):
    with pm.Model() as m:
        build(spec.dist, "beta", shape=2)
    assert "beta" in m.named_vars


@pytest.mark.parametrize("spec", ALL_SHRINKAGE)
def test_shrinkage_priors_refuse_positive_truncation(spec):
    with pm.Model():
        with pytest.raises(ValueError, match="positive=True"):
            spec.build("beta", positive=True)


@pytest.mark.parametrize("spec", ALL_SHRINKAGE)
def test_shrinkage_priors_have_no_single_pymc_distribution(spec):
    with pytest.raises(ValueError, match="build"):
        spec._pymc_dist()


# --------------------------------------------------------------------------
# build(): the constructions themselves
# --------------------------------------------------------------------------


def test_hs_sub_parameters_are_the_right_families_and_shapes():
    with pm.Model() as m:
        hs(df=3, global_df=2, global_scale=0.02, slab_df=5,
           slab_scale=1.5).build("beta", shape=4)
    ops = {v.name: v.owner.op.name for v in m.free_RVs}
    assert ops["beta_global"] == "halfstudentt"      # tau
    assert ops["beta_local"] == "halfstudentt"       # lambdas
    assert ops["beta_slab"] == "invgamma"            # c^2
    assert ops["beta_z"] == "normal"
    # global scale and slab are shared; the local scales are per-coefficient
    assert tuple(m["beta_global"].shape.eval()) == ()
    assert tuple(m["beta_slab"].shape.eval()) == ()
    assert tuple(m["beta_local"].shape.eval()) == (4,)


def test_hs_is_the_non_centred_regularised_horseshoe():
    """beta = z tau lambda_tilde, with lambda_tilde the slab-regularised local."""
    spec = hs(df=1, global_df=1, global_scale=0.01, slab_df=4, slab_scale=2.5)
    with pm.Model() as m:
        beta = spec.build("beta", shape=3)
    tau, lam, c2, z, b = pm.draw(
        [m["beta_global"], m["beta_local"], m["beta_slab"], m["beta_z"], beta],
        draws=1, random_seed=0)
    lam_t = np.sqrt(c2 * lam ** 2 / (c2 + tau ** 2 * lam ** 2))
    np.testing.assert_allclose(b, z * tau * lam_t)
    # the regularisation caps the local scale at the slab width
    assert np.all(tau * lam_t <= np.sqrt(c2) + 1e-12)


def test_hs_global_and_slab_use_the_hyperparameters_given():
    """Cheap exactness check: logp of the hyperpriors at a known point."""
    from scipy import stats

    spec = hs(global_df=2.0, global_scale=0.02, slab_df=5.0, slab_scale=1.5)
    with pm.Model() as m:
        spec.build("beta", shape=2)
    # tau ~ half-t(2, 0, 0.02): density is twice the t density
    logp_tau = pm.logp(m["beta_global"], 0.05).eval()
    expected = np.log(2) + stats.t.logpdf(0.05, df=2.0, scale=0.02)
    assert float(logp_tau) == pytest.approx(expected, rel=1e-6)
    # c^2 ~ InvGamma(slab_df/2, slab_df * slab_scale^2 / 2)
    logp_c2 = pm.logp(m["beta_slab"], 2.0).eval()
    expected = stats.invgamma.logpdf(2.0, a=2.5, scale=5.0 * 1.5 ** 2 / 2)
    assert float(logp_c2) == pytest.approx(expected, rel=1e-6)


def test_hs_local_scales_use_df():
    from scipy import stats

    with pm.Model() as m:
        hs(df=3.0).build("beta", shape=1)
    logp = float(pm.logp(m["beta_local"], np.array([0.7])).eval()[0])
    assert logp == pytest.approx(np.log(2) + stats.t.logpdf(0.7, df=3.0),
                                 rel=1e-6)


def test_hs_plus_has_two_nested_local_scales():
    spec = hs_plus(df1=1, df2=1)
    with pm.Model() as m:
        beta = spec.build("beta", shape=3)
    ops = {v.name: v.owner.op.name for v in m.free_RVs}
    assert ops["beta_local1"] == ops["beta_local2"] == "halfstudentt"
    tau, l1, l2, c2, z, b = pm.draw(
        [m["beta_global"], m["beta_local1"], m["beta_local2"], m["beta_slab"],
         m["beta_z"], beta], draws=1, random_seed=0)
    lam = l1 * l2
    lam_t = np.sqrt(c2 * lam ** 2 / (c2 + tau ** 2 * lam ** 2))
    np.testing.assert_allclose(b, z * tau * lam_t)


def test_hs_plus_local_dfs_are_not_swapped():
    from scipy import stats

    with pm.Model() as m:
        hs_plus(df1=3.0, df2=7.0).build("beta", shape=1)
    for var, df in [("beta_local1", 3.0), ("beta_local2", 7.0)]:
        logp = float(pm.logp(m[var], np.array([0.7])).eval()[0])
        assert logp == pytest.approx(np.log(2) + stats.t.logpdf(0.7, df=df),
                                     rel=1e-6)


def test_lasso_is_a_laplace_with_an_estimated_global_scale():
    spec = lasso(df=2.0, location=0.5, scale=0.25)
    with pm.Model() as m:
        beta = spec.build("beta", shape=3)
    ops = {v.name: v.owner.op.name for v in m.free_RVs}
    # PyMC implements ChiSquared as a Gamma, so the op reports "gamma"; that it
    # really is chi-square(df) is pinned down by the logp test below.
    assert ops["beta_global"] == "gamma"          # the estimated penalty
    assert ops["beta_z"] == "laplace"             # double-exponential
    g, z, b = pm.draw([m["beta_global"], m["beta_z"], beta], draws=1,
                      random_seed=0)
    np.testing.assert_allclose(b, 0.5 + 0.25 * g * z)
    # z is a *standard* Laplace; the scale lives in the deterministic
    assert float(pm.logp(m["beta_z"], np.zeros(3)).eval()[0]) == \
        pytest.approx(np.log(0.5))


def test_lasso_global_uses_df():
    from scipy import stats

    with pm.Model() as m:
        lasso(df=3.0).build("beta", shape=2)
    logp = float(pm.logp(m["beta_global"], 1.7).eval())
    assert logp == pytest.approx(stats.chi2.logpdf(1.7, df=3.0), rel=1e-6)


@pytest.mark.parametrize("k", [1, 2, 4])
def test_product_normal_is_a_product_of_k_standard_normals(k):
    spec = product_normal(num_terms=k, location=0.25, scale=2.0)
    with pm.Model() as m:
        beta = spec.build("beta", shape=3)
    names = [f"beta_z{i + 1}" for i in range(k)]
    assert [v.name for v in m.free_RVs] == names
    assert all(m[n].owner.op.name == "normal" for n in names)
    draws = pm.draw([m[n] for n in names] + [beta], draws=1, random_seed=0)
    zs, b = draws[:-1], draws[-1]
    # R's make_beta(): beta = prod(z) * scale^num_normals + location
    np.testing.assert_allclose(b, 0.25 + 2.0 ** k * np.prod(zs, axis=0))


def test_product_normal_with_one_term_is_just_a_normal():
    with pm.Model() as m:
        beta = product_normal(num_terms=1, location=1.0, scale=0.5).build("beta")
    z, b = pm.draw([m["beta_z1"], beta], draws=1, random_seed=0)
    np.testing.assert_allclose(b, 1.0 + 0.5 * z)


def test_shrinkage_priors_actually_shrink():
    """A sanity check on the prior itself: mass piled up at zero, heavy tails.

    Forward draws from the prior only -- no posterior, no sampler.
    """
    with pm.Model() as m:
        b_hs = hs().build("b_hs", shape=50)
        b_n = normal(0.0, 1.0).build("b_n", shape=50)
    draws_hs, draws_n = pm.draw([b_hs, b_n], draws=200, random_seed=1)
    # far more prior mass within 0.01 of zero than under a unit normal
    assert np.mean(np.abs(draws_hs) < 0.01) > np.mean(np.abs(draws_n) < 0.01)
    assert np.median(np.abs(draws_hs)) < np.median(np.abs(draws_n))
    assert set(m.named_vars) >= {"b_hs_global", "b_hs_local", "b_hs_slab"}


# --------------------------------------------------------------------------
# autoscale()
# --------------------------------------------------------------------------


@pytest.mark.parametrize("spec", [normal(0.0, 0.5), student_t(3, 0.0, 0.5),
                                  cauchy(0.0, 0.5), laplace(0.0, 0.5),
                                  lasso(1, 0.0, 0.5)])
def test_autoscale_divides_the_scale_by_the_predictor_scale(spec):
    """R: prior_scale <- prior_scale / apply(x, 2, x.scale)."""
    scaled = autoscale(spec, 2.0)
    assert type(scaled) is type(spec)
    assert scaled.scale == pytest.approx(0.25)
    # everything else is carried over untouched
    other = {k: v for k, v in spec.params().items() if k != "scale"}
    assert {k: v for k, v in scaled.params().items() if k != "scale"} == other
    # and the original is unchanged -- specs are frozen, autoscale copies
    assert spec.scale == 0.5


def test_autoscale_handles_a_vector_of_predictor_scales():
    """One scale per coefficient, as R gets from apply(xtemp, 2L, ...)."""
    scaled = autoscale(normal(0.0, 1.0), np.array([1.0, 2.0, 4.0]))
    np.testing.assert_allclose(scaled.scale, [1.0, 0.5, 0.25])
    assert scaled.location == 0.0


def test_autoscale_of_a_unit_scale_predictor_changes_nothing():
    assert autoscale(normal(0, 0.5), 1.0) == normal(0, 0.5)


def test_autoscale_floors_the_scale():
    """R's min_prior_scale: a huge predictor must not collapse the prior."""
    scaled = autoscale(normal(0.0, 1.0), 1e300)
    assert scaled.scale == MIN_PRIOR_SCALE


@pytest.mark.parametrize("spec", [exponential(0.03), shifted_gamma(1 / 6, 1, 0.1),
                                  hexp(), decov(), lkj(), hs(), hs_plus(),
                                  product_normal()])
def test_autoscale_is_a_no_op_outside_the_documented_families(spec):
    """R has no autoscale for these, or never divides their scale."""
    assert spec.dist not in AUTOSCALE_DISTS
    assert autoscale(spec, 4.0) is spec


def test_autoscale_families_match_what_r_autoscales():
    assert AUTOSCALE_DISTS == {"normal", "t", "cauchy", "laplace", "lasso"}


@pytest.mark.parametrize("bad", [0.0, -1.0, np.nan, np.inf,
                                 np.array([1.0, 0.0])])
def test_autoscale_rejects_a_non_positive_predictor_scale(bad):
    with pytest.raises(ValueError, match="positive"):
        autoscale(normal(0, 1), bad)


def test_autoscale_accepts_a_family_name():
    assert autoscale("normal", 4.0).scale == pytest.approx(0.25)


def test_autoscale_result_is_still_a_valid_spec_and_builds():
    with pm.Model():
        rv = build(autoscale(normal(0.0, 0.5), np.array([2.0, 4.0])), "beta",
                   shape=2)
    assert tuple(rv.shape.eval()) == (2,)
    # per-coefficient scales really did reach the distribution
    lp = pm.logp(rv, np.zeros(2)).eval()
    assert lp[0] == pytest.approx(-np.log(0.25 * np.sqrt(2 * np.pi)))
    assert lp[1] == pytest.approx(-np.log(0.125 * np.sqrt(2 * np.pi)))


# --------------------------------------------------------------------------
# predictor_scale(): R's rule, which is not simply sd()
# --------------------------------------------------------------------------


def test_predictor_scale_uses_the_range_for_a_binary_covariate():
    """R: num.categories == 2 -> diff(range(x)); an NPI indicator, typically."""
    x = np.array([0.0, 1.0, 1.0, 0.0, 1.0])
    assert predictor_scale(x) == pytest.approx(1.0)
    assert predictor_scale(np.array([2.0, 5.0, 5.0])) == pytest.approx(3.0)


def test_predictor_scale_uses_the_sample_sd_for_a_continuous_covariate():
    x = np.array([1.0, 2.0, 3.0, 4.0, 7.0])
    assert predictor_scale(x) == pytest.approx(np.std(x, ddof=1))  # R's sd()


def test_predictor_scale_of_a_constant_column_is_one():
    """Nothing to rescale, and it keeps autoscale from dividing by zero."""
    assert predictor_scale(np.ones(10)) == 1.0


def test_predictor_scale_works_column_by_column_on_a_design_matrix():
    x = np.column_stack([np.ones(5),                       # constant
                         [0.0, 1.0, 0.0, 1.0, 1.0],        # binary
                         [1.0, 2.0, 3.0, 4.0, 7.0]])       # continuous
    got = predictor_scale(x)
    np.testing.assert_allclose(
        got, [1.0, 1.0, np.std([1.0, 2, 3, 4, 7], ddof=1)])
    # which is exactly what autoscale() consumes
    scaled = autoscale(normal(0.0, 0.5), got)
    np.testing.assert_allclose(scaled.scale, 0.5 / got)


def test_predictor_scale_rejects_a_3d_array():
    with pytest.raises(ValueError, match="dimensional"):
        predictor_scale(np.zeros((2, 2, 2)))
