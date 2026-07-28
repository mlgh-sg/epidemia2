"""Sampler diagnostics for a fitted model.

The mirror of R's :func:`sampler_diagnostics`. nutpie already records
everything needed in ``idata.sample_stats`` -- ``diverging``,
``maxdepth_reached`` and ``energy`` -- but reading it means knowing which
variable carries which quantity, and nutpie's names differ from PyMC's. This
module summarises them the same way the R package does, so the two ports can be
checked against each other without translating between conventions.
"""

from __future__ import annotations

import warnings

import numpy as np

__all__ = ["sampler_diagnostics", "SamplerDiagnostics"]


# nutpie and PyMC disagree on what to call these; accept either.
_DIVERGING = ("diverging", "divergent")
_TREEDEPTH = ("maxdepth_reached", "reached_max_treedepth")


def _first_present(ds, names):
    for n in names:
        if n in ds.data_vars:
            return ds[n]
    return None


def _ebfmi(energy):
    """E-BFMI per chain.

    The estimator is ``mean(diff(E)^2) / var(E)`` within a chain. ArviZ exposes
    this as ``az.bfmi``, but computing it here keeps the result per-chain and
    avoids depending on which ArviZ version is installed.
    """
    e = np.asarray(energy, dtype=float)
    out = np.empty(e.shape[0])
    for c in range(e.shape[0]):
        ec = e[c]
        v = np.var(ec)
        out[c] = np.nan if v == 0 else np.mean(np.diff(ec) ** 2) / v
    return out


class SamplerDiagnostics:
    """HMC diagnostics for a fitted model, with a readable ``repr``.

    Attributes
    ----------
    per_chain : pandas.DataFrame
        One row per chain: ``chain``, ``divergent``, ``max_treedepth``, ``ebfmi``.
    draws_per_chain : int
        Post-warmup draws per chain.
    worst_rhat, worst_rhat_par : float, str
        The largest R-hat over all parameters, and which parameter it belongs to.
    min_ess_bulk, min_ess_bulk_par : float, str
        The smallest bulk effective sample size, and which parameter.
    min_ess_tail : float
        The smallest tail effective sample size.
    problems : list of str
        Human-readable warnings; empty when nothing needs attention.
    """

    def __init__(self, per_chain, draws_per_chain, worst_rhat, worst_rhat_par,
                 min_ess_bulk, min_ess_bulk_par, min_ess_tail, problems):
        self.per_chain = per_chain
        self.draws_per_chain = draws_per_chain
        self.worst_rhat = worst_rhat
        self.worst_rhat_par = worst_rhat_par
        self.min_ess_bulk = min_ess_bulk
        self.min_ess_bulk_par = min_ess_bulk_par
        self.min_ess_tail = min_ess_tail
        self.problems = problems

    @property
    def divergences(self) -> int:
        """Total divergent transitions across all chains."""
        return int(self.per_chain["divergent"].sum())

    @property
    def max_treedepth_hits(self) -> int:
        """Total iterations that saturated the maximum tree depth."""
        return int(self.per_chain["max_treedepth"].sum())

    @property
    def ok(self) -> bool:
        """``True`` when nothing needs attention."""
        return not self.problems

    def __repr__(self):
        n = len(self.per_chain)
        total = n * self.draws_per_chain
        lines = ["Sampler diagnostics",
                 f"{n} chains x {self.draws_per_chain} post-warmup draws = {total}",
                 "",
                 self.per_chain.to_string(index=False),
                 ""]
        pct = (lambda k: f" ({100 * k / total:.1f}%)") if total else (lambda k: "")
        lines.append(f"Divergent transitions: {self.divergences}{pct(self.divergences)}")
        lines.append(f"Hit max treedepth:     {self.max_treedepth_hits}"
                     f"{pct(self.max_treedepth_hits)}")
        lines.append(f"Lowest E-BFMI:         {self.per_chain['ebfmi'].min():.2f}")
        if self.worst_rhat is not None and np.isfinite(self.worst_rhat):
            lines.append(f"Worst R-hat:           {self.worst_rhat:.3f}  "
                         f"({self.worst_rhat_par})")
            lines.append(f"Lowest bulk ESS:       {self.min_ess_bulk:.0f}  "
                         f"({self.min_ess_bulk_par})")
            lines.append(f"Lowest tail ESS:       {self.min_ess_tail:.0f}")
        if self.problems:
            lines.append("")
            lines.append("Warnings:")
            lines.extend(f"* {p}" for p in self.problems)
        else:
            lines.append("")
            lines.append("No problems detected.")
        return "\n".join(lines)


def sampler_diagnostics(idata, warn: bool = False) -> SamplerDiagnostics:
    """Summarise the sampler diagnostics of a fitted model.

    Reports divergent transitions, iterations that saturated the maximum tree
    depth, and E-BFMI per chain, together with the worst R-hat and lowest
    effective sample size over all parameters. This is the Python mirror of R's
    ``sampler_diagnostics()`` and reports the same quantities under the same
    names, so a model fitted in either language can be checked the same way.

    What the numbers mean:

    - **Divergent transitions** mean the sampler could not follow the
      posterior's curvature. Even a handful biases the result, and drawing more
      samples does not fix it -- raise ``target_accept`` or reparameterise.
    - **Max treedepth** costs efficiency rather than correctness: the sampler
      was still making progress when it hit the limit. Raise ``max_treedepth``.
    - **E-BFMI** below roughly 0.2 suggests momentum resampling is not exploring
      the energy distribution, often a sign of heavy tails.

    Parameters
    ----------
    idata : arviz.InferenceData
        A fit returned by :func:`epidemia.fit`, :func:`epidemia.fit_multilevel`
        or :func:`epidemia.fit_epidemia`.
    warn : bool, default False
        Also emit a :class:`UserWarning` for each problem found. Useful in
        scripts that should not silently ignore a bad fit.

    Returns
    -------
    SamplerDiagnostics
        Print it, or read ``.divergences``, ``.max_treedepth_hits``,
        ``.per_chain`` and ``.ok`` directly.

    Raises
    ------
    ValueError
        If ``idata`` carries no ``sample_stats`` group -- which is the case for
        variational fits, since ADVI has no NUTS diagnostics.

    Examples
    --------
    >>> d = sampler_diagnostics(idata)   # doctest: +SKIP
    >>> d.ok                             # doctest: +SKIP
    True
    >>> d.divergences                    # doctest: +SKIP
    0
    """
    import pandas as pd

    if "sample_stats" not in idata.groups():
        raise ValueError(
            "this fit carries no 'sample_stats' group, so it has no NUTS "
            "diagnostics. Variational fits (fit_variational) never do -- ADVI "
            "is not a Hamiltonian sampler. Refit with NUTS to check convergence."
        )

    ss = idata.sample_stats
    div = _first_present(ss, _DIVERGING)
    td = _first_present(ss, _TREEDEPTH)
    if div is None:
        raise ValueError(
            "'sample_stats' has no divergence variable (looked for "
            f"{' or '.join(_DIVERGING)}); got {sorted(ss.data_vars)}"
        )

    div_a = np.asarray(div, dtype=bool)                    # (chain, draw)
    nchain, ndraw = div_a.shape
    td_a = (np.asarray(td, dtype=bool) if td is not None
            else np.zeros_like(div_a, dtype=bool))
    ebfmi = (_ebfmi(ss["energy"]) if "energy" in ss.data_vars
             else np.full(nchain, np.nan))

    per_chain = pd.DataFrame({
        "chain": np.arange(1, nchain + 1),
        "divergent": div_a.sum(axis=1).astype(int),
        "max_treedepth": td_a.sum(axis=1).astype(int),
        "ebfmi": np.round(ebfmi, 3),
    })

    # R-hat / ESS over the posterior, matching what R reports.
    worst_rhat = worst_rhat_par = None
    min_ess_bulk = min_ess_bulk_par = min_ess_tail = None
    try:
        import arviz as az
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            summ = az.summary(idata, kind="diagnostics")
        if len(summ):
            worst_rhat = float(summ["r_hat"].max())
            worst_rhat_par = str(summ["r_hat"].idxmax())
            min_ess_bulk = float(summ["ess_bulk"].min())
            min_ess_bulk_par = str(summ["ess_bulk"].idxmin())
            min_ess_tail = float(summ["ess_tail"].min())
    except Exception:  # pragma: no cover - arviz summary is best-effort
        pass

    total = nchain * ndraw
    problems = []
    ndiv = int(per_chain["divergent"].sum())
    ntd = int(per_chain["max_treedepth"].sum())
    if ndiv:
        problems.append(
            f"{ndiv} divergent transition{'' if ndiv == 1 else 's'}. These bias "
            f"the posterior and more draws will not help; raise target_accept "
            f"above its current value or reparameterise."
        )
    if ntd:
        problems.append(
            f"{ntd} iteration{'' if ntd == 1 else 's'} saturated max_treedepth. "
            f"This costs efficiency rather than correctness; raise max_treedepth."
        )
    if np.isfinite(ebfmi).any() and np.nanmin(ebfmi) < 0.2:
        problems.append(
            f"E-BFMI of {np.nanmin(ebfmi):.2f} is below 0.2, suggesting the "
            f"sampler is not exploring the energy distribution well."
        )
    # Compare the value as printed. Warning that "1.010 exceeds 1.01" because the
    # unrounded value is 1.0104 reads as a contradiction and trains people to
    # ignore the warning.
    if worst_rhat is not None and round(worst_rhat, 3) > 1.01:
        problems.append(
            f"R-hat of {worst_rhat:.3f} for {worst_rhat_par} exceeds 1.01, so "
            f"the chains have not mixed."
        )
    if min_ess_bulk is not None and min_ess_bulk < 100 * nchain:
        problems.append(
            f"Bulk ESS of {min_ess_bulk:.0f} for {min_ess_bulk_par} is "
            f"{min_ess_bulk / nchain:.0f} per chain, below the 100 per chain that "
            f"keeps posterior summaries stable."
        )

    if warn:
        for p in problems:
            warnings.warn(p, UserWarning, stacklevel=2)

    del total
    return SamplerDiagnostics(per_chain, ndraw, worst_rhat, worst_rhat_par,
                              min_ess_bulk, min_ess_bulk_par, min_ess_tail,
                              problems)
