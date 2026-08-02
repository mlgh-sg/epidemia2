#!/usr/bin/env python
"""Head-to-head benchmark: the R (CmdStanR) multilevel fit vs the Python port.

Runs the *same* model -- the eleven-country Europe/COVID multilevel model of
the multilevel NPI model that the europe-covid vignette used to carry --
through three engines and records wall-clock,
sampler diagnostics and effective sample size, so the comparison is on
efficiency rather than raw speed:

  R / cmdstanr        Stan's NUTS, diagonal metric
  Python / nutpie     diag adaptation      (Fisher-information diagonal)
  Python / nutpie     low_rank adaptation  (recommended for correlated covariates)

Usage
-----
    uv run --project python python benchmarks/run.py --quick       # pilot
    uv run --project python python benchmarks/run.py --draws 500 --tune 500

Everything is written to ``--out`` (a temp directory by default). Nothing is
left behind in the working tree: this repository is for testing the packages,
not for storing fits.

Timings are only meaningful on an otherwise idle machine -- the engines are run
strictly one after another, never concurrently.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
NPIS = [
    "public_events",
    "schools_universities",
    "self_isolating_if_ill",
    "social_distancing_encouraged",
    "lockdown",
]


def machine() -> dict:
    import multiprocessing

    return {
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "cpu_count": multiprocessing.cpu_count(),
        "python": platform.python_version(),
    }


def versions() -> dict:
    import importlib.metadata as md

    out = {}
    for pkg in ("pymc", "nutpie", "numba", "numpy", "pytensor", "arviz"):
        try:
            out[pkg] = md.version(pkg)
        except Exception:
            out[pkg] = None
    return out


def run_r(draws: int, tune: int, chains: int, seed: int, out_dir: Path) -> dict:
    """Invoke benchmarks/bench_r.R and read the JSON it writes."""
    json_path = out_dir / "bench_r.json"
    cmd = [
        "Rscript",
        str(REPO / "benchmarks" / "bench_r.R"),
        "--draws", str(draws),
        "--tune", str(tune),
        "--chains", str(chains),
        "--seed", str(seed),
        "--json", str(json_path),
    ]
    print(f"[R] {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    if proc.returncode != 0 or not json_path.exists():
        print(proc.stderr[-4000:], file=sys.stderr)
        raise SystemExit(f"R benchmark failed (exit {proc.returncode})")
    return json.loads(json_path.read_text())


def run_python(adaptation: str, draws: int, tune: int, chains: int,
               seed: int) -> dict:
    """Fit the same model with nutpie under one adaptation scheme."""
    import arviz as az
    import numpy as np

    import epidemia
    from epidemia.multilevel import (
        MultilevelConfig,
        build_multilevel_model,
        prepare_panel,
    )

    ec = epidemia.europe_covid2()
    data = prepare_panel(
        ec.data, NPIS, response="deaths", group="country", date="date",
        seed_offset=30, death_threshold=10, fit_until="2020-05-05",
    )
    config = MultilevelConfig(gen=ec.si, i2o=ec.inf2death)

    # Compilation is per-run for nutpie (numba JIT), unlike Stan's cached
    # executable, so it is timed separately and reported as its own column.
    model = build_multilevel_model(data, config)
    import nutpie

    t0 = time.perf_counter()
    compiled = nutpie.compile_pymc_model(model, backend="numba")
    t_compile = time.perf_counter() - t0

    t0 = time.perf_counter()
    idata = nutpie.sample(
        compiled,
        draws=draws, tune=tune, chains=chains, seed=seed,
        target_accept=0.95, adaptation=adaptation,
        progress_bar=False,
    )
    t_sample = time.perf_counter() - t0

    summ = az.summary(idata, var_names=["beta"], round_to=None)
    diverging = int(np.asarray(idata.sample_stats["diverging"]).sum())

    effects = {}
    for name, row in zip(NPIS, summ.itertuples()):
        effects[name] = {
            "mean": float(row.mean),
            "sd": float(row.sd),
            "ess_bulk": float(row.ess_bulk),
        }

    return {
        "engine": f"Python / nutpie ({adaptation})",
        "backend": "numba",
        "adaptation": adaptation,
        "draws": draws, "tune": tune, "chains": chains,
        "compile_seconds": t_compile,
        "sample_seconds": t_sample,
        "divergences": diverging,
        "max_treedepth_hits": None,
        "max_rhat": float(summ["r_hat"].max()),
        "min_ess_bulk": float(summ["ess_bulk"].min()),
        "median_ess_bulk": float(summ["ess_bulk"].median()),
        "effects": effects,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--draws", type=int, default=500)
    ap.add_argument("--tune", type=int, default=500)
    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--quick", action="store_true",
                    help="tiny pilot run, to estimate cost before committing to a full one")
    ap.add_argument("--skip-r", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    if args.quick:
        args.draws, args.tune, args.chains = 100, 100, 2

    out_dir = args.out or Path(os.environ.get("TMPDIR", "/tmp")) / "epidemia-bench"
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    # Strictly sequential: two engines running at once would compete for cores
    # and neither timing would mean anything.
    if not args.skip_r:
        results.append(run_r(args.draws, args.tune, args.chains, args.seed, out_dir))
    for adaptation in ("diag", "low_rank"):
        print(f"[python] nutpie adaptation={adaptation}", flush=True)
        results.append(
            run_python(adaptation, args.draws, args.tune, args.chains, args.seed)
        )

    payload = {
        "machine": machine(),
        "versions": versions(),
        "config": {
            "draws": args.draws, "tune": args.tune,
            "chains": args.chains, "seed": args.seed,
            "model": "Europe/COVID multilevel, 11 countries, 5 NPIs, deaths",
        },
        "results": results,
    }
    dest = out_dir / "benchmark.json"
    dest.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {dest}")

    # Console summary
    hdr = f"{'engine':32s} {'compile':>9s} {'sample':>9s} {'minESS':>8s} {'ESS/s':>8s} {'div':>5s}"
    print("\n" + hdr)
    print("-" * len(hdr))
    for r in results:
        ess_per_s = (r["min_ess_bulk"] / r["sample_seconds"]
                     if r["sample_seconds"] else float("nan"))
        print(f"{r['engine']:32s} {r['compile_seconds']:9.1f} "
              f"{r['sample_seconds']:9.1f} {r['min_ess_bulk']:8.0f} "
              f"{ess_per_s:8.2f} {str(r['divergences']):>5s}")


if __name__ == "__main__":
    main()
