#!/usr/bin/env python
"""Precompute ("bake") the tutorial notebooks into the documentation site.

The Python counterpart of ``vignettes/precompute.R``, and it exists for the same
reason: the tutorials fit real Stan/PyMC models, and a documentation build should
not. Each tutorial is authored as ``notebooks/<name>.py`` (a jupytext percent
notebook, so it is also a runnable script); this script executes it once and
writes the rendered markdown and figures into ``docs/tutorials/``, which is what
mkdocs publishes.

    uv run --group dev python scripts/precompute.py                 # all
    uv run --group dev python scripts/precompute.py multilevel-multi-obs

Run it whenever the modelling code or a notebook changes, then commit the
regenerated markdown and images. Nothing is fitted at ``mkdocs build`` time.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOKS = ROOT / "notebooks"
OUT = ROOT / "docs" / "tutorials"

# Order matters only for the printed summary.
ALL_TUTORIALS = ["europe-covid", "partial-pooling", "multilevel-multi-obs"]

TITLES = {
    "europe-covid": "Assessing the effects of interventions",
    "partial-pooling": "Partial pooling",
    "multilevel-multi-obs": "Multilevel models with several observation series",
}


KERNEL = "epidemia-precompute"


@contextmanager
def project_kernel():
    """A throwaway kernelspec pointing at THIS interpreter.

    Without it, jupytext/nbconvert resolve the notebook's declared ``python3``
    kernel to whatever is registered globally -- which is not the project
    virtualenv, so the notebook fails on ``import plotnine``. Installing into a
    temporary prefix and pointing ``JUPYTER_PATH`` at it keeps the user's real
    kernel list untouched; the directory goes away on exit.
    """
    tmp = tempfile.mkdtemp(prefix="epidemia-kernel-")
    try:
        subprocess.run(
            [sys.executable, "-m", "ipykernel", "install",
             "--prefix", tmp, "--name", KERNEL],
            check=True, capture_output=True,
        )
        env = dict(os.environ)
        share = Path(tmp) / "share" / "jupyter"
        env["JUPYTER_PATH"] = (
            f"{share}{os.pathsep}{env['JUPYTER_PATH']}"
            if env.get("JUPYTER_PATH") else str(share)
        )
        yield env
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def bake(name: str, timeout: int, env: dict) -> bool:
    """Execute one notebook and write markdown + figures into docs/tutorials."""
    src = NOTEBOOKS / f"{name}.py"
    if not src.exists():
        print(f"  !! {src} does not exist", file=sys.stderr)
        return False

    OUT.mkdir(parents=True, exist_ok=True)
    ipynb = NOTEBOOKS / f"{name}.ipynb"

    print(f"== baking {name} ==", flush=True)
    started = time.time()
    try:
        # jupytext converts the percent script to a notebook and runs it; the
        # kernel is this project's interpreter, so the package under development
        # is what gets exercised.
        subprocess.run(
            ["jupytext", "--to", "ipynb", "--execute",
             "--set-kernel", KERNEL, "-o", str(ipynb), str(src)],
            cwd=ROOT, check=True, timeout=timeout, env=env,
        )
        # nbconvert writes <name>.md plus a <name>_files/ directory of figures.
        subprocess.run(
            ["jupyter", "nbconvert", "--to", "markdown",
             "--output-dir", str(OUT), str(ipynb)],
            cwd=ROOT, check=True, timeout=600, env=env,
        )
    except subprocess.CalledProcessError as exc:
        print(f"  !! {name} failed (exit {exc.returncode})", file=sys.stderr)
        return False
    except subprocess.TimeoutExpired:
        print(f"  !! {name} timed out after {timeout}s", file=sys.stderr)
        return False
    finally:
        # The .ipynb is an intermediate: the .py is the source of truth and the
        # .md is the artefact. Leaving it behind invites editing the wrong file.
        ipynb.unlink(missing_ok=True)

    _prepend_title(OUT / f"{name}.md", TITLES.get(name, name))
    print(f"   wrote docs/tutorials/{name}.md  ({time.time() - started:.0f}s)")
    return True


def _prepend_title(path: Path, title: str) -> None:
    """Give mkdocs a top-level heading even if the notebook starts with prose."""
    if not path.exists():
        return
    text = path.read_text()
    if text.lstrip().startswith("# "):
        return
    path.write_text(f"# {title}\n\n{text}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("tutorials", nargs="*", default=None,
                    help="subset to bake; default is all of them")
    ap.add_argument("--timeout", type=int, default=3600,
                    help="per-notebook execution timeout in seconds")
    ap.add_argument("--clean", action="store_true",
                    help="remove docs/tutorials before baking")
    args = ap.parse_args()

    for tool in ("jupytext", "jupyter"):
        if shutil.which(tool) is None:
            raise SystemExit(
                f"{tool} not found. Install the dev group: uv sync --group dev"
            )

    wanted = args.tutorials or ALL_TUTORIALS
    unknown = set(wanted) - set(ALL_TUTORIALS)
    if unknown:
        raise SystemExit(
            f"unknown tutorial(s) {sorted(unknown)}; "
            f"choose from {ALL_TUTORIALS}"
        )

    if args.clean and OUT.exists():
        shutil.rmtree(OUT)

    with project_kernel() as env:
        ok = [bake(name, args.timeout, env) for name in wanted]
    failed = [n for n, good in zip(wanted, ok) if not good]
    if failed:
        raise SystemExit(f"failed: {', '.join(failed)}")
    print("\nDone. Commit docs/tutorials/ and its *_files/ directories.")


if __name__ == "__main__":
    main()
