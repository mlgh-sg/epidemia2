"""Regenerate docs/llms-full.txt: the full docs concatenated for LLM ingestion.

Run from the ``python/`` directory (``uv run python scripts/build_llms_full.py``).
The mkdocs build copies both ``docs/llms.txt`` (a curated index) and the
``docs/llms-full.txt`` written here to the site root.
"""

from __future__ import annotations

from pathlib import Path

# Follows the site nav. The TUTORIALS matter most for an LLM -- they are the
# worked examples, and omitting them (as this script used to) left llms-full.txt
# at a twelfth the size of the R package's, carrying the API but none of the
# usage. Kept explicit rather than globbed so the order is the reading order and
# a new page is a deliberate addition.
ORDER = [
    "index.md",
    "guide.md",
    "tutorials.md",
    "tutorials/flu.md",
    "tutorials/multiple-obs.md",
    "tutorials/partial-pooling.md",
    "tutorials/multilevel-multi-obs.md",
    "tutorials/flaxman.md",
    "tutorials/b117.md",
    "priors.md",
    "parity.md",
    "performance.md",
    "reference.md",
]
HEADER = (
    "# epidemia (Python) — full documentation\n\n"
    "> Concatenation of the epidemia Python docs for LLM ingestion. "
    "Source: https://mlgh-sg.com/epidemia2/python/\n\n"
    "The API reference below is rendered from NumPy-style docstrings on the "
    "site; here the `:::` lines are mkdocstrings directives naming the objects "
    "documented (see the package source for full signatures).\n\n---\n\n"
)


def main() -> None:
    docs = Path(__file__).resolve().parent.parent / "docs"
    parts = [HEADER]
    missing = [n for n in ORDER if not (docs / n).exists()]
    if missing:
        raise SystemExit(
            f"missing docs page(s): {missing}. Bake the tutorials first "
            "(uv run --group dev python scripts/precompute.py), or drop the "
            "page from ORDER if it is gone for good."
        )
    for name in ORDER:
        parts.append(f"\n\n<!-- ===== {name} ===== -->\n\n")
        parts.append((docs / name).read_text())
    out = docs / "llms-full.txt"
    out.write_text("".join(parts))
    print(f"wrote {out} ({out.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
