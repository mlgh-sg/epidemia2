"""Regenerate docs/llms-full.txt: the full docs concatenated for LLM ingestion.

Run from the ``python/`` directory (``uv run python scripts/build_llms_full.py``).
The mkdocs build copies both ``docs/llms.txt`` (a curated index) and the
``docs/llms-full.txt`` written here to the site root.
"""

from __future__ import annotations

from pathlib import Path

ORDER = ["index.md", "guide.md", "reference.md"]
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
    for name in ORDER:
        parts.append(f"\n\n<!-- ===== {name} ===== -->\n\n")
        parts.append((docs / name).read_text())
    out = docs / "llms-full.txt"
    out.write_text("".join(parts))
    print(f"wrote {out} ({out.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
