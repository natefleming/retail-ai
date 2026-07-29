#!/usr/bin/env python3
"""Assemble the Just the Docs staging tree for the documentation site.

The source docs (``docs/*.md``) are kept pristine — no YAML front matter is
committed to them, so they keep rendering cleanly when browsed on GitHub. Just
the Docs, however, drives its sidebar entirely from per-page front matter. This
script bridges the two: it copies the docs into a staging directory and stamps
``layout`` / ``title`` / ``nav_order`` front matter onto the copies. Jekyll then
builds from the staging tree.

Ordering and sidebar titles come from ``docs-nav.yml`` so re-ordering the nav is
a one-file edit rather than a code change.

Usage:
    python scripts/build-docs-site.py [--staging _site_src]

Then build/serve from the staging dir:
    bundle exec jekyll build --source _site_src --destination _site
    bundle exec jekyll serve  --source _site_src
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = REPO_ROOT / "docs"
NAV_FILE = REPO_ROOT / "docs-nav.yml"

# Files copied verbatim from the repo root into the staging tree.
ROOT_FILES = ["_config.yml", "index.md", "Gemfile", "Gemfile.lock"]

H1_RE = re.compile(r"^#\s+(.+?)\s*$", re.MULTILINE)


def load_nav() -> tuple[list[str], dict[str, str]]:
    """Return (ordered filenames, {filename: title-override})."""
    nav = yaml.safe_load(NAV_FILE.read_text())
    return nav.get("order", []), nav.get("titles", {})


def first_h1(markdown: str, fallback: str) -> str:
    """Extract the first level-1 heading, or fall back to the filename stem."""
    match = H1_RE.search(markdown)
    return match.group(1).strip() if match else fallback


def front_matter(title: str, nav_order: int) -> str:
    # Quote the title so colons / special chars in headings stay valid YAML.
    safe_title = title.replace('"', '\\"')
    return (
        "---\n"
        "layout: default\n"
        f'title: "{safe_title}"\n'
        f"nav_order: {nav_order}\n"
        "---\n\n"
    )


def build_staging(staging: Path) -> int:
    if not NAV_FILE.exists():
        sys.exit(f"error: missing {NAV_FILE}")

    order, title_overrides = load_nav()

    if staging.exists():
        shutil.rmtree(staging)
    (staging / "docs").mkdir(parents=True)

    # Copy verbatim root files (skip Gemfile.lock if it hasn't been generated).
    for name in ROOT_FILES:
        src = REPO_ROOT / name
        if src.exists():
            shutil.copy2(src, staging / name)

    # Copy images so relative `images/*.png` links resolve on every page.
    images = DOCS_DIR / "images"
    if images.is_dir():
        shutil.copytree(images, staging / "docs" / "images")

    ordered = {name: i for i, name in enumerate(order, start=1)}

    # Any docs/*.md not listed in docs-nav.yml still gets built, appended after
    # the ordered pages (so a newly added doc is never silently dropped).
    present = sorted(p.name for p in DOCS_DIR.glob("*.md"))
    unlisted = [n for n in present if n not in ordered]
    for offset, name in enumerate(unlisted, start=1):
        ordered[name] = len(order) + offset
        print(f"note: {name} not in docs-nav.yml — appended at end of nav")

    count = 0
    for name, nav_order in sorted(ordered.items(), key=lambda kv: kv[1]):
        src = DOCS_DIR / name
        if not src.exists():
            print(f"warning: {name} listed in docs-nav.yml but not found — skipped")
            continue
        body = src.read_text()
        title = title_overrides.get(name) or first_h1(body, src.stem)
        # The source docs are plain markdown never written for Jekyll, yet they
        # contain `{{ ... }}` (secret refs, eval `{{ inputs }}`/`{{ outputs }}`,
        # SQL params) that Jekyll's Liquid engine would consume and blank out.
        # Wrap the whole body in {% raw %} so every page renders literally.
        # jekyll-relative-links rewrites `](foo.md)` links before Liquid runs,
        # so link rewriting is unaffected.
        staged = f"{front_matter(title, nav_order)}{{% raw %}}\n{body}\n{{% endraw %}}\n"
        (staging / "docs" / name).write_text(staged)
        count += 1

    print(f"staged {count} docs pages into {staging.relative_to(REPO_ROOT)}/")
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--staging",
        default="_site_src",
        help="staging directory to assemble (default: _site_src)",
    )
    args = parser.parse_args()
    build_staging(REPO_ROOT / args.staging)


if __name__ == "__main__":
    main()
