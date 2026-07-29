source "https://rubygems.org"

# Jekyll + Just the Docs power the GitHub Pages documentation site.
# Built via GitHub Actions (.github/workflows/publish-docs.yaml), not the
# GitHub Pages native Jekyll build, so gem versions are pinned here.
gem "jekyll", "~> 4.3"
gem "just-the-docs", "~> 0.12.0"

group :jekyll_plugins do
  gem "jekyll-relative-links"   # ](foo.md) -> page URL rewriting
  gem "jekyll-seo-tag"
end

# Ruby 3.4+ dropped several gems from the default set that Jekyll relies on.
gem "csv"
gem "base64"
gem "bigdecimal"
gem "logger"
