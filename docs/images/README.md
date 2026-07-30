# dao-ai Image Assets

Brand and documentation imagery for the **dao-ai** library. All illustrated assets were extracted and cleaned from the AI-generated brand sheet (`brand/dao-ai-brand-sheet.png`).

Each asset ships in four files: `name.png` (1x), `name@2x.png` (retina), and `.webp` versions of both. Assets marked *transparent* have their backgrounds removed. `manifest.json` indexes everything with sizes and transparency flags.

## Folders

### `brand/`
| Asset | Transparent | Notes |
|---|---|---|
| `logo-lockup` | yes | D-mark + "DAO-ai" wordmark (white text — dark backgrounds) |
| `logo-lockup-lightbg` | yes | Dark navy text for light backgrounds |
| `logo-lockup-tagline` | yes | Lockup + "Orchestrate. Collaborate. Automate." (dark bg) |
| `logo-lockup-tagline-lightbg` | yes | Light-background version |
| `logo-symbol` | yes | D-mark only — favicons, avatars, bullets |
| `color-palette` | yes | Rebuilt crisp swatch strip |
| `dao-ai-brand-sheet` | no | Original source sheet |

### `mascots/`
Transparent mascot poses for callouts: `wand-thumbs-up`, `yaml-scroll` (holding `agents.yaml`), `laptop`, `heart-thumbs-up`.

### `hero/`
`main-hero-panel` — full hero illustration (opaque) for README/landing. `hero-mascot` — transparent multi-arm mascot cutout with dissolve fade at the feet.

### `icons/`
App icons (rounded/circular masks, transparent corners): `mascot-square`, `mascot-round`, `logo-square-purple`, `logo-square-blue`.
Doc line icons (transparent, best on dark backgrounds): `multi-agent-collaboration`, `dynamic-workflows`, `tool-code-execution`, `yaml-driven`, `extensible-design`, `secure-local-first`, `plug-in-everything`.

### `stickers/`
Die-cut badges, transparent: `lets-build`, `yaml-driven`, `team-of-agents`, `automate-everything`.

### `banners/`
`terminal-banner` — terminal window with tagline and ASCII mascot, rounded corners.
Feature cards (opaque, rounded corners): `yaml-first`, `python-powered`, `modular-extensible`, `observability`.

### `diagrams/`
Documentation diagrams (not derived from the brand sheet), organized by topic:

- `architecture/` — `dao-architecture-layers`, `dao-system-dataflow`, `dao-agent-bricks-kasal-composition`
- `orchestration-patterns/` — `supervisor-pattern`, `swarm-pattern`, `parallel-fan-out-pattern`, `background-agents-at-a-glance`, `background-agents-three-ops`
- `retrieval/` — `instructed-retrieval-pipeline`, `vector-search-rerank-funnel`
- `genie-cache/` — `genie-cache-hierarchy`, `genie-cache-circuit-breaker`, `genie-cache-prompt-history-mutation`

## Brand color palette

| Hex | Role |
|---|---|
| `#7C3AED` | Violet — primary |
| `#A855F7` | Purple — secondary |
| `#3B82F6` | Blue — accent |
| `#22D3EE` | Cyan — accent / researcher |
| `#10B981` | Green — success |
| `#F59E0B` | Amber — warning / writer |
| `#0F172A` | Midnight — background |

## Notes

- Source sheet is 1254×1254, so 1x sizes are modest (icons ≈ 90–160 px). `@2x` files are sharpened upscales — fine at doc sizes; regenerate from a larger source for print/hero sizes.
- Line icons, non-`lightbg` lockups, and the terminal banner are designed for dark backgrounds.
