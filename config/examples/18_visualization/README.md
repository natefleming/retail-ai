# 18 - Visualization (Vega-Lite)

Generate portable Vega-Lite chart specs from structured data. Specs are delivered
to clients via `custom_outputs.visualizations`, tagged with a `message_id` so the
UI can associate each chart with the correct response message.

## Prerequisites

- Databricks workspace with a SQL warehouse (only needed for the SQL data tool)
- `DATABRICKS_WAREHOUSE_ID` environment variable set

## Quick Start

```bash
dao-ai chat -c config/examples/18_visualization/vega_lite_visualization.yaml
```

## How It Works

1. The **visualization factory tool** (`create_visualization_tool`) is configured
   in YAML with defaults for chart dimensions, color scheme, and chart type.
2. At runtime the LLM calls the tool with data, field names, and optional
   overrides (chart type, aggregation, sort, color field, title).
3. The tool returns a Vega-Lite JSON spec as its tool output.
4. The DAO AI response pipeline detects the Vega-Lite spec in the tool messages
   and places it in `custom_outputs.visualizations` tagged with the current
   `message_id`.
5. The client renders the spec using `vega-embed` (or any Vega-Lite renderer).

## Chart Types

| Type      | Mark    | Best For                          |
|-----------|---------|-----------------------------------|
| `bar`     | bar     | Categorical comparisons           |
| `line`    | line    | Trends over time                  |
| `scatter` | point   | Correlations between two measures |
| `area`    | area    | Volume over time                  |
| `arc`     | arc     | Proportional breakdowns (pie)     |
| `heatmap` | rect    | Density or matrix data            |

## Factory Arguments

| Argument             | Default       | Description                             |
|----------------------|---------------|-----------------------------------------|
| `name`               | `visualization_tool` | Tool name visible to the LLM     |
| `description`        | Auto-generated| Tool description for the LLM            |
| `default_chart_type` | `bar`         | Fallback chart type                     |
| `width`              | `container`   | Chart width (responsive or fixed int)   |
| `height`             | `400`         | Chart height in pixels                  |
| `color_scheme`       | `tableau10`   | Vega-Lite named color scheme            |

## Client Rendering

Each visualization in `custom_outputs.visualizations` has the shape:

```json
{
  "spec": { "$schema": "...", "data": {...}, "mark": "bar", ... },
  "message_id": "msg_abc12345"
}
```

Match `message_id` to the response output item `id` to render the chart
alongside the correct message.
