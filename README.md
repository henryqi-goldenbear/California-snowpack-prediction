# California Snowpack Prediction

A California winter outlook pipeline and **Mistral Winter Lab** dashboard that turn Pacific climate indices into temperature, precipitation, and snowfall guidance across seven regions.

**Live demo:** [https://mistral-winter-lab.henryqi.workers.dev](https://mistral-winter-lab.henryqi.workers.dev)

No sign-in required. Adjust the climate sliders and Mistral returns a statewide outlook in about 15-20 seconds.

## Overview

The project combines:

- **Mistral Winter Lab** — an interactive dashboard where users adjust ENSO/ONI, PDO, AO, and PNA sliders and receive Mistral-generated statewide, seasonal, and regional outlooks
- **CaliforniaWinterOutlook** — a Python model that predicts regional Nov-Apr precipitation and snowfall from numeric climate indices
- **Optional Mistral agents** — Python helpers for experiment design and forecast explanation in offline workflows

Mistral is **not** used to parse free-text scenario descriptions in the shipped demo. Climate inputs are numeric indices; Mistral generates the narrative forecast, early/mid/late season analysis, regional winter-risk text, water allocation outlook, and wildfire carryover assessment from those values.

## Features

### Mistral Winter Lab (web dashboard)

- Adjust **ENSO/ONI, PDO, AO, and PNA** with sliders for a selected water year
- Toggle **metric or imperial** units for temperature, precipitation, and snowfall
- View Mistral's **statewide winter read** plus **early (Nov-Dec), mid (Jan-Feb), and late (Mar-Apr)** season breakdowns
- Explore a **monthly precipitation/snowfall trajectory** and **seven regional outlooks**
- Read **region-specific winter risks** (flooding, drought, snowpack deficit, etc.) for each area
- See **water allocation** outlook (reservoirs, agriculture, cities, ecosystems) and **wildfire carryover** risk for the following fire season
- Recent winters are weighted more heavily in Mistral's reasoning (2023 ARs, 2020-2022 drought, etc.)

### CaliforniaWinterOutlook (Python)

- Causal chain: climate indices → temperature anomaly → precipitation → rain/snow partition → snowfall
- **Recency-weighted training** — recent water years count more when fitting (7-year half-life by default)
- **Recency-weighted baselines** — "normal" precipitation and snow reflect recent winters, not a flat long-run mean
- Regional forecasts for seven California zones and statewide summaries
- Held-out water-year backtesting with scikit-learn random forests

### Optional Python agents

These modules support offline experimentation; they are **not** wired into the dashboard:

- **Experiment Agent** — suggests features, models, and evaluation setups from dataset summaries
- **Forecast Explainer** — generates plain-language explanations of model outputs for stakeholders

## Project Structure

```
California-snowpack-prediction/
├── src/
│   ├── App.tsx                 # Mistral Winter Lab dashboard
│   ├── models/                 # CaliforniaWinterOutlook + baseline models
│   ├── agent/                  # Optional Mistral Python agents
│   └── data/                   # Data loading and preprocessing
├── worker/                     # Cloudflare Worker / local API (Mistral forecast)
├── scripts/                    # Build, staging, and local dev helpers
├── tests/                      # Python E2E tests
├── wrangler.toml               # Cloudflare Workers deploy config
├── winter_outlook.py           # CLI for statewide winter outlook
├── package.json                # Frontend (Vite + React)
└── requirements.txt            # Python dependencies
```

## Getting Started

### Prerequisites

- **Node.js 20+** for the dashboard and Cloudflare deploy
- **Python 3.9+** for the offline outlook model and agents
- **Mistral API key** for live forecasts (`MISTRAL_API_KEY`)

### Frontend setup

```bash
git clone https://github.com/henryqi-goldenbear/California-snowpack-prediction.git
cd California-snowpack-prediction
npm install
cp .env.example .env   # add your MISTRAL_API_KEY
```

### Local dashboard

```bash
# Terminal 1 — local Mistral API (port 8787)
npm run demo

# Terminal 2 — dashboard (port 5173)
npm run dev
```

Open [http://127.0.0.1:5173](http://127.0.0.1:5173), adjust the climate sliders, and click **Refresh with Mistral**.

`MISTRAL_MODEL` in `.env` optionally overrides the default `mistral-small-latest`.

### Local production server

Serve the built dashboard and API on one port (default `8080`):

```bash
npm run start:public
```

Open [http://127.0.0.1:8080](http://127.0.0.1:8080). On your LAN, use `http://<your-ip>:8080`.

### Deploy to Cloudflare Workers (permanent host)

1. Log in once: `npx wrangler@3 login`
2. Register a `workers.dev` subdomain in the [Cloudflare dashboard](https://dash.cloudflare.com/) if prompted
3. Store your API key: `npx wrangler@3 secret put MISTRAL_API_KEY`
4. Deploy: `npm run deploy`

The app publishes to `https://mistral-winter-lab.<your-subdomain>.workers.dev`.

### Python outlook CLI

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

python winter_outlook.py --water-year 2025 --enso 1.2 --pdo 0.5 \
  --ao -0.2 --pna 0.4 --quick
```

The bundled history generator is deterministic and intended for offline E2E validation. For operational forecasts, train `CaliforniaWinterOutlook` with observed regional monthly NOAA climate indices and CDEC precipitation/snowfall.

## API

The Worker exposes a single forecast endpoint:

```http
POST /api/forecast
Content-Type: application/json

{
  "year": 2027,
  "enso": 0.8,
  "pdo": 0.35,
  "ao": 0.0,
  "pna": 0.2
}
```

Returns JSON with `summary`, `category`, statewide metrics, `seasonPhases`, monthly `trajectory`, seven `details` (with `risks`), `waterAllocation`, and `wildfireRisk`.

## Model Architecture

### Mistral Winter Lab API (`worker/index.js`)

- Accepts numeric climate indices (ENSO, PDO, AO, PNA, water year)
- Calls Mistral with a structured prompt anchored on recent California winter precedents
- Returns validated JSON for the dashboard

### CaliforniaWinterOutlook (Python)

- Random-forest regressors with recency-weighted training
- Separate regional precipitation and snowfall models
- Season-phase and impact summaries for offline use

### Optional Python agents

- **Experiment Agent** — LLM-based experiment design suggestions
- **Forecast Explainer** — natural-language explanations of tabular model outputs

## Roadmap

- [x] Mistral Winter Lab dashboard with climate-index sliders
- [x] Mistral API forecast with seasonal and regional analysis
- [x] Metric/imperial unit toggle
- [x] Regional winter-risk descriptions
- [x] Water allocation and wildfire carryover analysis
- [x] Recency-weighted outlook reasoning
- [x] Public Cloudflare Workers deployment
- [ ] Connect live NOAA/CDEC observations to replace demo history
- [ ] Optional Python agents integrated into a unified CLI

## License

MIT License — see [LICENSE](LICENSE).

## Acknowledgments

- Mistral AI for the forecast API
- California Department of Water Resources, NOAA, and USGS for hydrologic and climate data used in model development

---

**Built with Mistral AI, React, and Python**
