# California Snowpack Prediction

A California winter outlook pipeline and **Mistral Winter Lab** dashboard that turn Pacific climate indices into temperature, precipitation, and snowfall guidance across seven regions.

## Overview

The project combines:

- **Mistral Winter Lab** — an interactive dashboard where users adjust ENSO/ONI, PDO, AO, and PNA sliders and receive Mistral-generated statewide, seasonal, and regional outlooks
- **CaliforniaWinterOutlook** — a Python model that predicts regional Nov–Apr precipitation and snowfall from numeric climate indices
- **Optional Mistral agents** — Python helpers for experiment design and forecast explanation in offline workflows

Mistral is **not** used to parse free-text scenario descriptions in the shipped demo. Climate inputs are numeric indices; Mistral generates the narrative forecast, early/mid/late season analysis, and regional winter-risk text from those values.

## Features

### Mistral Winter Lab (web dashboard)

- Adjust **ENSO/ONI, PDO, AO, and PNA** with sliders for a selected water year
- Toggle **metric or imperial** units for temperature, precipitation, and snowfall
- View Mistral’s **statewide winter read** plus **early (Nov–Dec), mid (Jan–Feb), and late (Mar–Apr)** season breakdowns
- Explore a **monthly precipitation/snowfall trajectory** and **seven regional outlooks**
- Read **region-specific winter risks** (flooding, drought, snowpack deficit, etc.) for each area
- See **water allocation** outlook (reservoirs, agriculture, cities, ecosystems) and **wildfire carryover** risk for the following fire season

### CaliforniaWinterOutlook (Python)

- Causal chain: climate indices → temperature anomaly → precipitation → rain/snow partition → snowfall
- **Recency-weighted training** — recent water years count more when fitting (7-year half-life by default)
- **Recency-weighted baselines** — “normal” precipitation and snow reflect recent winters, not a flat long-run mean
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
├── worker/                     # Sites / local API handler (Mistral forecast)
├── scripts/                    # Build and local dev helpers
├── tests/                      # Python E2E tests
├── winter_outlook.py           # CLI for statewide winter outlook
├── package.json                # Frontend (Vite + React)
└── requirements.txt            # Python dependencies
```

## AI for Climate and Environment

Snowpack is a critical water resource and hazard driver:
- **Water supply planning** for California's agricultural and urban needs
- **Flood risk assessment** from rapid snowmelt
- **Drought forecasting** based on snowpack deficits
- **Ski industry planning** for resort operations

Mistral's high-performance, efficient LLMs are ideal for:
- Agentic workflows in domain-specific applications
- Customization for environmental datasets and APIs
- Decision support for operations (reservoir management, resort planning)

## Getting Started

### Prerequisites

- Python 3.9+
- Mistral AI API key (for agent functionality)
- Required Python packages (see requirements.txt)

### Installation

```bash
# Clone the repository
git clone https://github.com/henryqi-goldenbear/California-snowpack-prediction.git
cd California-snowpack-prediction

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Quick Start

```python
from src.agent.experiment_agent import ExperimentAgent
from src.data.data_loader import load_snowpack_data

# Load data
data = load_snowpack_data()

# Initialize agent
agent = ExperimentAgent(api_key="your-mistral-api-key")

# Get experiment suggestions
suggestions = agent.suggest_experiments(data_summary=data.describe())
print(suggestions)
```

## Usage Examples

### Climate-driven statewide winter outlook

Run a November-April forecast for every California region using ENSO, PDO, AO,
and PNA. The indices first predict temperature; predicted temperature then
controls the rain/snow partition and snowfall. The command reports statewide winter
wetness, precipitation and snowfall totals, the monthly trajectory, regional
detail, and a held-out-year backtest:

```bash
python winter_outlook.py --water-year 2025 --enso 1.2 --pdo 0.5 \
  --ao -0.2 --pna 0.4 --quick
```

The bundled history generator is deterministic and intended for offline E2E
validation. For operational forecasts, train `CaliforniaWinterOutlook` with
observed regional monthly NOAA climate indices and CDEC precipitation/snowfall;
the required schema is enforced by `validate_history`.

### Interactive frontend demo

The **Mistral Winter Lab** dashboard sends your slider values to the Mistral API and renders the returned outlook: statewide summary, early/mid/late season cards, monthly trajectory, seven regional totals, and regional winter risks. Choose **metric or imperial** units in the header.

```bash
npm install

# Terminal 1 — local Mistral API
npm run demo

# Terminal 2 — dashboard
npm run dev
```

Open `http://127.0.0.1:5173`, adjust the climate sliders, and click **Refresh with Mistral**.

Set `MISTRAL_API_KEY` in `.env` for local runs. `MISTRAL_MODEL` optionally overrides the default `mistral-small-latest`.

### Public demo (not localhost)

Build and serve the dashboard plus Mistral API on one port (default `8080`, all interfaces):

```bash
npm run start:public
```

Then expose it with a Cloudflare quick tunnel:

```bash
npx cloudflared tunnel --url http://127.0.0.1:8080
```

Share the `https://*.trycloudflare.com` URL. Anyone with the link can use the dashboard; the tunnel stays up while both processes keep running.

On your local network you can also open `http://<your-lan-ip>:8080` without a tunnel.

### 1. Running a Baseline Forecast

```python
from src.models.baseline.xgboost_model import XGBoostForecaster
from src.data.data_loader import load_snowpack_data

# Load and preprocess data
data = load_snowpack_data()
X, y = preprocess_data(data)

# Train and predict
model = XGBoostForecaster()
model.fit(X, y)
predictions = model.predict_future(horizon=12)
```

### 2. Using the Mistral Agent

```python
from src.agent.experiment_agent import ExperimentAgent

# Initialize agent
agent = ExperimentAgent()

# Get feature engineering suggestions
feature_suggestions = agent.suggest_features(
    variables=data.columns.tolist(),
    target="swe"  # Snow Water Equivalent
)

# Get model recommendations
model_suggestions = agent.suggest_models(
    data_shape=X.shape,
    forecast_horizon=12
)
```

### 3. Generating Forecast Reports (optional Python agent)

```python
from src.agent.forecast_explainer import ForecastExplainer

explainer = ForecastExplainer()
report = explainer.explain_forecast(
    predictions=predictions,
    feature_importance=model.feature_importances_,
    historical_context=data,
)
print(report.natural_language_summary)
```

## Data Sources

The project uses the following data sources:

1. **California Data Exchange Center (CDEC)**
   - Snow water equivalent (SWE) measurements
   - Snow depth measurements
   - Precipitation data

2. **NOAA Climate Data**
   - Temperature records
   - Precipitation records
   - Climate indices (ENSO, PDO)

3. **USGS Water Data**
   - Streamflow data
   - Reservoir levels

4. **NASA/NOAA Remote Sensing**
   - MODIS snow cover extent
   - Satellite-derived snow metrics

## Model Architecture

### Baseline Models

1. **XGBoost**
   - Gradient boosting for tabular data
   - Handles missing values and outliers well
   - Feature importance analysis

2. **LSTM (Long Short-Term Memory)**
   - Recurrent neural network for time series
   - Captures temporal dependencies
   - Good for sequential snowpack data

3. **Prophet**
   - Facebook's time series forecasting
   - Handles seasonality and holidays
   - Robust to missing data

### Mistral-Powered Components

1. **Mistral Winter Lab API** (`worker/index.js`)
   - Accepts numeric climate indices (ENSO, PDO, AO, PNA, water year)
   - Returns structured JSON: statewide outlook, seasonal phases, trajectory, regional detail, and regional risks

2. **Experiment Agent** (optional, Python)
   - LLM-based experiment design suggestions for offline modeling workflows

3. **Forecast Explainer** (optional, Python)
   - Natural language explanations of tabular model outputs

## Evaluation Metrics

- **RMSE (Root Mean Squared Error)**: Primary metric for forecast accuracy
- **MAE (Mean Absolute Error)**: Robust to outliers
- **R² (R-squared)**: Explained variance
- **CRPS (Continuous Ranked Probability Score)**: For probabilistic forecasts
- **NSE (Nash-Sutcliffe Efficiency)**: Hydrology-specific metric

## Project Roadmap

- [x] Mistral Winter Lab dashboard with climate-index sliders
- [x] Mistral API forecast with seasonal and regional analysis
- [x] CaliforniaWinterOutlook Python model with E2E tests
- [x] Metric/imperial unit toggle
- [x] Regional winter-risk descriptions
- [ ] Connect live NOAA/CDEC observations to replace demo history
- [ ] Optional Python agents integrated into a unified CLI

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Mistral AI for providing the foundation models
- California Department of Water Resources for snowpack data
- NOAA for climate data
- USGS for hydrological data

## Contact

For questions or collaboration opportunities, please open an issue or contact the project maintainers.

---

**Built with Mistral AI and Python**
