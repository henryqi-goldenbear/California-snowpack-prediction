# California Snowpack Prediction

A forecasting pipeline that predicts California snowpack (e.g., SWE, depth) using historical weather and hydrology data, with a Mistral-based agent that orchestrates data preprocessing, model selection, and forecast interpretation.

## Overview

This project integrates **Mistral AI models** as an agentic layer over classical time-series forecasting. Mistral is used to:

- **Orchestrate data preprocessing** and feature engineering
- **Suggest model architectures** and experiment configurations
- **Generate natural language explanations** of forecast outputs
- **Provide stakeholder-specific analyses** (water managers, ski resorts, etc.)

## Features

### 1. Experiment Agent
A Mistral model acts as an agent that:
- Takes a JSON summary of the dataset (variables, ranges, missingness)
- Proposes feature engineering steps (lag features, moving averages, cumulative precipitation, ENSO indices)
- Recommends model choices (XGBoost, LSTM, Transformer)
- Suggests train/validation splits and evaluation metrics (RMSE, MAE, CRPS)

### 2. Forecast Explainer
After a model predicts snowpack for the next season:
- Explains why the model predicts a low/high year in plain language
- Highlights which drivers (temperature anomalies, precipitation patterns) were most influential
- Generates stakeholder-specific summaries

### 3. Scenario Simulator
Allows users to specify scenarios in natural language:
- "strong El Ni\u00f1o, 1-2\u00b0C warmer, 10% more precipitation"
- Parses text into parameter adjustments
- Runs forecasts for adjusted inputs
- Presents comparative impact on snowpack

## Project Structure

```
california-snowpack-prediction/
├── data/
│   ├── raw/                    # Original datasets
│   ├── processed/              # Cleaned and engineered features
│   └── external/               # External data sources (ENSO, etc.)
├── models/
│   ├── baseline/               # Baseline models (XGBoost, LSTM)
│   ├── mistral_agent/          # Mistral-based agent components
│   └── saved_models/           # Trained model artifacts
├── notebooks/                  # Exploratory analysis
├── src/
│   ├── data/                   # Data loading and preprocessing
│   ├── features/               # Feature engineering
│   ├── models/                 # Model definitions
│   ├── agent/                  # Mistral agent implementation
│   └── visualization/          # Plotting and reporting
├── tests/                      # Unit and integration tests
├── configs/                    # Configuration files
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Containerization
└── README.md
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

### 3. Generating Forecast Reports

```python
from src.agent.forecast_explainer import ForecastExplainer

# Generate explanation
explainer = ForecastExplainer()
report = explainer.explain_forecast(
    predictions=predictions,
    feature_importance=model.feature_importances_,
    historical_context=data
)
print(report)
```

### 4. Scenario Simulation

```python
from src.agent.scenario_simulator import ScenarioSimulator

# Simulate El Niño scenario
simulator = ScenarioSimulator()
scenario = "strong El Niño with 2°C warmer temperatures and 15% more precipitation"
adjusted_data, scenario_predictions = simulator.run_scenario(
    base_data=data,
    scenario_description=scenario
)
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

1. **Experiment Agent**
   - LLM-based experiment design
   - Automated feature engineering suggestions
   - Model architecture recommendations

2. **Forecast Explainer**
   - Natural language explanations
   - Stakeholder-specific reporting
   - Impact analysis

3. **Scenario Simulator**
   - Natural language scenario parsing
   - Parameter adjustment
   - Comparative analysis

## Evaluation Metrics

- **RMSE (Root Mean Squared Error)**: Primary metric for forecast accuracy
- **MAE (Mean Absolute Error)**: Robust to outliers
- **R² (R-squared)**: Explained variance
- **CRPS (Continuous Ranked Probability Score)**: For probabilistic forecasts
- **NSE (Nash-Sutcliffe Efficiency)**: Hydrology-specific metric

## Project Roadmap

- [ ] Initial data pipeline setup
- [ ] Baseline XGBoost model implementation
- [ ] LSTM model implementation
- [ ] Mistral experiment agent
- [ ] Forecast explainer
- [ ] Scenario simulator
- [ ] Comprehensive evaluation framework
- [ ] Docker containerization
- [ ] API deployment
- [ ] Web interface for stakeholders

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
