"""Climate-index driven winter outlooks for California.

The module is intentionally data-source agnostic: callers can train it with observed
regional monthly data, while ``generate_demo_history`` provides a deterministic E2E
fixture until the NOAA/CDEC ingestion layer is connected to real observations.
"""

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


WINTER_MONTHS = (11, 12, 1, 2, 3, 4)
CLIMATE_INDICES = ("enso", "pdo", "ao", "pna")
CLIMATE_FEATURES = (*CLIMATE_INDICES, "temperature_anomaly")
REGIONS = {
    "north_coast": {"latitude": 40.2, "elevation_m": 900, "area_weight": 0.16},
    "shasta_cascades": {"latitude": 41.0, "elevation_m": 1700, "area_weight": 0.12},
    "northern_sierra": {"latitude": 39.6, "elevation_m": 2100, "area_weight": 0.14},
    "central_sierra": {"latitude": 38.3, "elevation_m": 2300, "area_weight": 0.15},
    "southern_sierra": {"latitude": 36.8, "elevation_m": 2400, "area_weight": 0.13},
    "central_coast_valleys": {"latitude": 36.7, "elevation_m": 250, "area_weight": 0.18},
    "southern_california": {"latitude": 34.2, "elevation_m": 750, "area_weight": 0.12},
}


@dataclass(frozen=True)
class ClimateScenario:
    """Seasonal climate-index assumptions used for an outlook."""

    enso: float
    pdo: float
    ao: float = 0.0
    pna: float = 0.0
    def as_dict(self) -> Dict[str, float]:
        return {name: float(getattr(self, name)) for name in CLIMATE_INDICES}


def validate_history(history: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize observed regional monthly training data."""
    required = {
        "water_year", "month", "region", "precipitation_mm", "snowfall_cm",
        *CLIMATE_FEATURES,
    }
    missing = sorted(required.difference(history.columns))
    if missing:
        raise ValueError(f"History is missing required columns: {missing}")
    result = history.copy()
    if result.empty:
        raise ValueError("History must contain at least one winter")
    if not set(result["month"].unique()).issubset(WINTER_MONTHS):
        raise ValueError("History may only contain November through April")
    unknown = sorted(set(result["region"]) - set(REGIONS))
    if unknown:
        raise ValueError(f"Unknown California regions: {unknown}")
    numeric = ["water_year", "month", "precipitation_mm", "snowfall_cm", *CLIMATE_FEATURES]
    if result[numeric].isna().any().any():
        raise ValueError("History contains missing numeric values")
    if (result[["precipitation_mm", "snowfall_cm"]] < 0).any().any():
        raise ValueError("Precipitation and snowfall cannot be negative")
    return result


class CaliforniaWinterOutlook:
    """Causal temperature -> precipitation -> snowfall regional forecast."""

    base_features = ["water_year", "month", "region", "latitude", "elevation_m", *CLIMATE_INDICES]
    precipitation_features = [*base_features, "temperature_anomaly"]
    snowfall_features = [*precipitation_features, "precipitation_mm"]

    def __init__(self, n_estimators: int = 180, random_state: int = 42):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.temperature_model = self._make_pipeline()
        self.precipitation_model = self._make_pipeline()
        self.snowfall_model = self._make_pipeline()
        self._fitted = False

    def _make_pipeline(self) -> Pipeline:
        encoder = ColumnTransformer(
            [("region", OneHotEncoder(handle_unknown="ignore"), ["region"])],
            remainder="passthrough",
        )
        forest = RandomForestRegressor(
            n_estimators=self.n_estimators, min_samples_leaf=2,
            random_state=self.random_state, n_jobs=-1,
        )
        return Pipeline([("features", encoder), ("model", forest)])

    @staticmethod
    def _add_geography(frame: pd.DataFrame) -> pd.DataFrame:
        result = frame.copy()
        result["latitude"] = result["region"].map(lambda r: REGIONS[r]["latitude"])
        result["elevation_m"] = result["region"].map(lambda r: REGIONS[r]["elevation_m"])
        return result

    def fit(self, history: pd.DataFrame) -> "CaliforniaWinterOutlook":
        history = self._add_geography(validate_history(history))
        self.temperature_model.fit(history[self.base_features], history["temperature_anomaly"])
        self.precipitation_model.fit(history[self.precipitation_features], history["precipitation_mm"])
        self.snowfall_model.fit(history[self.snowfall_features], history["snowfall_cm"])
        self._fitted = True
        return self

    def _predict_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        result = self._add_geography(frame)
        result["temperature_anomaly"] = self.temperature_model.predict(result[self.base_features])
        result["precipitation_mm"] = np.maximum(
            0.0, self.precipitation_model.predict(result[self.precipitation_features])
        )
        result["snowfall_cm"] = np.maximum(
            0.0, self.snowfall_model.predict(result[self.snowfall_features])
        )
        return result

    def predict(self, water_year: int, scenario: ClimateScenario) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("Call fit before predict")
        rows = []
        for region in REGIONS:
            for month in WINTER_MONTHS:
                rows.append({"water_year": water_year, "region": region, "month": month, **scenario.as_dict()})
        frame = self._predict_frame(pd.DataFrame(rows))
        frame["period"] = frame.apply(
            lambda row: f"{water_year - 1 if row.month >= 10 else water_year}-{int(row.month):02d}", axis=1
        )
        frame["area_weight"] = frame["region"].map(lambda r: REGIONS[r]["area_weight"])
        return frame[["water_year", "period", "month", "region", "area_weight", *CLIMATE_FEATURES,
                      "precipitation_mm", "snowfall_cm"]]

    def evaluate_last_years(self, history: pd.DataFrame, years: int = 5) -> Dict[str, float]:
        """Backtest on whole unseen water years to prevent temporal leakage."""
        history = validate_history(history)
        water_years = sorted(history["water_year"].unique())
        if len(water_years) <= years:
            raise ValueError("Not enough water years for the requested holdout")
        test_years = water_years[-years:]
        train = history[~history["water_year"].isin(test_years)]
        test = self._add_geography(history[history["water_year"].isin(test_years)])
        candidate = CaliforniaWinterOutlook(self.n_estimators, self.random_state).fit(train)
        predicted = candidate._predict_frame(test[candidate.base_features])
        return {
            "temperature_mae_c": float(mean_absolute_error(test["temperature_anomaly"], predicted["temperature_anomaly"])),
            "temperature_r2": float(r2_score(test["temperature_anomaly"], predicted["temperature_anomaly"])),
            "precipitation_mae_mm": float(mean_absolute_error(test["precipitation_mm"], predicted["precipitation_mm"])),
            "snowfall_mae_cm": float(mean_absolute_error(test["snowfall_cm"], predicted["snowfall_cm"])),
            "precipitation_r2": float(r2_score(test["precipitation_mm"], predicted["precipitation_mm"])),
            "snowfall_r2": float(r2_score(test["snowfall_cm"], predicted["snowfall_cm"])),
        }


def summarize_outlook(forecast: pd.DataFrame, climatology: pd.DataFrame) -> Dict[str, object]:
    """Return statewide wetness/snow totals plus the Nov-Apr statewide trajectory."""
    climate = validate_history(climatology)
    normal = climate.groupby(["region", "month"])[["precipitation_mm", "snowfall_cm"]].mean()
    compared = forecast.join(normal, on=["region", "month"], rsuffix="_normal")
    for variable in ("precipitation_mm", "snowfall_cm"):
        compared[f"{variable}_pct_normal"] = 100 * compared[variable] / compared[f"{variable}_normal"].clip(lower=0.01)

    weighted = compared.assign(
        weighted_precip=lambda x: x.precipitation_mm * x.area_weight,
        weighted_normal=lambda x: x.precipitation_mm_normal * x.area_weight,
        weighted_snow=lambda x: x.snowfall_cm * x.area_weight,
    )
    trajectory = weighted.groupby(["period", "month"], as_index=False).agg(
        precipitation_mm=("weighted_precip", "sum"),
        snowfall_cm=("weighted_snow", "sum"),
    )
    precip_pct = 100 * weighted.weighted_precip.sum() / weighted.weighted_normal.sum()
    category = "wet" if precip_pct >= 110 else "dry" if precip_pct <= 90 else "near_normal"
    return {
        "statewide_wetness": category,
        "statewide_precipitation_pct_normal": float(precip_pct),
        "statewide_precipitation_mm": float(weighted.weighted_precip.sum()),
        "statewide_snowfall_cm": float(weighted.weighted_snow.sum()),
        "statewide_temperature_anomaly_c": float(
            (weighted.temperature_anomaly * weighted.area_weight).sum() / weighted.area_weight.sum()
        ),
        "trajectory": trajectory.to_dict(orient="records"),
        "regional_monthly": compared.to_dict(orient="records"),
    }


def generate_demo_history(start_water_year: int = 1981, end_water_year: int = 2024,
                          random_state: int = 42) -> pd.DataFrame:
    """Create reproducible climate-sensitive history for offline demos/tests."""
    rng = np.random.default_rng(random_state)
    rows = []
    month_factor = {11: .65, 12: 1.15, 1: 1.3, 2: 1.1, 3: .9, 4: .45}
    for wy in range(start_water_year, end_water_year + 1):
        t = wy - start_water_year
        enso = 1.15 * np.sin(2 * np.pi * t / 4.3) + rng.normal(0, .3)
        pdo = .9 * np.sin(2 * np.pi * t / 13 + .8) + rng.normal(0, .25)
        ao, pna = rng.normal(0, .75, 2)
        temp = .018 * t + .35 * enso + rng.normal(0, .25)
        for region, geo in REGIONS.items():
            south = (40.5 - geo["latitude"]) / 7
            base = 85 + 90 * (1 - south) + 20 * (geo["elevation_m"] / 2400)
            teleconnection = 1 + (.11 * enso * (2 * south - .55)) + .1 * pdo + .04 * pna - .025 * ao
            for month in WINTER_MONTHS:
                precipitation = max(2, base * month_factor[month] * teleconnection * rng.lognormal(0, .13))
                snow_fraction = np.clip(.1 + geo["elevation_m"] / 2800 - .09 * temp - .035 * south, .03, .96)
                snowfall = max(0, precipitation * snow_fraction * 1.05 + rng.normal(0, 5))
                rows.append({"water_year": wy, "month": month, "region": region,
                             "enso": enso, "pdo": pdo, "ao": ao, "pna": pna,
                             "temperature_anomaly": temp, "precipitation_mm": precipitation,
                             "snowfall_cm": snowfall})
    return pd.DataFrame(rows)
