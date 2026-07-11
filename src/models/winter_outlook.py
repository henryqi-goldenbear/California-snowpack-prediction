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
SEASON_PHASES = {
    "early": {"label": "Early season", "months": "Nov–Dec", "month_values": (11, 12)},
    "mid": {"label": "Mid season", "months": "Jan–Feb", "month_values": (1, 2)},
    "late": {"label": "Late season", "months": "Mar–Apr", "month_values": (3, 4)},
}
CLIMATE_INDICES = ("enso", "pdo", "ao", "pna")
CLIMATE_FEATURES = (*CLIMATE_INDICES, "temperature_anomaly")
REGIONS = {
    "north_coast": {"latitude": 40.2, "elevation_m": 900, "area_weight": 0.16, "label": "North Coast"},
    "shasta_cascades": {"latitude": 41.0, "elevation_m": 1700, "area_weight": 0.12, "label": "Shasta & Cascades"},
    "northern_sierra": {"latitude": 39.6, "elevation_m": 2100, "area_weight": 0.14, "label": "Northern Sierra"},
    "central_sierra": {"latitude": 38.3, "elevation_m": 2300, "area_weight": 0.15, "label": "Central Sierra"},
    "southern_sierra": {"latitude": 36.8, "elevation_m": 2400, "area_weight": 0.13, "label": "Southern Sierra"},
    "central_coast_valleys": {"latitude": 36.7, "elevation_m": 250, "area_weight": 0.18, "label": "Central Coast & Valleys"},
    "southern_california": {"latitude": 34.2, "elevation_m": 750, "area_weight": 0.12, "label": "Southern California"},
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


def _describe_regional_risks(region: str, precip_pct: float, snow_pct: float, temp_c: float) -> str:
    """Summarize plausible winter hazards from regional forecast anomalies."""
    risks = []
    if precip_pct >= 115:
        risks.append("elevated flood and debris-flow risk during atmospheric river events")
    elif precip_pct <= 85:
        risks.append("drought stress and reduced reservoir recharge")
    if snow_pct <= 85 or temp_c >= 0.8:
        risks.append("below-normal snowpack and weaker spring runoff")
    if temp_c >= 1.0:
        risks.append("rain-on-snow and early melt at mid elevations")
    if temp_c <= -0.8 and region in {"central_coast_valleys", "southern_california"}:
        risks.append("agricultural frost and cold-sensitive crop stress")
    if region in {"north_coast", "shasta_cascades", "northern_sierra"} and precip_pct >= 110:
        risks.append("landslide and coastal erosion concerns on saturated slopes")
    if region in {"central_coast_valleys", "southern_california"} and precip_pct >= 110:
        risks.append("urban flooding and post-fire burn-scar runoff")
    if not risks:
        risks.append("typical winter variability with no dominant extreme hazard signal")
    return "; ".join(risks[:3]).capitalize() + "."


def summarize_outlook(forecast: pd.DataFrame, climatology: pd.DataFrame,
                      units: str = "metric") -> Dict[str, object]:
    """Return statewide wetness/snow totals, Nov-Apr trajectory, and seasonal phases."""
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
    category = _wetness_category(precip_pct)
    season_phases = []
    for phase_id, phase in SEASON_PHASES.items():
        phase_frame = weighted[weighted["month"].isin(phase["month_values"])]
        phase_precip = float(phase_frame.weighted_precip.sum())
        phase_normal = float(phase_frame.weighted_normal.sum())
        phase_pct = 100 * phase_precip / phase_normal if phase_normal else 100.0
        season_phases.append({
            "id": phase_id,
            "label": phase["label"],
            "months": phase["months"],
            "category": _wetness_category(phase_pct),
            "predicted_temp_c": float(
                (phase_frame.temperature_anomaly * phase_frame.area_weight).sum()
                / phase_frame.area_weight.sum()
            ),
            "precipitation_mm": phase_precip,
            "precipitation_pct_normal": float(phase_pct),
            "snowfall_cm": float(phase_frame.weighted_snow.sum()),
        })
    regional_summary = []
    for region, geo in REGIONS.items():
        region_frame = compared[compared["region"] == region]
        precip_total = float(region_frame.precipitation_mm.sum())
        normal_total = float(region_frame.precipitation_mm_normal.sum())
        snow_total = float(region_frame.snowfall_cm.sum())
        snow_normal = float(region_frame.snowfall_cm_normal.sum())
        temp_mean = float(region_frame.temperature_anomaly.mean())
        precip_pct = 100 * precip_total / normal_total if normal_total else 100.0
        snow_pct = 100 * snow_total / snow_normal if snow_normal else 100.0
        regional_summary.append({
            "name": geo["label"],
            "precipitation_mm": precip_total,
            "normal_mm": normal_total,
            "snowfall_cm": snow_total,
            "precipitation_pct_normal": float(precip_pct),
            "risks": _describe_regional_risks(region, precip_pct, snow_pct, temp_mean),
        })
    return {
        "statewide_wetness": category,
        "statewide_precipitation_pct_normal": float(precip_pct),
        "statewide_precipitation_mm": float(weighted.weighted_precip.sum()),
        "statewide_snowfall_cm": float(weighted.weighted_snow.sum()),
        "statewide_temperature_anomaly_c": float(
            (weighted.temperature_anomaly * weighted.area_weight).sum() / weighted.area_weight.sum()
        ),
        "trajectory": trajectory.to_dict(orient="records"),
        "season_phases": season_phases,
        "regional_summary": regional_summary,
        "regional_monthly": compared.to_dict(orient="records"),
        "units": units,
        "display": format_outlook_units({
            "statewide_precipitation_mm": float(weighted.weighted_precip.sum()),
            "statewide_snowfall_cm": float(weighted.weighted_snow.sum()),
            "statewide_temperature_anomaly_c": float(
                (weighted.temperature_anomaly * weighted.area_weight).sum() / weighted.area_weight.sum()
            ),
            "season_phases": season_phases,
        }, units),
    }


def _wetness_category(precip_pct: float) -> str:
    if precip_pct >= 110:
        return "wet"
    if precip_pct <= 90:
        return "dry"
    return "near_normal"


def format_outlook_units(values: Dict[str, object], units: str = "metric") -> Dict[str, object]:
    """Convert metric model outputs for imperial or metric presentation."""
    if units not in {"metric", "imperial"}:
        raise ValueError("units must be 'metric' or 'imperial'")

    def convert_temp(c: float) -> Dict[str, float]:
        return {"c": c, "f_delta": c * 9 / 5}

    def convert_precip(mm: float) -> Dict[str, float]:
        return {"mm": mm, "in": mm / 25.4}

    def convert_snow(cm: float) -> Dict[str, float]:
        return {"cm": cm, "in": cm / 2.54}

    display = {
        "temperature": convert_temp(float(values["statewide_temperature_anomaly_c"])),
        "precipitation": convert_precip(float(values["statewide_precipitation_mm"])),
        "snowfall": convert_snow(float(values["statewide_snowfall_cm"])),
        "season_phases": [],
    }
    for phase in values.get("season_phases", []):
        display["season_phases"].append({
            "id": phase["id"],
            "label": phase["label"],
            "months": phase["months"],
            "category": phase["category"],
            "temperature": convert_temp(float(phase["predicted_temp_c"])),
            "precipitation": convert_precip(float(phase["precipitation_mm"])),
            "snowfall": convert_snow(float(phase["snowfall_cm"])),
            "precipitation_pct_normal": phase["precipitation_pct_normal"],
        })
    return display


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
