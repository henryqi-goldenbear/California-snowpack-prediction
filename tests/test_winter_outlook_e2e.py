import numpy as np

from src.models.winter_outlook import (
    REGIONS, WINTER_MONTHS, CaliforniaWinterOutlook, ClimateScenario,
    generate_demo_history, summarize_outlook,
)


def test_statewide_winter_outlook_e2e():
    history = generate_demo_history(1981, 2022, random_state=7)
    model = CaliforniaWinterOutlook(n_estimators=35, random_state=7)
    metrics = model.evaluate_last_years(history, years=4)
    model.fit(history)
    forecast = model.predict(2023, ClimateScenario(enso=1.2, pdo=.5))
    summary = summarize_outlook(forecast, history)

    assert len(forecast) == len(REGIONS) * len(WINTER_MONTHS)
    assert set(forecast.region) == set(REGIONS)
    assert set(forecast.month) == set(WINTER_MONTHS)
    assert np.isfinite(forecast[["temperature_anomaly", "precipitation_mm", "snowfall_cm"]]).all().all()
    assert (forecast[["precipitation_mm", "snowfall_cm"]] >= 0).all().all()
    assert len(summary["trajectory"]) == 6
    assert len(summary["season_phases"]) == 3
    assert len(summary["regional_summary"]) == 7
    assert all(region["risks"] for region in summary["regional_summary"])
    assert {phase["id"] for phase in summary["season_phases"]} == {"early", "mid", "late"}
    assert summary["display"]["temperature"]["c"] == summary["statewide_temperature_anomaly_c"]
    assert summary["statewide_wetness"] in {"dry", "near_normal", "wet"}
    assert metrics["precipitation_r2"] > 0.45
    assert metrics["snowfall_r2"] > 0.45
    assert metrics["temperature_mae_c"] < 0.65


def test_climate_indices_change_the_forecast():
    history = generate_demo_history(1981, 2022, random_state=11)
    model = CaliforniaWinterOutlook(n_estimators=35, random_state=11).fit(history)
    cold_pdo = model.predict(2023, ClimateScenario(enso=-1.5, pdo=-1.0))
    warm_pdo = model.predict(2023, ClimateScenario(enso=1.5, pdo=1.0))
    assert not np.allclose(cold_pdo.precipitation_mm, warm_pdo.precipitation_mm)
    assert not np.allclose(cold_pdo.snowfall_cm, warm_pdo.snowfall_cm)
    assert not np.allclose(cold_pdo.temperature_anomaly, warm_pdo.temperature_anomaly)


def test_recency_weighting_favors_recent_years():
    history = generate_demo_history(1981, 2022, random_state=13)
    recent = CaliforniaWinterOutlook(n_estimators=35, random_state=13, recency_half_life_years=5.0).fit(history)
    uniform = CaliforniaWinterOutlook(n_estimators=35, random_state=13, recency_half_life_years=0.0).fit(history)
    scenario = ClimateScenario(enso=1.4, pdo=0.6, ao=-0.3, pna=0.5)
    recent_forecast = recent.predict(2023, scenario)
    uniform_forecast = uniform.predict(2023, scenario)
    assert not np.allclose(recent_forecast.precipitation_mm, uniform_forecast.precipitation_mm)


def test_recency_weighted_climatology():
    history = generate_demo_history(1981, 2022, random_state=17)
    summary = summarize_outlook(
        CaliforniaWinterOutlook(n_estimators=35, random_state=17).fit(history).predict(2023, ClimateScenario(0.5, 0.2)),
        history,
        recency_half_life_years=6.0,
    )
    assert summary["recency_half_life_years"] == 6.0


def test_predicted_temperature_controls_snowfall():
    history = generate_demo_history(1981, 2022, random_state=19)
    model = CaliforniaWinterOutlook(n_estimators=45, random_state=19).fit(history)
    colder = model.predict(2023, ClimateScenario(enso=-1.8, pdo=-1.0, ao=1.0, pna=-1.0))
    warmer = model.predict(2023, ClimateScenario(enso=1.8, pdo=1.0, ao=-1.0, pna=1.0))
    assert warmer.temperature_anomaly.mean() > colder.temperature_anomaly.mean()
    assert warmer.snowfall_cm.sum() != colder.snowfall_cm.sum()
