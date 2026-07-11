#!/usr/bin/env python3
"""Run a complete offline California winter outlook and backtest."""

import argparse
import json

from src.models.winter_outlook import (
    CaliforniaWinterOutlook, ClimateScenario, generate_demo_history, summarize_outlook,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Climate-driven California winter outlook")
    parser.add_argument("--water-year", type=int, default=2025)
    parser.add_argument("--enso", type=float, default=0.0, help="Oceanic Nino Index anomaly")
    parser.add_argument("--pdo", type=float, default=0.0, help="Pacific Decadal Oscillation index")
    parser.add_argument("--ao", type=float, default=0.0)
    parser.add_argument("--pna", type=float, default=0.0)
    parser.add_argument("--quick", action="store_true", help="Use fewer trees for an E2E smoke run")
    args = parser.parse_args()

    history = generate_demo_history(end_water_year=args.water_year - 1)
    model = CaliforniaWinterOutlook(n_estimators=40 if args.quick else 180)
    metrics = model.evaluate_last_years(history)
    model.fit(history)
    scenario = ClimateScenario(args.enso, args.pdo, args.ao, args.pna)
    result = summarize_outlook(model.predict(args.water_year, scenario), history)
    result["backtest"] = metrics
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
