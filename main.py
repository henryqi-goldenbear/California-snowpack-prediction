#!/usr/bin/env python3
"""
California Snowpack Prediction - Main Script

This script demonstrates the core functionality of the snowpack prediction pipeline.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import argparse
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data.data_loader import load_snowpack_data, load_climate_data, load_enso_data
from data.data_cleaner import clean_data
from data.data_preprocessor import DataPreprocessor, train_test_split
from models.baseline.xgboost_model import XGBoostForecaster, XGBoostConfig
from agent.experiment_agent import ExperimentAgent
from agent.forecast_explainer import ForecastExplainer
from models.winter_outlook import CaliforniaWinterOutlook, ClimateScenario, generate_demo_history, summarize_outlook


def main():
    """Main function to run the snowpack prediction pipeline."""
    
    parser = argparse.ArgumentParser(description='California Snowpack Prediction Pipeline')
    parser.add_argument('--start-date', type=str, default='2010-01-01', 
                        help='Start date for data loading (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default='2023-12-31',
                        help='End date for data loading (YYYY-MM-DD)')
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='Test size for train-test split')
    parser.add_argument('--quick', action='store_true',
                        help='Run quick demo with smaller dataset')
    parser.add_argument('--no-agent', action='store_true',
                        help='Skip Mistral agent functionality')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("California Snowpack Prediction Pipeline")
    print("=" * 60)
    print(f"Start date: {args.start_date}")
    print(f"End date: {args.end_date}")
    print(f"Test size: {args.test_size}")
    print()
    
    # Load data
    print("1. Loading data...")
    if args.quick:
        # Use smaller date range for quick demo
        start_date = (datetime.now().year - 2).__str__() + "-01-01"
        end_date = datetime.now().strftime("%Y-%m-%d")
    else:
        start_date = args.start_date
        end_date = args.end_date
    
    snowpack_data = load_snowpack_data(
        source="cdec",
        start_date=start_date,
        end_date=end_date
    )
    
    climate_data = load_climate_data(
        source="noaa",
        start_date=start_date,
        end_date=end_date
    )
    
    enso_data = load_enso_data(
        start_date=start_date,
        end_date=end_date
    )
    
    print(f"   - Snowpack data: {snowpack_data.shape[0]} samples")
    print(f"   - Climate data: {climate_data.shape[0]} samples")
    print(f"   - ENSO data: {enso_data.shape[0]} samples")
    print()
    
    # Clean data
    print("2. Cleaning data...")
    cleaned_snowpack = clean_data(
        snowpack_data,
        target_columns=['swe', 'snow_depth']
    )
    
    cleaned_climate = clean_data(climate_data)
    
    print(f"   - Cleaned snowpack: {cleaned_snowpack.shape[0]} samples")
    print(f"   - Cleaned climate: {cleaned_climate.shape[0]} samples")
    print()
    
    # Merge and preprocess data
    print("3. Preprocessing data...")
    merged_data = pd.merge(
        cleaned_snowpack,
        cleaned_climate,
        on=['date', 'station_id'],
        how='left'
    )
    
    merged_data = pd.merge(
        merged_data,
        enso_data,
        on='date',
        how='left'
    )
    
    # Create features
    preprocessor = DataPreprocessor(config={
        'target_column': 'swe',
        'date_column': 'date',
        'station_column': 'station_id',
        'create_time_features': True,
        'create_lag_features': True,
        'lag_periods': [1, 7, 30],
        'create_rolling_features': True,
        'rolling_windows': [7, 30],
        'create_climate_features': True,
        'create_enso_features': True
    })
    
    features_data = preprocessor.create_features(merged_data, enso_data)
    X, y = preprocessor.preprocess_data(features_data)
    
    print(f"   - Features created: {X.shape[1]} features")
    print(f"   - Target variable: swe")
    print()
    
    # Train-test split
    print("4. Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=args.test_size,
        temporal_split=True,
        random_state=42
    )
    
    print(f"   - Training set: {X_train.shape[0]} samples")
    print(f"   - Test set: {X_test.shape[0]} samples")
    print()
    
    # Train model
    print("5. Training XGBoost model...")
    xgb_config = XGBoostConfig(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        random_state=42
    )
    
    xgb_model = XGBoostForecaster(xgb_config)
    xgb_model.fit(X_train, y_train, validation_data=(X_test, y_test))
    
    # Evaluate
    metrics = xgb_model.evaluate(X_test, y_test)
    print("   Model Metrics:")
    for metric, value in metrics.items():
        print(f"     - {metric.upper()}: {value:.4f}")
    
    # Feature importance
    feature_importance = xgb_model.get_feature_importance()
    print("\n   Top 5 Feature Importances:")
    for feature, importance in sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"     - {feature}: {importance:.4f}")
    print()
    
    # Use Mistral agents (if not skipped)
    if not args.no_agent:
        try:
            # Experiment agent
            print("6. Using Mistral Experiment Agent...")
            agent = ExperimentAgent()
            
            # Create data summary
            data_summary = {
                'num_samples': X_train.shape[0],
                'num_features': X_train.shape[1],
                'target_variable': 'swe',
                'date_range': {
                    'start': str(start_date),
                    'end': str(end_date)
                }
            }
            
            # Get suggestions
            feature_suggestions = agent.suggest_features(
                variables=list(X_train.columns),
                target_variable='swe',
                data_summary=data_summary
            )
            
            print(f"   - Got {len(feature_suggestions)} feature suggestions")
            for i, suggestion in enumerate(feature_suggestions[:2], 1):
                print(f"     {i}. {suggestion.feature_name} ({suggestion.importance})")
            
            agent.close()
            print()
            
            # Forecast explainer
            print("7. Using Mistral Forecast Explainer...")
            explainer = ForecastExplainer()
            
            test_predictions = xgb_model.predict(X_test)
            explanation = explainer.explain_forecast(
                predictions=test_predictions,
                feature_importance=feature_importance,
                historical_context=merged_data,
                forecast_horizon="30 days",
                target_variable="swe"
            )
            
            print(f"   - Forecast: {explanation.predicted_value:.2f} inches")
            print(f"   - Confidence: {explanation.confidence}")
            print(f"   - Risk Level: {explanation.risk_level}")
            print(f"   - Summary: {explanation.natural_language_summary[:100]}...")
            
            explainer.close()
            print()
            
        except Exception as e:
            print(f"   Mistral agent error (API key may be required): {e}")
            print()
    
    # Climate-index winter outlook (supported workflow)
    print("8. Running California winter outlook...")
    try:
        history = generate_demo_history(1981, 2022, random_state=7)
        outlook = CaliforniaWinterOutlook(n_estimators=40 if args.quick else 120, random_state=7).fit(history)
        scenario = ClimateScenario(enso=1.2, pdo=0.5, ao=-0.2, pna=0.4)
        forecast = outlook.predict(2025, scenario)
        summary = summarize_outlook(forecast, history)

        print(f"   - Statewide wetness: {summary['statewide_wetness']}")
        print(f"   - Precipitation vs normal: {summary['statewide_precipitation_pct_normal']:.1f}%")
        print(f"   - Temperature anomaly: {summary['statewide_temperature_anomaly_c']:+.2f} °C")
        print(f"   - Season phases: {len(summary['season_phases'])}")
        print(f"   - Regional outlooks: {len(summary['regional_summary'])}")
    except Exception as e:
        print(f"   Winter outlook error: {e}")
    
    print()
    print("=" * 60)
    print("Pipeline completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
