"""Forecast models.

Baseline libraries are optional and can be large, so imports are resolved only
when a caller asks for a particular legacy model.
"""

from importlib import import_module

__all__ = [
    'XGBoostForecaster',
    'LSTMPredictor',
    'ProphetForecaster',
    'RandomForestForecaster',
    'LinearRegressionForecaster'
]

_MODEL_MODULES = {
    "XGBoostForecaster": ".baseline.xgboost_model",
    "LSTMPredictor": ".baseline.lstm_model",
    "ProphetForecaster": ".baseline.prophet_model",
    "RandomForestForecaster": ".baseline.random_forest_model",
    "LinearRegressionForecaster": ".baseline.linear_model",
}


def __getattr__(name):
    if name not in _MODEL_MODULES:
        raise AttributeError(name)
    return getattr(import_module(_MODEL_MODULES[name], __name__), name)
