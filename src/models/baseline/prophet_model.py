"""
Prophet Model for Snowpack Prediction

Implements a Facebook Prophet-based forecaster for California snowpack prediction.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from prophet import Prophet
import joblib
import warnings


@dataclass
class ProphetConfig:
    """Configuration for Prophet model."""
    growth: str = "linear"  # 'linear' or 'logistic'
    seasonality_mode: str = "additive"  # 'additive' or 'multiplicative'
    yearly_seasonality: bool = True
    weekly_seasonality: bool = False
    daily_seasonality: bool = False
    seasonality_prior_scale: float = 10.0
    changepoint_prior_scale: float = 0.05
    n_changepoints: int = 25
    changepoint_range: float = 0.8
    interval_width: float = 0.95
    random_state: int = 42


class ProphetForecaster(BaseEstimator, RegressorMixin):
    """
    Prophet-based forecaster for snowpack prediction.
    
    This model uses Facebook's Prophet for time series forecasting,
    which is particularly good at handling seasonality and holidays.
    """
    
    def __init__(self, config: Optional[ProphetConfig] = None):
        """
        Initialize the Prophet forecaster.
        
        Args:
            config: Configuration for the Prophet model
        """
        self.config = config or ProphetConfig()
        self.model = None
        self.feature_columns_ = None
        self.target_column_ = None
        
    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        date_column: str = "date"
    ) -> 'ProphetForecaster':
        """
        Fit the Prophet model to the training data.
        
        Args:
            X: Training features (must include date column)
            y: Target values
            date_column: Name of the date column
            
        Returns:
            Self
        """
        if isinstance(X, pd.DataFrame):
            df = X.copy()
            self.feature_columns_ = X.columns.tolist()
        else:
            # Convert numpy array to DataFrame
            df = pd.DataFrame(X)
            self.feature_columns_ = [f"feature_{i}" for i in range(X.shape[1])]
        
        if isinstance(y, pd.Series):
            target = y.values
        else:
            target = y
        
        # Ensure date column exists
        if date_column not in df.columns:
            raise ValueError(f"Date column '{date_column}' not found in features.")
        
        self.target_column_ = 'y'
        
        # Create Prophet DataFrame
        prophet_df = pd.DataFrame({
            'ds': df[date_column],
            'y': target
        })
        
        # Add additional regressors (features other than date)
        additional_regressors = [col for col in df.columns if col != date_column]
        
        # Create and fit the model
        self.model = Prophet(
            growth=self.config.growth,
            seasonality_mode=self.config.seasonality_mode,
            yearly_seasonality=self.config.yearly_seasonality,
            weekly_seasonality=self.config.weekly_seasonality,
            daily_seasonality=self.config.daily_seasonality,
            seasonality_prior_scale=self.config.seasonality_prior_scale,
            changepoint_prior_scale=self.config.changepoint_prior_scale,
            n_changepoints=self.config.n_changepoints,
            changepoint_range=self.config.changepoint_range,
            interval_width=self.config.interval_width
        )
        
        # Add additional regressors
        for reg in additional_regressors:
            prophet_df[reg] = df[reg].values
            self.model.add_regressor(reg)
        
        # Fit the model
        self.model.fit(prophet_df)
        
        return self
    
    def predict(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        date_column: str = "date"
    ) -> np.ndarray:
        """
        Make predictions using the trained model.
        
        Args:
            X: Input features
            date_column: Name of the date column
            
        Returns:
            Predicted values
        """
        if self.model is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        
        if isinstance(X, pd.DataFrame):
            df = X.copy()
        else:
            df = pd.DataFrame(X, columns=self.feature_columns_)
        
        # Create future DataFrame
        future = pd.DataFrame({
            'ds': df[date_column]
        })
        
        # Add additional regressors
        additional_regressors = [col for col in df.columns if col != date_column]
        for reg in additional_regressors:
            future[reg] = df[reg].values
        
        # Make predictions
        forecast = self.model.predict(future)
        
        return forecast['yhat'].values
    
    def predict_future(
        self,
        periods: int = 30,
        freq: str = "D",
        date_column: str = "date"
    ) -> np.ndarray:
        """
        Make future predictions beyond the training data.
        
        Args:
            periods: Number of periods to forecast
            freq: Frequency of periods ('D' for daily, 'W' for weekly, etc.)
            date_column: Name of the date column
            
        Returns:
            Array of future predictions
        """
        if self.model is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        
        # Create future DataFrame
        future = self.model.make_future_dataframe(periods=periods, freq=freq)
        
        # Make predictions
        forecast = self.model.predict(future)
        
        # Return only the future predictions (not the historical ones)
        return forecast['yhat'].values[-periods:]
    
    def evaluate(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        date_column: str = "date"
    ) -> Dict[str, float]:
        """
        Evaluate the model on test data.
        
        Args:
            X: Test features
            y: True target values
            date_column: Name of the date column
            
        Returns:
            Dictionary of evaluation metrics
        """
        predictions = self.predict(X, date_column)
        
        if isinstance(y, pd.Series):
            y = y.values
        
        metrics = {
            'rmse': float(np.sqrt(mean_squared_error(y, predictions))),
            'mae': float(mean_absolute_error(y, predictions)),
            'r2': float(r2_score(y, predictions)),
            'mse': float(mean_squared_error(y, predictions))
        }
        
        return metrics
    
    def get_forecast_components(self) -> Dict:
        """
        Get the components of the forecast (trend, seasonality, etc.).
        
        Returns:
            Dictionary with forecast components
        """
        if self.model is None:
            raise ValueError("Model has not been fitted yet.")
        
        # This would require the model to have been used for prediction
        # For now, return empty dict
        return {}
    
    def save(self, filepath: str) -> None:
        """
        Save the model to a file.
        
        Args:
            filepath: Path to save the model
        """
        if self.model is None:
            raise ValueError("Model has not been fitted yet.")
        
        # Save the model using joblib
        joblib.dump({
            'model': self.model,
            'config': self.config,
            'feature_columns_': self.feature_columns_,
            'target_column_': self.target_column_
        }, filepath)
    
    @classmethod
    def load(cls, filepath: str) -> 'ProphetForecaster':
        """
        Load a saved model from a file.
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            ProphetForecaster instance
        """
        data = joblib.load(filepath)
        
        forecaster = cls(data['config'])
        forecaster.model = data['model']
        forecaster.feature_columns_ = data['feature_columns_']
        forecaster.target_column_ = data['target_column_']
        
        return forecaster
    
    def get_params(self, deep: bool = True) -> Dict:
        """Get parameters for this estimator."""
        return {
            'growth': self.config.growth,
            'seasonality_mode': self.config.seasonality_mode,
            'yearly_seasonality': self.config.yearly_seasonality,
            'weekly_seasonality': self.config.weekly_seasonality,
            'daily_seasonality': self.config.daily_seasonality,
            'seasonality_prior_scale': self.config.seasonality_prior_scale,
            'changepoint_prior_scale': self.config.changepoint_prior_scale,
            'n_changepoints': self.config.n_changepoints,
            'changepoint_range': self.config.changepoint_range,
            'interval_width': self.config.interval_width,
            'random_state': self.config.random_state
        }
    
    def set_params(self, **params) -> 'ProphetForecaster':
        """Set parameters for this estimator."""
        for param, value in params.items():
            if hasattr(self.config, param):
                setattr(self.config, param, value)
        return self


# Module-level functions
def train_prophet(
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray],
    config: Optional[ProphetConfig] = None,
    date_column: str = "date"
) -> ProphetForecaster:
    """
    Train a Prophet forecaster.
    
    Args:
        X: Training features
        y: Target values
        config: Optional configuration
        date_column: Name of the date column
        
    Returns:
        Trained ProphetForecaster
    """
    forecaster = ProphetForecaster(config)
    forecaster.fit(X, y, date_column)
    return forecaster


def predict_prophet(
    model: ProphetForecaster,
    X: Union[pd.DataFrame, np.ndarray],
    date_column: str = "date"
) -> np.ndarray:
    """
    Make predictions with a Prophet forecaster.
    
    Args:
        model: Trained ProphetForecaster
        X: Input features
        date_column: Name of the date column
        
    Returns:
        Predictions
    """
    return model.predict(X, date_column)
