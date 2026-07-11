"""
Random Forest Model for Snowpack Prediction

Implements a Random Forest-based forecaster for California snowpack prediction.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import warnings


@dataclass
class RandomForestConfig:
    """Configuration for Random Forest model."""
    n_estimators: int = 100
    max_depth: Optional[int] = None
    min_samples_split: int = 5
    min_samples_leaf: int = 2
    max_features: str = "auto"  # 'auto', 'sqrt', 'log2'
    bootstrap: bool = True
    random_state: int = 42
    n_jobs: int = -1  # Use all available cores
    warm_start: bool = False


class RandomForestForecaster(BaseEstimator, RegressorMixin):
    """
    Random Forest-based forecaster for snowpack prediction.
    
    This model uses an ensemble of decision trees to predict snowpack metrics
    and provides feature importance for interpretability.
    """
    
    def __init__(self, config: Optional[RandomForestConfig] = None):
        """
        Initialize the Random Forest forecaster.
        
        Args:
            config: Configuration for the Random Forest model
        """
        self.config = config or RandomForestConfig()
        self.model = None
        self.feature_importances_ = None
        
    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        sample_weight: Optional[Union[pd.Series, np.ndarray]] = None
    ) -> 'RandomForestForecaster':
        """
        Fit the Random Forest model to the training data.
        
        Args:
            X: Training features
            y: Target values
            sample_weight: Optional sample weights
            
        Returns:
            Self
        """
        # Convert inputs if needed
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
        if sample_weight is not None and isinstance(sample_weight, pd.Series):
            sample_weight = sample_weight.values
        
        # Create and fit the model
        self.model = RandomForestRegressor(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            min_samples_split=self.config.min_samples_split,
            min_samples_leaf=self.config.min_samples_leaf,
            max_features=self.config.max_features,
            bootstrap=self.config.bootstrap,
            random_state=self.config.random_state,
            n_jobs=self.config.n_jobs,
            warm_start=self.config.warm_start
        )
        
        self.model.fit(X, y, sample_weight=sample_weight)
        
        # Store feature importances
        self.feature_importances_ = dict(zip(
            range(X.shape[1]), 
            self.model.feature_importances_
        )) if hasattr(self.model, 'feature_importances_') else {}
        
        return self
    
    def predict(
        self,
        X: Union[pd.DataFrame, np.ndarray]
    ) -> np.ndarray:
        """
        Make predictions using the trained model.
        
        Args:
            X: Input features
            
        Returns:
            Predicted values
        """
        if self.model is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        return self.model.predict(X)
    
    def predict_future(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        horizon: int = 30
    ) -> np.ndarray:
        """
        Make future predictions (for time series forecasting).
        
        This method assumes X contains lag features and can be used
        for multi-step forecasting.
        
        Args:
            X: Input features for the initial time step
            horizon: Number of time steps to forecast ahead
            
        Returns:
            Array of future predictions
        """
        if self.model is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        
        predictions = []
        current_X = X.copy()
        
        if isinstance(current_X, pd.DataFrame):
            current_X = current_X.values
        
        for _ in range(horizon):
            # Predict next value
            pred = self.predict(current_X)
            predictions.append(pred[0] if len(pred) == 1 else pred.mean())
            
            # Update X with the prediction (for lag features)
            # This is a simplified approach - in practice, you'd need to
            # properly update all lag features
            if current_X.shape[0] > 1:
                # For multiple samples, this is more complex
                break
        
        return np.array(predictions)
    
    def evaluate(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray]
    ) -> Dict[str, float]:
        """
        Evaluate the model on test data.
        
        Args:
            X: Test features
            y: True target values
            
        Returns:
            Dictionary of evaluation metrics
        """
        predictions = self.predict(X)
        
        if isinstance(y, pd.Series):
            y = y.values
        
        metrics = {
            'rmse': float(np.sqrt(mean_squared_error(y, predictions))),
            'mae': float(mean_absolute_error(y, predictions)),
            'r2': float(r2_score(y, predictions)),
            'mse': float(mean_squared_error(y, predictions))
        }
        
        return metrics
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importances from the trained model.
        
        Returns:
            Dictionary of feature indices and importance scores
        """
        if self.feature_importances_ is None:
            return {}
        
        # Normalize importances to sum to 1
        total = sum(self.feature_importances_.values())
        if total == 0:
            return {k: 0.0 for k in self.feature_importances_.keys()}
        
        return {k: v / total for k, v in self.feature_importances_.items()}
    
    def save(self, filepath: str) -> None:
        """
        Save the model to a file.
        
        Args:
            filepath: Path to save the model
        """
        if self.model is None:
            raise ValueError("Model has not been fitted yet.")
        
        joblib.dump({
            'model': self.model,
            'config': self.config,
            'feature_importances_': self.feature_importances_
        }, filepath)
    
    @classmethod
    def load(cls, filepath: str) -> 'RandomForestForecaster':
        """
        Load a saved model from a file.
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            RandomForestForecaster instance
        """
        data = joblib.load(filepath)
        
        forecaster = cls(data['config'])
        forecaster.model = data['model']
        forecaster.feature_importances_ = data['feature_importances_']
        
        return forecaster
    
    def get_params(self, deep: bool = True) -> Dict:
        """Get parameters for this estimator."""
        return {
            'n_estimators': self.config.n_estimators,
            'max_depth': self.config.max_depth,
            'min_samples_split': self.config.min_samples_split,
            'min_samples_leaf': self.config.min_samples_leaf,
            'max_features': self.config.max_features,
            'bootstrap': self.config.bootstrap,
            'random_state': self.config.random_state,
            'n_jobs': self.config.n_jobs,
            'warm_start': self.config.warm_start
        }
    
    def set_params(self, **params) -> 'RandomForestForecaster':
        """Set parameters for this estimator."""
        for param, value in params.items():
            if hasattr(self.config, param):
                setattr(self.config, param, value)
        return self


# Module-level functions
def train_random_forest(
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray],
    config: Optional[RandomForestConfig] = None,
    sample_weight: Optional[Union[pd.Series, np.ndarray]] = None
) -> RandomForestForecaster:
    """
    Train a Random Forest forecaster.
    
    Args:
        X: Training features
        y: Target values
        config: Optional configuration
        sample_weight: Optional sample weights
        
    Returns:
        Trained RandomForestForecaster
    """
    forecaster = RandomForestForecaster(config)
    forecaster.fit(X, y, sample_weight)
    return forecaster


def predict_random_forest(
    model: RandomForestForecaster,
    X: Union[pd.DataFrame, np.ndarray]
) -> np.ndarray:
    """
    Make predictions with a Random Forest forecaster.
    
    Args:
        model: Trained RandomForestForecaster
        X: Input features
        
    Returns:
        Predictions
    """
    return model.predict(X)
