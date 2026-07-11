"""
Linear Regression Model for Snowpack Prediction

Implements a linear regression-based forecaster for California snowpack prediction.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import warnings


@dataclass
class LinearModelConfig:
    """Configuration for Linear model."""
    model_type: str = "linear"  # 'linear', 'ridge', 'lasso', 'elasticnet'
    alpha: float = 1.0  # Regularization strength for ridge, lasso, elasticnet
    l1_ratio: float = 0.5  # Ratio of L1 regularization for elasticnet
    fit_intercept: bool = True
    normalize: bool = False
    random_state: int = 42


class LinearRegressionForecaster(BaseEstimator, RegressorMixin):
    """
    Linear regression-based forecaster for snowpack prediction.
    
    This model uses linear regression (with optional regularization)
    to predict snowpack metrics and provides interpretable coefficients.
    """
    
    def __init__(self, config: Optional[LinearModelConfig] = None):
        """
        Initialize the Linear Regression forecaster.
        
        Args:
            config: Configuration for the Linear model
        """
        self.config = config or LinearModelConfig()
        self.model = None
        self.coefficients_ = None
        self.intercept_ = None
        self.feature_names_ = None
        
    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        sample_weight: Optional[Union[pd.Series, np.ndarray]] = None
    ) -> 'LinearRegressionForecaster':
        """
        Fit the Linear model to the training data.
        
        Args:
            X: Training features
            y: Target values
            sample_weight: Optional sample weights
            
        Returns:
            Self
        """
        # Store feature names
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = X.columns.tolist()
            X = X.values
        else:
            self.feature_names_ = [f"feature_{i}" for i in range(X.shape[1])]
        
        if isinstance(y, pd.Series):
            y = y.values
        if sample_weight is not None and isinstance(sample_weight, pd.Series):
            sample_weight = sample_weight.values
        
        # Create the appropriate model
        if self.config.model_type == "linear":
            self.model = LinearRegression(
                fit_intercept=self.config.fit_intercept,
                normalize=self.config.normalize
            )
        elif self.config.model_type == "ridge":
            self.model = Ridge(
                alpha=self.config.alpha,
                fit_intercept=self.config.fit_intercept,
                normalize=self.config.normalize,
                random_state=self.config.random_state
            )
        elif self.config.model_type == "lasso":
            self.model = Lasso(
                alpha=self.config.alpha,
                fit_intercept=self.config.fit_intercept,
                normalize=self.config.normalize,
                random_state=self.config.random_state
            )
        elif self.config.model_type == "elasticnet":
            self.model = ElasticNet(
                alpha=self.config.alpha,
                l1_ratio=self.config.l1_ratio,
                fit_intercept=self.config.fit_intercept,
                normalize=self.config.normalize,
                random_state=self.config.random_state
            )
        else:
            raise ValueError(f"Unknown model type: {self.config.model_type}")
        
        # Fit the model
        self.model.fit(X, y, sample_weight=sample_weight)
        
        # Store coefficients
        if hasattr(self.model, 'coef_'):
            self.coefficients_ = dict(zip(range(len(self.model.coef_)), self.model.coef_))
        if hasattr(self.model, 'intercept_'):
            self.intercept_ = self.model.intercept_
        
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
    
    def get_coefficients(self) -> Dict[str, float]:
        """
        Get model coefficients with feature names.
        
        Returns:
            Dictionary of feature names and coefficients
        """
        if self.coefficients_ is None or self.feature_names_ is None:
            return {}
        
        return {name: self.coefficients_[i] for i, name in enumerate(self.feature_names_)}
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance (absolute value of coefficients).
        
        Returns:
            Dictionary of feature names and importance scores
        """
        coefficients = self.get_coefficients()
        if not coefficients:
            return {}
        
        # Use absolute values for importance
        importances = {k: abs(v) for k, v in coefficients.items()}
        
        # Normalize to sum to 1
        total = sum(importances.values())
        if total == 0:
            return {k: 0.0 for k in importances.keys()}
        
        return {k: v / total for k, v in importances.items()}
    
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
            'coefficients_': self.coefficients_,
            'intercept_': self.intercept_,
            'feature_names_': self.feature_names_
        }, filepath)
    
    @classmethod
    def load(cls, filepath: str) -> 'LinearRegressionForecaster':
        """
        Load a saved model from a file.
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            LinearRegressionForecaster instance
        """
        data = joblib.load(filepath)
        
        forecaster = cls(data['config'])
        forecaster.model = data['model']
        forecaster.coefficients_ = data['coefficients_']
        forecaster.intercept_ = data['intercept_']
        forecaster.feature_names_ = data['feature_names_']
        
        return forecaster
    
    def get_params(self, deep: bool = True) -> Dict:
        """Get parameters for this estimator."""
        return {
            'model_type': self.config.model_type,
            'alpha': self.config.alpha,
            'l1_ratio': self.config.l1_ratio,
            'fit_intercept': self.config.fit_intercept,
            'normalize': self.config.normalize,
            'random_state': self.config.random_state
        }
    
    def set_params(self, **params) -> 'LinearRegressionForecaster':
        """Set parameters for this estimator."""
        for param, value in params.items():
            if hasattr(self.config, param):
                setattr(self.config, param, value)
        return self


# Module-level functions
def train_linear(
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray],
    config: Optional[LinearModelConfig] = None,
    sample_weight: Optional[Union[pd.Series, np.ndarray]] = None
) -> LinearRegressionForecaster:
    """
    Train a Linear Regression forecaster.
    
    Args:
        X: Training features
        y: Target values
        config: Optional configuration
        sample_weight: Optional sample weights
        
    Returns:
        Trained LinearRegressionForecaster
    """
    forecaster = LinearRegressionForecaster(config)
    forecaster.fit(X, y, sample_weight)
    return forecaster


def predict_linear(
    model: LinearRegressionForecaster,
    X: Union[pd.DataFrame, np.ndarray]
) -> np.ndarray:
    """
    Make predictions with a Linear Regression forecaster.
    
    Args:
        model: Trained LinearRegressionForecaster
        X: Input features
        
    Returns:
        Predictions
    """
    return model.predict(X)
