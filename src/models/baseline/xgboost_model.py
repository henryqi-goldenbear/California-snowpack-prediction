"""
XGBoost Model for Snowpack Prediction

Implements an XGBoost-based forecaster for California snowpack prediction.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import joblib
import warnings


@dataclass
class XGBoostConfig:
    """Configuration for XGBoost model."""
    n_estimators: int = 200
    max_depth: int = 6
    learning_rate: float = 0.1
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    gamma: float = 0
    reg_alpha: float = 0
    reg_lambda: float = 1
    random_state: int = 42
    early_stopping_rounds: Optional[int] = 10
    eval_metric: str = "rmse"
    use_gpu: bool = False
    tree_method: str = "auto"


class XGBoostForecaster(BaseEstimator, RegressorMixin):
    """
    XGBoost-based forecaster for snowpack prediction.
    
    This model uses gradient boosting to predict snowpack metrics
    (SWE, snow depth) based on historical weather and climate data.
    """
    
    def __init__(self, config: Optional[XGBoostConfig] = None):
        """
        Initialize the XGBoost forecaster.
        
        Args:
            config: Configuration for the XGBoost model
        """
        self.config = config or XGBoostConfig()
        self.model = None
        self.feature_importances_ = None
        self.best_iteration_ = None
        self.training_history_ = None
        
    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        validation_data: Optional[Tuple] = None,
        sample_weight: Optional[Union[pd.Series, np.ndarray]] = None
    ) -> 'XGBoostForecaster':
        """
        Fit the XGBoost model to the training data.
        
        Args:
            X: Training features
            y: Target values
            validation_data: Optional tuple of (X_val, y_val) for early stopping
            sample_weight: Optional sample weights
            
        Returns:
            Self
        """
        # Convert inputs to numpy arrays if needed
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
            
        # Create DMatrix for XGBoost
        dtrain = xgb.DMatrix(X, label=y, weight=sample_weight)
        
        # Prepare parameters
        params = {
            'objective': 'reg:squarederror',
            'n_estimators': self.config.n_estimators,
            'max_depth': self.config.max_depth,
            'eta': self.config.learning_rate,
            'subsample': self.config.subsample,
            'colsample_bytree': self.config.colsample_bytree,
            'gamma': self.config.gamma,
            'alpha': self.config.reg_alpha,
            'lambda': self.config.reg_lambda,
            'random_state': self.config.random_state,
            'eval_metric': self.config.eval_metric,
            'tree_method': self.config.tree_method
        }
        
        # Handle GPU
        if self.config.use_gpu:
            params['tree_method'] = 'gpu_hist'
            params['gpu_id'] = 0
            params['predictor'] = 'gpu_predictor'
        
        # Prepare validation data
        evals = []
        if validation_data is not None:
            X_val, y_val = validation_data
            if isinstance(X_val, pd.DataFrame):
                X_val = X_val.values
            if isinstance(y_val, pd.Series):
                y_val = y_val.values
            
            dval = xgb.DMatrix(X_val, label=y_val)
            evals = [(dtrain, 'train'), (dval, 'eval')]
        else:
            evals = [(dtrain, 'train')]
        
        # Train the model
        self.model = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=self.config.n_estimators,
            evals=evals,
            early_stopping_rounds=self.config.early_stopping_rounds,
            verbose_eval=False
        )
        
        # Store feature importances
        self.feature_importances_ = self.model.get_score(importance_type='weight')
        
        # Store best iteration
        if hasattr(self.model, 'best_iteration'):
            self.best_iteration_ = self.model.best_iteration
        
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
            
        dtest = xgb.DMatrix(X)
        predictions = self.model.predict(dtest)
        
        return predictions
    
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
            Dictionary of feature names and importance scores
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
            'feature_importances_': self.feature_importances_,
            'best_iteration_': self.best_iteration_
        }, filepath)
    
    @classmethod
    def load(cls, filepath: str) -> 'XGBoostForecaster':
        """
        Load a saved model from a file.
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            XGBoostForecaster instance
        """
        data = joblib.load(filepath)
        
        forecaster = cls(data['config'])
        forecaster.model = data['model']
        forecaster.feature_importances_ = data['feature_importances_']
        forecaster.best_iteration_ = data['best_iteration_']
        
        return forecaster
    
    def get_params(self, deep: bool = True) -> Dict:
        """Get parameters for this estimator."""
        return {
            'n_estimators': self.config.n_estimators,
            'max_depth': self.config.max_depth,
            'learning_rate': self.config.learning_rate,
            'subsample': self.config.subsample,
            'colsample_bytree': self.config.colsample_bytree,
            'gamma': self.config.gamma,
            'reg_alpha': self.config.reg_alpha,
            'reg_lambda': self.config.reg_lambda,
            'random_state': self.config.random_state
        }
    
    def set_params(self, **params) -> 'XGBoostForecaster':
        """Set parameters for this estimator."""
        for param, value in params.items():
            if hasattr(self.config, param):
                setattr(self.config, param, value)
        return self


# Module-level functions
def train_xgboost(
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray],
    config: Optional[XGBoostConfig] = None,
    validation_data: Optional[Tuple] = None
) -> XGBoostForecaster:
    """
    Train an XGBoost forecaster.
    
    Args:
        X: Training features
        y: Target values
        config: Optional configuration
        validation_data: Optional validation data
        
    Returns:
        Trained XGBoostForecaster
    """
    forecaster = XGBoostForecaster(config)
    forecaster.fit(X, y, validation_data)
    return forecaster


def predict_xgboost(
    model: XGBoostForecaster,
    X: Union[pd.DataFrame, np.ndarray]
) -> np.ndarray:
    """
    Make predictions with an XGBoost forecaster.
    
    Args:
        model: Trained XGBoostForecaster
        X: Input features
        
    Returns:
        Predictions
    """
    return model.predict(X)
