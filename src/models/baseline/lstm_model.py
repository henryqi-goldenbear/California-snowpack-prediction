"""
LSTM Model for Snowpack Prediction

Implements an LSTM-based predictor for California snowpack time series forecasting.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import joblib
import warnings


@dataclass
class LSTMConfig:
    """Configuration for LSTM model."""
    layers: List[int] = field(default_factory=lambda: [64, 32])
    dropout_rate: float = 0.2
    recurrent_dropout_rate: float = 0.2
    batch_size: int = 32
    epochs: int = 100
    learning_rate: float = 0.001
    validation_split: float = 0.2
    early_stopping_patience: int = 10
    reduce_lr_patience: int = 5
    loss: str = "mse"
    optimizer: str = "adam"
    random_state: int = 42
    use_batch_norm: bool = True
    return_sequences: bool = False


class LSTMPredictor(BaseEstimator, RegressorMixin):
    """
    LSTM-based predictor for snowpack time series forecasting.
    
    This model uses Long Short-Term Memory networks to capture temporal
    dependencies in snowpack data.
    """
    
    def __init__(self, config: Optional[LSTMConfig] = None):
        """
        Initialize the LSTM predictor.
        
        Args:
            config: Configuration for the LSTM model
        """
        self.config = config or LSTMConfig()
        self.model = None
        self.scaler = None
        self.feature_names_ = None
        self.target_scaler = None
        self.history_ = None
        self.input_shape_ = None
        
    def _create_model(self, input_shape: tuple) -> tf.keras.Model:
        """
        Create the LSTM model architecture.
        
        Args:
            input_shape: Shape of input data (timesteps, features)
            
        Returns:
            Compiled Keras model
        """
        model = Sequential()
        
        # Input layer
        model.add(LSTM(
            units=self.config.layers[0],
            input_shape=input_shape,
            return_sequences=len(self.config.layers) > 1 or self.config.return_sequences,
            dropout=self.config.dropout_rate,
            recurrent_dropout=self.config.recurrent_dropout_rate
        ))
        
        if self.config.use_batch_norm:
            model.add(BatchNormalization())
        
        # Hidden layers
        for i, units in enumerate(self.config.layers[1:]):
            return_seq = self.config.return_sequences and i < len(self.config.layers) - 1
            model.add(LSTM(
                units=units,
                return_sequences=return_seq,
                dropout=self.config.dropout_rate,
                recurrent_dropout=self.config.recurrent_dropout_rate
            ))
            if self.config.use_batch_norm:
                model.add(BatchNormalization())
        
        # Output layer
        model.add(Dense(1))
        
        # Compile the model
        optimizer = Adam(learning_rate=self.config.learning_rate)
        model.compile(
            optimizer=optimizer,
            loss=self.config.loss,
            metrics=['mae']
        )
        
        return model
    
    def _prepare_data(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        timesteps: int = 1
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data for LSTM training.
        
        Args:
            X: Input features
            y: Target values
            timesteps: Number of timesteps for time series
            
        Returns:
            Tuple of (X_train, y_train, X_val, y_val)
        """
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = X.columns.tolist()
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
        
        # Scale the data
        if self.scaler is None:
            self.scaler = MinMaxScaler()
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        
        # Scale target
        if self.target_scaler is None:
            self.target_scaler = MinMaxScaler()
            y_scaled = self.target_scaler.fit_transform(y.reshape(-1, 1)).flatten()
        else:
            y_scaled = self.target_scaler.transform(y.reshape(-1, 1)).flatten()
        
        # Create time series data
        X_ts, y_ts = self._create_timeseries_data(X_scaled, y_scaled, timesteps)
        
        # Split into train and validation
        split_idx = int(len(X_ts) * (1 - self.config.validation_split))
        
        X_train, X_val = X_ts[:split_idx], X_ts[split_idx:]
        y_train, y_val = y_ts[:split_idx], y_ts[split_idx:]
        
        return X_train, y_train, X_val, y_val
    
    def _create_timeseries_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
        timesteps: int = 1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create time series data for LSTM.
        
        Args:
            X: Input features
            y: Target values
            timesteps: Number of timesteps
            
        Returns:
            Tuple of (X_ts, y_ts) where X_ts has shape (n_samples, timesteps, n_features)
        """
        X_ts, y_ts = [], []
        
        for i in range(timesteps, len(X)):
            X_ts.append(X[i-timesteps:i])
            y_ts.append(y[i])
        
        return np.array(X_ts), np.array(y_ts)
    
    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        timesteps: int = 1,
        validation_data: Optional[Tuple] = None
    ) -> 'LSTMPredictor':
        """
        Fit the LSTM model to the training data.
        
        Args:
            X: Training features
            y: Target values
            timesteps: Number of timesteps for time series
            validation_data: Optional validation data
            
        Returns:
            Self
        """
        # Prepare the data
        X_train, y_train, X_val, y_val = self._prepare_data(X, y, timesteps)
        
        # Store input shape
        self.input_shape_ = (X_train.shape[1], X_train.shape[2])
        
        # Create the model
        self.model = self._create_model(self.input_shape_)
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=self.config.early_stopping_patience,
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                patience=self.config.reduce_lr_patience,
                factor=0.5,
                min_lr=1e-7
            )
        ]
        
        # Train the model
        self.history_ = self.model.fit(
            X_train, y_train,
            batch_size=self.config.batch_size,
            epochs=self.config.epochs,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=0
        )
        
        return self
    
    def predict(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        timesteps: int = 1
    ) -> np.ndarray:
        """
        Make predictions using the trained model.
        
        Args:
            X: Input features
            timesteps: Number of timesteps used in training
            
        Returns:
            Predicted values
        """
        if self.model is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        # Scale the input
        X_scaled = self.scaler.transform(X)
        
        # Create time series data
        X_ts, _ = self._create_timeseries_data(X_scaled, np.zeros(len(X)), timesteps)
        
        # Predict
        predictions_scaled = self.model.predict(X_ts).flatten()
        
        # Inverse transform the predictions
        predictions = self.target_scaler.inverse_transform(predictions_scaled.reshape(-1, 1)).flatten()
        
        return predictions
    
    def predict_future(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        horizon: int = 30,
        timesteps: int = 1
    ) -> np.ndarray:
        """
        Make future predictions (multi-step forecasting).
        
        Args:
            X: Input features for the initial time step
            horizon: Number of time steps to forecast ahead
            timesteps: Number of timesteps used in training
            
        Returns:
            Array of future predictions
        """
        if self.model is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        
        predictions = []
        current_X = X.copy()
        
        if isinstance(current_X, pd.DataFrame):
            current_X = current_X.values
        
        # Scale the initial input
        current_X_scaled = self.scaler.transform(current_X)
        
        for _ in range(horizon):
            # Create time series input
            X_ts, _ = self._create_timeseries_data(current_X_scaled, np.zeros(len(current_X_scaled)), timesteps)
            
            # Predict next value
            pred_scaled = self.model.predict(X_ts[-1:]).flatten()
            pred = self.target_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()[0]
            predictions.append(pred)
            
            # Update current_X with the prediction
            # This is a simplified approach - in practice, you'd need to
            # properly update the time series with the new prediction
            if len(current_X_scaled) >= timesteps:
                current_X_scaled = np.vstack([current_X_scaled[1:], pred_scaled.reshape(1, -1)])
            else:
                current_X_scaled = np.vstack([current_X_scaled, pred_scaled.reshape(1, -1)])
        
        return np.array(predictions)
    
    def evaluate(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        timesteps: int = 1
    ) -> Dict[str, float]:
        """
        Evaluate the model on test data.
        
        Args:
            X: Test features
            y: True target values
            timesteps: Number of timesteps used in training
            
        Returns:
            Dictionary of evaluation metrics
        """
        predictions = self.predict(X, timesteps)
        
        if isinstance(y, pd.Series):
            y = y.values
        
        metrics = {
            'rmse': float(np.sqrt(mean_squared_error(y, predictions))),
            'mae': float(mean_absolute_error(y, predictions)),
            'r2': float(r2_score(y, predictions)),
            'mse': float(mean_squared_error(y, predictions))
        }
        
        return metrics
    
    def save(self, filepath: str) -> None:
        """
        Save the model to a file.
        
        Args:
            filepath: Path to save the model
        """
        if self.model is None:
            raise ValueError("Model has not been fitted yet.")
        
        # Save the Keras model
        model_path = filepath + "_model.h5"
        self.model.save(model_path)
        
        # Save the scalers and config
        joblib.dump({
            'config': self.config,
            'scaler': self.scaler,
            'target_scaler': self.target_scaler,
            'feature_names_': self.feature_names_,
            'input_shape_': self.input_shape_,
            'model_path': model_path
        }, filepath)
    
    @classmethod
    def load(cls, filepath: str) -> 'LSTMPredictor':
        """
        Load a saved model from a file.
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            LSTMPredictor instance
        """
        data = joblib.load(filepath)
        
        predictor = cls(data['config'])
        predictor.scaler = data['scaler']
        predictor.target_scaler = data['target_scaler']
        predictor.feature_names_ = data['feature_names_']
        predictor.input_shape_ = data['input_shape_']
        
        # Load the Keras model
        model_path = data.get('model_path', filepath + "_model.h5")
        predictor.model = tf.keras.models.load_model(model_path)
        
        return predictor
    
    def get_params(self, deep: bool = True) -> Dict:
        """Get parameters for this estimator."""
        return {
            'layers': self.config.layers,
            'dropout_rate': self.config.dropout_rate,
            'recurrent_dropout_rate': self.config.recurrent_dropout_rate,
            'batch_size': self.config.batch_size,
            'epochs': self.config.epochs,
            'learning_rate': self.config.learning_rate,
            'validation_split': self.config.validation_split,
            'early_stopping_patience': self.config.early_stopping_patience,
            'reduce_lr_patience': self.config.reduce_lr_patience,
            'loss': self.config.loss,
            'optimizer': self.config.optimizer,
            'random_state': self.config.random_state,
            'use_batch_norm': self.config.use_batch_norm
        }
    
    def set_params(self, **params) -> 'LSTMPredictor':
        """Set parameters for this estimator."""
        for param, value in params.items():
            if hasattr(self.config, param):
                setattr(self.config, param, value)
        return self


# Module-level functions
def train_lstm(
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray],
    config: Optional[LSTMConfig] = None,
    timesteps: int = 7
) -> LSTMPredictor:
    """
    Train an LSTM predictor.
    
    Args:
        X: Training features
        y: Target values
        config: Optional configuration
        timesteps: Number of timesteps for time series
        
    Returns:
        Trained LSTMPredictor
    """
    predictor = LSTMPredictor(config)
    predictor.fit(X, y, timesteps)
    return predictor


def predict_lstm(
    model: LSTMPredictor,
    X: Union[pd.DataFrame, np.ndarray],
    timesteps: int = 7
) -> np.ndarray:
    """
    Make predictions with an LSTM predictor.
    
    Args:
        model: Trained LSTMPredictor
        X: Input features
        timesteps: Number of timesteps used in training
        
    Returns:
        Predictions
    """
    return model.predict(X, timesteps)
