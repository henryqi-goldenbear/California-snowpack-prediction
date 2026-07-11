"""
Data Preprocessor Module

Handles feature engineering, data transformation, and train-test splitting.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from sklearn.model_selection import train_test_split as sklearn_train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from datetime import datetime, timedelta
import warnings


class DataPreprocessor:
    """Main data preprocessor class for California snowpack prediction."""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the data preprocessor.
        
        Args:
            config: Configuration dictionary for preprocessing parameters
        """
        self.config = config or {}
        self._default_config = {
            'target_column': 'swe',
            'date_column': 'date',
            'station_column': 'station_id',
            'test_size': 0.2,
            'random_state': 42,
            'scaling_method': 'standard',  # 'standard', 'minmax', 'robust', 'none'
            'create_time_features': True,
            'create_lag_features': True,
            'lag_periods': [1, 7, 30, 90],
            'create_rolling_features': True,
            'rolling_windows': [7, 30, 90],
            'create_climate_features': True,
            'create_enso_features': True,
            'forecast_horizon': 30  # Days to forecast ahead
        }
        self.config = {**self._default_config, **self.config}
        self.scaler = None
        self.preprocessor = None
        
    def preprocess_data(
        self,
        df: pd.DataFrame,
        target_column: Optional[str] = None,
        date_column: Optional[str] = None,
        station_column: Optional[str] = None
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Preprocess data for modeling.
        
        Args:
            df: Input DataFrame
            target_column: Name of the target column
            date_column: Name of the date column
            station_column: Name of the station column
            
        Returns:
            Tuple of (features DataFrame, target Series)
        """
        # Update config
        if target_column:
            self.config['target_column'] = target_column
        if date_column:
            self.config['date_column'] = date_column
        if station_column:
            self.config['station_column'] = station_column
            
        df_processed = df.copy()
        
        # Preprocessing pipeline
        df_processed = self._ensure_date_column(df_processed)
        df_processed = self._create_time_features(df_processed)
        df_processed = self._create_lag_features(df_processed)
        df_processed = self._create_rolling_features(df_processed)
        df_processed = self._create_climate_features(df_processed)
        df_processed = self._create_enso_features(df_processed)
        df_processed = self._handle_stations(df_processed)
        
        # Separate features and target
        X = df_processed.drop(columns=[self.config['target_column']])
        y = df_processed[self.config['target_column']]
        
        # Scale features
        X = self._scale_features(X)
        
        return X, y
    
    def _ensure_date_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure date column exists and is datetime type."""
        if self.config['date_column'] not in df.columns:
            # Try to find a date-like column
            date_cols = [col for col in df.columns if 'date' in col.lower()]
            if date_cols:
                self.config['date_column'] = date_cols[0]
            else:
                raise ValueError(f"No date column found in DataFrame. Columns: {df.columns.tolist()}")
        
        df[self.config['date_column']] = pd.to_datetime(df[self.config['date_column']])
        return df
    
    def _create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features."""
        if not self.config['create_time_features']:
            return df
            
        date_col = self.config['date_column']
        
        # Basic time features
        if 'year' not in df.columns:
            df['year'] = df[date_col].dt.year
        if 'month' not in df.columns:
            df['month'] = df[date_col].dt.month
        if 'day' not in df.columns:
            df['day'] = df[date_col].dt.day
        if 'day_of_year' not in df.columns:
            df['day_of_year'] = df[date_col].dt.dayofyear
        if 'week_of_year' not in df.columns:
            df['week_of_year'] = df[date_col].dt.isocalendar().week
            
        # Seasonal features
        if 'is_winter' not in df.columns:
            df['is_winter'] = df['month'].isin([12, 1, 2]).astype(int)
        if 'is_spring' not in df.columns:
            df['is_spring'] = df['month'].isin([3, 4, 5]).astype(int)
        if 'is_summer' not in df.columns:
            df['is_summer'] = df['month'].isin([6, 7, 8]).astype(int)
        if 'is_fall' not in df.columns:
            df['is_fall'] = df['month'].isin([9, 10, 11]).astype(int)
            
        # Cyclical encoding for month and day
        if 'month_sin' not in df.columns:
            df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        if 'month_cos' not in df.columns:
            df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        if 'day_sin' not in df.columns:
            df['day_sin'] = np.sin(2 * np.pi * df['day'] / 31)
        if 'day_cos' not in df.columns:
            df['day_cos'] = np.cos(2 * np.pi * df['day'] / 31)
            
        return df
    
    def _create_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create lag features for time series."""
        if not self.config['create_lag_features']:
            return df
            
        # Sort by date and station
        date_col = self.config['date_column']
        station_col = self.config['station_column']
        target_col = self.config['target_column']
        
        # Sort by station and date
        df = df.sort_values([station_col, date_col]).reset_index(drop=True)
        
        # Create lag features for target and key variables
        lag_periods = self.config['lag_periods']
        
        for period in lag_periods:
            # Lag target variable
            lag_col = f'{target_col}_lag_{period}'
            if lag_col not in df.columns:
                df[lag_col] = df.groupby(station_col)[target_col].shift(period)
                
            # Lag temperature
            if 'temperature' in df.columns:
                lag_col = f'temperature_lag_{period}'
                if lag_col not in df.columns:
                    df[lag_col] = df.groupby(station_col)['temperature'].shift(period)
                    
            # Lag precipitation
            if 'precipitation' in df.columns:
                lag_col = f'precipitation_lag_{period}'
                if lag_col not in df.columns:
                    df[lag_col] = df.groupby(station_col)['precipitation'].shift(period)
                    
        return df
    
    def _create_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create rolling window features."""
        if not self.config['create_rolling_features']:
            return df
            
        date_col = self.config['date_column']
        station_col = self.config['station_column']
        target_col = self.config['target_column']
        
        rolling_windows = self.config['rolling_windows']
        
        for window in rolling_windows:
            # Rolling mean of target
            roll_col = f'{target_col}_roll_mean_{window}'
            if roll_col not in df.columns:
                df[roll_col] = df.groupby(station_col)[target_col].transform(
                    lambda x: x.rolling(window, min_periods=1).mean()
                )
                
            # Rolling std of target
            roll_col = f'{target_col}_roll_std_{window}'
            if roll_col not in df.columns:
                df[roll_col] = df.groupby(station_col)[target_col].transform(
                    lambda x: x.rolling(window, min_periods=1).std()
                )
                
            # Rolling sum of precipitation
            if 'precipitation' in df.columns:
                roll_col = f'precipitation_roll_sum_{window}'
                if roll_col not in df.columns:
                    df[roll_col] = df.groupby(station_col)['precipitation'].transform(
                        lambda x: x.rolling(window, min_periods=1).sum()
                    )
                    
            # Rolling mean of temperature
            if 'temperature' in df.columns:
                roll_col = f'temperature_roll_mean_{window}'
                if roll_col not in df.columns:
                    df[roll_col] = df.groupby(station_col)['temperature'].transform(
                        lambda x: x.rolling(window, min_periods=1).mean()
                    )
                    
        return df
    
    def _create_climate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create climate-related features."""
        if not self.config['create_climate_features']:
            return df
            
        # Add cumulative precipitation (water year starts Oct 1)
        if 'precipitation' in df.columns and self.config['date_column'] in df.columns:
            date_col = self.config['date_column']
            station_col = self.config['station_column']
            
            # Water year starts October 1
            df['water_year'] = df[date_col].apply(
                lambda x: x.year if x.month >= 10 else x.year - 1
            )
            df['water_year_start'] = df['water_year'].astype(str) + '-10-01'
            df['water_year_start'] = pd.to_datetime(df['water_year_start'])
            
            # Days since water year start
            df['days_since_wy_start'] = (df[date_col] - df['water_year_start']).dt.days
            
            # Cumulative precipitation since water year start
            if 'precipitation_cum_wy' not in df.columns:
                df['precipitation_cum_wy'] = df.groupby([station_col, 'water_year'])['precipitation'].cumsum()
                
            # Cumulative precipitation since Oct 1 (for current water year)
            if 'precipitation_cum_oct1' not in df.columns:
                oct1_dates = df[date_col].apply(
                    lambda x: datetime(x.year - 1, 10, 1) if x.month < 10 else datetime(x.year, 10, 1)
                )
                df['days_since_oct1'] = (df[date_col] - oct1_dates).dt.days
                df['precipitation_cum_oct1'] = df.groupby(station_col)['precipitation'].apply(
                    lambda x: x.cumsum() - x.shift(1).fillna(0).cumsum()
                )
                
        # Add temperature anomaly features
        if 'temperature' in df.columns and 'month' in df.columns:
            # Monthly temperature climatology (simplified)
            monthly_climatology = df.groupby('month')['temperature'].transform('mean')
            df['temperature_anomaly'] = df['temperature'] - monthly_climatology
            
        return df
    
    def _create_enso_features(self, df: pd.DataFrame, enso_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Create ENSO-related features."""
        if not self.config['create_enso_features']:
            return df
            
        # If ENSO data is provided, merge it
        if enso_data is not None and self.config['date_column'] in df.columns:
            date_col = self.config['date_column']
            
            # Ensure enso_data has date column
            if 'date' in enso_data.columns:
                enso_date_col = 'date'
            else:
                # Try to find date column
                date_cols = [col for col in enso_data.columns if 'date' in col.lower()]
                if date_cols:
                    enso_date_col = date_cols[0]
                else:
                    return df
                    
            # Merge ENSO data
            df = df.merge(
                enso_data, 
                left_on=date_col, 
                right_on=enso_date_col, 
                how='left'
            )
            
            # Add ENSO phase as categorical feature
            if 'enso_phase' in df.columns:
                df = pd.get_dummies(df, columns=['enso_phase'], prefix='enso')
                
        return df
    
    def _handle_stations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle station-specific features."""
        station_col = self.config['station_column']
        
        if station_col in df.columns:
            # One-hot encode stations
            df = pd.get_dummies(df, columns=[station_col], prefix='station')
            
            # Station-specific statistics
            target_col = self.config['target_column']
            if target_col in df.columns:
                # Station mean SWE
                station_mean = df.groupby(station_col)[target_col].transform('mean')
                df['station_mean_swe'] = station_mean
                
                # Station std SWE
                station_std = df.groupby(station_col)[target_col].transform('std')
                df['station_std_swe'] = station_std
                
        return df
    
    def _scale_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Scale features using specified method."""
        method = self.config['scaling_method']
        
        if method == 'none':
            return X
            
        # Get numeric columns
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        
        if method == 'standard':
            self.scaler = StandardScaler()
        elif method == 'minmax':
            self.scaler = MinMaxScaler()
        elif method == 'robust':
            self.scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaling method: {method}")
            
        # Fit and transform
        X[numeric_cols] = self.scaler.fit_transform(X[numeric_cols])
        
        return X
    
    def train_test_split(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: Optional[float] = None,
        random_state: Optional[int] = None,
        temporal_split: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split data into training and testing sets.
        
        Args:
            X: Features DataFrame
            y: Target Series
            test_size: Proportion of data for testing
            random_state: Random seed
            temporal_split: Whether to use temporal split (recommended for time series)
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        test_size = test_size or self.config['test_size']
        random_state = random_state or self.config['random_state']
        
        if temporal_split and self.config['date_column'] in X.columns:
            # Temporal split - use most recent data for testing
            date_col = self.config['date_column']
            
            # Sort by date
            X = X.sort_values(date_col).reset_index(drop=True)
            y = y.loc[X.index]
            
            # Find split point
            split_idx = int(len(X) * (1 - test_size))
            
            X_train = X.iloc[:split_idx]
            X_test = X.iloc[split_idx:]
            y_train = y.iloc[:split_idx]
            y_test = y.iloc[split_idx:]
            
        else:
            # Random split
            X_train, X_test, y_train, y_test = sklearn_train_test_split(
                X, y, 
                test_size=test_size, 
                random_state=random_state
            )
            
        return X_train, X_test, y_train, y_test
    
    def create_features(
        self,
        df: pd.DataFrame,
        enso_data: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Create all features without separating target.
        
        Args:
            df: Input DataFrame
            enso_data: Optional ENSO data to merge
            
        Returns:
            DataFrame with all features created
        """
        df_features = df.copy()
        
        df_features = self._ensure_date_column(df_features)
        df_features = self._create_time_features(df_features)
        df_features = self._create_lag_features(df_features)
        df_features = self._create_rolling_features(df_features)
        df_features = self._create_climate_features(df_features)
        df_features = self._create_enso_features(df_features, enso_data)
        df_features = self._handle_stations(df_features)
        
        return df_features


# Module-level functions
def preprocess_data(
    df: pd.DataFrame,
    target_column: str = 'swe',
    date_column: str = 'date',
    station_column: str = 'station_id',
    config: Optional[Dict] = None
) -> Tuple[pd.DataFrame, pd.Series]:
    """Preprocess data with default settings."""
    preprocessor = DataPreprocessor(config)
    return preprocessor.preprocess_data(df, target_column, date_column, station_column)


def create_features(
    df: pd.DataFrame,
    enso_data: Optional[pd.DataFrame] = None,
    config: Optional[Dict] = None
) -> pd.DataFrame:
    """Create features without separating target."""
    preprocessor = DataPreprocessor(config)
    return preprocessor.create_features(df, enso_data)


def train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
    temporal_split: bool = True,
    config: Optional[Dict] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split data into train and test sets."""
    preprocessor = DataPreprocessor(config)
    return preprocessor.train_test_split(X, y, test_size, random_state, temporal_split)
