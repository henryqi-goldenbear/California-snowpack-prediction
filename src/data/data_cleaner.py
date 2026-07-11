"""
Data Cleaner Module

Handles data cleaning, missing value imputation, and outlier detection.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.ensemble import IsolationForest
from scipy import stats
import warnings


class DataCleaner:
    """Main data cleaner class for California snowpack data."""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the data cleaner.
        
        Args:
            config: Configuration dictionary for cleaning parameters
        """
        self.config = config or {}
        self._default_config = {
            'missing_value_threshold': 0.3,  # Drop columns with >30% missing
            'outlier_std_threshold': 3.0,    # Z-score threshold for outliers
            'imputation_strategy': 'median',  # 'mean', 'median', 'knn', 'interpolate'
            'knn_neighbors': 5,
            'date_column': 'date',
            'target_columns': ['swe', 'snow_depth'],
            'feature_columns': None  # Auto-detected
        }
        self.config = {**self._default_config, **self.config}
        
    def clean_data(
        self, 
        df: pd.DataFrame,
        target_columns: Optional[List[str]] = None,
        feature_columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Clean a DataFrame with comprehensive cleaning pipeline.
        
        Args:
            df: Input DataFrame
            target_columns: Columns to treat as targets (special handling)
            feature_columns: Columns to treat as features
            
        Returns:
            Cleaned DataFrame
        """
        # Update config with provided columns
        if target_columns:
            self.config['target_columns'] = target_columns
        if feature_columns:
            self.config['feature_columns'] = feature_columns
            
        # Make a copy to avoid modifying original
        df_clean = df.copy()
        
        # Cleaning pipeline
        df_clean = self._convert_dtypes(df_clean)
        df_clean = self._handle_dates(df_clean)
        df_clean = self._remove_duplicate_rows(df_clean)
        df_clean = self._handle_missing_values(df_clean)
        df_clean = self._remove_outliers(df_clean)
        df_clean = self._remove_constant_columns(df_clean)
        df_clean = self._remove_high_cardinality_categorical(df_clean)
        
        return df_clean
    
    def _convert_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert data types appropriately."""
        # Convert numeric columns that might be strings
        for col in df.columns:
            if df[col].dtype == 'object':
                # Try to convert to numeric
                try:
                    df[col] = pd.to_numeric(df[col])
                except (ValueError, TypeError):
                    pass
        
        # Convert date columns
        date_cols = [col for col in df.columns if 'date' in col.lower()]
        for col in date_cols:
            try:
                df[col] = pd.to_datetime(df[col])
            except (ValueError, TypeError):
                pass
                
        return df
    
    def _handle_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle date-related features."""
        if self.config['date_column'] in df.columns:
            # Ensure datetime type
            df[self.config['date_column']] = pd.to_datetime(
                df[self.config['date_column']]
            )
            
            # Sort by date
            df = df.sort_values(self.config['date_column']).reset_index(drop=True)
            
            # Add time-based features if not present
            date_col = self.config['date_column']
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
                
        return df
    
    def _remove_duplicate_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate rows."""
        # Check for duplicates based on all columns
        initial_count = len(df)
        df = df.drop_duplicates()
        final_count = len(df)
        
        if initial_count != final_count:
            print(f"Removed {initial_count - final_count} duplicate rows")
            
        return df
    
    def handle_missing_values(
        self, 
        df: pd.DataFrame,
        strategy: Optional[str] = None,
        threshold: Optional[float] = None
    ) -> pd.DataFrame:
        """
        Handle missing values in the DataFrame.
        
        Args:
            df: Input DataFrame
            strategy: Imputation strategy ('mean', 'median', 'knn', 'interpolate', 'drop')
            threshold: Drop columns with missing values above this threshold
            
        Returns:
            DataFrame with missing values handled
        """
        strategy = strategy or self.config['imputation_strategy']
        threshold = threshold or self.config['missing_value_threshold']
        
        df_clean = df.copy()
        
        # Calculate missing percentages
        missing_percent = df_clean.isnull().mean()
        
        # Drop columns with too many missing values
        cols_to_drop = missing_percent[missing_percent > threshold].index.tolist()
        if cols_to_drop:
            print(f"Dropping columns with >{threshold*100}% missing: {cols_to_drop}")
            df_clean = df_clean.drop(columns=cols_to_drop)
            
        # Handle remaining missing values
        if strategy == 'drop':
            # Drop rows with any missing values
            initial_count = len(df_clean)
            df_clean = df_clean.dropna()
            final_count = len(df_clean)
            if initial_count != final_count:
                print(f"Dropped {initial_count - final_count} rows with missing values")
                
        elif strategy == 'mean':
            # Impute with mean
            imputer = SimpleImputer(strategy='mean')
            numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
            df_clean[numeric_cols] = imputer.fit_transform(df_clean[numeric_cols])
            
        elif strategy == 'median':
            # Impute with median
            imputer = SimpleImputer(strategy='median')
            numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
            df_clean[numeric_cols] = imputer.fit_transform(df_clean[numeric_cols])
            
        elif strategy == 'knn':
            # KNN imputation
            imputer = KNNImputer(n_neighbors=self.config['knn_neighbors'])
            numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
            df_clean[numeric_cols] = imputer.fit_transform(df_clean[numeric_cols])
            
        elif strategy == 'interpolate':
            # Time-based interpolation
            if self.config['date_column'] in df_clean.columns:
                df_clean = df_clean.set_index(self.config['date_column'])
                df_clean = df_clean.interpolate(method='time')
                df_clean = df_clean.reset_index()
            else:
                df_clean = df_clean.interpolate()
                
        # Handle categorical columns (if any)
        categorical_cols = df_clean.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df_clean[col].isnull().any():
                # Fill with mode for categorical
                mode_val = df_clean[col].mode()[0]
                df_clean[col] = df_clean[col].fillna(mode_val)
                
        return df_clean
    
    def _remove_outliers(
        self, 
        df: pd.DataFrame,
        target_columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Remove outliers using statistical methods.
        
        Args:
            df: Input DataFrame
            target_columns: Specific columns to check for outliers
            
        Returns:
            DataFrame with outliers removed
        """
        target_cols = target_columns or self.config['target_columns']
        df_clean = df.copy()
        
        # Get numeric columns
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        
        # Focus on target columns if specified, otherwise all numeric
        cols_to_check = [col for col in target_cols if col in numeric_cols]
        if not cols_to_check:
            cols_to_check = numeric_cols.tolist()
            
        # Remove outliers using z-score method
        for col in cols_to_check:
            if col in df_clean.columns:
                z_scores = np.abs(stats.zscore(df_clean[col].dropna()))
                threshold = self.config['outlier_std_threshold']
                outliers = z_scores > threshold
                
                if outliers.any():
                    # Cap outliers instead of removing to preserve data
                    upper_bound = df_clean[col].mean() + threshold * df_clean[col].std()
                    lower_bound = df_clean[col].mean() - threshold * df_clean[col].std()
                    df_clean[col] = df_clean[col].clip(lower_bound, upper_bound)
                    print(f"Capped {outliers.sum()} outliers in column '{col}'")
        
        return df_clean
    
    def remove_outliers(
        self, 
        df: pd.DataFrame,
        method: str = 'zscore',
        threshold: Optional[float] = None
    ) -> pd.DataFrame:
        """
        Remove outliers using specified method.
        
        Args:
            df: Input DataFrame
            method: Method for outlier detection ('zscore', 'iqr', 'isolation_forest')
            threshold: Threshold for outlier detection
            
        Returns:
            DataFrame with outliers removed
        """
        threshold = threshold or self.config['outlier_std_threshold']
        df_clean = df.copy()
        
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        
        if method == 'zscore':
            for col in numeric_cols:
                z_scores = np.abs(stats.zscore(df_clean[col].dropna()))
                outliers = z_scores > threshold
                if outliers.any():
                    upper = df_clean[col].mean() + threshold * df_clean[col].std()
                    lower = df_clean[col].mean() - threshold * df_clean[col].std()
                    df_clean[col] = df_clean[col].clip(lower, upper)
                    
        elif method == 'iqr':
            for col in numeric_cols:
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                df_clean[col] = df_clean[col].clip(lower_bound, upper_bound)
                
        elif method == 'isolation_forest':
            # Use Isolation Forest for multivariate outlier detection
            iso_forest = IsolationForest(contamination=0.05, random_state=42)
            numeric_data = df_clean[numeric_cols].values
            outliers = iso_forest.fit_predict(numeric_data)
            
            # Remove outlier rows
            df_clean = df_clean[outliers == 1].reset_index(drop=True)
            
        return df_clean
    
    def _remove_constant_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove columns with constant values."""
        constant_cols = df.columns[df.nunique() == 1]
        if len(constant_cols) > 0:
            print(f"Removing constant columns: {list(constant_cols)}")
            df = df.drop(columns=constant_cols)
        return df
    
    def _remove_high_cardinality_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove or encode high cardinality categorical columns."""
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            if df[col].nunique() > 50:  # High cardinality threshold
                # For ID columns, just drop them
                if 'id' in col.lower() or 'station' in col.lower():
                    df = df.drop(columns=[col])
                    print(f"Dropped high cardinality column: {col}")
                else:
                    # For other high cardinality columns, we might want to encode
                    # For now, just drop to keep it simple
                    df = df.drop(columns=[col])
                    print(f"Dropped high cardinality column: {col}")
                    
        return df
    
    def get_cleaning_report(self, df_original: pd.DataFrame, df_cleaned: pd.DataFrame) -> Dict:
        """
        Generate a report comparing original and cleaned data.
        
        Args:
            df_original: Original DataFrame
            df_cleaned: Cleaned DataFrame
            
        Returns:
            Dictionary with cleaning statistics
        """
        report = {
            'original_shape': df_original.shape,
            'cleaned_shape': df_cleaned.shape,
            'rows_removed': len(df_original) - len(df_cleaned),
            'columns_removed': len(df_original.columns) - len(df_cleaned.columns),
            'missing_values_original': df_original.isnull().sum().to_dict(),
            'missing_values_cleaned': df_cleaned.isnull().sum().to_dict(),
            'numeric_columns_summary': {}
        }
        
        # Add summary statistics for numeric columns
        numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            report['numeric_columns_summary'][col] = {
                'mean': df_cleaned[col].mean(),
                'std': df_cleaned[col].std(),
                'min': df_cleaned[col].min(),
                'max': df_cleaned[col].max(),
                'missing_count': df_cleaned[col].isnull().sum()
            }
            
        return report


# Module-level functions
def clean_data(
    df: pd.DataFrame,
    target_columns: Optional[List[str]] = None,
    feature_columns: Optional[List[str]] = None,
    config: Optional[Dict] = None
) -> pd.DataFrame:
    """Clean a DataFrame with default settings."""
    cleaner = DataCleaner(config)
    return cleaner.clean_data(df, target_columns, feature_columns)


def handle_missing_values(
    df: pd.DataFrame,
    strategy: str = 'median',
    threshold: float = 0.3
) -> pd.DataFrame:
    """Handle missing values in a DataFrame."""
    cleaner = DataCleaner()
    return cleaner.handle_missing_values(df, strategy, threshold)


def remove_outliers(
    df: pd.DataFrame,
    method: str = 'zscore',
    threshold: float = 3.0
) -> pd.DataFrame:
    """Remove outliers from a DataFrame."""
    cleaner = DataCleaner()
    return cleaner.remove_outliers(df, method, threshold)
