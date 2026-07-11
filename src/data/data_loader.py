"""
Data Loader Module

Handles loading of snowpack, climate, and ENSO data from various sources.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
from pathlib import Path
import requests
import json
from datetime import datetime
import os


class DataLoader:
    """Main data loader class for California snowpack prediction."""
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize the data loader.
        
        Args:
            data_dir: Path to the data directory
        """
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        self.external_dir = self.data_dir / "external"
        
        # Ensure directories exist
        self._ensure_directories()
        
    def _ensure_directories(self):
        """Create necessary directories if they don't exist."""
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.external_dir.mkdir(parents=True, exist_ok=True)
    
    def load_snowpack_data(
        self, 
        source: str = "cdec",
        stations: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Load snowpack data from specified source.
        
        Args:
            source: Data source ('cdec', 'usgs', 'nrcs')
            stations: List of station IDs to load
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            DataFrame with snowpack measurements
        """
        if source == "cdec":
            return self._load_cdec_snowpack(stations, start_date, end_date)
        elif source == "usgs":
            return self._load_usgs_snowpack(stations, start_date, end_date)
        elif source == "nrcs":
            return self._load_nrcs_snowpack(stations, start_date, end_date)
        else:
            raise ValueError(f"Unknown source: {source}. Use 'cdec', 'usgs', or 'nrcs'.")
    
    def _load_cdec_snowpack(
        self, 
        stations: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Load snowpack data from California Data Exchange Center (CDEC).
        
        Args:
            stations: List of CDEC station IDs
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            DataFrame with columns: date, station_id, swe, snow_depth, temperature, precipitation
        """
        # Default stations (major snow courses in California)
        if stations is None:
            stations = [
                'CAC001', 'CAC002', 'CAC003',  # Central Sierra
                'NAC001', 'NAC002', 'NAC003',  # Northern Sierra
                'SAC001', 'SAC002', 'SAC003',  # Southern Sierra
            ]
        
        # Default date range (last 20 years)
        if start_date is None:
            start_date = (datetime.now().year - 20).__str__() + "-01-01"
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")
        
        # In a real implementation, this would fetch from CDEC API
        # For now, we'll create a mock dataset
        print(f"Loading CDEC snowpack data for stations {stations} from {start_date} to {end_date}")
        
        # Generate mock data
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        data = []
        
        for date in dates:
            for station in stations:
                # Generate realistic mock data with seasonality
                day_of_year = date.dayofyear
                month = date.month
                
                # Snowpack typically builds Nov-Apr, melts May-Jun
                if month in [11, 12, 1, 2, 3, 4]:
                    # Winter accumulation
                    swe = np.random.normal(20, 5) + (day_of_year - 300) * 0.5
                    snow_depth = swe * 3  # Approximate conversion
                elif month in [5, 6]:
                    # Spring melt
                    swe = np.random.normal(15, 3) - (day_of_year - 120) * 0.3
                    snow_depth = swe * 3 if swe > 0 else 0
                else:
                    # Summer/fall - minimal snow
                    swe = np.random.normal(2, 1)
                    snow_depth = swe * 3 if swe > 0 else 0
                
                # Add some randomness and ensure non-negative
                swe = max(0, swe + np.random.normal(0, 2))
                snow_depth = max(0, snow_depth + np.random.normal(0, 1))
                temperature = np.random.normal(5, 10) - (day_of_year - 180) * 0.2
                precipitation = np.random.exponential(0.2)
                
                data.append({
                    'date': date,
                    'station_id': station,
                    'swe': round(swe, 2),
                    'snow_depth': round(snow_depth, 2),
                    'temperature': round(temperature, 2),
                    'precipitation': round(precipitation, 2)
                })
        
        df = pd.DataFrame(data)
        df['date'] = pd.to_datetime(df['date'])
        
        # Add derived features
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
        df['day_of_year'] = df['date'].dt.dayofyear
        
        return df
    
    def _load_usgs_snowpack(
        self, 
        stations: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """Load snowpack data from USGS."""
        # Implementation would fetch from USGS API
        raise NotImplementedError("USGS data loading not yet implemented")
    
    def _load_nrcs_snowpack(
        self, 
        stations: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """Load snowpack data from NRCS SNOTEL."""
        # Implementation would fetch from NRCS API
        raise NotImplementedError("NRCS data loading not yet implemented")
    
    def load_climate_data(
        self,
        source: str = "noaa",
        stations: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Load climate data (temperature, precipitation).
        
        Args:
            source: Data source ('noaa', 'cimis')
            stations: List of station IDs
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            DataFrame with climate measurements
        """
        if source == "noaa":
            return self._load_noaa_climate(stations, start_date, end_date)
        elif source == "cimis":
            return self._load_cimis_climate(stations, start_date, end_date)
        else:
            raise ValueError(f"Unknown climate source: {source}")
    
    def _load_noaa_climate(
        self,
        stations: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """Load climate data from NOAA."""
        # Generate mock climate data
        if stations is None:
            stations = ['USC0004', 'USC0005', 'USC0006']
        
        if start_date is None:
            start_date = (datetime.now().year - 20).__str__() + "-01-01"
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")
        
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        data = []
        
        for date in dates:
            for station in stations:
                day_of_year = date.dayof_year
                month = date.month
                
                # Seasonal temperature pattern
                base_temp = 15 + 10 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
                temperature = base_temp + np.random.normal(0, 5)
                
                # Seasonal precipitation pattern (winter wet, summer dry)
                if month in [12, 1, 2]:
                    precipitation = np.random.exponential(0.4)
                elif month in [3, 4, 5, 11]:
                    precipitation = np.random.exponential(0.2)
                else:
                    precipitation = np.random.exponential(0.05)
                
                humidity = np.random.normal(60, 15)
                wind_speed = np.random.exponential(3)
                
                data.append({
                    'date': date,
                    'station_id': station,
                    'temperature': round(temperature, 2),
                    'precipitation': round(precipitation, 2),
                    'humidity': round(humidity, 2),
                    'wind_speed': round(wind_speed, 2)
                })
        
        df = pd.DataFrame(data)
        df['date'] = pd.to_datetime(df['date'])
        
        return df
    
    def _load_cimis_climate(
        self,
        stations: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """Load climate data from CIMIS."""
        raise NotImplementedError("CIMIS data loading not yet implemented")
    
    def load_enso_data(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Load ENSO (El Niño Southern Oscillation) data.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            DataFrame with ENSO indices (ONI, MEI, etc.)
        """
        if start_date is None:
            start_date = (datetime.now().year - 50).__str__() + "-01-01"
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")
        
        # Generate mock ENSO data with realistic patterns
        dates = pd.date_range(start=start_date, end=end_date, freq='M')
        
        # Create realistic ENSO cycles (3-7 year periods)
        n_months = len(dates)
        time = np.arange(n_months)
        
        # Multiple ENSO cycles with different periods
        enso_cycle_1 = 5 * np.sin(2 * np.pi * time / (36))  # 3-year cycle
        enso_cycle_2 = 3 * np.sin(2 * np.pi * time / (84))  # 7-year cycle
        enso_cycle_3 = 2 * np.sin(2 * np.pi * time / (60))  # 5-year cycle
        
        # Combine cycles with noise
        oni = enso_cycle_1 + enso_cycle_2 + enso_cycle_3 + np.random.normal(0, 0.5, n_months)
        
        # MEI (Multivariate ENSO Index) - correlated with ONI
        mei = 0.8 * oni + np.random.normal(0, 0.3, n_months)
        
        # Create DataFrame
        df = pd.DataFrame({
            'date': dates,
            'oni': round(oni, 3),  # Oceanic Niño Index
            'mei': round(mei, 3),  # Multivariate ENSO Index
            'nino34': round(oni * 0.9 + np.random.normal(0, 0.2, n_months), 3),
            'soi': round(-0.7 * oni + np.random.normal(0, 0.4, n_months), 3)
        })
        
        # Add ENSO phase classification
        df['enso_phase'] = pd.cut(
            df['oni'], 
            bins=[-np.inf, -0.5, 0.5, np.inf],
            labels=['La Niña', 'Neutral', 'El Niño']
        )
        
        # Add rolling averages for trend analysis
        df['oni_3month_avg'] = df['oni'].rolling(window=3, min_periods=1).mean()
        df['oni_12month_avg'] = df['oni'].rolling(window=12, min_periods=1).mean()
        
        return df
    
    def load_all_data(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Load all data sources and return as dictionary.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            Dictionary with keys: 'snowpack', 'climate', 'enso'
        """
        return {
            'snowpack': self.load_snowpack_data(start_date=start_date, end_date=end_date),
            'climate': self.load_climate_data(start_date=start_date, end_date=end_date),
            'enso': self.load_enso_data(start_date=start_date, end_date=end_date)
        }


# Module-level functions for convenience
def load_snowpack_data(
    source: str = "cdec",
    stations: Optional[List[str]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> pd.DataFrame:
    """Load snowpack data from specified source."""
    loader = DataLoader()
    return loader.load_snowpack_data(source, stations, start_date, end_date)


def load_climate_data(
    source: str = "noaa",
    stations: Optional[List[str]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> pd.DataFrame:
    """Load climate data from specified source."""
    loader = DataLoader()
    return loader.load_climate_data(source, stations, start_date, end_date)


def load_enso_data(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> pd.DataFrame:
    """Load ENSO data."""
    loader = DataLoader()
    return loader.load_enso_data(start_date, end_date)


def load_all_data(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> Dict[str, pd.DataFrame]:
    """Load all data sources."""
    loader = DataLoader()
    return loader.load_all_data(start_date, end_date)
