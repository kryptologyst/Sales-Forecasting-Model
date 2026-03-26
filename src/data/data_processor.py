"""Data processing and synthetic data generation for sales forecasting."""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict, Any
from datetime import datetime, timedelta
from pathlib import Path
import logging

from ..utils.config import DataConfig
from ..utils.utils import ensure_dir, get_project_root

logger = logging.getLogger(__name__)


class DataProcessor:
    """Handles data loading, processing, and synthetic data generation."""
    
    def __init__(self, config: DataConfig):
        """Initialize data processor.
        
        Args:
            config: Data configuration
        """
        self.config = config
        self.project_root = get_project_root()
        
    def generate_synthetic_data(self) -> pd.DataFrame:
        """Generate synthetic sales data for demonstration.
        
        Returns:
            DataFrame with synthetic sales data
        """
        logger.info("Generating synthetic sales data...")
        
        # Create date range
        start_date = pd.to_datetime(self.config.synthetic_start_date)
        end_date = pd.to_datetime(self.config.synthetic_end_date)
        dates = pd.date_range(start=start_date, end=end_date, freq=self.config.frequency.replace('M', 'ME'))
        
        n_periods = len(dates)
        
        # Generate trend component
        trend = np.linspace(100, 100 * (1 + self.config.synthetic_trend) ** n_periods, n_periods)
        
        # Generate seasonal component (yearly seasonality)
        seasonal_period = 12  # Monthly data, yearly seasonality
        seasonal_component = self.config.synthetic_seasonality * 100 * np.sin(
            2 * np.pi * np.arange(n_periods) / seasonal_period
        )
        
        # Generate noise
        noise = np.random.normal(0, self.config.synthetic_noise * 100, n_periods)
        
        # Combine components
        sales = trend + seasonal_component + noise
        
        # Ensure positive values
        sales = np.maximum(sales, 10)
        
        # Create DataFrame
        df = pd.DataFrame({
            self.config.date_column: dates,
            self.config.target_column: sales.round(2)
        })
        
        # Add some additional features for demonstration
        df['month'] = df[self.config.date_column].dt.month
        df['quarter'] = df[self.config.date_column].dt.quarter
        df['year'] = df[self.config.date_column].dt.year
        
        # Add some external features (simulated)
        np.random.seed(42)  # For reproducibility
        df['promotion'] = np.random.choice([0, 1], size=len(df), p=[0.8, 0.2])
        df['holiday'] = np.random.choice([0, 1], size=len(df), p=[0.9, 0.1])
        df['competitor_price'] = np.random.normal(100, 10, len(df))
        
        logger.info(f"Generated {len(df)} records of synthetic data")
        return df
    
    def load_data(self, file_path: Optional[str] = None) -> pd.DataFrame:
        """Load data from file or generate synthetic data.
        
        Args:
            file_path: Optional path to data file
            
        Returns:
            Loaded or generated DataFrame
        """
        if file_path and Path(file_path).exists():
            logger.info(f"Loading data from {file_path}")
            df = pd.read_csv(file_path)
            
            # Ensure date column is datetime
            df[self.config.date_column] = pd.to_datetime(df[self.config.date_column])
            
            return df
        elif self.config.generate_synthetic:
            logger.info("No data file found, generating synthetic data")
            return self.generate_synthetic_data()
        else:
            raise FileNotFoundError("No data file found and synthetic generation disabled")
    
    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for modeling.
        
        Args:
            df: Raw DataFrame
            
        Returns:
            Prepared DataFrame
        """
        logger.info("Preparing data for modeling...")
        
        # Sort by date
        df = df.sort_values(self.config.date_column).reset_index(drop=True)
        
        # Ensure target column is numeric
        df[self.config.target_column] = pd.to_numeric(df[self.config.target_column], errors='coerce')
        
        # Handle missing values
        df = df.dropna(subset=[self.config.target_column])
        
        # Add lag features
        for lag in [1, 2, 3, 6, 12]:
            df[f'sales_lag_{lag}'] = df[self.config.target_column].shift(lag)
        
        # Add rolling statistics
        for window in [3, 6, 12]:
            df[f'sales_ma_{window}'] = df[self.config.target_column].rolling(window=window).mean()
            df[f'sales_std_{window}'] = df[self.config.target_column].rolling(window=window).std()
        
        # Add trend features
        df['trend'] = np.arange(len(df))
        df['trend_squared'] = df['trend'] ** 2
        
        logger.info(f"Data prepared: {len(df)} records, {len(df.columns)} features")
        return df
    
    def split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split data into train, validation, and test sets.
        
        Args:
            df: Prepared DataFrame
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        logger.info("Splitting data into train/validation/test sets...")
        
        # Ensure minimum training length
        min_train_length = max(self.config.min_train_length, int(len(df) * self.config.train_ratio))
        
        # Calculate split indices
        train_end = min_train_length
        val_end = train_end + int(len(df) * self.config.val_ratio)
        
        train_df = df.iloc[:train_end].copy()
        val_df = df.iloc[train_end:val_end].copy()
        test_df = df.iloc[val_end:].copy()
        
        logger.info(f"Data split - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        
        return train_df, val_df, test_df
    
    def save_data(self, df: pd.DataFrame, filename: str) -> None:
        """Save DataFrame to file.
        
        Args:
            df: DataFrame to save
            filename: Output filename
        """
        output_path = self.project_root / self.config.processed_data_path / filename
        ensure_dir(output_path.parent)
        
        df.to_csv(output_path, index=False)
        logger.info(f"Data saved to {output_path}")
    
    def load_processed_data(self, filename: str) -> pd.DataFrame:
        """Load processed data from file.
        
        Args:
            filename: Input filename
            
        Returns:
            Loaded DataFrame
        """
        input_path = self.project_root / self.config.processed_data_path / filename
        
        if not input_path.exists():
            raise FileNotFoundError(f"Processed data file not found: {input_path}")
        
        df = pd.read_csv(input_path)
        df[self.config.date_column] = pd.to_datetime(df[self.config.date_column])
        
        logger.info(f"Processed data loaded from {input_path}")
        return df
