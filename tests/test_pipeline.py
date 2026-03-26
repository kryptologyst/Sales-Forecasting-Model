"""Unit tests for sales forecasting pipeline."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.utils.config import Config, DataConfig, ModelConfig, EvaluationConfig
from src.data.data_processor import DataProcessor
from src.models.forecasters import ForecastingModelFactory, LinearForecaster
from src.eval.evaluator import ForecastingEvaluator
from src.utils.utils import set_seed


class TestConfig:
    """Test configuration management."""
    
    def test_data_config_defaults(self):
        """Test DataConfig default values."""
        config = DataConfig()
        assert config.date_column == "date"
        assert config.target_column == "sales"
        assert config.frequency == "M"
        assert config.train_ratio == 0.7
        assert config.generate_synthetic is True
    
    def test_model_config_defaults(self):
        """Test ModelConfig default values."""
        config = ModelConfig()
        assert config.random_state == 42
        assert config.xgb_n_estimators == 100
        assert config.lstm_hidden_size == 50
        assert "linear" in config.models
    
    def test_evaluation_config_defaults(self):
        """Test EvaluationConfig default values."""
        config = EvaluationConfig()
        assert "mae" in config.metrics
        assert "service_level" in config.business_metrics
        assert config.stockout_cost == 10.0
        assert config.service_level_target == 0.95


class TestDataProcessor:
    """Test data processing functionality."""
    
    def test_synthetic_data_generation(self):
        """Test synthetic data generation."""
        config = DataConfig()
        processor = DataProcessor(config)
        
        df = processor.generate_synthetic_data()
        
        assert len(df) > 0
        assert "date" in df.columns
        assert "sales" in df.columns
        assert df["sales"].min() > 0  # All sales should be positive
        
        # Check date range
        date_range = df["date"].max() - df["date"].min()
        assert date_range.days > 365  # At least one year of data
    
    def test_data_preparation(self):
        """Test data preparation."""
        config = DataConfig()
        processor = DataProcessor(config)
        
        # Generate synthetic data
        df = processor.generate_synthetic_data()
        
        # Prepare data
        df_prepared = processor.prepare_data(df)
        
        assert len(df_prepared) > 0
        assert len(df_prepared.columns) > len(df.columns)  # Should have more features
        
        # Check for lag features
        lag_features = [col for col in df_prepared.columns if "lag" in col]
        assert len(lag_features) > 0
    
    def test_data_splitting(self):
        """Test data splitting."""
        config = DataConfig()
        processor = DataProcessor(config)
        
        # Generate and prepare data
        df = processor.generate_synthetic_data()
        df_prepared = processor.prepare_data(df)
        
        # Split data
        train_df, val_df, test_df = processor.split_data(df_prepared)
        
        assert len(train_df) > 0
        assert len(val_df) > 0
        assert len(test_df) > 0
        
        # Check that splits are non-overlapping
        total_length = len(train_df) + len(val_df) + len(test_df)
        assert total_length <= len(df_prepared)


class TestForecasters:
    """Test forecasting models."""
    
    def test_model_factory(self):
        """Test model factory."""
        available_models = ForecastingModelFactory.get_available_models()
        assert "linear" in available_models
        assert "arima" in available_models
        assert "xgboost" in available_models
    
    def test_linear_forecaster(self):
        """Test linear regression forecaster."""
        config = ModelConfig()
        forecaster = LinearForecaster(config)
        
        # Create sample data
        X = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100)
        })
        y = pd.Series(np.random.randn(100))
        
        # Fit model
        forecaster.fit(X, y)
        assert forecaster.is_fitted is True
        
        # Make predictions
        predictions = forecaster.predict(X, horizon=5)
        assert len(predictions) == len(X)
        assert isinstance(predictions, np.ndarray)
        
        # Check feature importance
        importance = forecaster.get_feature_importance()
        assert importance is not None
        assert len(importance) == len(X.columns)


class TestEvaluator:
    """Test evaluation functionality."""
    
    def test_mae_calculation(self):
        """Test MAE calculation."""
        config = EvaluationConfig()
        evaluator = ForecastingEvaluator(config)
        
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1.1, 1.9, 3.1, 3.9, 5.1])
        
        mae = evaluator.calculate_mae(y_true, y_pred)
        assert mae > 0
        assert mae < 1  # Should be small for this example
    
    def test_rmse_calculation(self):
        """Test RMSE calculation."""
        config = EvaluationConfig()
        evaluator = ForecastingEvaluator(config)
        
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1.1, 1.9, 3.1, 3.9, 5.1])
        
        rmse = evaluator.calculate_rmse(y_true, y_pred)
        assert rmse > 0
        assert rmse > evaluator.calculate_mae(y_true, y_pred)  # RMSE >= MAE
    
    def test_mape_calculation(self):
        """Test MAPE calculation."""
        config = EvaluationConfig()
        evaluator = ForecastingEvaluator(config)
        
        y_true = np.array([100, 200, 300, 400, 500])
        y_pred = np.array([110, 190, 310, 390, 510])
        
        mape = evaluator.calculate_mape(y_true, y_pred)
        assert mape > 0
        assert mape < 10  # Should be around 2% for this example
    
    def test_business_metrics(self):
        """Test business metrics calculation."""
        config = EvaluationConfig()
        evaluator = ForecastingEvaluator(config)
        
        y_true = np.array([100, 120, 110, 130, 125])
        y_pred = np.array([105, 115, 115, 125, 130])
        
        metrics = evaluator.calculate_business_metrics(y_true, y_pred)
        
        assert "service_level" in metrics
        assert "inventory_turns" in metrics
        assert "total_cost" in metrics
        assert "cost_efficiency" in metrics
        
        # Check reasonable ranges
        assert 0 <= metrics["service_level"] <= 100
        assert metrics["inventory_turns"] > 0
        assert metrics["total_cost"] >= 0


class TestUtils:
    """Test utility functions."""
    
    def test_seed_setting(self):
        """Test random seed setting."""
        set_seed(42)
        
        # Generate some random numbers
        np_rand1 = np.random.randn(5)
        
        # Reset seed and generate again
        set_seed(42)
        np_rand2 = np.random.randn(5)
        
        # Should be identical
        np.testing.assert_array_equal(np_rand1, np_rand2)


# Integration tests
class TestIntegration:
    """Integration tests for the full pipeline."""
    
    def test_small_pipeline(self):
        """Test a small version of the pipeline."""
        # Create minimal config
        config = Config()
        config.data.synthetic_start_date = "2023-01-01"
        config.data.synthetic_end_date = "2023-12-31"
        config.model.models = ["linear"]  # Only test linear model
        config.data.forecast_horizon = 3
        
        # This would test the full pipeline if we had all dependencies
        # For now, just test that we can create the components
        from src.data.data_processor import DataProcessor
        from src.models.forecasters import ForecastingModelFactory
        
        processor = DataProcessor(config.data)
        model = ForecastingModelFactory.create_model("linear", config.model)
        
        assert processor is not None
        assert model is not None


if __name__ == "__main__":
    pytest.main([__file__])
