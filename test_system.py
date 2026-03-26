#!/usr/bin/env python3
"""Simple test script to verify the sales forecasting system works."""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        from src.utils.config import Config
        from src.data.data_processor import DataProcessor
        from src.models.forecasters import ForecastingModelFactory
        from src.eval.evaluator import ForecastingEvaluator
        from src.viz.visualizer import ForecastingVisualizer
        from src.forecasting.pipeline import SalesForecastingPipeline
        print("✓ All imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False

def test_config():
    """Test configuration loading."""
    print("Testing configuration...")
    
    try:
        from src.utils.config import Config
        config_path = Path(__file__).parent / "configs" / "config.yaml"
        config = Config.from_yaml(str(config_path))
        
        assert config.data.date_column == "date"
        assert config.data.target_column == "sales"
        assert len(config.model.models) > 0
        print("✓ Configuration loaded successfully")
        return True
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        return False

def test_data_generation():
    """Test synthetic data generation."""
    print("Testing data generation...")
    
    try:
        from src.utils.config import Config
        from src.data.data_processor import DataProcessor
        
        config = Config()
        config.data.synthetic_start_date = "2023-01-01"
        config.data.synthetic_end_date = "2023-12-31"
        
        processor = DataProcessor(config.data)
        df = processor.generate_synthetic_data()
        
        assert len(df) > 0
        assert "date" in df.columns
        assert "sales" in df.columns
        assert df["sales"].min() > 0
        
        print(f"✓ Generated {len(df)} records of synthetic data")
        return True
    except Exception as e:
        print(f"✗ Data generation test failed: {e}")
        return False

def test_model_creation():
    """Test model creation."""
    print("Testing model creation...")
    
    try:
        from src.utils.config import Config
        from src.models.forecasters import ForecastingModelFactory
        
        config = Config()
        config.model.models = ["linear"]  # Test with simple model
        
        model = ForecastingModelFactory.create_model("linear", config.model)
        assert model is not None
        
        print("✓ Model creation successful")
        return True
    except Exception as e:
        print(f"✗ Model creation test failed: {e}")
        return False

def test_evaluation():
    """Test evaluation metrics."""
    print("Testing evaluation metrics...")
    
    try:
        from src.eval.evaluator import ForecastingEvaluator
        from src.utils.config import Config
        import numpy as np
        
        config = Config()
        evaluator = ForecastingEvaluator(config.evaluation)
        
        y_true = np.array([100, 120, 110, 130, 125])
        y_pred = np.array([105, 115, 115, 125, 130])
        
        mae = evaluator.calculate_mae(y_true, y_pred)
        rmse = evaluator.calculate_rmse(y_true, y_pred)
        
        assert mae > 0
        assert rmse >= mae  # RMSE >= MAE
        
        print(f"✓ Evaluation metrics working (MAE: {mae:.2f}, RMSE: {rmse:.2f})")
        return True
    except Exception as e:
        print(f"✗ Evaluation test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 50)
    print("Sales Forecasting System - Test Suite")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_config,
        test_data_generation,
        test_model_creation,
        test_evaluation
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            print()
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with exception: {e}")
            print()
    
    print("=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The system is ready to use.")
        print("\nNext steps:")
        print("1. Run: python scripts/run_pipeline.py")
        print("2. Launch demo: streamlit run demo/app.py")
        print("3. Open notebook: jupyter notebook notebooks/sales_forecasting_example.ipynb")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
