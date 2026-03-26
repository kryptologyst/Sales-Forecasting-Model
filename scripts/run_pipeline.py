"""Main script to run the sales forecasting pipeline."""

import sys
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.utils.config import Config
from src.utils.utils import setup_logging
from src.forecasting.pipeline import SalesForecastingPipeline


def main():
    """Main function to run the forecasting pipeline."""
    
    # Set up logging
    logger = setup_logging(log_level="INFO")
    logger.info("Starting Sales Forecasting Pipeline")
    
    try:
        # Load configuration
        config_path = Path(__file__).parent / "configs" / "config.yaml"
        config = Config.from_yaml(str(config_path))
        
        # Initialize pipeline
        pipeline = SalesForecastingPipeline(config)
        
        # Run full pipeline
        report = pipeline.run_full_pipeline()
        
        # Print results
        print("\n" + "="*50)
        print("EVALUATION REPORT")
        print("="*50)
        print(report.round(4))
        
        # Get best model
        best_model_name, best_model = pipeline.get_best_model()
        print(f"\nBest Model: {best_model_name}")
        print(f"MAE: {report.loc[best_model_name, 'mae']:.4f}")
        
        # Make future forecast
        future_forecast = pipeline.forecast_future(best_model_name, 6)
        print(f"\nFuture 6-month forecast: {future_forecast}")
        
        logger.info("Pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()
