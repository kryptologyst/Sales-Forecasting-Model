"""Main forecasting pipeline for sales prediction."""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from pathlib import Path
import joblib

from ..data.data_processor import DataProcessor
from ..models.forecasters import ForecastingModelFactory, BaseForecaster
from ..eval.evaluator import ForecastingEvaluator
from ..viz.visualizer import ForecastingVisualizer
from ..utils.config import Config
from ..utils.utils import set_seed, setup_logging, ensure_dir

logger = logging.getLogger(__name__)


class SalesForecastingPipeline:
    """Main pipeline for sales forecasting with multiple models."""
    
    def __init__(self, config: Config):
        """Initialize forecasting pipeline.
        
        Args:
            config: Configuration object
        """
        self.config = config
        
        # Set up reproducibility
        set_seed(config.seed)
        
        # Initialize components
        self.data_processor = DataProcessor(config.data)
        self.evaluator = ForecastingEvaluator(config.evaluation)
        self.visualizer = ForecastingVisualizer(config.output_dir)
        
        # Storage for results
        self.models: Dict[str, BaseForecaster] = {}
        self.predictions: Dict[str, np.ndarray] = {}
        self.evaluation_results: Dict[str, Dict[str, float]] = {}
        self.feature_importance: Dict[str, Dict[str, float]] = {}
        
        # Data storage
        self.train_data: Optional[pd.DataFrame] = None
        self.val_data: Optional[pd.DataFrame] = None
        self.test_data: Optional[pd.DataFrame] = None
        
    def load_and_prepare_data(self, data_path: Optional[str] = None) -> None:
        """Load and prepare data for modeling.
        
        Args:
            data_path: Optional path to data file
        """
        logger.info("Loading and preparing data...")
        
        # Load data
        df = self.data_processor.load_data(data_path)
        
        # Prepare data
        df_prepared = self.data_processor.prepare_data(df)
        
        # Split data
        self.train_data, self.val_data, self.test_data = self.data_processor.split_data(df_prepared)
        
        # Save processed data
        self.data_processor.save_data(self.train_data, "train_data.csv")
        self.data_processor.save_data(self.val_data, "val_data.csv")
        self.data_processor.save_data(self.test_data, "test_data.csv")
        
        logger.info("Data preparation completed")
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features and target for modeling.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Tuple of (features, target)
        """
        # Select feature columns (exclude date and target)
        feature_columns = [col for col in df.columns 
                          if col not in [self.config.data.date_column, self.config.data.target_column]]
        
        X = df[feature_columns].fillna(0)
        y = df[self.config.data.target_column]
        
        return X, y
    
    def train_models(self) -> None:
        """Train all configured models."""
        logger.info("Training forecasting models...")
        
        if self.train_data is None:
            raise ValueError("Data must be loaded before training models")
        
        # Prepare training features
        X_train, y_train = self.prepare_features(self.train_data)
        
        # Train each model
        for model_name in self.config.model.models:
            try:
                logger.info(f"Training {model_name} model...")
                
                # Create model
                model = ForecastingModelFactory.create_model(model_name, self.config.model)
                
                # Train model
                model.fit(X_train, y_train)
                
                # Store model
                self.models[model_name] = model
                
                # Get feature importance if available
                importance = model.get_feature_importance()
                if importance:
                    self.feature_importance[model_name] = importance
                
                logger.info(f"{model_name} model trained successfully")
                
            except Exception as e:
                logger.error(f"Failed to train {model_name} model: {e}")
                continue
        
        logger.info(f"Training completed for {len(self.models)} models")
    
    def make_predictions(self, horizon: Optional[int] = None) -> None:
        """Make predictions using trained models.
        
        Args:
            horizon: Forecast horizon (uses config default if None)
        """
        if horizon is None:
            horizon = self.config.data.forecast_horizon
        
        logger.info(f"Making predictions with horizon {horizon}...")
        
        if not self.models:
            raise ValueError("Models must be trained before making predictions")
        
        # Prepare test features
        X_test, y_test = self.prepare_features(self.test_data)
        
        # Make predictions for each model
        for model_name, model in self.models.items():
            try:
                logger.info(f"Making predictions with {model_name}...")
                
                # Make predictions
                predictions = model.predict(X_test, horizon)
                
                # Store predictions
                self.predictions[model_name] = predictions
                
                logger.info(f"{model_name} predictions completed")
                
            except Exception as e:
                logger.error(f"Failed to make predictions with {model_name}: {e}")
                continue
        
        logger.info(f"Predictions completed for {len(self.predictions)} models")
    
    def evaluate_models(self) -> None:
        """Evaluate all trained models."""
        logger.info("Evaluating models...")
        
        if not self.predictions:
            raise ValueError("Predictions must be made before evaluation")
        
        # Prepare test target
        _, y_test = self.prepare_features(self.test_data)
        _, y_train = self.prepare_features(self.train_data)
        
        # Evaluate each model
        for model_name, predictions in self.predictions.items():
            try:
                logger.info(f"Evaluating {model_name}...")
                
                # Calculate metrics
                metrics = self.evaluator.evaluate_model(
                    y_true=y_test.values,
                    y_pred=predictions,
                    y_train=y_train.values
                )
                
                # Store results
                self.evaluation_results[model_name] = metrics
                
                logger.info(f"{model_name} evaluation completed")
                
            except Exception as e:
                logger.error(f"Failed to evaluate {model_name}: {e}")
                continue
        
        logger.info(f"Evaluation completed for {len(self.evaluation_results)} models")
    
    def create_visualizations(self) -> None:
        """Create visualizations for results."""
        logger.info("Creating visualizations...")
        
        if not self.predictions or not self.evaluation_results:
            raise ValueError("Predictions and evaluation results must be available")
        
        # Prepare data for visualization
        viz_data = self.test_data[[self.config.data.date_column, self.config.data.target_column]].copy()
        viz_data.columns = ['date', 'sales']
        
        # Create forecast comparison plot
        self.visualizer.plot_forecast_comparison(
            df=viz_data,
            models=self.predictions,
            title="Sales Forecast Comparison",
            save_path="forecast_comparison.png"
        )
        
        # Create model performance plot
        self.visualizer.plot_model_performance(
            evaluation_results=self.evaluation_results,
            title="Model Performance Comparison",
            save_path="model_performance.png"
        )
        
        # Create business metrics plot
        self.visualizer.plot_business_metrics(
            evaluation_results=self.evaluation_results,
            title="Business Metrics Comparison",
            save_path="business_metrics.png"
        )
        
        # Create feature importance plots for each model
        for model_name, importance in self.feature_importance.items():
            self.visualizer.plot_feature_importance(
                feature_importance=importance,
                model_name=model_name,
                save_path=f"feature_importance_{model_name}.png"
            )
        
        logger.info("Visualizations created successfully")
    
    def generate_report(self) -> pd.DataFrame:
        """Generate comprehensive evaluation report.
        
        Returns:
            DataFrame with evaluation results
        """
        if not self.evaluation_results:
            raise ValueError("Evaluation results must be available")
        
        logger.info("Generating evaluation report...")
        
        # Create evaluation report
        report = self.evaluator.create_evaluation_report(self.evaluation_results)
        
        # Add model rankings
        rankings = self.evaluator.generate_model_ranking(self.evaluation_results)
        report['rank'] = report.index.map(rankings)
        
        # Save report
        report_path = Path(self.config.output_dir) / "evaluation_report.csv"
        ensure_dir(report_path.parent)
        report.to_csv(report_path)
        
        logger.info(f"Evaluation report saved to {report_path}")
        
        return report
    
    def save_models(self) -> None:
        """Save trained models."""
        logger.info("Saving trained models...")
        
        models_dir = Path(self.config.output_dir) / "models"
        ensure_dir(models_dir)
        
        for model_name, model in self.models.items():
            model_path = models_dir / f"{model_name}_model.joblib"
            joblib.dump(model, model_path)
            logger.info(f"{model_name} model saved to {model_path}")
        
        logger.info("All models saved successfully")
    
    def load_models(self, models_dir: str) -> None:
        """Load pre-trained models.
        
        Args:
            models_dir: Directory containing saved models
        """
        logger.info(f"Loading models from {models_dir}...")
        
        models_path = Path(models_dir)
        
        for model_name in self.config.model.models:
            model_path = models_path / f"{model_name}_model.joblib"
            
            if model_path.exists():
                try:
                    model = joblib.load(model_path)
                    self.models[model_name] = model
                    logger.info(f"{model_name} model loaded successfully")
                except Exception as e:
                    logger.error(f"Failed to load {model_name} model: {e}")
            else:
                logger.warning(f"Model file not found: {model_path}")
        
        logger.info(f"Loaded {len(self.models)} models")
    
    def run_full_pipeline(self, data_path: Optional[str] = None) -> pd.DataFrame:
        """Run the complete forecasting pipeline.
        
        Args:
            data_path: Optional path to data file
            
        Returns:
            Evaluation report DataFrame
        """
        logger.info("Starting full forecasting pipeline...")
        
        try:
            # Load and prepare data
            self.load_and_prepare_data(data_path)
            
            # Train models
            self.train_models()
            
            # Make predictions
            self.make_predictions()
            
            # Evaluate models
            self.evaluate_models()
            
            # Create visualizations
            self.create_visualizations()
            
            # Generate report
            report = self.generate_report()
            
            # Save models
            self.save_models()
            
            logger.info("Full pipeline completed successfully")
            
            return report
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise
    
    def get_best_model(self) -> Tuple[str, BaseForecaster]:
        """Get the best performing model based on MAE.
        
        Returns:
            Tuple of (model_name, model)
        """
        if not self.evaluation_results:
            raise ValueError("Evaluation results must be available")
        
        # Find model with lowest MAE
        best_model_name = min(self.evaluation_results.keys(), 
                            key=lambda x: self.evaluation_results[x].get('mae', float('inf')))
        
        return best_model_name, self.models[best_model_name]
    
    def forecast_future(self, model_name: str, periods: int) -> np.ndarray:
        """Make future forecasts using a specific model.
        
        Args:
            model_name: Name of the model to use
            periods: Number of periods to forecast
            
        Returns:
            Future predictions
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self.models[model_name]
        
        # Use the last available data for forecasting
        if self.test_data is not None:
            last_data = self.test_data.iloc[-1:]
        elif self.val_data is not None:
            last_data = self.val_data.iloc[-1:]
        else:
            last_data = self.train_data.iloc[-1:]
        
        X_last, _ = self.prepare_features(last_data)
        
        # Make predictions
        predictions = model.predict(X_last, periods)
        
        return predictions
