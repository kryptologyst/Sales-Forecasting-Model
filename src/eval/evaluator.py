"""Evaluation metrics and business KPIs for sales forecasting."""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging

from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy import stats

from ..utils.config import EvaluationConfig

logger = logging.getLogger(__name__)


class ForecastingEvaluator:
    """Evaluator for forecasting models with business metrics."""
    
    def __init__(self, config: EvaluationConfig):
        """Initialize evaluator.
        
        Args:
            config: Evaluation configuration
        """
        self.config = config
        
    def calculate_mae(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Mean Absolute Error."""
        return mean_absolute_error(y_true, y_pred)
    
    def calculate_rmse(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Root Mean Square Error."""
        return np.sqrt(mean_squared_error(y_true, y_pred))
    
    def calculate_mape(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Mean Absolute Percentage Error."""
        # Avoid division by zero
        mask = y_true != 0
        if not np.any(mask):
            return np.inf
        
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    
    def calculate_smape(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Symmetric Mean Absolute Percentage Error."""
        numerator = np.abs(y_true - y_pred)
        denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
        
        # Avoid division by zero
        mask = denominator != 0
        if not np.any(mask):
            return np.inf
        
        return np.mean(numerator[mask] / denominator[mask]) * 100
    
    def calculate_mase(self, y_true: np.ndarray, y_pred: np.ndarray, 
                      y_train: np.ndarray) -> float:
        """Calculate Mean Absolute Scaled Error."""
        # Calculate naive forecast error (seasonal naive)
        seasonal_period = 12  # Monthly data
        if len(y_train) < seasonal_period:
            seasonal_period = 1
        
        naive_errors = np.abs(np.diff(y_train[-seasonal_period:]))
        scale = np.mean(naive_errors) if len(naive_errors) > 0 else 1
        
        mae = self.calculate_mae(y_true, y_pred)
        return mae / scale if scale > 0 else np.inf
    
    def calculate_bias(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate forecast bias."""
        return np.mean(y_pred - y_true)
    
    def calculate_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray, 
                          threshold: float = 0.1) -> float:
        """Calculate accuracy within threshold."""
        errors = np.abs((y_true - y_pred) / y_true)
        return np.mean(errors <= threshold) * 100
    
    def calculate_coverage(self, y_true: np.ndarray, y_pred_lower: np.ndarray, 
                          y_pred_upper: np.ndarray) -> float:
        """Calculate prediction interval coverage."""
        mask = (y_true >= y_pred_lower) & (y_true <= y_pred_upper)
        return np.mean(mask) * 100
    
    def calculate_business_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                                 inventory_levels: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Calculate business-specific metrics.
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
            inventory_levels: Optional inventory levels for calculations
            
        Returns:
            Dictionary of business metrics
        """
        metrics = {}
        
        # Service Level (assuming we order based on predictions)
        if inventory_levels is not None:
            stockouts = np.maximum(0, y_true - inventory_levels)
            service_level = 1 - np.mean(stockouts > 0)
            metrics['service_level'] = service_level * 100
            
            # Stockout rate
            metrics['stockout_rate'] = np.mean(stockouts > 0) * 100
            
            # Average stockout quantity
            metrics['avg_stockout_qty'] = np.mean(stockouts[stockouts > 0]) if np.any(stockouts > 0) else 0
        else:
            # Simplified service level calculation
            over_forecast = np.maximum(0, y_pred - y_true)
            under_forecast = np.maximum(0, y_true - y_pred)
            
            # Service level based on forecast accuracy
            total_error = np.sum(over_forecast) + np.sum(under_forecast)
            if total_error > 0:
                service_level = 1 - np.sum(under_forecast) / total_error
                metrics['service_level'] = max(0, service_level) * 100
            else:
                metrics['service_level'] = 100
        
        # Inventory turns (simplified calculation)
        if inventory_levels is not None:
            avg_inventory = np.mean(inventory_levels)
            if avg_inventory > 0:
                metrics['inventory_turns'] = np.sum(y_true) / avg_inventory
            else:
                metrics['inventory_turns'] = 0
        else:
            # Simplified calculation
            metrics['inventory_turns'] = np.sum(y_true) / np.mean(y_pred)
        
        # Cost metrics
        overage_cost = np.sum(np.maximum(0, y_pred - y_true)) * self.config.holding_cost
        underage_cost = np.sum(np.maximum(0, y_true - y_pred)) * self.config.stockout_cost
        total_cost = overage_cost + underage_cost
        
        metrics['total_cost'] = total_cost
        metrics['overage_cost'] = overage_cost
        metrics['underage_cost'] = underage_cost
        
        # Cost efficiency (lower is better)
        baseline_cost = np.sum(y_true) * self.config.stockout_cost  # Worst case: always stockout
        if baseline_cost > 0:
            metrics['cost_efficiency'] = (1 - total_cost / baseline_cost) * 100
        else:
            metrics['cost_efficiency'] = 100
        
        return metrics
    
    def evaluate_model(self, y_true: np.ndarray, y_pred: np.ndarray,
                      y_train: Optional[np.ndarray] = None,
                      y_pred_lower: Optional[np.ndarray] = None,
                      y_pred_upper: Optional[np.ndarray] = None,
                      inventory_levels: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Comprehensive model evaluation.
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
            y_train: Training data for MASE calculation
            y_pred_lower: Lower prediction intervals
            y_pred_upper: Upper prediction intervals
            inventory_levels: Inventory levels for business metrics
            
        Returns:
            Dictionary of evaluation metrics
        """
        metrics = {}
        
        # Statistical metrics
        if "mae" in self.config.metrics:
            metrics['mae'] = self.calculate_mae(y_true, y_pred)
        
        if "rmse" in self.config.metrics:
            metrics['rmse'] = self.calculate_rmse(y_true, y_pred)
        
        if "mape" in self.config.metrics:
            metrics['mape'] = self.calculate_mape(y_true, y_pred)
        
        if "smape" in self.config.metrics:
            metrics['smape'] = self.calculate_smape(y_true, y_pred)
        
        if "mase" in self.config.metrics and y_train is not None:
            metrics['mase'] = self.calculate_mase(y_true, y_pred, y_train)
        
        # Additional metrics
        metrics['bias'] = self.calculate_bias(y_true, y_pred)
        metrics['accuracy_10pct'] = self.calculate_accuracy(y_true, y_pred, 0.1)
        metrics['accuracy_20pct'] = self.calculate_accuracy(y_true, y_pred, 0.2)
        
        # Prediction interval coverage
        if y_pred_lower is not None and y_pred_upper is not None:
            metrics['coverage'] = self.calculate_coverage(y_true, y_pred_lower, y_pred_upper)
        
        # Business metrics
        business_metrics = self.calculate_business_metrics(y_true, y_pred, inventory_levels)
        metrics.update(business_metrics)
        
        return metrics
    
    def create_evaluation_report(self, results: Dict[str, Dict[str, float]]) -> pd.DataFrame:
        """Create evaluation report comparing multiple models.
        
        Args:
            results: Dictionary with model names as keys and metrics as values
            
        Returns:
            DataFrame with evaluation results
        """
        df = pd.DataFrame(results).T
        
        # Round numeric columns
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].round(4)
        
        # Sort by primary metric (MAE)
        if 'mae' in df.columns:
            df = df.sort_values('mae')
        
        return df
    
    def calculate_feature_importance_metrics(self, feature_importance: Dict[str, float]) -> Dict[str, Any]:
        """Calculate metrics for feature importance analysis.
        
        Args:
            feature_importance: Dictionary of feature importance scores
            
        Returns:
            Dictionary of feature importance metrics
        """
        if not feature_importance:
            return {}
        
        importance_values = list(feature_importance.values())
        
        metrics = {
            'num_features': len(feature_importance),
            'max_importance': max(importance_values),
            'min_importance': min(importance_values),
            'mean_importance': np.mean(importance_values),
            'std_importance': np.std(importance_values),
            'top_feature': max(feature_importance, key=feature_importance.get),
            'top_feature_importance': max(importance_values)
        }
        
        # Calculate concentration metrics
        total_importance = sum(importance_values)
        if total_importance > 0:
            normalized_importance = [v / total_importance for v in importance_values]
            metrics['concentration_ratio'] = sum(sorted(normalized_importance, reverse=True)[:3])
            metrics['gini_coefficient'] = self._calculate_gini_coefficient(normalized_importance)
        
        return metrics
    
    def _calculate_gini_coefficient(self, values: List[float]) -> float:
        """Calculate Gini coefficient for concentration measurement."""
        if len(values) == 0:
            return 0
        
        sorted_values = sorted(values)
        n = len(sorted_values)
        cumsum = np.cumsum(sorted_values)
        
        return (n + 1 - 2 * sum((n + 1 - i) * y for i, y in enumerate(cumsum, 1))) / (n * sum(sorted_values))
    
    def generate_model_ranking(self, evaluation_results: Dict[str, Dict[str, float]]) -> Dict[str, int]:
        """Generate model ranking based on multiple metrics.
        
        Args:
            evaluation_results: Dictionary with model names and their metrics
            
        Returns:
            Dictionary with model rankings
        """
        rankings = {}
        
        # Define metric weights (lower is better for most metrics)
        metric_weights = {
            'mae': 0.25,
            'rmse': 0.20,
            'mape': 0.15,
            'smape': 0.15,
            'mase': 0.10,
            'bias': 0.05,
            'total_cost': 0.10
        }
        
        # Calculate weighted scores
        model_scores = {}
        for model_name, metrics in evaluation_results.items():
            score = 0
            total_weight = 0
            
            for metric, weight in metric_weights.items():
                if metric in metrics:
                    # Normalize metric (assuming lower is better)
                    metric_values = [m.get(metric, np.inf) for m in evaluation_results.values()]
                    if metric_values and max(metric_values) > 0:
                        normalized_score = metrics[metric] / max(metric_values)
                        score += weight * normalized_score
                        total_weight += weight
            
            if total_weight > 0:
                model_scores[model_name] = score / total_weight
            else:
                model_scores[model_name] = np.inf
        
        # Rank models
        sorted_models = sorted(model_scores.items(), key=lambda x: x[1])
        for rank, (model_name, _) in enumerate(sorted_models, 1):
            rankings[model_name] = rank
        
        return rankings
