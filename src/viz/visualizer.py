"""Visualization utilities for sales forecasting results."""

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from pathlib import Path

from ..utils.utils import ensure_dir

logger = logging.getLogger(__name__)

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class ForecastingVisualizer:
    """Visualizer for forecasting results and analysis."""
    
    def __init__(self, output_dir: str = "assets"):
        """Initialize visualizer.
        
        Args:
            output_dir: Directory to save plots
        """
        self.output_dir = Path(output_dir)
        ensure_dir(self.output_dir)
        
    def plot_forecast_comparison(self, 
                               df: pd.DataFrame,
                               models: Dict[str, np.ndarray],
                               title: str = "Sales Forecast Comparison",
                               save_path: Optional[str] = None) -> None:
        """Plot comparison of different forecasting models.
        
        Args:
            df: DataFrame with date and actual values
            models: Dictionary with model names and predictions
            title: Plot title
            save_path: Optional path to save plot
        """
        fig, ax = plt.subplots(figsize=(15, 8))
        
        # Plot actual values
        ax.plot(df['date'], df['sales'], 'o-', label='Actual', linewidth=2, markersize=6)
        
        # Plot predictions for each model
        colors = plt.cm.tab10(np.linspace(0, 1, len(models)))
        for i, (model_name, predictions) in enumerate(models.items()):
            ax.plot(df['date'], predictions, '--', 
                   label=f'{model_name.title()} Forecast', 
                   color=colors[i], linewidth=2)
        
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Sales', fontsize=12)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # Rotate x-axis labels
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(self.output_dir / save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Forecast comparison plot saved to {save_path}")
        
        plt.show()
    
    def plot_interactive_forecast(self,
                                df: pd.DataFrame,
                                models: Dict[str, np.ndarray],
                                title: str = "Interactive Sales Forecast") -> go.Figure:
        """Create interactive forecast plot with Plotly.
        
        Args:
            df: DataFrame with date and actual values
            models: Dictionary with model names and predictions
            title: Plot title
            
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        # Add actual values
        fig.add_trace(go.Scatter(
            x=df['date'],
            y=df['sales'],
            mode='lines+markers',
            name='Actual',
            line=dict(color='black', width=3),
            marker=dict(size=8)
        ))
        
        # Add predictions for each model
        colors = px.colors.qualitative.Set1
        for i, (model_name, predictions) in enumerate(models.items()):
            fig.add_trace(go.Scatter(
                x=df['date'],
                y=predictions,
                mode='lines',
                name=f'{model_name.title()} Forecast',
                line=dict(color=colors[i % len(colors)], width=2, dash='dash')
            ))
        
        fig.update_layout(
            title=dict(text=title, font=dict(size=20)),
            xaxis_title='Date',
            yaxis_title='Sales',
            hovermode='x unified',
            template='plotly_white',
            width=1000,
            height=600
        )
        
        return fig
    
    def plot_model_performance(self,
                              evaluation_results: Dict[str, Dict[str, float]],
                              metrics: List[str] = None,
                              title: str = "Model Performance Comparison",
                              save_path: Optional[str] = None) -> None:
        """Plot model performance comparison.
        
        Args:
            evaluation_results: Dictionary with model metrics
            metrics: List of metrics to plot
            title: Plot title
            save_path: Optional path to save plot
        """
        if metrics is None:
            metrics = ['mae', 'rmse', 'mape', 'smape']
        
        # Prepare data
        df_results = pd.DataFrame(evaluation_results).T
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics[:4]):
            if metric in df_results.columns:
                ax = axes[i]
                
                # Sort by metric value
                sorted_results = df_results.sort_values(metric)
                
                bars = ax.bar(range(len(sorted_results)), sorted_results[metric])
                ax.set_title(f'{metric.upper()}', fontweight='bold')
                ax.set_xlabel('Models')
                ax.set_ylabel(metric.upper())
                ax.set_xticks(range(len(sorted_results)))
                ax.set_xticklabels(sorted_results.index, rotation=45)
                
                # Color bars based on performance
                colors = plt.cm.RdYlGn_r(np.linspace(0, 1, len(sorted_results)))
                for bar, color in zip(bars, colors):
                    bar.set_color(color)
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(self.output_dir / save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Model performance plot saved to {save_path}")
        
        plt.show()
    
    def plot_residuals_analysis(self,
                              y_true: np.ndarray,
                              y_pred: np.ndarray,
                              model_name: str = "Model",
                              save_path: Optional[str] = None) -> None:
        """Plot residual analysis for model diagnostics.
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
            model_name: Name of the model
            save_path: Optional path to save plot
        """
        residuals = y_true - y_pred
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Residuals vs Fitted
        axes[0, 0].scatter(y_pred, residuals, alpha=0.6)
        axes[0, 0].axhline(y=0, color='red', linestyle='--')
        axes[0, 0].set_title('Residuals vs Fitted Values')
        axes[0, 0].set_xlabel('Fitted Values')
        axes[0, 0].set_ylabel('Residuals')
        
        # Q-Q Plot
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[0, 1])
        axes[0, 1].set_title('Q-Q Plot of Residuals')
        
        # Histogram of residuals
        axes[1, 0].hist(residuals, bins=20, alpha=0.7, edgecolor='black')
        axes[1, 0].set_title('Distribution of Residuals')
        axes[1, 0].set_xlabel('Residuals')
        axes[1, 0].set_ylabel('Frequency')
        
        # Residuals over time
        axes[1, 1].plot(residuals, alpha=0.7)
        axes[1, 1].axhline(y=0, color='red', linestyle='--')
        axes[1, 1].set_title('Residuals Over Time')
        axes[1, 1].set_xlabel('Time Index')
        axes[1, 1].set_ylabel('Residuals')
        
        plt.suptitle(f'Residual Analysis - {model_name}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(self.output_dir / save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Residual analysis plot saved to {save_path}")
        
        plt.show()
    
    def plot_feature_importance(self,
                               feature_importance: Dict[str, float],
                               model_name: str = "Model",
                               top_n: int = 15,
                               save_path: Optional[str] = None) -> None:
        """Plot feature importance.
        
        Args:
            feature_importance: Dictionary of feature importance scores
            model_name: Name of the model
            top_n: Number of top features to show
            save_path: Optional path to save plot
        """
        if not feature_importance:
            logger.warning("No feature importance data available")
            return
        
        # Sort features by importance
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        top_features = sorted_features[:top_n]
        
        features, importance = zip(*top_features)
        
        plt.figure(figsize=(12, 8))
        bars = plt.barh(range(len(features)), importance)
        plt.yticks(range(len(features)), features)
        plt.xlabel('Feature Importance')
        plt.title(f'Top {top_n} Feature Importance - {model_name}', fontweight='bold')
        plt.gca().invert_yaxis()
        
        # Color bars
        colors = plt.cm.viridis(np.linspace(0, 1, len(features)))
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(self.output_dir / save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Feature importance plot saved to {save_path}")
        
        plt.show()
    
    def plot_business_metrics(self,
                             evaluation_results: Dict[str, Dict[str, float]],
                             title: str = "Business Metrics Comparison",
                             save_path: Optional[str] = None) -> None:
        """Plot business-specific metrics.
        
        Args:
            evaluation_results: Dictionary with model metrics
            title: Plot title
            save_path: Optional path to save plot
        """
        business_metrics = ['service_level', 'inventory_turns', 'total_cost', 'cost_efficiency']
        
        # Filter available metrics
        available_metrics = []
        for metric in business_metrics:
            if any(metric in results for results in evaluation_results.values()):
                available_metrics.append(metric)
        
        if not available_metrics:
            logger.warning("No business metrics available for plotting")
            return
        
        # Prepare data
        df_results = pd.DataFrame(evaluation_results).T
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, metric in enumerate(available_metrics[:4]):
            if metric in df_results.columns:
                ax = axes[i]
                
                # Sort by metric value
                sorted_results = df_results.sort_values(metric)
                
                bars = ax.bar(range(len(sorted_results)), sorted_results[metric])
                ax.set_title(f'{metric.replace("_", " ").title()}', fontweight='bold')
                ax.set_xlabel('Models')
                ax.set_ylabel(metric.replace("_", " ").title())
                ax.set_xticks(range(len(sorted_results)))
                ax.set_xticklabels(sorted_results.index, rotation=45)
                
                # Color bars
                colors = plt.cm.viridis(np.linspace(0, 1, len(sorted_results)))
                for bar, color in zip(bars, colors):
                    bar.set_color(color)
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(self.output_dir / save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Business metrics plot saved to {save_path}")
        
        plt.show()
    
    def create_dashboard(self,
                        df: pd.DataFrame,
                        models: Dict[str, np.ndarray],
                        evaluation_results: Dict[str, Dict[str, float]],
                        title: str = "Sales Forecasting Dashboard") -> go.Figure:
        """Create comprehensive dashboard with multiple plots.
        
        Args:
            df: DataFrame with date and actual values
            models: Dictionary with model names and predictions
            evaluation_results: Dictionary with model metrics
            title: Dashboard title
            
        Returns:
            Plotly figure with subplots
        """
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Forecast Comparison', 'Model Performance (MAE)', 
                          'Model Performance (RMSE)', 'Business Metrics'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Forecast comparison
        fig.add_trace(
            go.Scatter(x=df['date'], y=df['sales'], mode='lines+markers', 
                      name='Actual', line=dict(color='black', width=3)),
            row=1, col=1
        )
        
        colors = px.colors.qualitative.Set1
        for i, (model_name, predictions) in enumerate(models.items()):
            fig.add_trace(
                go.Scatter(x=df['date'], y=predictions, mode='lines',
                          name=f'{model_name.title()} Forecast',
                          line=dict(color=colors[i % len(colors)], width=2, dash='dash')),
                row=1, col=1
            )
        
        # Model performance metrics
        df_results = pd.DataFrame(evaluation_results).T
        
        if 'mae' in df_results.columns:
            fig.add_trace(
                go.Bar(x=df_results.index, y=df_results['mae'], name='MAE'),
                row=1, col=2
            )
        
        if 'rmse' in df_results.columns:
            fig.add_trace(
                go.Bar(x=df_results.index, y=df_results['rmse'], name='RMSE'),
                row=2, col=1
            )
        
        if 'service_level' in df_results.columns:
            fig.add_trace(
                go.Bar(x=df_results.index, y=df_results['service_level'], name='Service Level'),
                row=2, col=2
            )
        
        fig.update_layout(
            title=dict(text=title, font=dict(size=20)),
            height=800,
            showlegend=True,
            template='plotly_white'
        )
        
        return fig
    
    def save_all_plots(self, prefix: str = "forecast") -> None:
        """Save all generated plots with a common prefix.
        
        Args:
            prefix: Prefix for saved files
        """
        logger.info(f"Saving all plots with prefix: {prefix}")
        # This method would be called after generating plots to save them
        # Implementation depends on specific plotting methods used
