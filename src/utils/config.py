"""Configuration management for sales forecasting model."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from omegaconf import OmegaConf
import yaml
from pathlib import Path


@dataclass
class DataConfig:
    """Configuration for data processing."""
    
    # Data paths
    raw_data_path: str = "data/raw/"
    processed_data_path: str = "data/processed/"
    
    # Time series parameters
    date_column: str = "date"
    target_column: str = "sales"
    frequency: str = "M"  # Monthly
    
    # Data splitting
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    
    # Time series specific
    min_train_length: int = 12  # Minimum months for training
    forecast_horizon: int = 6   # Months to forecast ahead
    
    # Synthetic data generation
    generate_synthetic: bool = True
    synthetic_start_date: str = "2020-01-01"
    synthetic_end_date: str = "2024-12-31"
    synthetic_trend: float = 0.05  # 5% monthly growth
    synthetic_seasonality: float = 0.1  # 10% seasonal variation
    synthetic_noise: float = 0.05  # 5% random noise


@dataclass
class ModelConfig:
    """Configuration for forecasting models."""
    
    # Model selection
    models: List[str] = None
    
    # Common parameters
    random_state: int = 42
    
    # ARIMA parameters
    arima_order: tuple = (1, 1, 1)
    arima_seasonal_order: tuple = (1, 1, 1, 12)
    
    # Prophet parameters
    prophet_yearly_seasonality: bool = True
    prophet_weekly_seasonality: bool = False
    prophet_daily_seasonality: bool = False
    prophet_changepoint_prior_scale: float = 0.05
    
    # XGBoost parameters
    xgb_n_estimators: int = 100
    xgb_max_depth: int = 6
    xgb_learning_rate: float = 0.1
    
    # LSTM parameters
    lstm_sequence_length: int = 12
    lstm_hidden_size: int = 50
    lstm_num_layers: int = 2
    lstm_dropout: float = 0.2
    lstm_epochs: int = 100
    lstm_batch_size: int = 32
    
    def __post_init__(self):
        if self.models is None:
            self.models = ["linear", "arima", "prophet", "xgboost", "lstm"]


@dataclass
class EvaluationConfig:
    """Configuration for model evaluation."""
    
    # Metrics to compute
    metrics: List[str] = None
    
    # Business KPIs
    business_metrics: List[str] = None
    
    # Evaluation settings
    cross_validation_folds: int = 5
    confidence_intervals: bool = True
    confidence_level: float = 0.95
    
    # Cost parameters for business evaluation
    stockout_cost: float = 10.0  # Cost per unit stockout
    holding_cost: float = 1.0    # Cost per unit per month
    service_level_target: float = 0.95  # 95% service level
    
    def __post_init__(self):
        if self.metrics is None:
            self.metrics = ["mae", "rmse", "mape", "smape", "mase"]
        
        if self.business_metrics is None:
            self.business_metrics = ["service_level", "inventory_turns", "stockout_rate"]


@dataclass
class Config:
    """Main configuration class."""
    
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    
    # General settings
    project_name: str = "sales_forecasting"
    output_dir: str = "outputs"
    log_level: str = "INFO"
    
    # Reproducibility
    seed: int = 42
    
    @classmethod
    def from_yaml(cls, config_path: str) -> "Config":
        """Load configuration from YAML file."""
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Convert nested dictionaries to config objects
        if 'data' in config_dict:
            config_dict['data'] = DataConfig(**config_dict['data'])
        if 'model' in config_dict:
            config_dict['model'] = ModelConfig(**config_dict['model'])
        if 'evaluation' in config_dict:
            config_dict['evaluation'] = EvaluationConfig(**config_dict['evaluation'])
        
        return cls(**config_dict)
    
    def to_yaml(self, output_path: str) -> None:
        """Save configuration to YAML file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        config_dict = OmegaConf.structured(self)
        with open(output_path, 'w') as f:
            yaml.dump(OmegaConf.to_yaml(config_dict), f, default_flow_style=False)
    
    def save(self, output_path: str) -> None:
        """Save configuration using OmegaConf."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        config_dict = OmegaConf.structured(self)
        OmegaConf.save(config_dict, output_path)
