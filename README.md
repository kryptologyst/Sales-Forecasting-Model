# Sales Forecasting Model

A comprehensive sales forecasting system with multiple algorithms, business metrics, and interactive visualization. This project demonstrates best practices in time series forecasting, model evaluation, and business analytics.

## ⚠️ IMPORTANT DISCLAIMER

**This is a research and educational tool only.**

- This forecasting model is for demonstration and learning purposes
- **Do not use these predictions for automated business decisions without human review**
- Always validate predictions with domain experts and additional data sources
- Consider external factors, market conditions, and business context
- This tool does not guarantee accuracy or suitability for production use

## Quick Start

### Prerequisites

- Python 3.10 or higher
- pip or conda package manager

### Installation

1. Clone the repository:
```bash
git clone https://github.com/kryptologyst/Sales-Forecasting-Model.git
cd Sales-Forecasting-Model
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the forecasting pipeline:
```bash
python scripts/run_pipeline.py
```

4. Launch the interactive demo:
```bash
streamlit run demo/app.py
```

## Features

### Forecasting Models
- **Linear Regression**: Simple trend-based forecasting
- **ARIMA**: Classical time series with seasonality
- **XGBoost**: Gradient boosting for complex patterns
- **LightGBM**: Fast gradient boosting
- **LSTM**: Deep learning for sequential data

### Evaluation Metrics
- **Statistical**: MAE, RMSE, MAPE, SMAPE, MASE
- **Business**: Service level, inventory turns, cost efficiency
- **Model Diagnostics**: Residual analysis, feature importance

### Data Pipeline
- Synthetic data generation for demonstration
- Time series preprocessing and feature engineering
- Proper train/validation/test splitting
- Lag features and rolling statistics

### Visualization
- Interactive forecast comparisons
- Model performance dashboards
- Business metrics analysis
- Feature importance plots

## Project Structure

```
sales-forecasting-model/
├── src/                          # Source code
│   ├── data/                     # Data processing
│   │   └── data_processor.py
│   ├── models/                    # Forecasting models
│   │   └── forecasters.py
│   ├── eval/                     # Evaluation metrics
│   │   └── evaluator.py
│   ├── viz/                      # Visualization
│   │   └── visualizer.py
│   ├── forecasting/              # Main pipeline
│   │   └── pipeline.py
│   └── utils/                    # Utilities
│       ├── config.py
│       └── utils.py
├── configs/                      # Configuration files
│   └── config.yaml
├── scripts/                      # Execution scripts
│   └── run_pipeline.py
├── demo/                         # Streamlit demo
│   └── app.py
├── tests/                        # Unit tests
├── data/                         # Data storage
│   ├── raw/
│   └── processed/
├── outputs/                      # Results and models
├── assets/                       # Generated plots
├── requirements.txt              # Dependencies
├── pyproject.toml               # Project configuration
└── README.md                    # This file
```

## 🔧 Configuration

The system uses YAML configuration files for easy customization. Key settings include:

### Data Configuration
- Data paths and column names
- Train/validation/test ratios
- Synthetic data parameters
- Forecast horizon

### Model Configuration
- Model selection and parameters
- Hyperparameter settings
- Random seeds for reproducibility

### Evaluation Configuration
- Metrics to compute
- Business cost parameters
- Confidence intervals

## Usage Examples

### Basic Usage

```python
from src.utils.config import Config
from src.forecasting.pipeline import SalesForecastingPipeline

# Load configuration
config = Config.from_yaml("configs/config.yaml")

# Initialize pipeline
pipeline = SalesForecastingPipeline(config)

# Run full pipeline
report = pipeline.run_full_pipeline()

# Get best model
best_model_name, best_model = pipeline.get_best_model()
print(f"Best model: {best_model_name}")
```

### Custom Configuration

```python
# Modify configuration
config.data.forecast_horizon = 12
config.model.models = ["arima", "xgboost"]
config.evaluation.stockout_cost = 15.0

# Run with custom settings
pipeline = SalesForecastingPipeline(config)
report = pipeline.run_full_pipeline()
```

### Future Forecasting

```python
# Make future predictions
future_forecast = pipeline.forecast_future("xgboost", 6)
print(f"Next 6 months forecast: {future_forecast}")
```

## Dataset Schema

### Input Data Format

The system expects time series data with the following structure:

```csv
date,sales,month,quarter,year,promotion,holiday,competitor_price
2020-01-31,100.5,1,1,2020,0,0,95.2
2020-02-29,105.3,2,1,2020,1,0,94.8
...
```

### Required Columns
- `date`: Time series date column
- `sales`: Target variable (sales values)

### Optional Columns
- `month`, `quarter`, `year`: Time-based features
- `promotion`, `holiday`: Binary indicators
- `competitor_price`: External factors

### Synthetic Data Generation

If no data file is provided, the system generates synthetic data with:
- Configurable trend (default: 5% monthly growth)
- Seasonal patterns (default: 10% variation)
- Random noise (default: 5% variation)
- Additional features for demonstration

## Business Metrics

The system evaluates models using business-relevant metrics:

### Service Level
- Percentage of demand satisfied without stockouts
- Target: 95% service level

### Inventory Turns
- Annual inventory turnover ratio
- Higher values indicate efficient inventory management

### Cost Metrics
- **Stockout Cost**: Cost per unit when demand exceeds supply
- **Holding Cost**: Cost per unit per month for inventory
- **Total Cost**: Combined overage and underage costs

### Cost Efficiency
- Percentage cost reduction compared to worst-case scenario
- Measures the value of forecasting accuracy

## Model Evaluation

### Statistical Metrics
- **MAE**: Mean Absolute Error
- **RMSE**: Root Mean Square Error
- **MAPE**: Mean Absolute Percentage Error
- **SMAPE**: Symmetric Mean Absolute Percentage Error
- **MASE**: Mean Absolute Scaled Error

### Model Diagnostics
- Residual analysis plots
- Q-Q plots for normality testing
- Feature importance analysis
- Cross-validation results

### Business Impact
- Service level achievement
- Inventory optimization
- Cost reduction analysis
- Risk assessment

## Advanced Features

### Model Comparison
- Automated model selection
- Performance ranking
- Statistical significance testing
- Ensemble methods

### Hyperparameter Tuning
- Grid search optimization
- Bayesian optimization
- Cross-validation
- Early stopping

### Uncertainty Quantification
- Prediction intervals
- Confidence bounds
- Scenario analysis
- Risk assessment

## Testing

Run the test suite:

```bash
pytest tests/ -v
```

Run with coverage:

```bash
pytest tests/ --cov=src --cov-report=html
```

## Development

### Code Quality
- Type hints throughout
- Google/NumPy docstrings
- Black code formatting
- Ruff linting
- Pre-commit hooks

### Contributing
1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Run linting and tests
5. Submit a pull request

## Privacy & Compliance

### Data Handling
- No PII collection or storage
- Synthetic data for demonstration
- Configurable data retention
- Audit trail logging

### Model Governance
- Reproducible experiments
- Version control for models
- Performance monitoring
- Bias detection

## References

### Academic Papers
- Hyndman, R.J. & Athanasopoulos, G. (2021). Forecasting: principles and practice
- Makridakis, S. et al. (2020). The M5 competition: Background, organization, and results

### Technical Documentation
- [Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)

## Support

For questions, issues, or contributions:
- Create an issue on GitHub
- Review the documentation
- Check the examples in `notebooks/`

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Scikit-learn team for machine learning tools
- Plotly team for interactive visualizations
- Streamlit team for the demo framework
- The open-source forecasting community

---

**Remember: This tool is for research and education only. Always validate predictions with domain experts before making business decisions.**
# Sales-Forecasting-Model
