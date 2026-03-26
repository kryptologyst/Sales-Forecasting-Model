"""Streamlit demo application for sales forecasting."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys
from pathlib import Path
import logging

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.utils.config import Config
from src.utils.utils import setup_logging
from src.forecasting.pipeline import SalesForecastingPipeline
from src.data.data_processor import DataProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Sales Forecasting Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Disclaimer
st.markdown("""
<div class="warning-box">
    <h4>⚠️ IMPORTANT DISCLAIMER</h4>
    <p><strong>This is a research and educational tool only.</strong></p>
    <ul>
        <li>This forecasting model is for demonstration and learning purposes</li>
        <li>Do not use these predictions for automated business decisions without human review</li>
        <li>Always validate predictions with domain experts and additional data sources</li>
        <li>Consider external factors, market conditions, and business context</li>
        <li>This tool does not guarantee accuracy or suitability for production use</li>
    </ul>
</div>
""", unsafe_allow_html=True)

# Main header
st.markdown('<h1 class="main-header">📈 Sales Forecasting Dashboard</h1>', unsafe_allow_html=True)

# Sidebar configuration
st.sidebar.header("Configuration")

# Load configuration
try:
    config_path = Path(__file__).parent.parent / "configs" / "config.yaml"
    config = Config.from_yaml(str(config_path))
except Exception as e:
    st.error(f"Failed to load configuration: {e}")
    st.stop()

# Sidebar controls
st.sidebar.subheader("Model Settings")
selected_models = st.sidebar.multiselect(
    "Select Models to Compare",
    options=config.model.models,
    default=config.model.models[:3]  # Default to first 3 models
)

st.sidebar.subheader("Forecast Settings")
forecast_horizon = st.sidebar.slider(
    "Forecast Horizon (months)",
    min_value=1,
    max_value=12,
    value=config.data.forecast_horizon
)

st.sidebar.subheader("Data Settings")
generate_synthetic = st.sidebar.checkbox(
    "Generate Synthetic Data",
    value=config.data.generate_synthetic
)

if generate_synthetic:
    trend = st.sidebar.slider(
        "Trend (% monthly growth)",
        min_value=0.0,
        max_value=0.2,
        value=config.data.synthetic_trend,
        step=0.01
    )
    
    seasonality = st.sidebar.slider(
        "Seasonality (% variation)",
        min_value=0.0,
        max_value=0.3,
        value=config.data.synthetic_seasonality,
        step=0.01
    )
    
    noise = st.sidebar.slider(
        "Noise (% random variation)",
        min_value=0.0,
        max_value=0.2,
        value=config.data.synthetic_noise,
        step=0.01
    )

# Update config with user selections
config.model.models = selected_models
config.data.forecast_horizon = forecast_horizon
config.data.generate_synthetic = generate_synthetic

if generate_synthetic:
    config.data.synthetic_trend = trend
    config.data.synthetic_seasonality = seasonality
    config.data.synthetic_noise = noise

# Main content
if st.button("🚀 Run Forecasting Pipeline", type="primary"):
    
    with st.spinner("Running forecasting pipeline..."):
        try:
            # Initialize pipeline
            pipeline = SalesForecastingPipeline(config)
            
            # Run pipeline
            report = pipeline.run_full_pipeline()
            
            st.success("Pipeline completed successfully!")
            
            # Display results
            st.header("📊 Results")
            
            # Metrics overview
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                best_model = report.index[0]  # Assuming sorted by MAE
                st.metric("Best Model", best_model)
            
            with col2:
                best_mae = report.loc[best_model, 'mae']
                st.metric("Best MAE", f"{best_mae:.2f}")
            
            with col3:
                best_rmse = report.loc[best_model, 'rmse']
                st.metric("Best RMSE", f"{best_rmse:.2f}")
            
            with col4:
                if 'service_level' in report.columns:
                    best_service = report.loc[best_model, 'service_level']
                    st.metric("Service Level", f"{best_service:.1f}%")
            
            # Detailed results table
            st.subheader("📋 Model Performance Comparison")
            st.dataframe(report.round(4), use_container_width=True)
            
            # Visualizations
            st.header("📈 Visualizations")
            
            # Load data for plotting
            test_data = pipeline.test_data
            predictions = pipeline.predictions
            
            if test_data is not None and predictions:
                # Prepare data for plotting
                viz_data = test_data[[config.data.date_column, config.data.target_column]].copy()
                viz_data.columns = ['date', 'sales']
                
                # Forecast comparison plot
                fig = go.Figure()
                
                # Add actual values
                fig.add_trace(go.Scatter(
                    x=viz_data['date'],
                    y=viz_data['sales'],
                    mode='lines+markers',
                    name='Actual',
                    line=dict(color='black', width=3),
                    marker=dict(size=8)
                ))
                
                # Add predictions
                colors = px.colors.qualitative.Set1
                for i, (model_name, pred_values) in enumerate(predictions.items()):
                    fig.add_trace(go.Scatter(
                        x=viz_data['date'],
                        y=pred_values,
                        mode='lines',
                        name=f'{model_name.title()} Forecast',
                        line=dict(color=colors[i % len(colors)], width=2, dash='dash')
                    ))
                
                fig.update_layout(
                    title="Sales Forecast Comparison",
                    xaxis_title="Date",
                    yaxis_title="Sales",
                    hovermode='x unified',
                    template='plotly_white',
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Model performance comparison
                fig_perf = make_subplots(
                    rows=1, cols=2,
                    subplot_titles=('MAE Comparison', 'RMSE Comparison')
                )
                
                # MAE comparison
                fig_perf.add_trace(
                    go.Bar(x=report.index, y=report['mae'], name='MAE'),
                    row=1, col=1
                )
                
                # RMSE comparison
                fig_perf.add_trace(
                    go.Bar(x=report.index, y=report['rmse'], name='RMSE'),
                    row=1, col=2
                )
                
                fig_perf.update_layout(
                    title="Model Performance Comparison",
                    height=400,
                    showlegend=False,
                    template='plotly_white'
                )
                
                st.plotly_chart(fig_perf, use_container_width=True)
            
            # Future forecasting
            st.header("🔮 Future Forecasting")
            
            if selected_models:
                selected_model = st.selectbox(
                    "Select Model for Future Forecast",
                    options=selected_models
                )
                
                future_periods = st.slider(
                    "Forecast Periods",
                    min_value=1,
                    max_value=12,
                    value=6
                )
                
                if st.button("Generate Future Forecast"):
                    try:
                        future_forecast = pipeline.forecast_future(selected_model, future_periods)
                        
                        # Create future dates
                        last_date = test_data[config.data.date_column].iloc[-1]
                        future_dates = pd.date_range(
                            start=last_date + pd.DateOffset(months=1),
                            periods=future_periods,
                            freq='M'
                        )
                        
                        # Plot future forecast
                        fig_future = go.Figure()
                        
                        # Add historical data
                        fig_future.add_trace(go.Scatter(
                            x=viz_data['date'],
                            y=viz_data['sales'],
                            mode='lines+markers',
                            name='Historical',
                            line=dict(color='black', width=2)
                        ))
                        
                        # Add future forecast
                        fig_future.add_trace(go.Scatter(
                            x=future_dates,
                            y=future_forecast,
                            mode='lines+markers',
                            name=f'{selected_model.title()} Forecast',
                            line=dict(color='red', width=2, dash='dash')
                        ))
                        
                        fig_future.update_layout(
                            title=f"Future Forecast - {selected_model.title()}",
                            xaxis_title="Date",
                            yaxis_title="Sales",
                            template='plotly_white',
                            height=500
                        )
                        
                        st.plotly_chart(fig_future, use_container_width=True)
                        
                        # Display forecast values
                        st.subheader("Forecast Values")
                        forecast_df = pd.DataFrame({
                            'Date': future_dates,
                            'Forecast': future_forecast.round(2)
                        })
                        st.dataframe(forecast_df, use_container_width=True)
                        
                    except Exception as e:
                        st.error(f"Failed to generate future forecast: {e}")
            
            # Feature importance
            if pipeline.feature_importance:
                st.header("🎯 Feature Importance")
                
                selected_model_importance = st.selectbox(
                    "Select Model for Feature Importance",
                    options=list(pipeline.feature_importance.keys())
                )
                
                importance_data = pipeline.feature_importance[selected_model_importance]
                
                if importance_data:
                    # Sort by importance
                    sorted_features = sorted(importance_data.items(), key=lambda x: x[1], reverse=True)
                    top_features = sorted_features[:10]  # Top 10 features
                    
                    features, importance = zip(*top_features)
                    
                    fig_importance = go.Figure(go.Bar(
                        x=list(importance),
                        y=list(features),
                        orientation='h',
                        marker_color='lightblue'
                    ))
                    
                    fig_importance.update_layout(
                        title=f"Top 10 Feature Importance - {selected_model_importance.title()}",
                        xaxis_title="Importance",
                        yaxis_title="Features",
                        template='plotly_white',
                        height=500
                    )
                    
                    st.plotly_chart(fig_importance, use_container_width=True)
        
        except Exception as e:
            st.error(f"Pipeline failed: {e}")
            logger.error(f"Pipeline failed: {e}")

# Information section
st.sidebar.header("ℹ️ Information")
st.sidebar.info("""
This dashboard demonstrates various forecasting models for sales prediction:

- **Linear Regression**: Simple trend-based forecasting
- **ARIMA**: Time series model with seasonality
- **XGBoost**: Gradient boosting for complex patterns
- **LightGBM**: Fast gradient boosting
- **LSTM**: Deep learning for sequential data

The models are evaluated using multiple metrics including MAE, RMSE, MAPE, and business KPIs.
""")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Sales Forecasting Dashboard | Research & Educational Tool</p>
    <p><em>Remember: This is for demonstration purposes only. Always validate predictions with domain experts.</em></p>
</div>
""", unsafe_allow_html=True)
