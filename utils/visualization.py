import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

def plot_model_comparison(df):
    """
    Create a bar chart comparing models across metrics
    
    Args:
        df: DataFrame with columns Model, Metric, Score
        
    Returns:
        Plotly figure
    """
    fig = px.bar(
        df,
        x="Metric",
        y="Score",
        color="Model",
        barmode="group",
        title="Model Performance Comparison",
        height=500,
    )
    
    fig.update_layout(
        xaxis_title="Metric",
        yaxis_title="Score",
        yaxis=dict(range=[0, 1.05]),
        legend_title="Model",
        legend=dict(orientation="h", y=-0.2),
    )
    
    return fig

def plot_radar_chart(df):
    """
    Create a radar chart for model comparison across metrics
    
    Args:
        df: DataFrame with columns Model, Metric, Score
        
    Returns:
        Plotly figure
    """
    # Pivot data for radar chart
    pivot_df = df.pivot(index="Model", columns="Metric", values="Score").reset_index()
    
    fig = go.Figure()
    
    # Get all metrics from columns
    metrics = [col for col in pivot_df.columns if col != "Model"]
    
    # Add traces for each model
    for i, model in enumerate(pivot_df["Model"]):
        values = pivot_df.iloc[i][metrics].tolist()
        
        # Close the loop by repeating first value
        values.append(values[0])
        metrics_plot = metrics + [metrics[0]]
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=metrics_plot,
            fill='toself',
            name=model
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        title="Model Performance Radar Chart",
        showlegend=True,
        height=600
    )
    
    return fig

def plot_modality_performance(df):
    """
    Create a grouped bar chart for modality performance
    
    Args:
        df: DataFrame with columns Model and modality columns (Text, Image, etc.)
        
    Returns:
        Plotly figure
    """
    # Melt the dataframe to long format
    modalities = [col for col in df.columns if col != "Model"]
    
    if not modalities:
        return go.Figure()
    
    df_melt = pd.melt(
        df,
        id_vars=["Model"],
        value_vars=modalities,
        var_name="Modality",
        value_name="Score"
    )
    
    fig = px.bar(
        df_melt,
        x="Model",
        y="Score",
        color="Modality",
        barmode="group",
        title="Model Performance by Modality",
        height=500,
    )
    
    fig.update_layout(
        xaxis_title="Model",
        yaxis_title="Average Score",
        yaxis=dict(range=[0, 1.05]),
        legend_title="Modality",
        legend=dict(orientation="h", y=-0.2),
    )
    
    return fig

def plot_benchmark_history(df):
    """
    Create a line chart showing benchmark performance over time
    
    Args:
        df: DataFrame with columns Date, Model, Score
        
    Returns:
        Plotly figure
    """
    fig = px.line(
        df,
        x="Date",
        y="Score",
        color="Model",
        markers=True,
        title="Benchmark Performance Over Time",
        height=500,
    )
    
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Score",
        yaxis=dict(range=[0, 1.05]),
        legend_title="Model",
        legend=dict(orientation="h", y=-0.2),
    )
    
    return fig

def plot_metric_distribution(df, metric_name):
    """
    Create a box plot showing distribution of a specific metric
    
    Args:
        df: DataFrame with columns Model and metric values
        metric_name: Name of the metric to plot
        
    Returns:
        Plotly figure
    """
    fig = px.box(
        df,
        x="Model",
        y=metric_name,
        title=f"{metric_name} Distribution by Model",
        height=500,
        points="all",
    )
    
    fig.update_layout(
        xaxis_title="Model",
        yaxis_title=metric_name,
        yaxis=dict(range=[0, 1.05]),
    )
    
    return fig
