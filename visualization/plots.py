import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Any, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def plot_topic_growth_leaderboard(trend_df: pd.DataFrame) -> go.Figure:
    """
    Create a leaderboard plot showing topic growth rates.

    Args:
        trend_df (pd.DataFrame): DataFrame with topic trends including 'topic_id', 'growth_rate'

    Returns:
        go.Figure: Plotly figure object for topic growth leaderboard
    """
    try:
        if trend_df is None or trend_df.empty:
            logger.warning("Empty data provided for growth leaderboard")
            return go.Figure()

        # Sort by growth rate descending
        sorted_df = trend_df.sort_values('growth_rate', ascending=False)

        # Create bar chart
        fig = go.Figure(go.Bar(
            x=sorted_df['growth_rate'],
            y=sorted_df['topic_id'],
            orientation='h',
            marker_color=['red' if rate < 0 else 'green' for rate in sorted_df['growth_rate']],
            text=sorted_df['growth_rate'].round(2),
            textposition='auto'
        ))

        fig.update_layout(
            title='Topic Growth Rate Leaderboard',
            xaxis_title='Growth Rate',
            yaxis_title='Topic ID',
            height=600,
            showlegend=False
        )

        return fig

    except Exception as e:
        logger.error(f"Error in plot_topic_growth_leaderboard: {str(e)}")
        raise

def plot_opportunity_rankings(trend_df: pd.DataFrame) -> go.Figure:
    """
    Create a ranking plot showing research opportunity scores.

    Args:
        trend_df (pd.DataFrame): DataFrame with topic trends including 'topic_id', 'opportunity_score'

    Returns:
        go.Figure: Plotly figure object for opportunity rankings
    """
    try:
        if trend_df is None or trend_df.empty:
            logger.warning("Empty data provided for opportunity rankings")
            return go.Figure()

        # Sort by opportunity score descending
        sorted_df = trend_df.sort_values('opportunity_score', ascending=False)

        # Create bar chart with color coding based on opportunity level
        colors = []
        for level in sorted_df['opportunity_level']:
            if level == 'High':
                colors.append('red')
            elif level == 'Medium':
                colors.append('orange')
            elif level == 'Low':
                colors.append('yellow')
            else:
                colors.append('gray')

        fig = go.Figure(go.Bar(
            x=sorted_df['opportunity_score'],
            y=sorted_df['topic_id'],
            orientation='h',
            marker_color=colors,
            text=sorted_df['opportunity_score'].round(1),
            textposition='auto'
        ))

        fig.update_layout(
            title='Research Opportunity Rankings',
            xaxis_title='Opportunity Score (0-10)',
            yaxis_title='Topic ID',
            height=600,
            showlegend=False
        )

        return fig

    except Exception as e:
        logger.error(f"Error in plot_opportunity_rankings: {str(e)}")
        raise

def plot_topic_distribution(trend_df: pd.DataFrame) -> go.Figure:
    """
    Create a distribution plot showing topic sizes and growth.

    Args:
        trend_df (pd.DataFrame): DataFrame with topic trends including 'topic_id', 'recent_count', 'growth_rate'

    Returns:
        go.Figure: Plotly figure object for topic distribution
    """
    try:
        if trend_df is None or trend_df.empty:
            logger.warning("Empty data provided for topic distribution")
            return go.Figure()

        # Create scatter plot
        fig = go.Figure()

        # Add points for each topic
        fig.add_trace(go.Scatter(
            x=trend_df['recent_count'],
            y=trend_df['growth_rate'],
            mode='markers+text',
            marker=dict(
                size=trend_df['recent_count'] * 2,
                color=['red' if rate < 0 else 'green' for rate in trend_df['growth_rate']],
                opacity=0.6
            ),
            text=trend_df['topic_id'],
            textposition="middle center",
            hovertemplate='<b>Topic %{text}</b><br>' +
                         'Size: %{x}<br>' +
                         'Growth Rate: %{y:.2f}<extra></extra>'
        ))

        fig.update_layout(
            title='Topic Distribution (Size vs Growth)',
            xaxis_title='Topic Size (Recent Papers)',
            yaxis_title='Growth Rate',
            height=500
        )

        return fig

    except Exception as e:
        logger.error(f"Error in plot_topic_distribution: {str(e)}")
        raise

def plot_emerging_topics(trend_df: pd.DataFrame) -> go.Figure:
    """
    Create a plot showing emerging topics (positive growth rate).

    Args:
        trend_df (pd.DataFrame): DataFrame with topic trends including 'topic_id', 'growth_rate'

    Returns:
        go.Figure: Plotly figure object for emerging topics
    """
    try:
        if trend_df is None or trend_df.empty:
            logger.warning("Empty data provided for emerging topics")
            return go.Figure()

        # Filter for emerging topics (positive growth rate)
        emerging_df = trend_df[trend_df['growth_rate'] > 0].sort_values('growth_rate', ascending=False)

        if emerging_df.empty:
            logger.info("No emerging topics found")
            return go.Figure()

        # Create horizontal bar chart
        fig = go.Figure(go.Bar(
            x=emerging_df['growth_rate'],
            y=emerging_df['topic_id'],
            orientation='h',
            marker_color='green',
            text=emerging_df['growth_rate'].round(2),
            textposition='auto'
        ))

        fig.update_layout(
            title='Emerging Topics (Positive Growth)',
            xaxis_title='Growth Rate',
            yaxis_title='Topic ID',
            height=500,
            showlegend=False
        )

        return fig

    except Exception as e:
        logger.error(f"Error in plot_emerging_topics: {str(e)}")
        raise
