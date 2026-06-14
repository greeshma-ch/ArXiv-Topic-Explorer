import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import logging
from datetime import datetime, timedelta
from collections import Counter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TrendCalculator:
    """
    Class for analyzing topic growth trends and calculating research opportunity scores.
    """

    def __init__(self):
        """
        Initialize the TrendCalculator.
        """
        self.trend_data = None

    def calculate_topic_trends(self, df: pd.DataFrame, topics: List[int],
                             time_window_days: int = 30) -> pd.DataFrame:
        """
        Calculate topic trends based on recent and previous counts.

        Args:
            df (pd.DataFrame): DataFrame with paper data including 'published' column
            topics (List[int]): Topic assignments for each paper
            time_window_days (int): Number of days to consider as recent

        Returns:
            pd.DataFrame: DataFrame with topic trends information
        """
        try:
            # Handle empty data
            if df is None or df.empty or not topics:
                logger.warning("Empty data provided for trend calculation")
                return pd.DataFrame()

            # Ensure we have the required columns
            if 'published' not in df.columns:
                raise ValueError("DataFrame must contain 'published' column")

            # Convert published dates to datetime if needed
            if not isinstance(df['published'].iloc[0], datetime):
                df = df.copy()
                df['published'] = pd.to_datetime(df['published'])

            # Create a copy of the data with topics
            trend_df = df.copy()
            trend_df['topic'] = topics

            # Remove papers without topics or invalid dates
            trend_df = trend_df.dropna(subset=['topic', 'published'])
            trend_df = trend_df[trend_df['topic'] != -1]  # Remove outlier topic

            if trend_df.empty:
                logger.warning("No valid papers for trend calculation")
                return pd.DataFrame()

            # Calculate time windows
            current_date = trend_df['published'].max()
            recent_threshold = current_date - timedelta(days=time_window_days)
            previous_threshold = recent_threshold - timedelta(days=time_window_days)

            # Filter data by time periods
            recent_papers = trend_df[trend_df['published'] >= recent_threshold]
            previous_papers = trend_df[(trend_df['published'] >= previous_threshold) &
                                     (trend_df['published'] < recent_threshold)]

            # Count papers per topic in each period
            recent_counts = Counter(recent_papers['topic'])
            previous_counts = Counter(previous_papers['topic'])

            # Get all unique topics
            all_topics = set(list(recent_counts.keys()) + list(previous_counts.keys()))

            # Calculate trends for each topic
            trends = []

            for topic_id in all_topics:
                recent_count = recent_counts.get(topic_id, 0)
                previous_count = previous_counts.get(topic_id, 0)

                # Calculate growth rate
                if previous_count == 0:
                    growth_rate = 100.0
                else:
                    growth_rate = (
                        (recent_count - previous_count)
                        / previous_count
                    ) * 100

                trends.append({
                    'topic_id': topic_id,
                    'topic_name': f'Topic_{topic_id}',
                    'recent_count': recent_count,
                    'previous_count': previous_count,
                    'growth_rate': growth_rate
                })

            # Create DataFrame
            trend_df = pd.DataFrame(trends)

            logger.info(f"Calculated trends for {len(trend_df)} topics")
            return trend_df

        except Exception as e:
            logger.error(f"Error in calculate_topic_trends: {str(e)}")
            raise

    def classify_topics(self, trend_df: pd.DataFrame) -> pd.DataFrame:
        """
        Classify topics based on their growth rates.

        Args:
            trend_df (pd.DataFrame): DataFrame with topic trends

        Returns:
            pd.DataFrame: DataFrame with trend classifications
        """
        try:
            if trend_df is None or trend_df.empty:
                logger.warning("Empty trend data for classification")
                return trend_df

            # Create a copy to avoid modifying original
            result_df = trend_df.copy()

            # Classify topics based on growth rate
            def classify_growth(rate):
                if pd.isna(rate):
                    return 'Unknown'
                elif rate > 0.30:
                    return 'Emerging'
                elif rate >= -0.30:
                    return 'Stable'
                else:
                    return 'Declining'

            result_df['trend_label'] = result_df['growth_rate'].apply(classify_growth)

            logger.info("Topic classification completed")
            return result_df

        except Exception as e:
            logger.error(f"Error in classify_topics: {str(e)}")
            raise

    def calculate_opportunity_score(self, trend_df: pd.DataFrame,
                                  topic_info: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate research opportunity scores for topics.

        Args:
            trend_df (pd.DataFrame): DataFrame with topic trends
            topic_info (pd.DataFrame): DataFrame with topic information

        Returns:
            pd.DataFrame: DataFrame with opportunity scores and levels
        """
        try:
            if trend_df is None or trend_df.empty:
                logger.warning("Empty trend data for opportunity score calculation")
                return trend_df

            # Create a copy to avoid modifying original
            result_df = trend_df.copy()

            # Get topic sizes (from topic_info if available)
            if 'topic_size' in topic_info.columns:
                topic_sizes = dict(zip(topic_info['Topic'], topic_info['Count']))
            else:
                # If we don't have topic info, use recent counts as proxy for size
                topic_sizes = dict(zip(result_df['topic_id'], result_df['recent_count']))

            # Normalize topic sizes (0-1 scale)
            max_size = max(topic_sizes.values()) if topic_sizes else 1
            normalized_sizes = {k: v / max_size if max_size > 0 else 0
                              for k, v in topic_sizes.items()}

            # Calculate opportunity score
            scores = []
            for _, row in result_df.iterrows():
                topic_id = row['topic_id']

                # Get normalized size
                size = normalized_sizes.get(topic_id, 0)

                # Get growth rate (handle infinity cases)
                growth_rate = row['growth_rate']
                if pd.isna(growth_rate) or np.isinf(growth_rate):
                    growth_rate = 0.0

                # Normalize growth rate to 0-1 scale
                # We'll cap the growth rate for very high values
                capped_growth = min(max(growth_rate, -2), 2)  # Cap between -2 and 2
                normalized_growth = (capped_growth + 2) / 4  # Scale to 0-1

                # Calculate weighted score (40% growth, 30% size, 30% recent activity)
                # Recent activity is the recent count
                recent_activity = row['recent_count']
                max_recent = max(result_df['recent_count']) if not result_df['recent_count'].empty else 1
                normalized_activity = recent_activity / max_recent if max_recent > 0 else 0

                opportunity_score = (0.4 * normalized_growth +
                                   0.3 * size +
                                   0.3 * normalized_activity) * 10

                scores.append(opportunity_score)

            result_df['opportunity_score'] = scores

            # Classify opportunity levels
            def classify_opportunity(score):
                if pd.isna(score):
                    return 'Unknown'
                elif score >= 8:
                    return 'High'
                elif score >= 6:
                    return 'Medium'
                elif score >= 4:
                    return 'Low'
                else:
                    return 'Very Low'

            result_df['opportunity_level'] = result_df['opportunity_score'].apply(classify_opportunity)

            logger.info("Opportunity scores calculated")
            return result_df

        except Exception as e:
            logger.error(f"Error in calculate_opportunity_score: {str(e)}")
            raise

    def get_top_opportunities(self, trend_df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
        """
        Get top N topics sorted by opportunity score.

        Args:
            trend_df (pd.DataFrame): DataFrame with opportunity scores
            n (int): Number of top opportunities to return

        Returns:
            pd.DataFrame: Top N topics sorted by opportunity score
        """
        try:
            if trend_df is None or trend_df.empty:
                logger.warning("Empty trend data for top opportunities")
                return pd.DataFrame()

            # Sort by opportunity score descending
            sorted_df = trend_df.sort_values('opportunity_score', ascending=False)

            # Return top N
            top_opportunities = sorted_df.head(n)

            logger.info(f"Retrieved top {len(top_opportunities)} opportunities")
            return top_opportunities

        except Exception as e:
            logger.error(f"Error in get_top_opportunities: {str(e)}")
            raise
