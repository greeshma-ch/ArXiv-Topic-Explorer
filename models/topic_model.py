import pandas as pd
from typing import Dict, List, Tuple, Any
import logging
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TopicModel:
    """
    Class for performing topic discovery on ArXiv paper abstracts using BERTopic.
    """

    def __init__(self):
        """
        Initialize the TopicModel with embedding model and BERTopic instance.
        """
        try:
            # Load sentence transformer model
            self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

            # Initialize BERTopic with the embedding model
            self.topic_model = BERTopic(
                   embedding_model=self.embedding_model,
                   min_topic_size=5,
                   nr_topics="auto",
                   verbose=True
)

            logger.info("BERTopic model initialized successfully")

        except Exception as e:
            logger.error(f"Error initializing TopicModel: {str(e)}")
            raise

    def fit_topics(self, df: pd.DataFrame) -> Tuple[BERTopic, List[int]]:
        """
        Fit BERTopic model on paper abstracts.

        Args:
            df (pd.DataFrame): DataFrame containing paper data with 'abstract' column

        Returns:
            Tuple[BERTopic, List[int]]: Fitted BERTopic model and topic assignments
        """
        try:
            # Handle empty DataFrame
            if df is None or df.empty:
                logger.warning("Empty DataFrame provided for topic modeling")
                return self.topic_model, []

            # Ensure we have the abstract column
            if 'abstract' not in df.columns:
                raise ValueError("DataFrame must contain 'abstract' column")

            # Get abstracts from DataFrame
            abstracts = df['abstract'].tolist()

            # Remove any empty abstracts
            abstracts = [abs for abs in abstracts if isinstance(abs, str) and len(abs.strip()) > 0]

            if not abstracts:
                logger.warning("No valid abstracts found for topic modeling")
                return self.topic_model, []

            logger.info(f"Fitting topics on {len(abstracts)} abstracts")

            # Fit BERTopic model
            topics, probs = self.topic_model.fit_transform(abstracts)

            logger.info(f"Topic modeling completed. Found {len(set(topics))} topics")
            return self.topic_model, topics

        except Exception as e:
            logger.error(f"Error in fit_topics: {str(e)}")
            raise

    def get_topic_info(self) -> pd.DataFrame:
        """
        Get information about all topics.

        Returns:
            pd.DataFrame: DataFrame with topic information
        """
        try:
            if self.topic_model is None:
                raise ValueError("Topic model not initialized")

            topic_info = self.topic_model.get_topic_info()
            return topic_info

        except Exception as e:
            logger.error(f"Error getting topic info: {str(e)}")
            raise

    def get_topics(self) -> Dict[int, Tuple[str, float]]:
        """
        Get all topics with their representations.

        Returns:
            Dict[int, Tuple[str, float]]: Dictionary mapping topic IDs to (representation, frequency)
        """
        try:
            if self.topic_model is None:
                raise ValueError("Topic model not initialized")

            # Get topics from BERTopic
            topics = self.topic_model.get_topic_info()

            # Create dictionary of topic representations
            topic_dict = {}
            for _, row in topics.iterrows():
                topic_id = int(row['Topic'])
                if topic_id != -1:  # Skip outlier topic
                    representation = row['Name']
                    frequency = row['Count']
                    topic_dict[topic_id] = (representation, frequency)

            return topic_dict

        except Exception as e:
            logger.error(f"Error getting topics: {str(e)}")
            raise

    def get_representative_docs(self, topic_id: int) -> List[str]:
        """
        Get representative documents for a specific topic.

        Args:
            topic_id (int): Topic ID to get representative documents for

        Returns:
            List[str]: List of representative document abstracts
        """
        try:
            if self.topic_model is None:
                raise ValueError("Topic model not initialized")

            # Get representative documents for the topic
            docs = self.topic_model.get_representative_docs(topic_id)
            return docs

        except Exception as e:
            logger.error(f"Error getting representative documents: {str(e)}")
            raise