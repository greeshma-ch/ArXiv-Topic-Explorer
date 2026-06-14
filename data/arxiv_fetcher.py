import arxiv
import pandas as pd
import logging
from typing import List, Dict, Any
from datetime import datetime, timedelta
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fetch_arxiv_papers(search_query: str = "machine learning", max_results: int = 100) -> pd.DataFrame:
    """
    Fetch papers from ArXiv based on search query.

    Args:
        search_query (str): ArXiv search query
        max_results (int): Maximum number of papers to fetch

    Returns:
        pd.DataFrame: DataFrame containing paper information
    """
    try:
        # Create search query with date filtering for recent papers
        search = arxiv.Search(
            query=search_query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate
        )

        papers = []
        for result in search.results():
            paper_info = {
                'id': result.entry_id,
                'title': result.title,
                'authors': ', '.join([author.name for author in result.authors]),
                'abstract': result.summary,
                'published': result.published,
                'categories': ', '.join(result.categories),
                'primary_category': result.primary_category,
                'journal_ref': result.journal_ref or '',
                'doi': result.doi or '',
                'links': [link.href for link in result.links]
            }
            papers.append(paper_info)

        df = pd.DataFrame(papers)
        logger.info(f"Fetched {len(df)} papers from ArXiv")
        return df

    except Exception as e:
        logger.error(f"Error fetching papers: {str(e)}")
        raise

def fetch_recent_papers(days_back: int = 30, max_results: int = 100) -> pd.DataFrame:
    """
    Fetch papers from the last specified number of days.

    Args:
        days_back (int): Number of days to look back
        max_results (int): Maximum number of papers to fetch

    Returns:
        pd.DataFrame: DataFrame containing recent paper information
    """
    try:
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)

        # Format date for ArXiv API
        search_query = f"cat:cs.AI OR cat:cs.LG OR cat:stat.ML"

        search = arxiv.Search(
            query=search_query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate
        )

        papers = []
        for result in search.results():
            # Filter by date range
            if result.published.date() >= start_date.date():
                paper_info = {
                'id': result.entry_id,
                'title': result.title,
                'authors': ', '.join([author.name for author in result.authors]),
                'abstract': result.summary,
                'published': result.published,
                'categories': ', '.join(result.categories),
                'primary_category': result.primary_category,
                'journal_ref': result.journal_ref or '',
                'doi': result.doi or '',
                'links': [link.href for link in result.links]

                }
                papers.append(paper_info)

        df = pd.DataFrame(papers)
        logger.info(f"Fetched {len(df)} recent papers from ArXiv")
        return df

    except Exception as e:
        logger.error(f"Error fetching recent papers: {str(e)}")
        raise

def preprocess_papers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess paper data for topic modeling.

    Args:
        df (pd.DataFrame): Raw paper DataFrame

    Returns:
        pd.DataFrame: Preprocessed DataFrame with combined text
    """
    try:
        # Combine title and abstract for embedding
        df['combined_text'] = df['title'] + ' ' + df['abstract']

        # Remove any rows with empty combined text
        df = df[df['combined_text'].str.len() > 0]

        logger.info(f"Preprocessed papers: {len(df)} valid papers")
        return df

    except Exception as e:
        logger.error(f"Error preprocessing papers: {str(e)}")
        raise

def save_papers_to_csv(df: pd.DataFrame, filename: str = "papers.csv") -> None:
    """
    Save paper DataFrame to CSV file.

    Args:
        df (pd.DataFrame): Paper DataFrame
        filename (str): Output filename
    """
    try:
        os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)
        df.to_csv(filename, index=False)
        logger.info(f"Saved papers to {filename}")

    except Exception as e:
        logger.error(f"Error saving papers to CSV: {str(e)}")
        raise

def load_papers_from_csv(filename: str = "papers.csv") -> pd.DataFrame:
    """
    Load paper DataFrame from CSV file.

    Args:
        filename (str): Input filename

    Returns:
        pd.DataFrame: Loaded paper DataFrame
    """
    try:
        df = pd.read_csv(filename)
        logger.info(f"Loaded {len(df)} papers from {filename}")
        return df

    except Exception as e:
        logger.error(f"Error loading papers from CSV: {str(e)}")
        raise

# Example usage
if __name__ == "__main__":
    # Fetch recent papers
    papers_df = fetch_recent_papers(days_back=7, max_results=50)

    # Preprocess papers
    processed_df = preprocess_papers(papers_df)

    # Save to CSV
    save_papers_to_csv(processed_df, "data/papers.csv")

    print(f"Successfully fetched and processed {len(processed_df)} papers")
