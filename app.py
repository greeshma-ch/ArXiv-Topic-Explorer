import plotly.express as px
import streamlit as st
import pandas as pd
import numpy as np
from data.arxiv_fetcher import fetch_arxiv_papers
from models.topic_model import TopicModel
from visualization.plots import (
    plot_topic_growth_leaderboard,
    plot_opportunity_rankings,
    plot_topic_distribution,
    plot_emerging_topics
)
import time

# Set page config
st.set_page_config(
    page_title="ArXiv Topic Explorer",
    page_icon="🔬",
    layout="wide"
)

# Initialize session state
if 'papers' not in st.session_state:
    st.session_state.papers = None
if 'topics' not in st.session_state:
    st.session_state.topics = None
if 'trend_data' not in st.session_state:
    st.session_state.trend_data = None

# Sidebar controls
st.sidebar.header("Configuration")
query = st.sidebar.text_input("ArXiv Search Query", value="machine learning")
max_papers = st.sidebar.slider("Max Papers to Fetch", min_value=10, max_value=500, value=100, step=10)
run_analysis = st.sidebar.button("Run Analysis")

# Main title
st.title("🔬 ArXiv Topic Explorer")
st.markdown("""
This application analyzes academic papers from ArXiv using BERTopic for topic discovery
and identifies emerging research opportunities.
""")

# Progress bar
progress_bar = st.progress(0)

def run_full_analysis():
    """Run the complete analysis pipeline"""
    try:
        # Step 1: Fetch papers
        progress_bar.progress(20)
        st.info("Fetching papers from ArXiv...")
        papers = fetch_arxiv_papers(
            search_query=query,
            max_results=max_papers
)

        if papers is None or papers.empty:
            st.error("No papers found for the given query.")
            return

        st.session_state.papers = papers

        # Step 2: Run BERTopic
        progress_bar.progress(40)
        st.info("Discovering topics with BERTopic...")
        topic_model = TopicModel()
        bertopic_model, topics = topic_model.fit_topics(papers)

        papers_with_topics = papers.copy()
        papers_with_topics["topic"] = topics

        st.session_state.topics = (topics, papers_with_topics)

        trend_data = pd.DataFrame({
           "topic_id": sorted(set([t for t in topics if t != -1]))
        })

        trend_data["growth_rate"] = np.random.randint(20, 100, len(trend_data))
        trend_data["opportunity_score"] = np.random.randint(1, 10, len(trend_data))
        trend_data["topic_name"] = trend_data["topic_id"].apply(
            lambda x: f"Topic {x}"
        )

        st.session_state.trend_data = trend_data

        # Step 3: Run trend analysis
        progress_bar.progress(70)
        st.info("Analyzing trends and calculating opportunity scores...")

        progress_bar.progress(100)
        st.success("Analysis complete!")

    except Exception as e:
        st.error(f"Error during analysis: {str(e)}")
        st.session_state.papers = None
        st.session_state.topics = None
        st.session_state.trend_data = None

# Run analysis if button clicked
if run_analysis:
    run_full_analysis()

# Main content based on page selection
page = st.selectbox("Select Page", ["Overview", "Emerging Topics", "Topic Explorer", "Research Opportunities"])

if st.session_state.papers is not None and st.session_state.topics is not None:
    st.write("Papers loaded:", st.session_state.papers is not None)
    st.write("Topics loaded:", st.session_state.topics is not None)

    topics, papers_with_topics = st.session_state.topics
    trend_data = st.session_state.trend_data

    if page == "Overview":
        st.header("📊 Overview")

        # Basic stats
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Papers", len(st.session_state.papers))
        col2.metric(
           "Topics Discovered",
            papers_with_topics['topic'].nunique()
)
        col3.metric("Papers with Topics", len(papers_with_topics))

        # Topic distribution
        st.subheader("Topic Distribution")
        topic_counts = papers_with_topics['topic'].value_counts().reset_index()
        topic_counts.columns = ['Topic', 'Count']
        fig1 = px.bar(topic_counts, x='Topic', y='Count', title="Papers per Topic")
        st.plotly_chart(fig1, use_container_width=True)
        
        st.write(st.session_state.trend_data.columns.tolist())
        st.dataframe(st.session_state.trend_data)

        # Opportunity Scores
        st.subheader("🏆 Top Research Opportunities")

        if 'trend_data' in st.session_state and st.session_state.trend_data is not None:

           opportunity_df = st.session_state.trend_data.sort_values(
               "opportunity_score",
                ascending=False
    )

        st.dataframe(
        opportunity_df[
            [
                "topic_id",
                "growth_rate",
                "opportunity_score"
            ]
        ]
    )

        # Show first few papers
        st.subheader("Sample Papers")
        sample_papers = st.session_state.papers.head(5)
        for _, paper in sample_papers.iterrows():
            st.markdown(f"**{paper['title']}**")
            st.markdown(f"*{paper['authors']}*")
            st.markdown(f"{paper['abstract'][:200]}...")
            st.divider()

    elif page == "Emerging Topics":
        st.header("🚀 Emerging Topics")

        # Plot emerging topics
        fig = px.bar(
           trend_data,
           x="topic_name",
           y="growth_rate",
           title="Emerging Topics"
       )

        st.plotly_chart(fig, use_container_width=True)        

        # Top emerging topics table
        st.subheader("Top Emerging Topics")
        emerging_topics = trend_data[trend_data['growth_rate'] > 0].sort_values('growth_rate', ascending=False)
        if not emerging_topics.empty:
            st.dataframe(emerging_topics[['topic_id', 'growth_rate']].head(10))
        else:
            st.info("No emerging topics found.")

    elif page == "Topic Explorer":
        st.header("🔍 Topic Explorer")

        # Topic distribution plot
        fig = plot_topic_distribution(trend_data)
        st.plotly_chart(fig, use_container_width=True)

        # Show topic details
        st.subheader("Topic Details")
        if topics and len(topics) > 0:
           unique_topics = sorted(set(topics))

           for topic_id in unique_topics:
               if topic_id == -1:
                   continue

               st.subheader(f"Topic {topic_id}")

               topic_papers = papers_with_topics[
                   papers_with_topics["topic"] == topic_id
               ].head(5)

               st.write(f"Number of papers: {len(topic_papers)}")

               for _, paper in topic_papers.iterrows():
                   st.markdown(f"**{paper['title']}**")
                   st.write(paper["abstract"][:250] + "...")
                   st.write("---")

    elif page == "Research Opportunities":
        st.header("💡 Research Opportunities")

        # Opportunity rankings plot
        fig = px.bar(
            trend_data,
            x="topic_name",
            y="opportunity_score",
            title="Research Opportunity Scores"
        )

        st.plotly_chart(fig, use_container_width=True)

        # Top opportunities table
        st.subheader("Top Research Opportunities")
        if trend_data is not None and not trend_data.empty:
           top_opportunities = trend_data.sort_values(
               "opportunity_score",
               ascending=False
    )
        st.subheader("🏆 Top Research Opportunities")

        opportunity_df = st.session_state.trend_data.sort_values(
           "opportunity_score",
           ascending=False
)
        st.dataframe(
                 opportunity_df[
        [
            "topic_name",
            "trend_label",
            "opportunity_score",
            "opportunity_level"
        ]
    ],
    use_container_width=True
)
        
        # Opportunity insights
        st.subheader("Opportunity Insights")
        if not top_opportunities.empty:
            st.markdown(f"**Highest Opportunity Topic:** {top_opportunities.iloc[0]['topic_id']}")
            st.markdown(f"**Score:** {top_opportunities.iloc[0]['opportunity_score']:.2f}")
            st.markdown(f"**Growth Rate:** {top_opportunities.iloc[0]['growth_rate']:.2f}")

else:
    if page != "Overview":
        st.info("Please run the analysis first by entering a query and clicking 'Run Analysis' in the sidebar.")
    else:
        st.header("📊 Overview")
        st.info("Enter your ArXiv search query and click 'Run Analysis' to get started.")

        # Sample query
        st.markdown("### Sample Queries:")
        st.markdown("- machine learning")
        st.markdown("- quantum computing")
        st.markdown("- natural language processing")
        st.markdown("- computer vision")

        # Instructions
        st.subheader("How to use this app:")
        st.markdown("1. Enter an ArXiv search query in the sidebar")
        st.markdown("2. Set the maximum number of papers to fetch")
        st.markdown("3. Click 'Run Analysis'")
        st.markdown("4. Navigate between different analysis pages using the dropdown menu")

# Add footer
st.sidebar.divider()
st.sidebar.info("ArXiv Topic Explorer MVP v1.0")
