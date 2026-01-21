"""
Shared State Management for Multi-Agent System
Defines the data structures that flow through the LangGraph workflow
"""

from typing import TypedDict, Optional, List, Dict, Any
from datetime import datetime


class AnalysisState(TypedDict, total=False):
    """
    Shared state that flows through all agents in the workflow.

    This state object is passed between agents and accumulates
    analysis results from each specialist agent.
    """

    # Input parameters
    ticker: str
    """Stock ticker symbol to analyze"""

    stock_price: Optional[float]
    """Current stock price (optional, for valuation metrics)"""

    user_query: Optional[str]
    """Original user query or question"""

    focus_areas: Optional[List[str]]
    """Specific areas to focus on (e.g., ['profitability', 'growth'])"""

    # Agent analysis results
    fundamental_analysis: Optional[Dict[str, Any]]
    """Results from fundamental analysis agent"""

    technical_analysis: Optional[Dict[str, Any]]
    """Results from technical analysis agent"""

    sentiment_analysis: Optional[Dict[str, Any]]
    """Results from sentiment analysis agent (future)"""

    risk_assessment: Optional[Dict[str, Any]]
    """Results from risk assessment agent (future)"""

    # Synthesis and final output
    final_recommendation: Optional[str]
    """Final investment recommendation (BUY/HOLD/SELL)"""

    confidence_score: Optional[float]
    """Confidence in the recommendation (0-1)"""

    synthesis_report: Optional[str]
    """Comprehensive report synthesizing all analyses"""

    # Workflow control
    completed_agents: List[str]
    """List of agents that have completed their analysis"""

    next_agent: Optional[str]
    """Next agent to run (for routing)"""

    errors: List[str]
    """Any errors encountered during analysis"""

    # Metadata
    analysis_start_time: Optional[datetime]
    """When the analysis started"""

    analysis_end_time: Optional[datetime]
    """When the analysis completed"""


def create_initial_state(ticker: str, **kwargs) -> AnalysisState:
    """
    Create an initial state object for a new analysis.

    Args:
        ticker: Stock ticker to analyze
        **kwargs: Additional state fields (stock_price, user_query, etc.)

    Returns:
        Initialized AnalysisState
    """
    state = AnalysisState(
        ticker=ticker.upper(),
        completed_agents=[],
        errors=[],
        analysis_start_time=datetime.now(),
    )

    # Add any additional fields
    for key, value in kwargs.items():
        if key in AnalysisState.__annotations__:
            state[key] = value

    return state
