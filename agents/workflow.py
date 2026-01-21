"""
LangGraph Workflow for Multi-Agent Investment Analysis
Orchestrates fundamental, technical, and other analysis agents
"""

import sys
from pathlib import Path
from typing import Literal
import logging

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from langgraph.graph import StateGraph, END
from agents.state import AnalysisState, create_initial_state
from agents.fundamental_agent import run_fundamental_analysis
from agents.technical_agent import run_technical_analysis

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def orchestrator_node(state: AnalysisState) -> AnalysisState:
    """
    Orchestrator decides which agent to run next.

    Args:
        state: Current analysis state

    Returns:
        Updated state with next_agent set
    """
    completed = state.get("completed_agents", [])
    logger.info(f"[Orchestrator] Completed agents: {completed}")

    # Define agent execution order
    if "fundamental" not in completed:
        state["next_agent"] = "fundamental"
    elif "technical" not in completed:
        state["next_agent"] = "technical"
    else:
        # All agents complete - move to synthesis
        state["next_agent"] = "synthesis"

    logger.info(f"[Orchestrator] Next agent: {state['next_agent']}")
    return state


def synthesis_node(state: AnalysisState) -> AnalysisState:
    """
    Synthesizes results from all agents into a final recommendation.

    Args:
        state: Current analysis state with all agent results

    Returns:
        Updated state with final recommendation
    """
    from datetime import datetime

    logger.info("[Synthesizer] Generating final recommendation")

    # Import agents for formatting
    from agents.fundamental_agent import fundamental_agent
    from agents.technical_agent import technical_agent

    # Build comprehensive summary
    lines = []
    lines.append("=" * 60)
    lines.append(f"INVESTMENT ANALYSIS REPORT: {state['ticker']}")
    lines.append("=" * 60)
    lines.append("")

    # Add fundamental analysis
    if state.get("fundamental_analysis"):
        lines.append("FUNDAMENTAL ANALYSIS")
        lines.append("-" * 60)
        lines.append(fundamental_agent.format_for_synthesis(state["fundamental_analysis"]))
        lines.append("")

    # Add technical analysis
    if state.get("technical_analysis"):
        lines.append("TECHNICAL ANALYSIS")
        lines.append("-" * 60)
        lines.append(technical_agent.format_for_synthesis(state["technical_analysis"]))
        lines.append("")

    # Simple recommendation logic (can be enhanced with LLM)
    recommendation = _generate_recommendation(state)
    lines.append("RECOMMENDATION")
    lines.append("-" * 60)
    lines.append(recommendation["text"])
    lines.append("")
    lines.append("=" * 60)

    state["synthesis_report"] = "\n".join(lines)
    state["final_recommendation"] = recommendation["rating"]
    state["confidence_score"] = recommendation["confidence"]
    state["analysis_end_time"] = datetime.now()

    logger.info(f"[Synthesizer] Final recommendation: {recommendation['rating']}")

    # Mark synthesis as complete
    state.setdefault("completed_agents", []).append("synthesis")

    return state


def _generate_recommendation(state: AnalysisState) -> dict:
    """
    Generate investment recommendation based on analyses.

    This is a simple rule-based approach. In production, you would
    use an LLM to synthesize all the information intelligently.

    Args:
        state: Analysis state with all results

    Returns:
        Dict with rating, confidence, and text
    """
    score = 0
    factors = []

    # Fundamental factors
    if state.get("fundamental_analysis"):
        fund = state["fundamental_analysis"]
        strengths_count = len(fund.get("key_strengths", []))
        concerns_count = len(fund.get("key_concerns", []))

        if strengths_count > concerns_count:
            score += 1
            factors.append("Strong fundamentals")
        elif strengths_count < concerns_count:
            score -= 1
            factors.append("Weak fundamentals")

    # Technical factors
    if state.get("technical_analysis"):
        tech = state["technical_analysis"]
        tech_score = tech.get("overall_score", 0)
        trend = tech.get("trend", "sideways")

        if tech_score > 0.3 and trend == "uptrend":
            score += 1
            factors.append("Positive technical indicators")
        elif tech_score < -0.3 and trend == "downtrend":
            score -= 1
            factors.append("Negative technical indicators")

    # Determine recommendation
    if score >= 1:
        rating = "BUY"
        confidence = 0.7
    elif score <= -1:
        rating = "SELL"
        confidence = 0.7
    else:
        rating = "HOLD"
        confidence = 0.6

    # Build recommendation text
    text_parts = [
        f"Rating: {rating}",
        f"Confidence: {confidence:.0%}",
        "",
        "Key Factors:"
    ]
    for factor in factors:
        text_parts.append(f"  • {factor}")

    if not factors:
        text_parts.append("  • Mixed signals across analyses")

    return {
        "rating": rating,
        "confidence": confidence,
        "text": "\n".join(text_parts)
    }


def router(state: AnalysisState) -> Literal["fundamental", "technical", "synthesis", "__end__"]:
    """
    Route to the next agent based on state.

    Args:
        state: Current analysis state

    Returns:
        Name of the next node to execute
    """
    next_agent = state.get("next_agent")

    if next_agent == "fundamental":
        return "fundamental"
    elif next_agent == "technical":
        return "technical"
    elif next_agent == "synthesis":
        return "synthesis"
    else:
        return "__end__"


def create_analysis_workflow() -> StateGraph:
    """
    Create the LangGraph workflow for investment analysis.

    Returns:
        Compiled StateGraph ready to execute
    """
    logger.info("Creating analysis workflow...")

    # Create the graph
    workflow = StateGraph(AnalysisState)

    # Add nodes
    workflow.add_node("orchestrator", orchestrator_node)
    workflow.add_node("fundamental", run_fundamental_analysis)
    workflow.add_node("technical", run_technical_analysis)
    workflow.add_node("synthesis", synthesis_node)

    # Set entry point
    workflow.set_entry_point("orchestrator")

    # Add conditional edges from orchestrator
    workflow.add_conditional_edges(
        "orchestrator",
        router,
        {
            "fundamental": "fundamental",
            "technical": "technical",
            "synthesis": "synthesis",
            "__end__": END
        }
    )

    # Each agent returns to orchestrator
    workflow.add_edge("fundamental", "orchestrator")
    workflow.add_edge("technical", "orchestrator")

    # Synthesis ends the workflow
    workflow.add_edge("synthesis", END)

    logger.info("Workflow created successfully")
    return workflow.compile()


def analyze_stock(ticker: str, stock_price: float = None, **kwargs) -> AnalysisState:
    """
    Analyze a stock using the multi-agent workflow.

    Args:
        ticker: Stock ticker symbol
        stock_price: Optional current stock price
        **kwargs: Additional parameters (focus_areas, etc.)

    Returns:
        Final analysis state with all results
    """
    logger.info(f"Starting analysis for {ticker}")

    # Create initial state
    initial_state = create_initial_state(
        ticker=ticker,
        stock_price=stock_price,
        **kwargs
    )

    # Create and run workflow
    workflow = create_analysis_workflow()
    final_state = workflow.invoke(initial_state)

    logger.info(f"Analysis complete for {ticker}")
    return final_state


# Demo/testing
if __name__ == "__main__":
    import sys

    # Get ticker from command line or use default
    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    price = float(sys.argv[2]) if len(sys.argv) > 2 else None

    print(f"\n{'='*60}")
    print(f"Multi-Agent Investment Analysis System")
    print(f"{'='*60}\n")

    # Run analysis
    result = analyze_stock(ticker, stock_price=price)

    # Print results
    if result.get("synthesis_report"):
        print(result["synthesis_report"])

    # Print any errors
    if result.get("errors"):
        print("\nERRORS ENCOUNTERED:")
        for error in result["errors"]:
            print(f"  ⚠ {error}")

    # Print timing
    if result.get("analysis_start_time") and result.get("analysis_end_time"):
        duration = result["analysis_end_time"] - result["analysis_start_time"]
        print(f"\nAnalysis Duration: {duration.total_seconds():.1f} seconds")
