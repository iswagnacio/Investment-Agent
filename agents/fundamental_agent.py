"""
Fundamental Analysis Agent
Wraps the IntelligentAnalyzer to work within LangGraph workflow
"""

import sys
from pathlib import Path
from typing import Dict, Any, Optional
import logging

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.state import AnalysisState
from fundamentals import IntelligentAnalyzer, IntelligentAnalysis
from config import config

logger = logging.getLogger(__name__)


class FundamentalAgent:
    """
    Agent that performs fundamental financial analysis.

    Wraps the existing IntelligentAnalyzer to make it compatible
    with the LangGraph multi-agent workflow.
    """

    def __init__(self):
        """Initialize the fundamental analysis agent"""
        self.analyzer = IntelligentAnalyzer()
        self.agent_name = "fundamental"

    def analyze(self, state: AnalysisState) -> AnalysisState:
        """
        Perform fundamental analysis and update state.

        Args:
            state: Current analysis state

        Returns:
            Updated state with fundamental analysis results
        """
        ticker = state["ticker"]
        logger.info(f"[FundamentalAgent] Analyzing {ticker}")

        try:
            # Get optional parameters from state
            stock_price = state.get("stock_price")
            focus_areas = state.get("focus_areas")

            # Run the analysis
            analysis = self.analyzer.analyze(
                ticker=ticker,
                stock_price=stock_price,
                focus_areas=focus_areas
            )

            if analysis:
                # Convert analysis to dict format for state
                state["fundamental_analysis"] = self._analysis_to_dict(analysis)
                logger.info(f"[FundamentalAgent] Successfully analyzed {ticker}")
            else:
                error_msg = f"Failed to fetch fundamental data for {ticker}"
                state.setdefault("errors", []).append(error_msg)
                logger.error(f"[FundamentalAgent] {error_msg}")

        except Exception as e:
            error_msg = f"Fundamental analysis error: {str(e)}"
            state.setdefault("errors", []).append(error_msg)
            logger.exception(f"[FundamentalAgent] {error_msg}")

        # Mark this agent as completed
        state.setdefault("completed_agents", []).append(self.agent_name)

        return state

    def _analysis_to_dict(self, analysis: IntelligentAnalysis) -> Dict[str, Any]:
        """
        Convert IntelligentAnalysis object to dictionary.

        Args:
            analysis: Analysis result object

        Returns:
            Dictionary representation suitable for state storage
        """
        return {
            "ticker": analysis.ticker,
            "company_name": analysis.company_name,
            "sector": analysis.sector.value,
            "analysis_date": analysis.analysis_date,
            "summary": analysis.summary,
            "key_strengths": analysis.key_strengths,
            "key_concerns": analysis.key_concerns,
            "sector_specific_notes": analysis.sector_specific_notes,
            "metrics": {
                name: {
                    "name": result.name,
                    "value": result.value,
                    "unit": result.unit,
                    "description": result.description,
                    "formula": result.formula_used,
                    "source": result.data_source,
                    "notes": result.notes
                }
                for name, result in analysis.metric_results.items()
            },
            "selected_metrics": analysis.selected_metrics,
            "available_data": analysis.available_data,
            "missing_data": analysis.missing_data
        }

    def format_for_synthesis(self, fundamental_data: Dict[str, Any]) -> str:
        """
        Format fundamental analysis data for synthesis agent.

        Args:
            fundamental_data: Dictionary from state['fundamental_analysis']

        Returns:
            Formatted string summary
        """
        lines = []
        lines.append(f"Company: {fundamental_data['company_name']} ({fundamental_data['ticker']})")
        lines.append(f"Sector: {fundamental_data['sector']}")
        lines.append(f"\nSummary: {fundamental_data['summary']}")

        if fundamental_data['key_strengths']:
            lines.append("\nStrengths:")
            for strength in fundamental_data['key_strengths']:
                lines.append(f"  • {strength}")

        if fundamental_data['key_concerns']:
            lines.append("\nConcerns:")
            for concern in fundamental_data['key_concerns']:
                lines.append(f"  • {concern}")

        # Include key metrics
        lines.append(f"\nMetrics Calculated: {len([m for m in fundamental_data['metrics'].values() if m['value'] is not None])}")

        return "\n".join(lines)


# Create singleton instance
fundamental_agent = FundamentalAgent()


def run_fundamental_analysis(state: AnalysisState) -> AnalysisState:
    """
    LangGraph node function for fundamental analysis.

    This is the function that gets called by LangGraph when
    the fundamental analysis node is reached in the workflow.

    Args:
        state: Current analysis state

    Returns:
        Updated state with fundamental analysis results
    """
    return fundamental_agent.analyze(state)
