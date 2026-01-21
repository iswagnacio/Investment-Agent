"""
Technical Analysis Agent
Wraps the MarketDataAnalyzer to work within LangGraph workflow
"""

import sys
from pathlib import Path
from typing import Dict, Any
import logging

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.state import AnalysisState
from market import MarketDataAnalyzer, MarketDataAnalysis
from config import config

logger = logging.getLogger(__name__)


class TechnicalAgent:
    """
    Agent that performs technical market analysis.

    Wraps the existing MarketDataAnalyzer to make it compatible
    with the LangGraph multi-agent workflow.
    """

    def __init__(self):
        """Initialize the technical analysis agent"""
        self.analyzer = MarketDataAnalyzer(
            lookback_days=config.MARKET_LOOKBACK_DAYS
        )
        self.agent_name = "technical"

    def analyze(self, state: AnalysisState) -> AnalysisState:
        """
        Perform technical analysis and update state.

        Args:
            state: Current analysis state

        Returns:
            Updated state with technical analysis results
        """
        ticker = state["ticker"]
        logger.info(f"[TechnicalAgent] Analyzing {ticker}")

        try:
            # Run the analysis
            analysis = self.analyzer.analyze(ticker)

            # Convert analysis to dict format for state
            state["technical_analysis"] = self._analysis_to_dict(analysis)
            logger.info(f"[TechnicalAgent] Successfully analyzed {ticker}")

        except Exception as e:
            error_msg = f"Technical analysis error: {str(e)}"
            state.setdefault("errors", []).append(error_msg)
            logger.exception(f"[TechnicalAgent] {error_msg}")

        # Mark this agent as completed
        state.setdefault("completed_agents", []).append(self.agent_name)

        return state

    def _analysis_to_dict(self, analysis: MarketDataAnalysis) -> Dict[str, Any]:
        """
        Convert MarketDataAnalysis object to dictionary.

        Args:
            analysis: Analysis result object

        Returns:
            Dictionary representation suitable for state storage
        """
        return {
            "symbol": analysis.symbol,
            "timestamp": analysis.timestamp.isoformat(),
            "current_price": analysis.current_price,
            "price_change_pct": analysis.price_change_pct,
            "trend": analysis.trend,
            "momentum": analysis.momentum,
            "volatility": analysis.volatility,
            "volume_trend": analysis.volume_trend,
            "overall_score": analysis.overall_score,
            "support_levels": analysis.support_levels,
            "resistance_levels": analysis.resistance_levels,
            "signals": [
                {
                    "indicator": signal.indicator,
                    "signal": signal.signal,
                    "value": signal.value,
                    "description": signal.description,
                    "strength": signal.strength
                }
                for signal in analysis.signals
            ]
        }

    def format_for_synthesis(self, technical_data: Dict[str, Any]) -> str:
        """
        Format technical analysis data for synthesis agent.

        Args:
            technical_data: Dictionary from state['technical_analysis']

        Returns:
            Formatted string summary
        """
        lines = []
        lines.append(f"Current Price: ${technical_data['current_price']:.2f} ({technical_data['price_change_pct']:+.2f}%)")
        lines.append(f"Trend: {technical_data['trend'].upper()}")
        lines.append(f"Momentum: {technical_data['momentum'].upper()}")
        lines.append(f"Volatility: {technical_data['volatility']:.2%} (annualized)")
        lines.append(f"Technical Score: {technical_data['overall_score']:+.2f}")

        if technical_data['signals']:
            lines.append(f"\nTechnical Signals ({len(technical_data['signals'])}):")
            for signal in technical_data['signals'][:5]:  # Limit to top 5
                lines.append(f"  • [{signal['signal'].upper()}] {signal['description']}")

        if technical_data['support_levels']:
            levels_str = ', '.join([f"${x:.2f}" for x in technical_data['support_levels']])
            lines.append(f"\nSupport Levels: {levels_str}")

        if technical_data['resistance_levels']:
            levels_str = ', '.join([f"${x:.2f}" for x in technical_data['resistance_levels']])
            lines.append(f"Resistance Levels: {levels_str}")

        return "\n".join(lines)


# Create singleton instance
technical_agent = TechnicalAgent()


def run_technical_analysis(state: AnalysisState) -> AnalysisState:
    """
    LangGraph node function for technical analysis.

    This is the function that gets called by LangGraph when
    the technical analysis node is reached in the workflow.

    Args:
        state: Current analysis state

    Returns:
        Updated state with technical analysis results
    """
    return technical_agent.analyze(state)
