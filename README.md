# Investment Analysis Platform

Multi-agent investment analysis system powered by **LangGraph** and **Anthropic Claude**.

## Overview

This platform performs comprehensive stock analysis using specialized AI agents that collaborate to provide investment insights:

- **Fundamental Analysis Agent**: SEC EDGAR data, financial metrics, ratios
- **Technical Analysis Agent**: Price trends, momentum, technical indicators
- **Synthesis Agent**: Combines analyses into actionable recommendations

## Architecture

```
Investment/
├── agents/                    # LangGraph multi-agent system
│   ├── __init__.py
│   ├── state.py              # Shared state management
│   ├── fundamental_agent.py  # Fundamental analysis agent
│   ├── technical_agent.py    # Technical analysis agent
│   └── workflow.py           # LangGraph orchestration
├── fundamentals/              # Fundamental analysis module
│   ├── analyzer.py           # Main fundamental analyzer
│   ├── edgar_fetch.py        # SEC EDGAR API client
│   ├── xbrl_parse.py         # XBRL financial data parser
│   └── metrics.py            # Financial metrics catalog
├── market/                    # Technical analysis module
│   └── analyzer.py           # Market data and technical indicators
├── prompts/                   # LLM prompt templates
│   ├── agent/                # System prompts for agents
│   └── user/                 # User-facing prompts
├── config.py                  # Centralized configuration
├── load_prompt.py            # Prompt loading utility
├── analyze.py                # Main entry point
├── requirements.txt          # Python dependencies
└── .env                      # Environment variables (API keys)
```

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

Create a `.env` file in the project root:

```bash
# Required
ANTHROPIC_API_KEY=your_api_key_here
USER_EMAIL=your_email@example.com

# Optional
DEFAULT_MODEL=claude-sonnet-4-5-20250929
FAST_MODEL=claude-haiku-4-5
MARKET_LOOKBACK_DAYS=90
```

### 3. Verify Setup

```bash
python -c "from config import config; print('✓ Configuration valid' if config.validate() else '✗ Check your .env file')"
```

## Usage

### Command Line Interface

```bash
# Basic analysis
python analyze.py AAPL

# With current stock price
python analyze.py TSLA --price 250.50

# Focus on specific areas
python analyze.py MSFT --focus profitability growth debt

# Verbose logging
python analyze.py GOOGL -v
```

### Python API

```python
from agents import analyze_stock

# Run analysis
result = analyze_stock(
    ticker="AAPL",
    stock_price=175.50,
    focus_areas=["profitability", "growth"]
)

# Access results
print(result["synthesis_report"])
print(f"Recommendation: {result['final_recommendation']}")
print(f"Confidence: {result['confidence_score']:.0%}")

# Access individual agent results
fundamental = result["fundamental_analysis"]
technical = result["technical_analysis"]
```

### Individual Analyzers

You can also use the analyzers independently:

```python
# Fundamental analysis only
from fundamentals import IntelligentAnalyzer

analyzer = IntelligentAnalyzer()
analysis = analyzer.analyze("AAPL", stock_price=175.50)
print(analyzer.format_report(analysis))

# Technical analysis only
from market import MarketDataAnalyzer

analyzer = MarketDataAnalyzer(lookback_days=90)
analysis = analyzer.analyze("AAPL")
print(analyzer.format_analysis(analysis))
```

## How It Works

### Multi-Agent Workflow

1. **Orchestrator** determines which agents to run based on the request
2. **Fundamental Agent** fetches SEC data and calculates financial metrics
3. **Technical Agent** analyzes price trends and technical indicators
4. **Synthesis Agent** combines all analyses into a final recommendation

The workflow uses **LangGraph** for flexible, graph-based agent orchestration.

### State Management

All agents share a common `AnalysisState` that flows through the workflow:

```python
{
    "ticker": "AAPL",
    "stock_price": 175.50,
    "fundamental_analysis": {...},
    "technical_analysis": {...},
    "final_recommendation": "BUY",
    "confidence_score": 0.75,
    "synthesis_report": "...",
    "completed_agents": ["fundamental", "technical", "synthesis"],
    "errors": []
}
```

## Features

### Fundamental Analysis
- Fetches data from SEC EDGAR API
- Calculates 30+ financial metrics
- Sector-aware metric selection using Claude
- LLM-generated insights and summaries

### Technical Analysis
- Price trends and momentum
- RSI, MACD, Bollinger Bands, Moving Averages
- Support/resistance levels
- Volume analysis
- Overall technical score (-1 to +1)

### Agent Orchestration
- Parallel agent execution where possible
- Graceful error handling
- Extensible architecture for new agents

## Extending the System

### Adding a New Agent

1. Create agent file in `agents/`:

```python
# agents/sentiment_agent.py
from agents.state import AnalysisState

class SentimentAgent:
    def __init__(self):
        self.agent_name = "sentiment"

    def analyze(self, state: AnalysisState) -> AnalysisState:
        # Your analysis logic
        state["sentiment_analysis"] = {...}
        state["completed_agents"].append(self.agent_name)
        return state

def run_sentiment_analysis(state: AnalysisState) -> AnalysisState:
    agent = SentimentAgent()
    return agent.analyze(state)
```

2. Update `agents/workflow.py`:

```python
from agents.sentiment_agent import run_sentiment_analysis

workflow.add_node("sentiment", run_sentiment_analysis)
workflow.add_edge("sentiment", "orchestrator")
```

3. Update orchestrator routing logic

### Adding New Metrics

Edit `fundamentals/metrics.py` to add metrics to `METRICS_CATALOG`:

```python
"your_metric": MetricDefinition(
    name="Your Metric",
    description="What it measures",
    required_inputs=["revenue", "net_income"],
    unit="ratio",
    formula_description="revenue / net_income",
    calculate=lambda data: data["revenue"] / data["net_income"],
    sectors=[Sector.TECHNOLOGY],
    priority=2
)
```

## Dependencies

- **anthropic**: Claude API integration
- **langgraph**: Multi-agent orchestration
- **pandas**: Data manipulation
- **yfinance**: Market data
- **requests**: SEC API calls
- **python-dotenv**: Environment configuration

## Project Status

✅ Fundamental analysis (SEC EDGAR + XBRL)
✅ Technical analysis (price + indicators)
✅ LangGraph multi-agent framework
✅ Agent orchestration and state management
✅ Simple synthesis and recommendations

🚧 Future enhancements:
- Sentiment analysis agent (news, social media)
- Risk assessment agent (portfolio correlation, volatility)
- LLM-powered synthesis (currently rule-based)
- Conversational interface
- Portfolio optimization

## License

Proprietary - Internal use only

## Contact

For questions or issues, contact the development team.
